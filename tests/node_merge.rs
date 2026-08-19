//! Integration tests for merging one node into another, against a real
//! Postgres.
//!
//! Every test owns a `tenant_id` derived from its own name, for the reason the
//! node and edge suites give: the suite runs in parallel against one shared
//! database, so isolation cannot come from truncating shared tables. Tenancy is
//! the isolation mechanism the schema already has.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use chrono::{DateTime, TimeZone, Utc};
use mundu::embed::DeterministicEmbedder;
use mundu::error::StoreError;
use mundu::store::edge::{EdgeInput, EdgeOutcome};
use mundu::store::node::{NodeInput, NodeUpsertOutcome};
use mundu::store::search::{SearchMethod, SearchOptions};
use mundu::store::{Provenance, Store};
use sqlx::Row;
use uuid::Uuid;

/// One [`Store`] per test, deliberately not a shared `static`, for the reason
/// the node suite documents: a sqlx pool is bound to the reactor that created
/// it, and every `#[tokio::test]` builds and tears down its own.
async fn store() -> Store {
    common::store().await
}

/// Names a tenant after the test that owns it and clears anything a previous
/// run left behind.
///
/// The order is the reverse of the reference graph: the audit rows point at
/// edges and at nodes, edges point at nodes and at each other, so the leaves
/// come off first.
async fn fresh_tenant(store: &Store, name: &str) -> String {
    let tenant = format!("test-node-merge-{name}");

    for statement in [
        "DELETE FROM node_merge_edge WHERE tenant_id = $1",
        "DELETE FROM node_merge WHERE tenant_id = $1",
        "DELETE FROM edge_rating WHERE tenant_id = $1",
        "DELETE FROM edge WHERE tenant_id = $1",
        "UPDATE node SET merged_into = NULL WHERE tenant_id = $1",
        "DELETE FROM node WHERE tenant_id = $1",
    ] {
        sqlx::query(statement)
            .bind(&tenant)
            .execute(store.pool())
            .await
            .expect("clearing the tenant should succeed");
    }

    tenant
}

fn embedder() -> DeterministicEmbedder {
    DeterministicEmbedder::new()
}

fn anonymous() -> Provenance {
    Provenance::default()
}

/// Mints nodes through the real upsert path, so the ids under test are the ids
/// the server would actually hand an agent.
async fn seed_nodes(store: &Store, tenant: &str, display_names: &[&str]) {
    for name in display_names {
        store
            .upsert_nodes(tenant, &[NodeInput::new(*name)], &embedder(), &anonymous())
            .await
            .expect("seeding nodes should succeed");
    }
}

/// The tombstone redirect as the database holds it.
async fn merged_into(store: &Store, tenant: &str, node_id: &str) -> Option<String> {
    let (merged_into,): (Option<String>,) =
        sqlx::query_as("SELECT merged_into FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(tenant)
            .bind(node_id)
            .fetch_one(store.pool())
            .await
            .expect("the node should be readable");

    merged_into
}

/// Entity resolution never deletes, and neither does a merge. Agents hold
/// `node_id`s across stateless calls, so a deleted id would dangle in a context
/// window nothing can reach or correct. The loser stays in the table as a
/// tombstone and keeps resolving — to its winner, and through a chain of them,
/// because a winner may itself be merged later.
#[tokio::test]
async fn the_loser_is_tombstoned_rather_than_deleted_and_still_resolves() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "tombstoned").await;

    seed_nodes(
        store,
        &tenant,
        &["Legacy Billing", "Billing Service", "Payments Platform"],
    )
    .await;

    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("merging one node into another should succeed");

    let (rows,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(&tenant)
            .bind("legacy-billing")
            .fetch_one(store.pool())
            .await
            .expect("counting should succeed");
    assert_eq!(rows, 1, "the loser must survive as a row, not be deleted");

    assert_eq!(
        merged_into(store, &tenant, "legacy-billing")
            .await
            .as_deref(),
        Some("billing-service"),
        "the tombstone must redirect to the winner"
    );
    assert_eq!(
        merged_into(store, &tenant, "billing-service").await,
        None,
        "the winner must stay live"
    );

    // A second merge lengthens the chain, and resolution has to walk it to the
    // end rather than stopping at the first hop.
    store
        .merge_nodes(
            &tenant,
            "billing-service",
            "payments-platform",
            &anonymous(),
        )
        .await
        .expect("merging the winner onwards should succeed");

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Legacy Billing")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting the retired name should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched {
            node_id: "payments-platform".to_string()
        },
        "follow_merges must walk the whole chain a merge built"
    );
}

/// Every edge the loser was an endpoint of, as the database holds it.
#[derive(Debug, Clone, PartialEq)]
struct EdgeRow {
    edge_id: Uuid,
    source: String,
    target: String,
    relationship_type: String,
    status: String,
    valid_from: DateTime<Utc>,
    valid_until: Option<DateTime<Utc>>,
    superseded_by_edge_id: Option<Uuid>,
}

/// Every edge in a tenant, oldest fact first, so assertions can name rows by
/// position without depending on insertion order.
async fn edges_of(store: &Store, tenant: &str) -> Vec<EdgeRow> {
    let rows = sqlx::query(
        "SELECT edge_id, source_node_id, target_node_id, relationship_type, \
                status::text AS status, valid_from, valid_until, superseded_by_edge_id \
         FROM edge WHERE tenant_id = $1 \
         ORDER BY valid_from, target_node_id, edge_id",
    )
    .bind(tenant)
    .fetch_all(store.pool())
    .await
    .expect("reading the tenant's edges should succeed");

    rows.into_iter()
        .map(|row| EdgeRow {
            edge_id: row.get("edge_id"),
            source: row.get("source_node_id"),
            target: row.get("target_node_id"),
            relationship_type: row.get("relationship_type"),
            status: row.get("status"),
            valid_from: row.get("valid_from"),
            valid_until: row.get("valid_until"),
            superseded_by_edge_id: row.get("superseded_by_edge_id"),
        })
        .collect()
}

/// The one edge with this id.
fn edge(edges: &[EdgeRow], edge_id: Uuid) -> &EdgeRow {
    edges
        .iter()
        .find(|edge| edge.edge_id == edge_id)
        .unwrap_or_else(|| panic!("edge {edge_id} should still exist"))
}

/// Writes edges through the real path and hands back their ids in input order.
async fn create_edges(store: &Store, tenant: &str, inputs: &[EdgeInput]) -> Vec<Uuid> {
    store
        .create_edges(tenant, inputs, &embedder(), &anonymous())
        .await
        .expect("creating edges should succeed")
        .into_iter()
        .map(|outcome| match outcome {
            EdgeOutcome::Created { edge_id, .. } | EdgeOutcome::Backfilled { edge_id, .. } => {
                edge_id
            }
            EdgeOutcome::Rejected { reason } => panic!("the seed edge was rejected: {reason:?}"),
        })
        .collect()
}

/// The database's clock, which is the only one anything in this schema is
/// timestamped against.
async fn server_now(store: &Store) -> DateTime<Utc> {
    let (now,): (DateTime<Utc>,) = sqlx::query_as("SELECT now()")
        .fetch_one(store.pool())
        .await
        .expect("reading the clock should succeed");

    now
}

fn instant(year: i32, month: u32, day: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, 12, 0, 0)
        .single()
        .expect("a valid civil timestamp")
}

/// A merge that left the loser's facts hanging off the tombstone would be the
/// fragmentation it exists to repair, wearing a redirect. Both directions move:
/// the loser is an endpoint, not a subject, so an edge pointing *at* it is
/// exactly as stranded as one pointing out of it.
#[tokio::test]
async fn edges_are_repointed_at_the_winner_as_both_source_and_target() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "repointed").await;

    seed_nodes(
        store,
        &tenant,
        &[
            "Legacy Billing",
            "Billing Service",
            "Postgres",
            "Ada Lovelace",
        ],
    )
    .await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "legacy-billing",
                "postgres",
                "USES",
                "Legacy Billing stores invoices in Postgres",
            ),
            EdgeInput::new(
                "ada-lovelace",
                "legacy-billing",
                "OWNS",
                "Ada owns Legacy Billing",
            ),
        ],
    )
    .await;
    let (out_of_loser, into_loser) = (ids[0], ids[1]);

    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("merging should succeed");

    let edges = edges_of(store, &tenant).await;

    assert_eq!(
        (
            edge(&edges, out_of_loser).source.as_str(),
            edge(&edges, out_of_loser).target.as_str()
        ),
        ("billing-service", "postgres"),
        "an edge out of the loser must now leave the winner"
    );
    assert_eq!(
        (
            edge(&edges, into_loser).source.as_str(),
            edge(&edges, into_loser).target.as_str()
        ),
        ("ada-lovelace", "billing-service"),
        "an edge into the loser must now arrive at the winner"
    );

    let (stranded,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM edge \
         WHERE tenant_id = $1 AND $2 IN (source_node_id, target_node_id)",
    )
    .bind(&tenant)
    .bind("legacy-billing")
    .fetch_one(store.pool())
    .await
    .expect("counting should succeed");
    assert_eq!(stranded, 0, "nothing may be left hanging off the tombstone");
}

/// Plants a node with a given alias list directly, bypassing resolution.
///
/// The upsert path cannot build this state, and that is the point: an alias
/// naming another node resolves onto that node at tier 2, which is the fusion
/// the resolver exists to perform. Two nodes whose names overlap this far are
/// what an operator's backfill, an import, or a resolver run under different
/// thresholds leaves behind — and they are exactly the duplicates a merge has
/// to absorb.
async fn plant_node(
    store: &Store,
    tenant: &str,
    node_id: &str,
    display_name: &str,
    aliases: &[&str],
) {
    let aliases: Vec<String> = aliases.iter().map(|alias| alias.to_string()).collect();

    sqlx::query(
        "INSERT INTO node (tenant_id, node_id, display_name, aliases) VALUES ($1, $2, $3, $4)",
    )
    .bind(tenant)
    .bind(node_id)
    .bind(display_name)
    .bind(&aliases)
    .execute(store.pool())
    .await
    .expect("planting a node should succeed");
}

/// Reads the alias array off a node.
async fn aliases_of(store: &Store, tenant: &str, node_id: &str) -> Vec<String> {
    let (aliases,): (Vec<String>,) =
        sqlx::query_as("SELECT aliases FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(tenant)
            .bind(node_id)
            .fetch_one(store.pool())
            .await
            .expect("the node should be readable");

    aliases
}

/// The loser's names are the whole reason the duplicate was reachable: some
/// agent called the entity that, and the next agent to call it that must land
/// on the winner. So every surface form the loser answered to — its display
/// name included, because after the merge it is a name for the winner and
/// nothing else — moves across.
///
/// Union, not concatenation. The two nodes were duplicates, so their alias
/// lists overlap by construction, and a merge that appended blindly would leave
/// the winner carrying the same name twice. Case-insensitively, by the rule the
/// upsert path already uses: a node must not carry an alias that merely recases
/// a name it already answers to.
#[tokio::test]
async fn the_losers_names_are_unioned_onto_the_winner_without_duplicates() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "aliases").await;

    store
        .upsert_nodes(
            &tenant,
            &[NodeInput::with_aliases(
                "Billing Service",
                ["Billing Svc", "Shared Alias"],
            )],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("seeding the winner should succeed");
    plant_node(
        store,
        &tenant,
        "legacy-billing",
        "Legacy Billing",
        &["Old Billing", "shared alias", "BILLING SERVICE"],
    )
    .await;

    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("merging should succeed");

    assert_eq!(
        aliases_of(store, &tenant, "billing-service").await,
        vec![
            "Billing Svc".to_string(),
            "Shared Alias".to_string(),
            "Legacy Billing".to_string(),
            "Old Billing".to_string(),
        ],
        "the loser's display name and its unseen aliases append, in order; \
         `shared alias` recases one the winner already has and `BILLING SERVICE` \
         recases the winner's own display name, so neither is written twice"
    );

    assert_eq!(
        aliases_of(store, &tenant, "legacy-billing").await,
        vec![
            "Old Billing".to_string(),
            "shared alias".to_string(),
            "BILLING SERVICE".to_string(),
        ],
        "nothing is stripped from the tombstone; the merge adds, it does not move"
    );
}

/// **The invariant a merge can break, and the reason this file exists.**
///
/// `create_edges` enforces at most one *active* edge per
/// `(tenant, source_node_id, relationship_type)` for a `one`-cardinality type,
/// and it resolves a contest by `valid_from`: the later fact is current, the
/// earlier becomes closed history pointing at its successor. Two such edges can
/// be legally active at the same instant as long as they hang off two different
/// source nodes — and a merge puts them on the same one. Nothing in
/// `create_edges` notices, because that code only runs on the write path.
///
/// So the merge has to resolve it, by the same rule and not a second one that
/// looks like it. Both directions are asserted: in one group the winner already
/// held the later fact, in the other the loser did. A rule that quietly favoured
/// the winner's own edge, or the one that arrived first, would pass one of these
/// and fail the other.
#[tokio::test]
async fn colliding_one_cardinality_edges_are_resolved_by_valid_from() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "cardinality-one").await;

    seed_nodes(
        store,
        &tenant,
        &[
            "Legacy Billing",
            "Billing Service",
            "Alice",
            "Bob",
            "Carol",
            "London",
            "Berlin",
        ],
    )
    .await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            // The loser's own history, already resolved against itself by
            // `create_edges`: carol is current, alice is closed behind her.
            EdgeInput::with_valid_from(
                "legacy-billing",
                "alice",
                "REPORTS_TO",
                "Legacy Billing reported to Alice in 2019",
                instant(2019, 3, 1),
            ),
            EdgeInput::with_valid_from(
                "legacy-billing",
                "carol",
                "REPORTS_TO",
                "Legacy Billing reported to Carol from 2021",
                instant(2021, 5, 1),
            ),
            // The winner's, holding the later fact.
            EdgeInput::with_valid_from(
                "billing-service",
                "bob",
                "REPORTS_TO",
                "Billing Service reports to Bob as of 2024",
                instant(2024, 6, 1),
            ),
            // And a group the other way round: the loser holds the later fact.
            EdgeInput::with_valid_from(
                "billing-service",
                "berlin",
                "LOCATED_IN",
                "Billing Service ran out of Berlin in 2019",
                instant(2019, 3, 1),
            ),
            EdgeInput::with_valid_from(
                "legacy-billing",
                "london",
                "LOCATED_IN",
                "Legacy Billing moved to London in 2024",
                instant(2024, 6, 1),
            ),
        ],
    )
    .await;
    let (alice, carol, bob, berlin, london) = (ids[0], ids[1], ids[2], ids[3], ids[4]);

    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("merging should succeed");

    let edges = edges_of(store, &tenant).await;

    // Group one: the winner's 2024 fact is the later one, so it stands.
    assert_eq!(
        edge(&edges, bob).status,
        "active",
        "the later fact keeps the current slot"
    );
    assert_eq!(edge(&edges, bob).valid_until, None);
    assert_eq!(edge(&edges, bob).superseded_by_edge_id, None);

    assert_eq!(
        (
            edge(&edges, carol).status.as_str(),
            edge(&edges, carol).valid_until,
            edge(&edges, carol).superseded_by_edge_id,
        ),
        ("superseded", Some(instant(2024, 6, 1)), Some(bob)),
        "the loser's current fact becomes closed history against its successor, \
         with valid_until set to that successor's valid_from"
    );

    assert_eq!(
        (
            edge(&edges, alice).status.as_str(),
            edge(&edges, alice).valid_until,
            edge(&edges, alice).superseded_by_edge_id,
        ),
        ("superseded", Some(instant(2021, 5, 1)), Some(carol)),
        "history that was already closed is not re-closed; only the active \
         edges contend, so alice still points at carol"
    );

    // Group two: this time the *loser* held the later fact, and it wins anyway.
    assert_eq!(
        edge(&edges, london).status,
        "active",
        "valid_from decides, not which side of the merge the edge came from"
    );
    assert_eq!(edge(&edges, london).valid_until, None);
    assert_eq!(edge(&edges, london).superseded_by_edge_id, None);

    assert_eq!(
        (
            edge(&edges, berlin).status.as_str(),
            edge(&edges, berlin).valid_until,
            edge(&edges, berlin).superseded_by_edge_id,
        ),
        ("superseded", Some(instant(2024, 6, 1)), Some(london)),
        "the winner's own earlier fact is closed exactly as the loser's would be"
    );

    // And the invariant itself, stated once against the table rather than
    // inferred from the rows above.
    let (violations,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM ( \
             SELECT 1 FROM edge e \
             WHERE e.tenant_id = $1 AND e.status = 'active' \
               AND EXISTS (SELECT 1 FROM relationship_type rt \
                           WHERE rt.name = e.relationship_type \
                             AND rt.tenant_id IN ('', $1) \
                             AND rt.cardinality = 'one') \
             GROUP BY e.source_node_id, e.relationship_type \
             HAVING count(*) > 1) AS broken",
    )
    .bind(&tenant)
    .fetch_one(store.pool())
    .await
    .expect("counting should succeed");
    assert_eq!(
        violations, 0,
        "no (source, one-cardinality type) may hold two active edges after a merge"
    );
}

/// **Self-edges are archived, and this is the case that decides it.**
///
/// An edge `loser -> winner` becomes `winner -> winner`. The fact did not
/// become false; it became unstatable. "Legacy Billing depends on Billing
/// Service" was a claim about two entities, and once they are one entity it is
/// a claim about nothing — a relationship needs two ends to be a relationship.
///
/// Leaving it active is not neutral, which is what settles it. Under a
/// `one`-cardinality type the tautology competes for the current slot on
/// `valid_from` like any other fact, and here it would *win*: the self-edge is
/// dated 2024 and the real fact 2019, so the graph would answer "who does
/// Billing Service report to" with "itself", and the true answer would be
/// closed history. Archiving is also the one disposition that keeps the row —
/// its proof, its writer, its dates — so the merge stays reversible, which
/// deleting would not.
///
/// This is not the ratings path's archival, which `rating.rs` refuses to
/// automate because `proof_excerpt` is untrusted text and an agent-reachable
/// erasure primitive is a hazard. Nothing here reads the proof. The trigger is
/// that two endpoint ids became equal, which is a fact about the merge an
/// operator asked for.
#[tokio::test]
async fn an_edge_that_the_merge_makes_self_referential_is_archived() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "self-edges").await;

    seed_nodes(
        store,
        &tenant,
        &["Legacy Billing", "Billing Service", "Bob"],
    )
    .await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "legacy-billing",
                "billing-service",
                "DEPENDS_ON",
                "Legacy Billing calls into Billing Service",
            ),
            // A `one`-cardinality edge across the pair, dated later than the
            // winner's real fact. If it survived the merge it would take the
            // current slot and answer the question with a tautology.
            EdgeInput::with_valid_from(
                "legacy-billing",
                "billing-service",
                "REPORTS_TO",
                "Legacy Billing rolled up to Billing Service from 2024",
                instant(2024, 6, 1),
            ),
            EdgeInput::with_valid_from(
                "billing-service",
                "bob",
                "REPORTS_TO",
                "Billing Service reported to Bob from 2019",
                instant(2019, 3, 1),
            ),
            // A fact the loser asserted about itself, which is a self-edge on
            // both sides of the merge rather than only after it.
            EdgeInput::new(
                "legacy-billing",
                "legacy-billing",
                "RELATED_TO",
                "Legacy Billing is cross-referenced with itself in the runbook",
            ),
        ],
    )
    .await;
    let (depends, rolls_up, reports, loops_back) = (ids[0], ids[1], ids[2], ids[3]);

    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("merging should succeed");

    let edges = edges_of(store, &tenant).await;

    for (edge_id, what) in [
        (depends, "loser -> winner under a many-cardinality type"),
        (rolls_up, "loser -> winner under a one-cardinality type"),
        (loops_back, "loser -> loser"),
    ] {
        let row = edge(&edges, edge_id);
        assert_eq!(
            (row.source.as_str(), row.target.as_str()),
            ("billing-service", "billing-service"),
            "{what} must still be re-pointed, not skipped"
        );
        assert_eq!(
            row.status, "archived",
            "{what} became a claim about one entity, so it is archived"
        );
        assert_eq!(
            row.valid_until, None,
            "{what} was not superseded by a later fact; nothing says when it \
             stopped holding, so valid_until stays empty"
        );
        assert_eq!(row.superseded_by_edge_id, None);
    }

    assert_eq!(
        edge(&edges, reports).status,
        "active",
        "the real fact keeps the current slot; a tautology dated later must not \
         take it"
    );
    assert_eq!(edge(&edges, reports).target, "bob");
}

/// Writes one edge under a named principal, so two assertions of the same fact
/// can be told apart by who made them.
async fn create_edge_as(store: &Store, tenant: &str, principal: &str, input: EdgeInput) -> Uuid {
    let provenance = Provenance {
        client: None,
        principal: Some(principal.to_string()),
    };

    match store
        .create_edges(tenant, &[input], &embedder(), &provenance)
        .await
        .expect("creating an edge should succeed")
        .remove(0)
    {
        EdgeOutcome::Created { edge_id, .. } | EdgeOutcome::Backfilled { edge_id, .. } => edge_id,
        EdgeOutcome::Rejected { reason } => panic!("the seed edge was rejected: {reason:?}"),
    }
}

/// **Exact duplicates under a `many` type are both kept, and that is a
/// decision rather than an omission.**
///
/// `loser -USES-> Postgres` and `winner -USES-> Postgres` become one edge
/// written twice. It is legal — `many` cardinality exists precisely because the
/// schema declines to call two assertions under one type a conflict — but it is
/// now the same claim standing on two proofs.
///
/// They stay. The proof and the writer are the whole value this store holds:
/// search returns `proof_excerpt`, ratings attach to an `edge_id`, and the
/// dispute quorum counts *distinct principals*. Collapsing two attestations
/// into one would throw away one quotation and one witness, and there is no
/// principled way to choose which — that is not deduplication, it is evidence
/// destruction, and it is the one thing a merge must not do quietly. Two rows
/// saying the same thing corroborate each other; one row does not.
///
/// It also keeps the merge reversible, which is the precondition for a worker
/// doing this unattended.
///
/// Note the asymmetry with a `one`-cardinality duplicate, which *is* collapsed
/// — see `colliding_one_cardinality_edges_are_resolved_by_valid_from`. That is
/// not a second policy: `place` closes an exact tie as a zero-length interval,
/// and a merge inherits the write path's rule rather than choosing one.
#[tokio::test]
async fn exact_duplicates_under_a_many_type_both_survive_the_merge() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "duplicates").await;

    seed_nodes(
        store,
        &tenant,
        &["Legacy Billing", "Billing Service", "Postgres"],
    )
    .await;

    let from_loser = create_edge_as(
        store,
        &tenant,
        "agent-one",
        EdgeInput::with_valid_from(
            "legacy-billing",
            "postgres",
            "USES",
            "Legacy Billing writes invoices to Postgres",
            instant(2024, 6, 1),
        ),
    )
    .await;
    let from_winner = create_edge_as(
        store,
        &tenant,
        "agent-two",
        EdgeInput::with_valid_from(
            "billing-service",
            "postgres",
            "USES",
            "the billing service persists to Postgres",
            instant(2024, 6, 1),
        ),
    )
    .await;

    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("merging should succeed");

    let edges = edges_of(store, &tenant).await;

    for edge_id in [from_loser, from_winner] {
        let row = edge(&edges, edge_id);
        assert_eq!(
            (
                row.source.as_str(),
                row.target.as_str(),
                row.status.as_str()
            ),
            ("billing-service", "postgres", "active"),
            "both attestations survive, both current"
        );
        assert_eq!(row.superseded_by_edge_id, None);
    }

    let proofs: Vec<(String, Option<String>)> = sqlx::query_as(
        "SELECT proof_excerpt, writer_principal FROM edge \
         WHERE tenant_id = $1 AND edge_id = ANY($2::uuid[]) ORDER BY writer_principal",
    )
    .bind(&tenant)
    .bind(vec![from_loser, from_winner])
    .fetch_all(store.pool())
    .await
    .expect("reading the proofs should succeed");

    assert_eq!(
        proofs,
        vec![
            (
                "Legacy Billing writes invoices to Postgres".to_string(),
                Some("agent-one".to_string())
            ),
            (
                "the billing service persists to Postgres".to_string(),
                Some("agent-two".to_string())
            ),
        ],
        "two witnesses and two quotations, neither discarded"
    );
}

/// Points one node's tombstone at another directly.
///
/// Used to build states `merge_nodes` refuses to build, so the refusals can be
/// tested against a graph that really is in that shape.
async fn plant_merge(store: &Store, tenant: &str, from: &str, into: &str) {
    sqlx::query("UPDATE node SET merged_into = $3 WHERE tenant_id = $1 AND node_id = $2")
        .bind(tenant)
        .bind(from)
        .bind(into)
        .execute(store.pool())
        .await
        .expect("planting a merge should succeed");
}

/// `merged_into` is `NULL` for every node in the tenant. Asserted after every
/// refusal: a merge that reported failure and tombstoned something anyway is
/// worse than one that failed loudly.
async fn nothing_is_tombstoned(store: &Store, tenant: &str) -> bool {
    let (count,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM node WHERE tenant_id = $1 AND merged_into IS NOT NULL",
    )
    .bind(tenant)
    .fetch_one(store.pool())
    .await
    .expect("counting should succeed");

    count == 0
}

/// The schema's `node_merge_not_self` CHECK would catch this as a constraint
/// violation, which is a database error rather than an answer. A caller that
/// asked for something incoherent should be told what it asked for.
#[tokio::test]
async fn merging_a_node_into_itself_is_refused() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "self").await;

    seed_nodes(store, &tenant, &["Billing Service"]).await;

    let error = store
        .merge_nodes(&tenant, "billing-service", "billing-service", &anonymous())
        .await
        .expect_err("a node cannot be merged into itself");

    assert!(
        matches!(&error, StoreError::SelfMerge { node_id, .. } if node_id == "billing-service"),
        "expected SelfMerge naming the node, got {error:?}"
    );
    assert!(nothing_is_tombstoned(store, &tenant).await);
}

/// Tenancy is derived from a validated credential, never from an argument, so
/// a merge is scoped to one tenant by construction and the other tenant's node
/// is simply not there.
///
/// It is reported as an unknown node rather than as a cross-tenant refusal, and
/// that wording is the point. A distinct error would answer "does this id exist
/// somewhere else in this database" for any id a caller cared to try, which is
/// a cross-tenant existence oracle assembled out of an error type — the same
/// reasoning `RatingRejection` gives for having exactly one variant.
#[tokio::test]
async fn merging_across_tenants_is_refused_as_an_unknown_node() {
    let store = &store().await;
    let acme = fresh_tenant(store, "cross-acme").await;
    let globex = fresh_tenant(store, "cross-globex").await;

    seed_nodes(store, &acme, &["Legacy Billing"]).await;
    seed_nodes(store, &globex, &["Billing Service"]).await;

    let error = store
        .merge_nodes(&acme, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect_err("the winner belongs to another tenant");

    assert!(
        matches!(
            &error,
            StoreError::UnknownMergeNode { tenant, node_id, role }
                if tenant == &acme && node_id == "billing-service" && *role == "winner"
        ),
        "expected an unknown-node refusal naming the caller's own tenant, got {error:?}"
    );
    assert!(nothing_is_tombstoned(store, &acme).await);
    assert!(
        nothing_is_tombstoned(store, &globex).await,
        "the other tenant's graph must not be touched at all"
    );
}

/// Either end may be the missing one, and the refusal has to say which: a
/// caller told only "unknown node" cannot tell whether it mistyped the id it is
/// retiring or the one it is keeping, and those are opposite mistakes.
#[tokio::test]
async fn an_id_this_tenant_does_not_have_is_refused_by_role() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "unknown").await;

    seed_nodes(store, &tenant, &["Billing Service"]).await;

    let error = store
        .merge_nodes(&tenant, "no-such-node", "billing-service", &anonymous())
        .await
        .expect_err("the loser does not exist");
    assert!(
        matches!(
            &error,
            StoreError::UnknownMergeNode { node_id, role, .. }
                if node_id == "no-such-node" && *role == "loser"
        ),
        "expected the loser to be named, got {error:?}"
    );

    let error = store
        .merge_nodes(&tenant, "billing-service", "no-such-node", &anonymous())
        .await
        .expect_err("the winner does not exist");
    assert!(
        matches!(
            &error,
            StoreError::UnknownMergeNode { node_id, role, .. }
                if node_id == "no-such-node" && *role == "winner"
        ),
        "expected the winner to be named, got {error:?}"
    );

    assert!(nothing_is_tombstoned(store, &tenant).await);
}

/// **The cycle this call could actually create.**
///
/// `A` is already merged into `B`. Merging `B` into `A` would write `B -> A`
/// alongside the `A -> B` that is already there, and `follow_merges` would then
/// walk between them forever — an unbounded loop inside an open transaction,
/// which does not fail a request so much as take the process with it. That is
/// the failure `follow_merges` guards against on the *read* side; this is the
/// write that would have caused it.
#[tokio::test]
async fn a_merge_that_would_close_a_cycle_is_refused() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "cycle").await;

    seed_nodes(store, &tenant, &["Legacy Billing", "Billing Service"]).await;

    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("the first merge should succeed");

    let error = store
        .merge_nodes(&tenant, "billing-service", "legacy-billing", &anonymous())
        .await
        .expect_err("merging the winner back into its own tombstone closes a loop");

    assert!(
        matches!(
            &error,
            StoreError::MergeCycle { tenant: named, node_id }
                if named == &tenant && node_id == "billing-service"
        ),
        "expected MergeCycle naming the node that would have been tombstoned, got {error:?}"
    );

    assert_eq!(
        merged_into(store, &tenant, "billing-service").await,
        None,
        "the winner of the first merge must still be live"
    );
}

/// A `merged_into` graph that is already a loop — two merges each reasonable on
/// its own, or an operator's UPDATE — is not something this call tries to act
/// inside. `follow_merges` raises on it, and the merge propagates that rather
/// than catching it and guessing which way round the loop the caller meant.
#[tokio::test]
async fn a_merge_inside_an_existing_cycle_is_refused_rather_than_followed() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "existing-cycle").await;

    seed_nodes(store, &tenant, &["Alpha", "Beta", "Gamma"]).await;
    plant_merge(store, &tenant, "alpha", "beta").await;
    plant_merge(store, &tenant, "beta", "alpha").await;

    let error = store
        .merge_nodes(&tenant, "gamma", "alpha", &anonymous())
        .await
        .expect_err("resolution cannot pick a winner inside a loop");

    assert!(
        matches!(error, StoreError::MergeCycle { .. }),
        "expected MergeCycle, got {error:?}"
    );
    assert_eq!(
        merged_into(store, &tenant, "gamma").await,
        None,
        "nothing may be written on the way to discovering the loop"
    );
}

/// A loser that is already a tombstone is refused rather than silently
/// redirected onto whatever it points at now.
///
/// The winner *is* followed to its live head — an edge must never be re-pointed
/// at a tombstone, which is the reason `resolve_endpoint` follows merges too.
/// The loser is not, and the asymmetry is deliberate. Naming a loser is naming
/// a specific row to retire; quietly retiring a different one because the named
/// one had moved is a semi-destructive call doing something larger than it was
/// asked. The caller — a consolidation worker, shortly — recomputes from the
/// graph anyway, and the honest answer is that its candidate has already gone.
#[tokio::test]
async fn a_loser_that_has_already_been_merged_away_is_refused() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "already-merged").await;

    seed_nodes(
        store,
        &tenant,
        &["Legacy Billing", "Billing Service", "Payments Platform"],
    )
    .await;
    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("the first merge should succeed");

    let error = store
        .merge_nodes(&tenant, "legacy-billing", "payments-platform", &anonymous())
        .await
        .expect_err("the loser is already a tombstone");

    assert!(
        matches!(
            &error,
            StoreError::AlreadyMerged { node_id, merged_into, .. }
                if node_id == "legacy-billing" && merged_into == "billing-service"
        ),
        "the refusal has to say where the node went, got {error:?}"
    );
    assert_eq!(
        merged_into(store, &tenant, "legacy-billing")
            .await
            .as_deref(),
        Some("billing-service"),
        "the existing redirect must not be rewritten"
    );
}

/// The tombstone must not be resurrected by the path that made the duplicate.
///
/// Tier 1 of resolution mints the id it would give this display name and checks
/// whether it is taken — and after a merge it *is* taken, by the tombstone. An
/// upsert that stopped there would hand the caller a dead id, and every edge
/// written against it would hang off a node entity resolution has retired,
/// which is the fragmentation the merge just repaired.
///
/// Nor may a second node be minted. The loser's display name is an alias of the
/// winner now, so tier 2 answers it even if tier 1 did not.
#[tokio::test]
async fn upserting_the_losers_name_afterwards_lands_on_the_winner() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "resolves-onward").await;

    seed_nodes(store, &tenant, &["Legacy Billing", "Billing Service"]).await;
    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
        .await
        .expect("merging should succeed");

    for name in ["Legacy Billing", "legacy billing", "  LEGACY BILLING  "] {
        let outcomes = store
            .upsert_nodes(&tenant, &[NodeInput::new(name)], &embedder(), &anonymous())
            .await
            .expect("upserting the retired name should succeed");

        assert_eq!(
            outcomes[0],
            NodeUpsertOutcome::Matched {
                node_id: "billing-service".to_string()
            },
            "{name:?} must resolve onward to the winner, not onto the tombstone"
        );
    }

    let (nodes,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting should succeed");
    assert_eq!(
        nodes, 2,
        "the retired name must not mint a third node either"
    );
}

/// `search` already filters `merged_into IS NULL` on the node side. That filter
/// has never had a real tombstone to exclude, because nothing in this codebase
/// wrote `merged_into` until now, so this asserts it against one.
///
/// Both halves of the claim matter. The tombstone must not appear — and the
/// loser's name must still find the facts, because the merge moved that name
/// onto the winner. A merge that made an entity unsearchable under the name
/// half the graph knows it by would be a worse outcome than the duplicate.
///
/// The two names share no word on purpose. `Phoenix Ledger` against
/// `Billing Service` means the only way the lexical half can reach the fact is
/// through the alias the merge wrote; a loser called `Legacy Billing` would
/// have matched the winner's own display name on `billing` and the assertion
/// would have held whether the merge moved anything or not.
#[tokio::test]
async fn search_returns_the_winner_and_never_the_tombstone() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "search").await;

    seed_nodes(
        store,
        &tenant,
        &["Phoenix Ledger", "Billing Service", "Postgres"],
    )
    .await;
    let ids = create_edges(
        store,
        &tenant,
        &[EdgeInput::new(
            "phoenix-ledger",
            "postgres",
            "USES",
            "the ledger writes every invoice straight to the database",
        )],
    )
    .await;

    store
        .merge_nodes(&tenant, "phoenix-ledger", "billing-service", &anonymous())
        .await
        .expect("merging should succeed");

    // The lexical half specifically. It is the half that matches on node names,
    // so it is the half the merge's alias union has to keep working — and the
    // half `merged_into IS NULL` filters. Left on hybrid, the semantic half
    // answers from `proof_excerpt` alone and the assertion would pass whether
    // the name moved across or not.
    let hits = store
        .search(
            &tenant,
            "Phoenix Ledger",
            &SearchOptions {
                search_method: Some(SearchMethod::Lexical),
                ..SearchOptions::default()
            },
            &embedder(),
        )
        .await
        .expect("searching should succeed");

    assert!(
        hits.iter().any(|hit| hit.edge_id == ids[0]),
        "the retired name still has to find the fact by name; the merge moved \
         it onto the winner as an alias, and the tombstone that used to carry \
         it is filtered out"
    );

    for hit in &hits {
        assert_ne!(
            hit.source_node_id, "phoenix-ledger",
            "no hit may name the tombstone"
        );
        assert_ne!(
            hit.target_node_id, "phoenix-ledger",
            "no hit may name the tombstone"
        );
    }

    let (tombstoned,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM node WHERE tenant_id = $1 AND merged_into IS NOT NULL",
    )
    .bind(&tenant)
    .fetch_one(store.pool())
    .await
    .expect("counting should succeed");
    assert_eq!(
        tombstoned, 1,
        "the tombstone is still there to be excluded, so the filter was exercised"
    );
}

/// Everything about a node a merge can change.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeRow {
    node_id: String,
    aliases: Vec<String>,
    merged_into: Option<String>,
}

/// The whole tenant, in the columns a merge touches. Compared before and after
/// a reconstruction, so "the audit is sufficient" is a claim about the graph
/// rather than about the audit's field list.
async fn snapshot(store: &Store, tenant: &str) -> (Vec<NodeRow>, Vec<EdgeRow>) {
    let nodes = sqlx::query(
        "SELECT node_id, aliases, merged_into FROM node \
         WHERE tenant_id = $1 ORDER BY node_id",
    )
    .bind(tenant)
    .fetch_all(store.pool())
    .await
    .expect("reading the tenant's nodes should succeed")
    .into_iter()
    .map(|row| NodeRow {
        node_id: row.get("node_id"),
        aliases: row.get("aliases"),
        merged_into: row.get("merged_into"),
    })
    .collect();

    (nodes, edges_of(store, tenant).await)
}

/// **The audit record has to be enough to undo the merge, and this checks it by
/// undoing one.**
///
/// A consolidation worker merging unattended is the next thing built on this,
/// and reversibility is the precondition that makes it safe rather than a
/// nicety. So the assertion is not that the table has the right columns — it is
/// that a reconstruction driven by *nothing but* those rows puts the graph back
/// exactly as it was. Anything the audit forgot shows up as a difference.
///
/// The state is chosen to exercise every way a merge can change a row: an edge
/// re-pointed by its source, one by its target, one collapsed into a self-edge
/// and archived, one retired by the `one`-cardinality contest, and — the case
/// that is easy to miss — an edge that was **never re-pointed at all** and was
/// retired anyway, because it already hung off the winner and lost the contest
/// to a fact that arrived with the loser.
#[tokio::test]
async fn the_audit_record_is_enough_to_reconstruct_an_unmerge() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "audit").await;

    seed_nodes(
        store,
        &tenant,
        &[
            "Legacy Billing",
            "Billing Service",
            "Postgres",
            "Ada Lovelace",
            "Berlin",
            "London",
        ],
    )
    .await;
    plant_node(
        store,
        &tenant,
        "legacy-billing-alias-carrier",
        "Legacy Billing Alias Carrier",
        &[],
    )
    .await;

    create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "legacy-billing",
                "postgres",
                "USES",
                "Legacy Billing writes invoices to Postgres",
            ),
            EdgeInput::new(
                "ada-lovelace",
                "legacy-billing",
                "OWNS",
                "Ada owns Legacy Billing",
            ),
            EdgeInput::new(
                "legacy-billing",
                "billing-service",
                "DEPENDS_ON",
                "Legacy Billing calls into Billing Service",
            ),
            // Never re-pointed: it is already on the winner. It loses the
            // `one`-cardinality contest to the loser's later fact, so the merge
            // changes it without ever having moved it.
            EdgeInput::with_valid_from(
                "billing-service",
                "berlin",
                "LOCATED_IN",
                "Billing Service ran out of Berlin in 2019",
                instant(2019, 3, 1),
            ),
            EdgeInput::with_valid_from(
                "legacy-billing",
                "london",
                "LOCATED_IN",
                "Legacy Billing moved to London in 2024",
                instant(2024, 6, 1),
            ),
        ],
    )
    .await;

    let before = snapshot(store, &tenant).await;

    let provenance = Provenance {
        client: Some("consolidation-worker/0.1".to_string()),
        principal: Some("operator@example.test".to_string()),
    };
    // The database's clock, not this process's. `merged_at` is server-set from
    // `now()`, and the two clocks are not the same clock — the reason
    // `create_edges` reads one shared clock for supersession applies here too.
    let started = server_now(store).await;
    store
        .merge_nodes(&tenant, "legacy-billing", "billing-service", &provenance)
        .await
        .expect("merging should succeed");

    assert_ne!(
        snapshot(store, &tenant).await,
        before,
        "the merge has to have changed something for the undo to mean anything"
    );

    // --- what the audit says --------------------------------------------

    let record = sqlx::query(
        "SELECT merge_id, loser_node_id, winner_node_id, loser_display_name, loser_aliases, \
                aliases_added, merged_at, merged_by_client, merged_by_principal \
         FROM node_merge WHERE tenant_id = $1",
    )
    .bind(&tenant)
    .fetch_one(store.pool())
    .await
    .expect("the merge should have left exactly one audit row");

    let merge_id: Uuid = record.get("merge_id");
    assert_eq!(
        record.get::<String, _>("loser_node_id"),
        "legacy-billing",
        "which node lost"
    );
    assert_eq!(
        record.get::<String, _>("winner_node_id"),
        "billing-service",
        "which node won"
    );
    assert_eq!(
        record.get::<String, _>("loser_display_name"),
        "Legacy Billing",
        "the name the loser answered to, which the winner now answers to"
    );
    assert_eq!(
        record.get::<Vec<String>, _>("aliases_added"),
        vec!["Legacy Billing".to_string()],
        "exactly what an undo has to take back off the winner, and no more"
    );

    let merged_at: DateTime<Utc> = record.get("merged_at");
    let finished = server_now(store).await;
    assert!(
        merged_at >= started && merged_at <= finished,
        "when it happened: {merged_at} is outside [{started}, {finished}]"
    );
    assert_eq!(
        (
            record.get::<Option<String>, _>("merged_by_client"),
            record.get::<Option<String>, _>("merged_by_principal"),
        ),
        (
            Some("consolidation-worker/0.1".to_string()),
            Some("operator@example.test".to_string()),
        ),
        "on whose authority, taken from the request rather than from an argument"
    );

    let changed = sqlx::query(
        "SELECT edge_id, previous_source_node_id, previous_target_node_id, \
                previous_status::text AS previous_status, previous_valid_until, \
                previous_superseded_by_edge_id \
         FROM node_merge_edge WHERE merge_id = $1 AND tenant_id = $2",
    )
    .bind(merge_id)
    .bind(&tenant)
    .fetch_all(store.pool())
    .await
    .expect("reading the edge audit should succeed");

    assert_eq!(
        changed.len(),
        5,
        "every edge the merge changed needs a row, including the one on the \
         winner that was never re-pointed but was still retired"
    );

    // --- the undo, driven by those rows and nothing else -----------------

    for row in &changed {
        sqlx::query(
            "UPDATE edge SET source_node_id = $3, target_node_id = $4, \
                             status = $5::edge_status, valid_until = $6, \
                             superseded_by_edge_id = $7 \
             WHERE tenant_id = $1 AND edge_id = $2",
        )
        .bind(&tenant)
        .bind(row.get::<Uuid, _>("edge_id"))
        .bind(row.get::<String, _>("previous_source_node_id"))
        .bind(row.get::<String, _>("previous_target_node_id"))
        .bind(row.get::<String, _>("previous_status"))
        .bind(row.get::<Option<DateTime<Utc>>, _>("previous_valid_until"))
        .bind(row.get::<Option<Uuid>, _>("previous_superseded_by_edge_id"))
        .execute(store.pool())
        .await
        .expect("restoring an edge should succeed");
    }

    sqlx::query(
        "UPDATE node SET aliases = ( \
             SELECT coalesce(array_agg(a ORDER BY ord), '{}'::text[]) \
             FROM unnest(node.aliases) WITH ORDINALITY AS kept(a, ord) \
             WHERE a <> ALL ($3::text[])) \
         WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(&tenant)
    .bind(record.get::<String, _>("winner_node_id"))
    .bind(record.get::<Vec<String>, _>("aliases_added"))
    .execute(store.pool())
    .await
    .expect("taking the added aliases back off should succeed");

    sqlx::query("UPDATE node SET merged_into = NULL WHERE tenant_id = $1 AND node_id = $2")
        .bind(&tenant)
        .bind(record.get::<String, _>("loser_node_id"))
        .execute(store.pool())
        .await
        .expect("lifting the tombstone should succeed");

    assert_eq!(
        snapshot(store, &tenant).await,
        before,
        "the audit record was not sufficient: the graph did not come back"
    );
}

/// Two agents merging the same pair in opposite directions at the same moment.
///
/// Both validate against a graph that does not yet contain the other's write,
/// so both see two live nodes and no loop. Whichever commits second would write
/// the second half of `X -> Y -> X`, and `follow_merges` would then walk
/// between them forever — an unbounded loop inside an open transaction, which
/// is the one failure here that takes the process rather than the request.
///
/// Exactly one of them may win, and the loser has to be told why rather than
/// silently doing nothing.
#[tokio::test]
async fn two_merges_of_one_pair_in_opposite_directions_cannot_close_a_loop() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "concurrent").await;

    seed_nodes(store, &tenant, &["Legacy Billing", "Billing Service"]).await;

    let forwards = {
        let store = store.clone();
        let tenant = tenant.clone();
        tokio::spawn(async move {
            store
                .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
                .await
        })
    };
    let backwards = {
        let store = store.clone();
        let tenant = tenant.clone();
        tokio::spawn(async move {
            store
                .merge_nodes(&tenant, "billing-service", "legacy-billing", &anonymous())
                .await
        })
    };

    let outcomes = [
        forwards.await.expect("the task should not panic"),
        backwards.await.expect("the task should not panic"),
    ];
    let winners = outcomes.iter().filter(|outcome| outcome.is_ok()).count();
    assert_eq!(
        winners, 1,
        "exactly one direction may be recorded, got {outcomes:?}"
    );

    let refusal = outcomes
        .iter()
        .find_map(|outcome| outcome.as_ref().err())
        .expect("the other direction has to be refused");
    assert!(
        matches!(
            refusal,
            StoreError::MergeCycle { .. } | StoreError::AlreadyMerged { .. }
        ),
        "the refusal has to name the reason, got {refusal:?}"
    );

    // The claim underneath all of it: resolution still terminates.
    let outcome = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Legacy Billing")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("resolution must not find a loop to walk");
    assert!(matches!(outcome[0], NodeUpsertOutcome::Matched { .. }));
}

/// The same race as above, but with the window held open instead of hoped for.
///
/// `merge_nodes` has to validate before it can know which rows to lock, so
/// there is a moment between "these two nodes are live and there is no loop"
/// and "both rows are held" in which another writer can commit. The test above
/// leaves it to the scheduler to land in that moment, which it usually does
/// not. This one puts a writer in it on purpose.
///
/// A transaction opened here takes the lock the merge will need, lets the merge
/// validate against the graph as it stands, then merges the winner away and
/// commits — so the merge wakes up holding a lock on a node that is now a
/// tombstone pointing back at its own loser. Writing `loser -> winner` from
/// there closes the loop.
#[tokio::test]
async fn a_winner_merged_away_inside_the_validation_window_is_caught() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "window").await;

    seed_nodes(store, &tenant, &["Legacy Billing", "Billing Service"]).await;

    // Held across the merge's validation, so the merge blocks at exactly the
    // point its answer is about to go stale.
    let mut blocker = store
        .pool()
        .begin()
        .await
        .expect("opening the blocking transaction should succeed");
    sqlx::query("SELECT node_id FROM node WHERE tenant_id = $1 AND node_id = $2 FOR NO KEY UPDATE")
        .bind(&tenant)
        .bind("billing-service")
        .fetch_one(&mut *blocker)
        .await
        .expect("taking the lock should succeed");

    let merging = {
        let store = store.clone();
        let tenant = tenant.clone();
        tokio::spawn(async move {
            store
                .merge_nodes(&tenant, "legacy-billing", "billing-service", &anonymous())
                .await
        })
    };

    // Long enough for the merge to read a graph with two live nodes and then
    // block on the lock above. If it has not got that far the merge simply sees
    // the committed state and refuses for the ordinary reason, which is also a
    // pass — the assertions below hold either way.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    sqlx::query("UPDATE node SET merged_into = $3 WHERE tenant_id = $1 AND node_id = $2")
        .bind(&tenant)
        .bind("billing-service")
        .bind("legacy-billing")
        .execute(&mut *blocker)
        .await
        .expect("merging the winner away should succeed");
    blocker
        .commit()
        .await
        .expect("committing should release the lock");

    let outcome = merging.await.expect("the task should not panic");

    assert!(
        outcome.is_err(),
        "the winner is a tombstone pointing at the loser by the time the lock \
         is held; recording this merge would close the loop, got {outcome:?}"
    );

    let (cycles,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM node a JOIN node b \
           ON b.tenant_id = a.tenant_id AND b.node_id = a.merged_into \
         WHERE a.tenant_id = $1 AND b.merged_into = a.node_id",
    )
    .bind(&tenant)
    .fetch_one(store.pool())
    .await
    .expect("counting should succeed");
    assert_eq!(cycles, 0, "no pair may point at each other");

    let outcome = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Billing Service")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("resolution must not find a loop to walk");
    assert!(matches!(outcome[0], NodeUpsertOutcome::Matched { .. }));
}
