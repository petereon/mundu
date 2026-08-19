//! Integration tests for edge creation and supersession, against a real
//! Postgres.
//!
//! Every test owns a `tenant_id` derived from its own name, for the reason the
//! node suite gives: the suite runs in parallel against one shared database, so
//! isolation cannot come from truncating shared tables. Tenancy is the isolation
//! mechanism the schema already has.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use chrono::{DateTime, TimeZone, Utc};
use mundu::embed::DeterministicEmbedder;
use mundu::error::StoreError;
use mundu::store::edge::{EdgeInput, EdgeOutcome, EdgeRejection, Endpoint};
use mundu::store::node::NodeInput;
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
/// run left behind, so "this call created an edge" means the same thing on the
/// first run and the hundredth.
///
/// Edges go first: they reference nodes, and they reference each other through
/// `superseded_by_edge_id`. One `DELETE` covering the whole tenant satisfies
/// both, because a foreign key check that runs at the end of the statement sees
/// the referencing row gone too.
async fn fresh_tenant(store: &Store, name: &str) -> String {
    let tenant = format!("test-edge-{name}");

    for statement in [
        "DELETE FROM edge_rating WHERE tenant_id = $1",
        "DELETE FROM edge WHERE tenant_id = $1",
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

/// Mints the nodes an edge test needs through the real upsert path, so the ids
/// under test are the ids the server would actually hand an agent.
async fn seed_nodes(store: &Store, tenant: &str, display_names: &[&str]) {
    let inputs: Vec<NodeInput> = display_names
        .iter()
        .map(|name| NodeInput::new(*name))
        .collect();

    store
        .upsert_nodes(tenant, &inputs, &embedder(), &anonymous())
        .await
        .expect("seeding nodes should succeed");
}

/// One edge as the database holds it.
#[derive(Debug, Clone, PartialEq)]
struct EdgeRow {
    edge_id: Uuid,
    source: String,
    target: String,
    relationship_type: String,
    relationship_type_tenant: String,
    raw_relationship_type: String,
    proof_excerpt: String,
    embedding_model: String,
    status: String,
    valid_from: DateTime<Utc>,
    valid_until: Option<DateTime<Utc>>,
    recorded_at: DateTime<Utc>,
    superseded_by_edge_id: Option<Uuid>,
    writer_client: Option<String>,
    writer_principal: Option<String>,
}

/// Every edge in a tenant, oldest fact first, so assertions can name rows by
/// position without depending on insertion order.
async fn edges_of(store: &Store, tenant: &str) -> Vec<EdgeRow> {
    let rows = sqlx::query(
        "SELECT edge_id, source_node_id, target_node_id, relationship_type, \
                relationship_type_tenant, raw_relationship_type, proof_excerpt, \
                embedding_model, status::text AS status, valid_from, valid_until, \
                recorded_at, superseded_by_edge_id, writer_client, writer_principal \
         FROM edge WHERE tenant_id = $1 \
         ORDER BY valid_from, target_node_id",
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
            relationship_type_tenant: row.get("relationship_type_tenant"),
            raw_relationship_type: row.get("raw_relationship_type"),
            proof_excerpt: row.get("proof_excerpt"),
            embedding_model: row.get("embedding_model"),
            status: row.get("status"),
            valid_from: row.get("valid_from"),
            valid_until: row.get("valid_until"),
            recorded_at: row.get("recorded_at"),
            superseded_by_edge_id: row.get("superseded_by_edge_id"),
            writer_client: row.get("writer_client"),
            writer_principal: row.get("writer_principal"),
        })
        .collect()
}

/// The database's clock, which is the only one anything in this schema is
/// timestamped against.
///
/// The same helper `node_merge` uses, for the same reason. `recorded_at` is the
/// column default, so it is `now()` evaluated inside Postgres; this process runs
/// on the host and Postgres runs in a container, and the two clocks are not the
/// same clock. Bracketing a server-set timestamp with `Utc::now()` therefore
/// compares two different clocks and fails whenever the skew exceeds however
/// long the write took — measured here at about 1.9 ms of skew against a window
/// a few hundred microseconds wide.
async fn server_now(store: &Store) -> DateTime<Utc> {
    let (now,): (DateTime<Utc>,) = sqlx::query_as("SELECT now()")
        .fetch_one(store.pool())
        .await
        .expect("reading the clock should succeed");

    now
}

/// A fixed instant, so a test that talks about "2019" is not quietly comparing
/// against the clock of whatever machine it runs on.
fn instant(year: i32, month: u32, day: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, 12, 0, 0)
        .single()
        .expect("a valid civil timestamp")
}

/// `USES` is `many` in the seeded vocabulary because "A uses B" and "A uses C"
/// are both legitimately true at once. Superseding here would delete a true
/// fact on the strength of a second true fact, and the deletion would be
/// invisible: search would simply stop returning the first one.
#[tokio::test]
async fn a_many_edge_appends_alongside_an_existing_edge_of_the_same_type() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "many-appends").await;
    seed_nodes(store, &tenant, &["Billing Service", "Kafka", "Postgres"]).await;

    store
        .create_edges(
            &tenant,
            &[EdgeInput::new(
                "billing-service",
                "kafka",
                "USES",
                "the billing service publishes settlement events to Kafka",
            )],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("the first edge should be written");

    store
        .create_edges(
            &tenant,
            &[EdgeInput::new(
                "billing-service",
                "postgres",
                "USES",
                "the billing service stores invoices in Postgres",
            )],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("the second edge should be written");

    let edges = edges_of(store, &tenant).await;
    assert_eq!(edges.len(), 2, "both edges should have been kept");

    for edge in &edges {
        assert_eq!(edge.relationship_type, "USES");
        assert_eq!(
            edge.status, "active",
            "a many-cardinality type must never supersede: {edge:?}"
        );
        assert_eq!(edge.valid_until, None, "an active edge is still open");
        assert_eq!(edge.superseded_by_edge_id, None);
    }

    let targets: Vec<&str> = edges.iter().map(|edge| edge.target.as_str()).collect();
    assert_eq!(targets, vec!["kafka", "postgres"]);
}

/// Writes one edge and returns the id the store minted for it, so a test can
/// talk about "the incumbent" without re-reading the table to find it.
async fn create_one(
    store: &Store,
    tenant: &str,
    input: EdgeInput,
    provenance: &Provenance,
) -> EdgeOutcome {
    let mut outcomes = store
        .create_edges(tenant, &[input], &embedder(), provenance)
        .await
        .expect("writing the edge should succeed");

    assert_eq!(outcomes.len(), 1, "one input, one outcome");
    outcomes.remove(0)
}

/// The id of an edge the store says it wrote, whether it landed as the current
/// fact or as closed history.
fn written_id(outcome: &EdgeOutcome) -> Uuid {
    match outcome {
        EdgeOutcome::Created { edge_id, .. } | EdgeOutcome::Backfilled { edge_id, .. } => *edge_id,
        other => panic!("expected a written edge, got {other:?}"),
    }
}

/// The row with this id, or a failure naming what is actually in the table.
fn row_with(edges: &[EdgeRow], edge_id: Uuid) -> &EdgeRow {
    edges
        .iter()
        .find(|edge| edge.edge_id == edge_id)
        .unwrap_or_else(|| panic!("no edge {edge_id} among {edges:#?}"))
}

/// `REPORTS_TO` is `one`: an employee reports to exactly one manager at a time.
/// A newer fact therefore retires the older one rather than sitting beside it,
/// and the retirement has to say all three things — that the edge is no longer
/// current, when it stopped being current, and what replaced it. An edge marked
/// `superseded` with no `valid_until` is a fact that was true until an
/// unspecified moment, which is not a bi-temporal record of anything.
#[tokio::test]
async fn a_newer_one_edge_supersedes_the_incumbent() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "newer-supersedes").await;
    seed_nodes(store, &tenant, &["Alice", "Bob", "Carol"]).await;

    let incumbent = written_id(
        &create_one(
            store,
            &tenant,
            EdgeInput::with_valid_from(
                "alice",
                "bob",
                "REPORTS_TO",
                "Alice reports to Bob, per the 2019 org chart",
                instant(2019, 3, 1),
            ),
            &anonymous(),
        )
        .await,
    );

    let outcome = create_one(
        store,
        &tenant,
        EdgeInput::with_valid_from(
            "alice",
            "carol",
            "REPORTS_TO",
            "Alice now reports to Carol after the June reorg",
            instant(2024, 6, 1),
        ),
        &anonymous(),
    )
    .await;

    let EdgeOutcome::Created {
        edge_id: successor,
        superseded_edge_id,
    } = &outcome
    else {
        panic!("expected the newer fact to be written as current, got {outcome:?}");
    };
    assert_eq!(
        superseded_edge_id.as_ref(),
        Some(&incumbent),
        "the caller has to be told which fact this retired"
    );

    let edges = edges_of(store, &tenant).await;
    assert_eq!(edges.len(), 2, "supersession retires, it never deletes");

    let retired = row_with(&edges, incumbent);
    assert_eq!(retired.status, "superseded");
    assert_eq!(
        retired.valid_until,
        Some(instant(2024, 6, 1)),
        "the incumbent stopped holding when the newer fact started"
    );
    assert_eq!(retired.superseded_by_edge_id, Some(*successor));

    let current = row_with(&edges, *successor);
    assert_eq!(current.status, "active");
    assert_eq!(current.valid_until, None);
    assert_eq!(current.superseded_by_edge_id, None);
}

/// The case the whole ordering rule exists for.
///
/// Writes arrive out of order: the server is stateless behind a load balancer
/// and many agents write at once, so an agent backfilling "Alice reported to
/// Bob back in 2019" can reach the database long after the fact that Alice now
/// reports to Carol. If supersession went by arrival rather than by
/// `valid_from`, that backfill would retire current truth and the graph would
/// assert a 2019 fact as the present one — with a `proof_excerpt` that reads
/// perfectly and survives an audit. Silent corruption of the one thing this
/// server is for.
///
/// So an older fact is inserted as history that is already closed: it holds
/// from its own `valid_from` until the incumbent's, points at the incumbent as
/// its successor, and leaves the incumbent exactly as it found it.
#[tokio::test]
async fn an_older_one_edge_is_stored_as_closed_history_and_the_incumbent_stays_active() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "older-backfill").await;
    seed_nodes(store, &tenant, &["Alice", "Bob", "Carol"]).await;

    let incumbent = written_id(
        &create_one(
            store,
            &tenant,
            EdgeInput::with_valid_from(
                "alice",
                "carol",
                "REPORTS_TO",
                "Alice reports to Carol after the June 2024 reorg",
                instant(2024, 6, 1),
            ),
            &anonymous(),
        )
        .await,
    );

    let outcome = create_one(
        store,
        &tenant,
        EdgeInput::with_valid_from(
            "alice",
            "bob",
            "REPORTS_TO",
            "Alice reported to Bob back in 2019, per the archived org chart",
            instant(2019, 3, 1),
        ),
        &anonymous(),
    )
    .await;
    let backfilled = written_id(&outcome);

    let edges = edges_of(store, &tenant).await;
    assert_eq!(edges.len(), 2, "the backfill is kept, not dropped");

    let current = row_with(&edges, incumbent);
    assert_eq!(
        current.status, "active",
        "a 2019 backfill must not retire the fact that holds today"
    );
    assert_eq!(current.valid_until, None, "the current fact is still open");
    assert_eq!(current.superseded_by_edge_id, None);

    let history = row_with(&edges, backfilled);
    assert_eq!(
        history.status, "superseded",
        "an older fact arrives already closed; it was never current"
    );
    assert_eq!(history.valid_from, instant(2019, 3, 1));
    assert_eq!(
        history.valid_until,
        Some(instant(2024, 6, 1)),
        "it held until the fact that displaced it began"
    );
    assert_eq!(
        history.superseded_by_edge_id,
        Some(incumbent),
        "history has to point at what replaced it, or the chain is unreadable"
    );

    assert_eq!(
        outcome,
        EdgeOutcome::Backfilled {
            edge_id: backfilled,
            superseded_by_edge_id: incumbent,
            valid_until: instant(2024, 6, 1),
        },
        "the caller must be told its write is history rather than current truth"
    );
}

/// Asserts two `REPORTS_TO` facts that both claim the same instant, in the
/// order given, and reports the target of whichever edge is left active.
async fn tie_between(store: &Store, tenant: &str, first: &str, second: &str) -> String {
    let claimed = instant(2024, 6, 1);

    for target in [first, second] {
        create_one(
            store,
            tenant,
            EdgeInput::with_valid_from(
                "alice",
                target,
                "REPORTS_TO",
                format!("Alice reports to {target} as of the June reorg"),
                claimed,
            ),
            &anonymous(),
        )
        .await;
    }

    let edges = edges_of(store, tenant).await;
    assert_eq!(edges.len(), 2, "both assertions are kept");

    let active: Vec<&EdgeRow> = edges
        .iter()
        .filter(|edge| edge.status == "active")
        .collect();
    assert_eq!(
        active.len(),
        1,
        "a one-cardinality type may leave exactly one active edge: {edges:#?}"
    );

    let closed = edges
        .iter()
        .find(|edge| edge.status == "superseded")
        .expect("the loser is kept as history");
    assert_eq!(
        closed.valid_until,
        Some(claimed),
        "a tie leaves a zero-length interval: the loser never held for any time"
    );
    assert_eq!(
        closed.superseded_by_edge_id,
        Some(active[0].edge_id),
        "history has to point at the fact that stands in its place"
    );

    active[0].target.clone()
}

/// Two agents can assert facts that claim the same instant, and then the
/// ordering key has nothing left to say. Something still has to give, because a
/// `one` type may leave only one active edge.
///
/// The tie is broken on `target_node_id`, greatest first. *Which* target wins is
/// arbitrary — there is no fact of the matter — but it must not be arrival
/// order: this server is stateless behind a round-robin load balancer, so
/// arrival order is a property of which replica the balancer picked, and a
/// graph whose current truth depends on that is a graph that answers the same
/// question differently on every replay, with nothing to show why. A
/// lexicographic comparison is a pure function of the two rows, so every replica
/// and every replay reaches the same answer.
#[tokio::test]
async fn an_exact_valid_from_tie_resolves_the_same_way_in_either_arrival_order() {
    let store = &store().await;

    let forwards = fresh_tenant(store, "tie-forwards").await;
    seed_nodes(store, &forwards, &["Alice", "Bob", "Carol"]).await;
    let backwards = fresh_tenant(store, "tie-backwards").await;
    seed_nodes(store, &backwards, &["Alice", "Bob", "Carol"]).await;

    assert_eq!(
        tie_between(store, &forwards, "bob", "carol").await,
        tie_between(store, &backwards, "carol", "bob").await,
        "the surviving fact must not depend on which write arrived first"
    );
}

/// The degenerate tie: the same fact asserted twice, from the same instant, by
/// two agents that each read it somewhere. There is nothing to choose between
/// them, so the incumbent stays and the newcomer is recorded as history — both
/// proofs and both writers are kept, and the graph still holds one current fact.
#[tokio::test]
async fn re_asserting_the_same_fact_at_the_same_instant_keeps_the_incumbent() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "tie-same-target").await;
    seed_nodes(store, &tenant, &["Alice", "Bob"]).await;

    let incumbent = written_id(
        &create_one(
            store,
            &tenant,
            EdgeInput::with_valid_from(
                "alice",
                "bob",
                "REPORTS_TO",
                "Alice reports to Bob, per the directory",
                instant(2024, 6, 1),
            ),
            &anonymous(),
        )
        .await,
    );

    let repeat = create_one(
        store,
        &tenant,
        EdgeInput::with_valid_from(
            "alice",
            "bob",
            "REPORTS_TO",
            "Bob is listed as Alice's manager in the handbook",
            instant(2024, 6, 1),
        ),
        &anonymous(),
    )
    .await;

    assert_eq!(
        repeat,
        EdgeOutcome::Backfilled {
            edge_id: written_id(&repeat),
            superseded_by_edge_id: incumbent,
            valid_until: instant(2024, 6, 1),
        },
        "with nothing to choose between them, the fact already recorded stands"
    );

    let edges = edges_of(store, &tenant).await;
    assert_eq!(row_with(&edges, incumbent).status, "active");
    assert_eq!(row_with(&edges, written_id(&repeat)).status, "superseded");
}

/// `valid_from` is the key supersession orders on, and it arrives as a tool
/// argument filled in by a model that has just read `proof_excerpt` text written
/// by some other agent — untrusted input, one step removed. A date in the future
/// would win every conflict from now until that date passes, and no later fact,
/// however true, could displace it.
///
/// So it is refused rather than clamped. Clamping would accept the write and
/// silently rewrite it, leaving the attempt indistinguishable from an honest
/// write in every column of the table; refusing puts it in front of the caller.
/// The tolerance exists only for ordinary clock skew between an agent's host and
/// this database.
#[tokio::test]
async fn a_valid_from_far_in_the_future_is_refused_rather_than_clamped() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "future-valid-from").await;
    seed_nodes(store, &tenant, &["Alice", "Bob"]).await;

    let far_future = Utc::now() + chrono::Duration::minutes(10);

    let error = store
        .create_edges(
            &tenant,
            &[EdgeInput::with_valid_from(
                "alice",
                "bob",
                "REPORTS_TO",
                "Alice will report to Bob, and has always reported to Bob",
                far_future,
            )],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect_err("a future-dated fact must not be written");

    let StoreError::FutureValidFrom {
        index, valid_from, ..
    } = &error
    else {
        panic!("expected a distinct future-dated error, got {error:?}");
    };
    assert_eq!(
        *index, 0,
        "the caller has to be told which item was refused"
    );
    assert_eq!(*valid_from, far_future);

    assert!(
        edges_of(store, &tenant).await.is_empty(),
        "a refused write must leave nothing behind, not a clamped row"
    );

    // The tolerance is for clock skew, so an instant a few seconds ahead is an
    // ordinary write rather than an attack.
    let slightly_ahead = Utc::now() + chrono::Duration::seconds(30);
    create_one(
        store,
        &tenant,
        EdgeInput::with_valid_from(
            "alice",
            "bob",
            "REPORTS_TO",
            "Alice reports to Bob as of this morning",
            slightly_ahead,
        ),
        &anonymous(),
    )
    .await;

    let edges = edges_of(store, &tenant).await;
    assert_eq!(edges.len(), 1);
    assert_eq!(
        edges[0].valid_from, slightly_ahead,
        "an accepted valid_from is stored as the caller gave it, unclamped"
    );
}

/// The endpoint a rejection is about, and what the caller probably meant.
fn unknown_endpoint(outcome: &EdgeOutcome) -> (Endpoint, &str, Vec<&str>) {
    let EdgeOutcome::Rejected {
        reason:
            EdgeRejection::UnknownEndpoint {
                endpoint,
                node_id,
                suggestions,
            },
    } = outcome
    else {
        panic!("expected a rejected endpoint, got {outcome:?}");
    };

    (
        *endpoint,
        node_id.as_str(),
        suggestions
            .iter()
            .map(|suggestion| suggestion.node_id.as_str())
            .collect(),
    )
}

/// A batch is a convenience, not a unit of truth. An agent that gets one
/// `node_id` wrong out of five has still learned four correct facts, and
/// throwing them away teaches it nothing about which one was wrong — it will
/// send the same five again.
///
/// So an unknown endpoint fails its own item and nothing else, and the failure
/// carries the ids it nearly matched, because the caller is a model that can act
/// on "did you mean `alice-whitfield`" and cannot act on "not found".
#[tokio::test]
async fn an_unknown_endpoint_fails_only_its_own_item() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "unknown-endpoint").await;
    seed_nodes(store, &tenant, &["Alice Whitfield", "Bob Nkemelu", "Kafka"]).await;

    let outcomes = store
        .create_edges(
            &tenant,
            &[
                EdgeInput::new(
                    "alicia-whitfield",
                    "bob-nkemelu",
                    "REPORTS_TO",
                    "Alicia reports to Bob",
                ),
                EdgeInput::new(
                    "alice-whitfield",
                    "kafka",
                    "USES",
                    "Alice's team publishes to Kafka",
                ),
                EdgeInput::new(
                    "alice-whitfield",
                    "bob-nkemelu-jr",
                    "REPORTS_TO",
                    "Alice reports to Bob Jr",
                ),
            ],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("a batch with a bad endpoint is not a failed call");

    assert_eq!(outcomes.len(), 3, "one outcome per input, in input order");

    let (endpoint, named, suggestions) = unknown_endpoint(&outcomes[0]);
    assert_eq!(endpoint, Endpoint::Source);
    assert_eq!(named, "alicia-whitfield");
    assert!(
        suggestions.contains(&"alice-whitfield"),
        "a near miss the caller can act on should be offered, got {suggestions:?}"
    );

    let sibling = written_id(&outcomes[1]);

    let (endpoint, named, suggestions) = unknown_endpoint(&outcomes[2]);
    assert_eq!(
        endpoint,
        Endpoint::Target,
        "the caller has to be told which end was wrong"
    );
    assert_eq!(named, "bob-nkemelu-jr");
    assert!(
        suggestions.contains(&"bob-nkemelu"),
        "expected bob-nkemelu among {suggestions:?}"
    );

    let edges = edges_of(store, &tenant).await;
    assert_eq!(
        edges.len(),
        1,
        "the good item commits and the bad ones write nothing: {edges:#?}"
    );
    assert_eq!(edges[0].edge_id, sibling);
    assert_eq!(edges[0].source, "alice-whitfield");
    assert_eq!(edges[0].target, "kafka");
}

/// The likeliest way an agent gets an endpoint wrong: it has the entity's name
/// in front of it and sends that, where the server wants the handle it minted.
/// Guessing that the two mean the same thing is how a fact ends up hanging off
/// an entity nobody asked for, so the write is refused — and the suggestion is
/// exactly the id the caller should have sent.
#[tokio::test]
async fn a_display_name_where_an_id_belongs_is_refused_with_the_id_suggested() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "display-name-endpoint").await;
    seed_nodes(store, &tenant, &["Alice Whitfield", "Kafka"]).await;

    let outcome = create_one(
        store,
        &tenant,
        EdgeInput::new(
            "Alice Whitfield",
            "kafka",
            "USES",
            "Alice's team publishes to Kafka",
        ),
        &anonymous(),
    )
    .await;

    let (endpoint, named, suggestions) = unknown_endpoint(&outcome);
    assert_eq!(endpoint, Endpoint::Source);
    assert_eq!(named, "Alice Whitfield");
    assert_eq!(suggestions, vec!["alice-whitfield"]);

    assert!(edges_of(store, &tenant).await.is_empty());
}

/// Many agents write into one graph, and an edge whose writer is unknown cannot
/// be judged: the rating loop, the dispute quorum and any operator reading a
/// suspicious proof all start from who asserted it. Provenance comes from the
/// request — `_meta` clientInfo and the validated credential — never from tool
/// arguments, so an edge that drops it cannot get it back.
#[tokio::test]
async fn the_writer_behind_a_fact_is_recorded_with_it() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "provenance").await;
    seed_nodes(store, &tenant, &["Alice", "Kafka"]).await;

    let provenance = Provenance {
        client: Some("claude-desktop/1.4".to_string()),
        principal: Some("agent-7@example.test".to_string()),
    };

    create_one(
        store,
        &tenant,
        EdgeInput::new("alice", "kafka", "USES", "Alice's team publishes to Kafka"),
        &provenance,
    )
    .await;

    // An unattributed write is still a write worth keeping, so it is recorded
    // as unattributed rather than refused — or, worse, attributed to whoever
    // wrote last.
    create_one(
        store,
        &tenant,
        EdgeInput::new("alice", "kafka", "DEPENDS_ON", "Alice's team needs Kafka"),
        &anonymous(),
    )
    .await;

    let edges = edges_of(store, &tenant).await;
    let attributed = edges
        .iter()
        .find(|edge| edge.relationship_type == "USES")
        .expect("the attributed edge should be readable");
    assert_eq!(
        attributed.writer_client.as_deref(),
        Some("claude-desktop/1.4")
    );
    assert_eq!(
        attributed.writer_principal.as_deref(),
        Some("agent-7@example.test")
    );

    let anonymous_edge = edges
        .iter()
        .find(|edge| edge.relationship_type == "DEPENDS_ON")
        .expect("the anonymous edge should be readable");
    assert_eq!(anonymous_edge.writer_client, None);
    assert_eq!(anonymous_edge.writer_principal, None);
}

/// Blocks until `expected` backends are stuck behind `blocker`, so a race can
/// be started from a known state rather than from a hopeful `sleep`.
///
/// The wait graph is walked transitively. Once the writers serialise on the
/// source node they queue behind *each other* rather than behind this test's
/// lock — `pg_blocking_pids` reports the holder immediately ahead — so counting
/// only direct waiters would see one writer and never two.
///
/// Naming the blocking pid keeps this confined to the sessions this test
/// parked: the rest of the suite runs against the same database in parallel and
/// cannot inflate the count.
async fn wait_until_blocked(store: &Store, blocker: i32, expected: i64) {
    for _ in 0..200 {
        let (waiting,): (i64,) = sqlx::query_as(
            "WITH RECURSIVE stuck(pid) AS ( \
                 SELECT pid FROM pg_stat_activity WHERE $1 = ANY(pg_blocking_pids(pid)) \
                 UNION \
                 SELECT waiter.pid FROM pg_stat_activity AS waiter, stuck \
                 WHERE stuck.pid = ANY(pg_blocking_pids(waiter.pid))) \
             SELECT count(*) FROM stuck",
        )
        .bind(blocker)
        .fetch_one(store.pool())
        .await
        .expect("reading the wait graph should succeed");

        if waiting >= expected {
            return;
        }

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    panic!("timed out waiting for {expected} writers to queue behind backend {blocker}");
}

/// Two agents asserting the same `one`-cardinality fact at the same moment must
/// not both end up current. Nothing in the schema can express that: a unique
/// index over `(tenant, source, type) WHERE status = 'active'` would be wrong
/// for every `many` type, and cardinality lives in another table that the index
/// cannot consult. So the mutual exclusion has to be taken explicitly, on the
/// source node, before the incumbent is read.
///
/// The race here is real rather than hoped for. `tokio::join!` on two futures
/// usually interleaves them at await points and proves nothing, so instead a
/// third transaction takes a conflicting lock on the source node and both
/// writers are confirmed — through `pg_blocking_pids`, not a sleep — to be
/// parked behind it. Releasing it starts them from the same instant, contending
/// for real, with the outcome decided by Postgres' lock queue.
///
/// Without the lock this fails: both writers read an empty incumbent, both
/// insert as active, and the graph asserts two managers for one person.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_writers_of_one_fact_leave_exactly_one_active_edge() {
    let store = store().await;
    let tenant = fresh_tenant(&store, "concurrent-writers").await;
    seed_nodes(&store, &tenant, &["Alice", "Bob", "Carol"]).await;

    let claimed = instant(2024, 6, 1);

    // Held open across the spawn below. Both writers have to reach into the
    // source node to serialise, so this is what they queue behind.
    let mut blocker = store
        .pool()
        .begin()
        .await
        .expect("opening the blocking transaction should succeed");

    let (blocker_pid,): (i32,) = sqlx::query_as("SELECT pg_backend_pid()")
        .fetch_one(&mut *blocker)
        .await
        .expect("reading the backend pid should succeed");

    sqlx::query("SELECT node_id FROM node WHERE tenant_id = $1 AND node_id = $2 FOR UPDATE")
        .bind(&tenant)
        .bind("alice")
        .fetch_one(&mut *blocker)
        .await
        .expect("locking the source node should succeed");

    let writers: Vec<_> = ["bob", "carol"]
        .into_iter()
        .map(|target| {
            let store = store.clone();
            let tenant = tenant.clone();

            tokio::spawn(async move {
                store
                    .create_edges(
                        &tenant,
                        &[EdgeInput::with_valid_from(
                            "alice",
                            target,
                            "REPORTS_TO",
                            format!("Alice reports to {target}, agreed at the June reorg"),
                            claimed,
                        )],
                        &embedder(),
                        &anonymous(),
                    )
                    .await
            })
        })
        .collect();

    wait_until_blocked(&store, blocker_pid, 2).await;

    blocker
        .rollback()
        .await
        .expect("releasing the blocking transaction should succeed");

    for writer in writers {
        writer
            .await
            .expect("neither writer should panic")
            .expect("neither writer should fail");
    }

    let edges = edges_of(&store, &tenant).await;
    assert_eq!(edges.len(), 2, "both assertions are kept: {edges:#?}");

    let active: Vec<&EdgeRow> = edges
        .iter()
        .filter(|edge| edge.status == "active")
        .collect();
    assert_eq!(
        active.len(),
        1,
        "two writers of a one-cardinality fact left {} active edges: {edges:#?}",
        active.len()
    );
    assert_eq!(
        active[0].target, "carol",
        "the tie-break decides the survivor, not who won the lock"
    );

    let closed = edges
        .iter()
        .find(|edge| edge.status == "superseded")
        .expect("the loser is kept as history");
    assert_eq!(closed.superseded_by_edge_id, Some(active[0].edge_id));
}

/// The two halves of a bi-temporal record answer different questions, and only
/// one of them is the caller's to answer. `valid_from` is when the fact held in
/// the world, which the agent knows and the server does not; `recorded_at` is
/// when this server came to believe it, which the server knows and the agent
/// cannot be allowed to state. An agent that could backdate `recorded_at` could
/// make a fact it wrote a minute ago look like something the graph has held all
/// along, which is precisely the provenance an auditor would lean on.
///
/// The struct literal below is deliberate: it names every field of `EdgeInput`,
/// so adding a caller-supplied `recorded_at` cannot be done without this test
/// refusing to compile.
///
/// The window is read off the database's clock rather than this process's.
/// `recorded_at` is the column default, so it is `now()` evaluated inside
/// Postgres; bracketing it with `Utc::now()` asks whether one clock's reading
/// falls between two readings of another, which is a question about NTP rather
/// than about this server. The window stays as narrow as the write itself —
/// widening it to swallow the skew would be the same as deleting the assertion,
/// because a `recorded_at` taken from the caller's 2019 date is exactly what
/// the narrowness is here to catch.
#[tokio::test]
async fn recorded_at_is_server_set_and_ignores_the_timestamps_a_caller_supplies() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "recorded-at").await;
    seed_nodes(store, &tenant, &["Alice", "Bob"]).await;

    let claimed = instant(2019, 3, 1);
    let before = server_now(store).await;

    create_one(
        store,
        &tenant,
        EdgeInput {
            source_node_id: "alice".to_string(),
            target_node_id: "bob".to_string(),
            relationship_type: "REPORTS_TO".to_string(),
            proof_excerpt: "Alice reported to Bob from March 2019".to_string(),
            valid_from: Some(claimed),
        },
        &anonymous(),
    )
    .await;

    let after = server_now(store).await;
    let edges = edges_of(store, &tenant).await;
    assert_eq!(edges.len(), 1);
    let edge = &edges[0];

    assert_eq!(
        edge.valid_from, claimed,
        "the world-time the caller stated is kept exactly"
    );
    assert!(
        edge.recorded_at >= before && edge.recorded_at <= after,
        "recorded_at is {}, which is not the moment this write happened \
         (between {before} and {after})",
        edge.recorded_at
    );
}

/// Coercion is a judgement call made by a normaliser, a lookup table and a
/// model, and it is applied to text this server does not trust. Keeping the
/// caller's own words beside the result is what makes a bad coercion visible
/// and reversible: without `raw_relationship_type`, a fact folded onto the wrong
/// type reads as though the agent had written that type itself.
#[tokio::test]
async fn the_callers_own_wording_is_kept_beside_the_canonical_type() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "raw-type").await;
    seed_nodes(store, &tenant, &["Alice", "Bob"]).await;

    create_one(
        store,
        &tenant,
        EdgeInput::new(
            "alice",
            "bob",
            "  reports-To ",
            "Alice's manager is Bob, per the directory",
        ),
        &anonymous(),
    )
    .await;

    let edges = edges_of(store, &tenant).await;
    assert_eq!(edges.len(), 1);

    assert_eq!(
        edges[0].relationship_type, "REPORTS_TO",
        "the vocabulary must not fragment across spellings"
    );
    assert_eq!(
        edges[0].raw_relationship_type, "  reports-To ",
        "the caller's wording is evidence, not a draft of the canonical name"
    );
    assert_eq!(
        edges[0].relationship_type_tenant, "",
        "a curated type is referenced in the curated scope"
    );
    assert_eq!(
        edges[0].embedding_model, "deterministic-768",
        "a vector is meaningless without the model that produced it"
    );
}

/// A relationship type with no usable characters is a caller bug, not a new
/// relation, and it is decidable from the argument alone. So it has to be found
/// before anything is written: an error returned after earlier items in the
/// batch have committed leaves the caller unable to tell what landed, and the
/// only safe thing it can do is send the whole batch again — which duplicates
/// every item that did land.
///
/// This is the rule the node upsert already follows for a blank display name,
/// and it is why an unknown endpoint is handled differently: that one is a
/// question about the graph, discovered against the database, and can be an
/// honest race with another agent's node write.
#[tokio::test]
async fn a_blank_relationship_type_is_refused_before_any_sibling_commits() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "blank-type").await;
    seed_nodes(store, &tenant, &["Alice", "Kafka"]).await;

    let error = store
        .create_edges(
            &tenant,
            &[
                EdgeInput::new("alice", "kafka", "USES", "Alice's team publishes to Kafka"),
                EdgeInput::new(
                    "alice",
                    "kafka",
                    "   ",
                    "Alice and Kafka are related somehow",
                ),
            ],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect_err("a blank relationship type must be refused");

    assert!(
        matches!(error, StoreError::BlankRelationshipType { .. }),
        "expected a blank-type error, got {error:?}"
    );

    assert!(
        edges_of(store, &tenant).await.is_empty(),
        "the sibling must not have committed under a call that reports failure"
    );
}

/// `tenant_id` is the whole isolation boundary, and it comes from a validated
/// credential rather than a tool argument. Two customers naming a project the
/// same thing is the ordinary case, so the ids collide by design — and a
/// supersession that read across the boundary would retire another customer's
/// current fact on the strength of a write they never saw, which is both a
/// corruption and a disclosure that this tenant has an edge there at all.
#[tokio::test]
async fn supersession_never_reaches_across_tenants() {
    let store = &store().await;
    let acme = fresh_tenant(store, "isolation-acme").await;
    let globex = fresh_tenant(store, "isolation-globex").await;

    for tenant in [&acme, &globex] {
        seed_nodes(store, tenant, &["Alice", "Bob", "Carol"]).await;

        create_one(
            store,
            tenant,
            EdgeInput::with_valid_from(
                "alice",
                "bob",
                "REPORTS_TO",
                "Alice reports to Bob",
                instant(2019, 3, 1),
            ),
            &anonymous(),
        )
        .await;
    }

    // Newer, and in one tenant only.
    create_one(
        store,
        &acme,
        EdgeInput::with_valid_from(
            "alice",
            "carol",
            "REPORTS_TO",
            "Alice reports to Carol now",
            instant(2024, 6, 1),
        ),
        &anonymous(),
    )
    .await;

    let neighbour = edges_of(store, &globex).await;
    assert_eq!(neighbour.len(), 1, "no row should have been written here");
    assert_eq!(
        neighbour[0].status, "active",
        "another tenant's write must not retire this one's current fact"
    );
    assert_eq!(neighbour[0].valid_until, None);
    assert_eq!(neighbour[0].superseded_by_edge_id, None);

    let mine = edges_of(store, &acme).await;
    assert_eq!(mine.len(), 2);
    assert_eq!(
        mine.iter().filter(|edge| edge.status == "active").count(),
        1,
        "and the tenant that did write still supersedes its own"
    );
}

/// A type nobody has registered is coined into the caller's own vocabulary, and
/// the edge has to reference it *there*: `edge` points at
/// `(relationship_type_tenant, relationship_type)`, and a private type recorded
/// under the curated scope is a foreign key into a row that does not exist.
#[tokio::test]
async fn an_edge_on_a_coined_type_references_the_tenants_own_vocabulary_scope() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "coined-type").await;
    seed_nodes(store, &tenant, &["Alice", "Bob"]).await;

    sqlx::query("DELETE FROM relationship_type WHERE tenant_id = $1")
        .bind(&tenant)
        .execute(store.pool())
        .await
        .expect("clearing the tenant's vocabulary should succeed");

    create_one(
        store,
        &tenant,
        EdgeInput::new(
            "alice",
            "bob",
            "onboarding buddy of",
            "Alice is Bob's onboarding buddy this quarter",
        ),
        &anonymous(),
    )
    .await;

    let edges = edges_of(store, &tenant).await;
    assert_eq!(edges[0].relationship_type, "ONBOARDING_BUDDY_OF");
    assert_eq!(edges[0].relationship_type_tenant, tenant);
    assert_eq!(edges[0].raw_relationship_type, "onboarding buddy of");
    assert_eq!(
        edges[0].status, "active",
        "a coined type defaults to many, so there is nothing for it to displace"
    );
}

/// A fact with no proof is a bare assertion wearing the costume of an evidenced
/// one. The tool's premise is that a claim comes with the words that justify it
/// — that quote is what a later reader uses to decide whether to believe a fact
/// it did not write — and an empty one is also embedded into a `proof_vector`
/// of essentially nothing, which then ranks against every query.
///
/// Refused for the whole batch before any sibling commits, exactly as a blank
/// relationship type is and for the same reason: it is decidable from the
/// argument alone, so a failure reported after earlier items landed would leave
/// the caller unable to tell what was written.
#[tokio::test]
async fn an_unevidenced_proof_excerpt_is_refused_before_any_sibling_commits() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "blank-proof").await;
    seed_nodes(store, &tenant, &["Alice", "Kafka"]).await;

    let error = store
        .create_edges(
            &tenant,
            &[
                EdgeInput::new("alice", "kafka", "USES", "Alice's team publishes to Kafka"),
                EdgeInput::new("alice", "kafka", "DEPENDS_ON", "   \t "),
            ],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect_err("a whitespace-only proof excerpt must be refused");

    assert!(
        matches!(
            error,
            StoreError::UnevidencedProofExcerpt {
                index: 1,
                chars: 0,
                ..
            }
        ),
        "expected item 1 to be named with nothing usable in it, got {error:?}"
    );

    assert!(
        edges_of(store, &tenant).await.is_empty(),
        "the sibling must not have committed under a call that reports failure"
    );
}
