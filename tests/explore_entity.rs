//! Integration tests for `explore_entity`, against a real Postgres.
//!
//! The fifth tool is the only one that *traverses*: `search_knowledge_graph`
//! ranks proof text and hands back edges, and nothing before this could answer
//! "what is connected to this entity". So these tests are mostly about shape
//! and about bounds — what comes back, how much of it, and what is said about
//! the part that did not.
//!
//! Every test owns a `tenant_id` derived from its own name, for the reason the
//! other suites give: the suite runs in parallel against one shared database,
//! so isolation cannot come from truncating shared tables. Tenancy is the
//! isolation mechanism the schema already has.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use mundu::auth::Principal;
use mundu::embed::{DeterministicEmbedder, Embedder};
use mundu::mcp::tools::{
    CreateRelationshipInput, EntityMatchReport, ExploreEntityInput, MatchedByReport, NodeSpec,
    RateMemoryUtilityInput, RelationshipReport, RelationshipSpec, SearchInput, Tools,
    UpsertNodeInput,
};
use mundu::store::Store;
use mundu::store::explore::EDGE_BUDGET;

/// One [`Store`] per test, deliberately not a shared `static`: a sqlx pool is
/// bound to the reactor that created it, and every `#[tokio::test]` builds and
/// tears down its own.
async fn store() -> Store {
    common::store().await
}

/// The tool layer under test, over a fresh pool and the deterministic embedder.
async fn tools() -> Tools {
    Tools::new(store().await, Arc::new(DeterministicEmbedder::new()))
}

/// Names a tenant after the test that owns it and clears anything a previous
/// run left behind.
async fn fresh_tenant(tools: &Tools, name: &str) -> String {
    let tenant = format!("test-explore-{name}");

    // The merge audit tables first, and `merged_into` before the nodes: one of
    // these tests merges, and both the audit rows and a tombstone's redirect
    // are foreign keys onto rows this is about to delete. Same order, and for
    // the same reason, as `tests/node_merge.rs`.
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
            .execute(tools.store().pool())
            .await
            .expect("clearing the tenant should succeed");
    }

    tenant
}

/// An authenticated caller. There is no constructor that takes a tenant from
/// anywhere but a credential, which is the property the whole layer rests on.
fn principal(tenant: &str, subject: &str) -> Principal {
    Principal {
        tenant: tenant.to_string(),
        subject: Some(subject.to_string()),
    }
}

/// Mints entities through the real tool path, so the ids the tests then quote
/// are exactly the ones an agent would have been handed.
async fn seed_nodes(tools: &Tools, caller: &Principal, names: &[&str]) {
    tools
        .upsert_node(
            caller,
            Some("test-harness/1.0"),
            UpsertNodeInput {
                nodes: names
                    .iter()
                    .map(|display_name| NodeSpec {
                        display_name: (*display_name).to_string(),
                        aliases: Vec::new(),
                    })
                    .collect(),
            },
        )
        .await
        .expect("seeding nodes should succeed");
}

/// Writes facts through the real tool path.
async fn seed_edges(tools: &Tools, caller: &Principal, edges: &[(&str, &str, &str, &str)]) {
    tools
        .create_provenance_relationship(
            caller,
            Some("test-harness/1.0"),
            CreateRelationshipInput {
                relationships: edges
                    .iter()
                    .map(|(source, target, kind, proof)| RelationshipSpec {
                        source_node_id: (*source).to_string(),
                        target_node_id: (*target).to_string(),
                        relationship_type: (*kind).to_string(),
                        proof_excerpt: (*proof).to_string(),
                        valid_from: None,
                    })
                    .collect(),
            },
        )
        .await
        .expect("seeding relationships should succeed");
}

/// Writes facts and hands back the ids they were recorded under.
async fn write_edges(
    tools: &Tools,
    caller: &Principal,
    edges: &[(String, String, String, String)],
) -> Vec<uuid::Uuid> {
    let written = tools
        .create_provenance_relationship(
            caller,
            Some("test-harness/1.0"),
            CreateRelationshipInput {
                relationships: edges
                    .iter()
                    .map(|(source, target, kind, proof)| RelationshipSpec {
                        source_node_id: source.clone(),
                        target_node_id: target.clone(),
                        relationship_type: kind.clone(),
                        proof_excerpt: proof.clone(),
                        valid_from: None,
                    })
                    .collect(),
            },
        )
        .await
        .expect("seeding relationships should succeed");

    written
        .iter()
        .map(|report| match report {
            RelationshipReport::Created { edge_id, .. }
            | RelationshipReport::Backfilled { edge_id, .. } => *edge_id,
            other => panic!("expected a written relationship, got {other:#?}"),
        })
        .collect()
}

/// The document-side vector of `text`, as the resolver would compute it.
async fn embed(text: &str) -> Vec<f32> {
    DeterministicEmbedder::new()
        .embed_documents(&[text.to_string()])
        .await
        .expect("the deterministic embedder cannot fail")
        .pop()
        .expect("one text in, one vector out")
}

/// Builds a unit vector whose cosine with `base` is exactly `cosine`, by the
/// method `tests/node_upsert.rs` documents: `other` is projected off `base`, so
/// the result is `cos * base + sin * u` for a genuine orthonormal `u`.
fn at_cosine(base: &[f32], other: &[f32], cosine: f64) -> Vec<f32> {
    let dot: f64 = base
        .iter()
        .zip(other)
        .map(|(b, o)| f64::from(*b) * f64::from(*o))
        .sum();

    let rejection: Vec<f64> = other
        .iter()
        .zip(base)
        .map(|(o, b)| f64::from(*o) - dot * f64::from(*b))
        .collect();
    let norm = rejection.iter().map(|r| r * r).sum::<f64>().sqrt();

    let sine = (1.0 - cosine * cosine).sqrt();
    base.iter()
        .zip(&rejection)
        .map(|(b, r)| (cosine * f64::from(*b) + sine * (r / norm)) as f32)
        .collect()
}

/// Replaces a node's `name_vector`, which is the only way to put a controlled
/// cosine in front of a resolver whose embedder hashes its input.
///
/// The shape a rename leaves behind, and the same fixture the upsert and
/// disambiguation suites use.
async fn set_name_vector(tools: &Tools, tenant: &str, node_id: &str, vector: &[f32]) {
    sqlx::query(
        "UPDATE node SET name_vector = $3::real[]::vector \
         WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(tenant)
    .bind(node_id)
    .bind(vector)
    .execute(tools.store().pool())
    .await
    .expect("moving a node's vector should succeed");
}

/// The arguments in their commonest shape: a node, no filter, no depth.
fn exploring(node: &str) -> ExploreEntityInput {
    ExploreEntityInput {
        node: node.to_string(),
        relationship_type: None,
        depth: None,
    }
}

/// The single match an unambiguous exploration returns, or a failure naming
/// what came back instead.
fn only_match(matches: Vec<EntityMatchReport>) -> EntityMatchReport {
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one match, got {matches:#?}"
    );

    matches.into_iter().next().expect("just counted one")
}

/// The first thing the tool has to do: take an id the server minted, hand back
/// the entity behind it and the facts touching it.
///
/// Both directions of every edge count. A caller asking what is connected to
/// the gateway means every fact the gateway appears in, not only the ones it
/// happens to be the subject of — an `edge` is directed because the *fact* is,
/// not because a neighbourhood is.
#[tokio::test]
async fn an_exact_node_id_comes_back_with_the_edges_on_both_sides_of_it() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "exact-id").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(
        &tools,
        &caller,
        &["Gateway Service", "Marble Communicator", "Postgres Cluster"],
    )
    .await;
    seed_edges(
        &tools,
        &caller,
        &[
            (
                "gateway-service",
                "marble-communicator",
                "PROXIES_TO",
                "the gateway proxies every agent-facing call to the marble communicator",
            ),
            (
                "postgres-cluster",
                "gateway-service",
                "PERSISTS_TO",
                "the cluster holds the gateway's session table",
            ),
        ],
    )
    .await;

    let matches = tools
        .explore_entity(&caller, exploring("gateway-service"))
        .await
        .expect("exploring a known id should succeed");

    let found = only_match(matches);

    assert_eq!(found.node_id, "gateway-service");
    assert_eq!(found.display_name, "Gateway Service");
    assert_eq!(
        found.matched_by,
        MatchedByReport::NodeId,
        "the id the server would mint for this text is taken, which is the \
         most certain tier there is"
    );
    assert!(
        found.similarity.is_none(),
        "an exact hit has no cosine to report, and a number here would be one \
         a model could weigh: {found:#?}"
    );
    assert_eq!(
        found.degree, 2,
        "degree counts every active edge touching the node, whichever end it is on"
    );
    assert_eq!(
        found.edges.len(),
        2,
        "and both of them come back: {found:#?}"
    );
    assert_eq!(found.edges_withheld, 0, "nothing was capped away");
    assert!(
        found.neighbours.is_none(),
        "depth 1 was not asked to look past the first hop: {found:#?}"
    );

    let outgoing = found
        .edges
        .iter()
        .find(|edge| edge.target_node_id == "marble-communicator")
        .expect("the edge the gateway is the source of");
    assert_eq!(outgoing.source_node_id, "gateway-service");
    assert_eq!(outgoing.relationship_type, "PROXIES_TO");
    assert_eq!(
        outgoing.proof, "the gateway proxies every agent-facing call to the marble communicator",
        "the proof is the words that justified the fact, quoted back"
    );
    assert!(!outgoing.proof_truncated, "this one fits whole");
    assert!(!outgoing.disputed);
    assert!(
        outgoing.salience > 0.9,
        "a fact written a moment ago has barely decayed, got {}",
        outgoing.salience
    );

    let incoming = found
        .edges
        .iter()
        .find(|edge| edge.source_node_id == "postgres-cluster")
        .expect("the edge the gateway is the target of");
    assert_eq!(incoming.target_node_id, "gateway-service");
    assert_eq!(incoming.relationship_type, "PERSISTS_TO");
}

/// A name, not an id — and one that only exists as an alias.
///
/// This is the tier that makes the tool usable from a conversation: an agent
/// that read "commsdk" in a log has no `node_id`, and the graph already knows
/// that name belongs to something. Reusing the upsert ladder is what gets it
/// for free.
#[tokio::test]
async fn an_alias_resolves_to_the_entity_that_answers_to_it() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "alias").await;
    let caller = principal(&tenant, "agent-a");

    tools
        .upsert_node(
            &caller,
            Some("test-harness/1.0"),
            UpsertNodeInput {
                nodes: vec![
                    NodeSpec {
                        display_name: "Marble Communicator".to_string(),
                        aliases: vec!["commsdk".to_string()],
                    },
                    NodeSpec {
                        display_name: "Gateway Service".to_string(),
                        aliases: Vec::new(),
                    },
                ],
            },
        )
        .await
        .expect("seeding should succeed");
    seed_edges(
        &tools,
        &caller,
        &[(
            "gateway-service",
            "marble-communicator",
            "PROXIES_TO",
            "the gateway proxies to the communicator",
        )],
    )
    .await;

    let found = only_match(
        tools
            .explore_entity(&caller, exploring("commsdk"))
            .await
            .expect("exploring by alias should succeed"),
    );

    assert_eq!(found.node_id, "marble-communicator");
    assert_eq!(found.display_name, "Marble Communicator");
    assert_eq!(
        found.matched_by,
        MatchedByReport::NameOrAlias,
        "an alias is a name some agent wrote down, not a cosine: {found:#?}"
    );
    assert!(found.similarity.is_none());
    assert_eq!(found.aliases, vec!["commsdk".to_string()]);
    assert_eq!(found.edges.len(), 1);
}

/// A name this memory nearly holds.
///
/// The third tier of the same ladder, and the one whose answer a model must not
/// read as certain — so the match reports how it was found and what the cosine
/// was. Thresholds are not restated here: the node sits at 0.95, above the
/// long-name confident bar the resolver already enforces, and the assertion is
/// on what came back rather than on the number that let it through.
#[tokio::test]
async fn a_near_name_resolves_by_similarity_and_reports_the_cosine() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "similarity").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(&tools, &caller, &["Marble Communicator", "Gateway Service"]).await;
    seed_edges(
        &tools,
        &caller,
        &[(
            "gateway-service",
            "marble-communicator",
            "PROXIES_TO",
            "the gateway proxies to the communicator",
        )],
    )
    .await;

    let asked = "the marble agent communicator";
    let near = at_cosine(
        &embed(asked).await,
        &embed("an unrelated anchor").await,
        0.95,
    );
    set_name_vector(&tools, &tenant, "marble-communicator", &near).await;

    let found = only_match(
        tools
            .explore_entity(&caller, exploring(asked))
            .await
            .expect("exploring by a near name should succeed"),
    );

    assert_eq!(found.node_id, "marble-communicator");
    assert_eq!(
        found.matched_by,
        MatchedByReport::Similarity,
        "nothing in the graph carries this name: {found:#?}"
    );

    let similarity = found
        .similarity
        .expect("a semantic match has to say how close it was");
    assert!(
        (similarity - 0.95).abs() < 1e-3,
        "the cosine the resolver acted on should be the cosine reported, got {similarity}"
    );
    assert_eq!(found.edges.len(), 1);
}

/// A name this memory has never heard of is not an error.
///
/// An empty answer is information — "there is nothing here" — and it costs one
/// call to learn. An error would make an agent retry, or worse, treat a missing
/// entity as a broken server.
#[tokio::test]
async fn a_name_this_memory_does_not_hold_comes_back_empty_rather_than_failing() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "unknown").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(&tools, &caller, &["Gateway Service"]).await;

    let matches = tools
        .explore_entity(&caller, exploring("Hydroelectric Dam Number Nine"))
        .await
        .expect("an unknown entity is an answer, not a failure");

    assert!(
        matches.is_empty(),
        "nothing in this tenant resembles that name, got {matches:#?}"
    );
}

/// The filter takes the caller's wording, not the server's spelling.
///
/// `create_provenance_relationship` folds `"depends on"` onto `DEPENDS_ON`
/// before it writes, so a filter that did not would be a filter that misses
/// every edge the write path just recorded. The same exact tiers run here, and
/// this is the test that would notice them being restated rather than reused.
///
/// `degree` is asserted alongside, because it is what tells a caller that the
/// filter — rather than the graph — is why the list is short.
#[tokio::test]
async fn a_relationship_type_filter_accepts_the_spelling_the_write_path_accepts() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "type-filter").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(
        &tools,
        &caller,
        &[
            "Gateway Service",
            "Postgres Cluster",
            "Redis Cache",
            "Marble Communicator",
        ],
    )
    .await;
    seed_edges(
        &tools,
        &caller,
        &[
            (
                "gateway-service",
                "postgres-cluster",
                "DEPENDS_ON",
                "the gateway cannot start without the cluster",
            ),
            (
                "gateway-service",
                "redis-cache",
                "depends on",
                "the gateway reads its session cache from redis",
            ),
            (
                "gateway-service",
                "marble-communicator",
                "PROXIES_TO",
                "the gateway proxies agent traffic to the communicator",
            ),
        ],
    )
    .await;

    let filtered = only_match(
        tools
            .explore_entity(
                &caller,
                ExploreEntityInput {
                    node: "gateway-service".to_string(),
                    relationship_type: Some("depends on".to_string()),
                    depth: None,
                },
            )
            .await
            .expect("filtering by a coerced spelling should succeed"),
    );

    assert_eq!(
        filtered.degree, 3,
        "degree is the whole neighbourhood, not the filtered view: {filtered:#?}"
    );
    assert_eq!(filtered.edges.len(), 2, "both dependencies: {filtered:#?}");
    assert!(
        filtered
            .edges
            .iter()
            .all(|edge| edge.relationship_type == "DEPENDS_ON"),
        "and nothing else: {filtered:#?}"
    );
    assert_eq!(
        filtered.edges_withheld, 0,
        "an edge a filter excluded was not withheld, it was not asked for"
    );

    let one_type = only_match(
        tools
            .explore_entity(
                &caller,
                ExploreEntityInput {
                    node: "gateway-service".to_string(),
                    relationship_type: Some("PROXIES_TO".to_string()),
                    depth: None,
                },
            )
            .await
            .expect("filtering by a canonical name should succeed"),
    );
    assert_eq!(one_type.edges.len(), 1);
    assert_eq!(one_type.edges[0].target_node_id, "marble-communicator");
}

/// A filter this vocabulary has never seen matches nothing — and, crucially,
/// **registers nothing**.
///
/// The write path coins a type when nothing fits, because an agent asserting a
/// fact has said something new. A caller filtering has not: it has typed a word.
/// Registering it would put junk in the vocabulary that later writes would then
/// be coerced onto, and would make a read-only tool a writer.
#[tokio::test]
async fn an_unknown_type_filters_everything_out_and_coins_nothing() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "unknown-type").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(&tools, &caller, &["Gateway Service", "Postgres Cluster"]).await;
    seed_edges(
        &tools,
        &caller,
        &[(
            "gateway-service",
            "postgres-cluster",
            "DEPENDS_ON",
            "the gateway cannot start without the cluster",
        )],
    )
    .await;

    sqlx::query("DELETE FROM relationship_type WHERE tenant_id = $1")
        .bind(&tenant)
        .execute(tools.store().pool())
        .await
        .expect("clearing this tenant's private vocabulary should succeed");

    let found = only_match(
        tools
            .explore_entity(
                &caller,
                ExploreEntityInput {
                    node: "gateway-service".to_string(),
                    relationship_type: Some("weathervanes for".to_string()),
                    depth: None,
                },
            )
            .await
            .expect("an unknown type is an answer, not a failure"),
    );

    assert!(
        found.edges.is_empty(),
        "no edge is of a type this memory does not hold: {found:#?}"
    );
    assert_eq!(
        found.degree, 1,
        "and the degree still says the entity is connected to something"
    );

    let (coined,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM relationship_type WHERE tenant_id = $1")
            .bind(&tenant)
            .fetch_one(tools.store().pool())
            .await
            .expect("counting this tenant's vocabulary should succeed");
    assert_eq!(
        coined, 0,
        "exploring must not teach the vocabulary a word the caller merely typed"
    );
}

/// **The measurement in `src/store/explore.rs` made into a test.**
///
/// The second hop returns neighbours, not the neighbours' facts. On the real
/// graph the difference is 100 KB of proof text against a few hundred bytes of
/// counts, and it is what makes `depth: 2` a thing a model can afford to ask
/// for. So the shape is asserted, not merely the contents: at depth 1 the field
/// is absent, at depth 2 each neighbour carries a degree and a breakdown by
/// type and nothing that scales with the size of the graph.
#[tokio::test]
async fn depth_two_summarises_the_neighbours_instead_of_returning_their_edges() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "depth-two").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(
        &tools,
        &caller,
        &[
            "Gateway Service",
            "Marble Communicator",
            "Postgres Cluster",
            "Redis Cache",
            "Nginx Container",
        ],
    )
    .await;
    seed_edges(
        &tools,
        &caller,
        &[
            (
                "gateway-service",
                "marble-communicator",
                "PROXIES_TO",
                "the gateway proxies agent traffic to the communicator",
            ),
            (
                "gateway-service",
                "postgres-cluster",
                "DEPENDS_ON",
                "the gateway cannot start without the cluster",
            ),
            (
                "marble-communicator",
                "postgres-cluster",
                "DEPENDS_ON",
                "the communicator writes its outbox to the cluster",
            ),
            (
                "marble-communicator",
                "redis-cache",
                "DEPENDS_ON",
                "the communicator keeps its dedupe window in redis",
            ),
            (
                "marble-communicator",
                "nginx-container",
                "USES",
                "the communicator is fronted by the nginx container",
            ),
        ],
    )
    .await;

    let shallow = only_match(
        tools
            .explore_entity(&caller, exploring("gateway-service"))
            .await
            .expect("depth 1 should succeed"),
    );
    assert!(
        shallow.neighbours.is_none(),
        "depth 1 was not asked to look past the first hop: {shallow:#?}"
    );

    let deep = only_match(
        tools
            .explore_entity(
                &caller,
                ExploreEntityInput {
                    node: "gateway-service".to_string(),
                    relationship_type: None,
                    depth: Some(2),
                },
            )
            .await
            .expect("depth 2 should succeed"),
    );

    assert_eq!(
        deep.edges.len(),
        2,
        "the first hop is unchanged by asking for a second: {deep:#?}"
    );

    let neighbours = deep
        .neighbours
        .as_ref()
        .expect("depth 2 has to say what is out there");
    assert_eq!(
        neighbours.len(),
        2,
        "one per first-hop edge: {neighbours:#?}"
    );

    let summarised = |node_id: &str| {
        neighbours
            .iter()
            .find(|neighbour| neighbour.node_id == node_id)
            .unwrap_or_else(|| panic!("{node_id} should be summarised, got {neighbours:#?}"))
            .clone()
    };

    let marble = summarised("marble-communicator");
    assert_eq!(marble.display_name, "Marble Communicator");
    assert_eq!(
        marble.degree, 4,
        "every active edge touching it, whichever end: {marble:#?}"
    );
    assert_eq!(
        marble.relationship_types,
        BTreeMap::from([
            ("DEPENDS_ON".to_string(), 2),
            ("PROXIES_TO".to_string(), 1),
            ("USES".to_string(), 1),
        ]),
        "counts by type are the whole of what a model needs to choose the next call"
    );
    assert_eq!(
        marble.relationship_types.values().sum::<i64>(),
        marble.degree,
        "the counts have to add up to the degree beside them"
    );

    let postgres = summarised("postgres-cluster");
    assert_eq!(postgres.degree, 2);
    assert_eq!(
        postgres.relationship_types,
        BTreeMap::from([("DEPENDS_ON".to_string(), 2)])
    );
}

/// A depth this memory will not serve is clamped, not refused.
///
/// A caller that asked for three hops asked a reasonable question badly, and an
/// error would cost it a round trip to learn a number the tool description
/// already gives it. What it gets is the deepest walk that is affordable, which
/// is the answer it would have asked for had it read the description.
#[tokio::test]
async fn a_depth_past_the_second_hop_is_clamped_rather_than_refused() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "depth-clamp").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(&tools, &caller, &["Gateway Service", "Postgres Cluster"]).await;
    seed_edges(
        &tools,
        &caller,
        &[(
            "gateway-service",
            "postgres-cluster",
            "DEPENDS_ON",
            "the gateway cannot start without the cluster",
        )],
    )
    .await;

    let asked_for_two = only_match(
        tools
            .explore_entity(
                &caller,
                ExploreEntityInput {
                    node: "gateway-service".to_string(),
                    relationship_type: None,
                    depth: Some(2),
                },
            )
            .await
            .expect("depth 2 should succeed"),
    );

    let asked_for_nine = only_match(
        tools
            .explore_entity(
                &caller,
                ExploreEntityInput {
                    node: "gateway-service".to_string(),
                    relationship_type: None,
                    depth: Some(9),
                },
            )
            .await
            .expect("an over-deep walk is clamped, not refused"),
    );

    assert_eq!(
        asked_for_nine.neighbours, asked_for_two.neighbours,
        "nine hops is two hops: {asked_for_nine:#?}"
    );
}

/// **The bound, and the number that says it was applied.**
///
/// A hub does not fit in a tool result, so something has to be left out. What
/// must not happen is leaving it out silently: a caller that cannot tell a
/// small neighbourhood from a truncated one will read the first as the second
/// and stop looking. So the cap is [`EDGE_BUDGET`], the order is salience —
/// the ranking this store already keeps — and the remainder is counted.
#[tokio::test]
async fn a_neighbourhood_past_the_budget_is_capped_by_salience_and_the_rest_is_counted() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "edge-cap").await;
    let caller = principal(&tenant, "agent-a");

    let over_budget = EDGE_BUDGET + 5;

    let mut names: Vec<String> = vec!["Gateway Service".to_string()];
    names.extend((0..over_budget).map(|index| format!("Dependency {index}")));
    let borrowed: Vec<&str> = names.iter().map(String::as_str).collect();
    seed_nodes(&tools, &caller, &borrowed).await;

    let facts: Vec<(String, String, String, String)> = (0..over_budget)
        .map(|index| {
            (
                "gateway-service".to_string(),
                format!("dependency-{index}"),
                "DEPENDS_ON".to_string(),
                format!("the gateway cannot start without dependency {index}"),
            )
        })
        .collect();
    let written = write_edges(&tools, &caller, &facts).await;

    // Five facts nobody found useful. Salience is what decides which edges
    // survive the cap, so these are the five that should not come back.
    let demoted: Vec<uuid::Uuid> = written.iter().rev().take(5).copied().collect();
    tools
        .rate_memory_utility(
            &caller,
            RateMemoryUtilityInput {
                edge_ids: demoted.clone(),
                utility_score: -1.0,
            },
        )
        .await
        .expect("rating should succeed");

    let found = only_match(
        tools
            .explore_entity(&caller, exploring("gateway-service"))
            .await
            .expect("exploring a hub should succeed"),
    );

    assert_eq!(
        found.degree, over_budget as i64,
        "the degree is the truth about the entity, whatever fits in the answer"
    );
    assert_eq!(
        found.edges.len(),
        EDGE_BUDGET,
        "one match spends the whole budget and no more"
    );
    assert_eq!(
        found.edges_withheld,
        5,
        "and says how many it did not send: {}",
        found.edges.len()
    );

    for edge in &found.edges {
        assert!(
            !demoted.contains(&edge.edge_id),
            "a demoted fact outranked one nobody rated down: {edge:#?}"
        );
    }
}

/// The budget is shared between contested candidates rather than granted to
/// each of them.
///
/// A name that resolves to five entities must not cost five times what a
/// certain one costs — that is the pathological case a per-match cap would
/// wave through.
#[tokio::test]
async fn several_matches_share_one_budget_between_them() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "shared-budget").await;
    let caller = principal(&tenant, "agent-a");

    let per_match = EDGE_BUDGET / 2;
    let over_budget = per_match + 3;

    let mut names: Vec<String> = vec!["Phoenix Alpha".to_string(), "Phoenix Beta".to_string()];
    names.extend((0..over_budget).map(|index| format!("Dependency {index}")));
    let borrowed: Vec<&str> = names.iter().map(String::as_str).collect();
    seed_nodes(&tools, &caller, &borrowed).await;

    let mut facts: Vec<(String, String, String, String)> = Vec::new();
    for owner in ["phoenix-alpha", "phoenix-beta"] {
        facts.extend((0..over_budget).map(|index| {
            (
                owner.to_string(),
                format!("dependency-{index}"),
                "DEPENDS_ON".to_string(),
                format!("{owner} cannot start without dependency {index}"),
            )
        }));
    }
    for chunk in facts.chunks(60) {
        write_edges(&tools, &caller, chunk).await;
    }

    // Two entities in the uncertain band around one name: the resolver commits
    // to neither and offers both, which is the only way this tool ever reports
    // more than one match.
    let asked = "Phoenix";
    let anchor = embed("an unrelated anchor").await;
    for node_id in ["phoenix-alpha", "phoenix-beta"] {
        let near = at_cosine(&embed(asked).await, &anchor, 0.87);
        set_name_vector(&tools, &tenant, node_id, &near).await;
    }

    let matches = tools
        .explore_entity(&caller, exploring(asked))
        .await
        .expect("exploring a contested name should succeed");

    assert_eq!(
        matches.len(),
        2,
        "the ladder offered two candidates and neither was certain: {matches:#?}"
    );

    let total: usize = matches.iter().map(|found| found.edges.len()).sum();
    assert_eq!(
        total, EDGE_BUDGET,
        "two matches share one search's worth of proof text: {matches:#?}"
    );

    for found in &matches {
        assert_eq!(found.edges.len(), per_match);
        assert_eq!(
            found.edges_withheld, 3,
            "each says what its own share left out: {found:#?}"
        );
        assert_eq!(
            found.matched_by,
            MatchedByReport::Similarity,
            "a candidate list is near misses, never a lookup: {found:#?}"
        );
    }
}

/// **The isolation boundary, on the one operation that could walk across it.**
///
/// A traversal is the tool most able to leave a tenant one edge at a time: the
/// resolution reads nodes, the first hop reads edges, the second hop reads
/// nodes again, and any one of those forgetting its `tenant_id` would hand
/// another customer's graph to whoever asked. So both halves are asserted — an
/// entity only the other tenant has is invisible, and an id both tenants hold
/// resolves to the caller's own.
#[tokio::test]
async fn one_tenants_entities_are_invisible_to_another() {
    let tools = tools().await;
    let theirs = fresh_tenant(&tools, "isolation-theirs").await;
    let ours = fresh_tenant(&tools, "isolation-ours").await;
    let them = principal(&theirs, "agent-them");
    let us = principal(&ours, "agent-us");

    seed_nodes(
        &tools,
        &them,
        &["Gateway Service", "Postgres Cluster", "Secret Warehouse"],
    )
    .await;
    seed_edges(
        &tools,
        &them,
        &[
            (
                "gateway-service",
                "postgres-cluster",
                "DEPENDS_ON",
                "their gateway depends on their cluster",
            ),
            (
                "gateway-service",
                "secret-warehouse",
                "PERSISTS_TO",
                "their gateway writes to a warehouse we should never hear about",
            ),
        ],
    )
    .await;

    seed_nodes(&tools, &us, &["Gateway Service", "Redis Cache"]).await;
    seed_edges(
        &tools,
        &us,
        &[(
            "gateway-service",
            "redis-cache",
            "USES",
            "our gateway uses our cache",
        )],
    )
    .await;

    let found = only_match(
        tools
            .explore_entity(
                &us,
                ExploreEntityInput {
                    node: "gateway-service".to_string(),
                    relationship_type: None,
                    depth: Some(2),
                },
            )
            .await
            .expect("exploring our own graph should succeed"),
    );

    assert_eq!(
        found.degree, 1,
        "the other tenant's two edges are not ours to count: {found:#?}"
    );
    assert_eq!(found.edges.len(), 1);
    assert_eq!(found.edges[0].target_node_id, "redis-cache");
    assert_eq!(
        found.neighbours.as_ref().map(|neighbours| neighbours.len()),
        Some(1),
        "and the second hop stays inside the tenant too: {found:#?}"
    );

    let serialised = serde_json::to_string(&found).expect("a report should serialise");
    assert!(
        !serialised.contains("secret-warehouse") && !serialised.contains("warehouse"),
        "nothing of theirs may appear anywhere in our answer: {serialised}"
    );

    let only_theirs = tools
        .explore_entity(&us, exploring("Secret Warehouse"))
        .await
        .expect("exploring a name we do not have is an answer, not a failure");
    assert!(
        only_theirs.is_empty(),
        "an entity only the other tenant holds does not exist for us: {only_theirs:#?}"
    );
}

/// History is real data and this is not the tool that shows it.
///
/// The graph is bi-temporal: a superseded fact is kept, not deleted, and an
/// archived one is an operator's judgement rather than an erasure. Neither is
/// what a caller asking "what is this connected to" is asking for, and a
/// retired fact reads exactly like a live one once it is in a model's context.
/// So the neighbourhood is the current picture — in the edges *and* in the
/// degree beside them, which is the number that would otherwise quietly count
/// history back in.
#[tokio::test]
async fn retired_facts_are_in_neither_the_edges_nor_the_degree() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "only-active").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(
        &tools,
        &caller,
        &["Gateway Service", "Postgres Cluster", "Redis Cache"],
    )
    .await;
    let written = write_edges(
        &tools,
        &caller,
        &[
            (
                "gateway-service".to_string(),
                "postgres-cluster".to_string(),
                "DEPENDS_ON".to_string(),
                "the gateway cannot start without the cluster".to_string(),
            ),
            (
                "gateway-service".to_string(),
                "redis-cache".to_string(),
                "DEPENDS_ON".to_string(),
                "the gateway used to keep sessions in redis".to_string(),
            ),
        ],
    )
    .await;

    sqlx::query("UPDATE edge SET status = 'superseded' WHERE tenant_id = $1 AND edge_id = $2")
        .bind(&tenant)
        .bind(written[1])
        .execute(tools.store().pool())
        .await
        .expect("retiring an edge should succeed");

    let found = only_match(
        tools
            .explore_entity(
                &caller,
                ExploreEntityInput {
                    node: "gateway-service".to_string(),
                    relationship_type: None,
                    depth: Some(2),
                },
            )
            .await
            .expect("exploring should succeed"),
    );

    assert_eq!(
        found.degree, 1,
        "a retired fact is not part of how connected an entity is now: {found:#?}"
    );
    assert_eq!(
        found
            .edges
            .iter()
            .map(|edge| edge.edge_id)
            .collect::<Vec<_>>(),
        vec![written[0]],
    );
    assert_eq!(
        found.neighbours.as_ref().map(|neighbours| neighbours.len()),
        Some(1),
        "and the entity the retired fact pointed at is not a neighbour: {found:#?}"
    );
}

/// A merged-away id still resolves, and the tombstone is never what comes back.
///
/// Agents hold `node_id`s in contexts this server cannot reach or correct, so a
/// merge leaves the old row behind as a redirect. The resolution ladder follows
/// it; this is the test that the traversal reports the entity it landed on
/// rather than the redirect it started from.
#[tokio::test]
async fn a_merged_away_id_explores_its_successor_rather_than_the_tombstone() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "tombstone").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(
        &tools,
        &caller,
        &["Marble Communicator", "Commsdk Marble", "Gateway Service"],
    )
    .await;
    seed_edges(
        &tools,
        &caller,
        &[(
            "gateway-service",
            "commsdk-marble",
            "PROXIES_TO",
            "the gateway proxies agent traffic to the communicator",
        )],
    )
    .await;

    tools
        .store()
        .merge_nodes(
            &tenant,
            "commsdk-marble",
            "marble-communicator",
            &Default::default(),
        )
        .await
        .expect("merging should succeed");

    let found = only_match(
        tools
            .explore_entity(&caller, exploring("commsdk-marble"))
            .await
            .expect("a retired id still resolves"),
    );

    assert_eq!(
        found.node_id, "marble-communicator",
        "the redirect leads to the entity, not to itself: {found:#?}"
    );
    assert_eq!(found.display_name, "Marble Communicator");
    assert_eq!(found.degree, 1, "the edge came with it: {found:#?}");
}

/// Proof text is snipped on exactly the budget a search snips it on.
///
/// Two answers to "how much of a proof does this memory show you" — differing
/// by which tool asked — is a difference a model cannot see and cannot correct
/// for. So this reuses `search`'s snippet rather than choosing a second length,
/// and the flag says the text was cut.
#[tokio::test]
async fn a_long_proof_comes_back_snipped_and_flagged_as_it_does_in_a_search() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "snippet").await;
    let caller = principal(&tenant, "agent-a");

    seed_nodes(&tools, &caller, &["Gateway Service", "Postgres Cluster"]).await;

    let long_proof = "the gateway cannot start without the cluster, and here is why: ".repeat(20);
    assert!(
        long_proof.chars().count() > 400,
        "this fixture only tests anything if the proof is longer than a snippet"
    );
    seed_edges(
        &tools,
        &caller,
        &[(
            "gateway-service",
            "postgres-cluster",
            "DEPENDS_ON",
            &long_proof,
        )],
    )
    .await;

    let found = only_match(
        tools
            .explore_entity(&caller, exploring("gateway-service"))
            .await
            .expect("exploring should succeed"),
    );

    let edge = &found.edges[0];
    assert!(edge.proof_truncated, "the caller has to be told: {edge:#?}");
    assert!(
        edge.proof.chars().count() < long_proof.chars().count(),
        "and the text has to actually be shorter: {edge:#?}"
    );

    let hits = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: "the gateway cannot start without the cluster".to_string(),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");

    assert_eq!(
        edge.proof,
        hits.first()
            .expect("the search should find the fact this test just wrote")
            .proof,
        "one memory, one snippet length"
    );
}
