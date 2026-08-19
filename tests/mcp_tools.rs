//! Integration tests for the MCP tool layer, against a real Postgres.
//!
//! These exercise the *operations* — the plain async methods the `#[tool]`
//! adapters delegate to — rather than a live MCP session, because the
//! transport does not exist yet. That is the whole point of the split: the
//! logic is reachable with nothing but a `Principal` and a `Store`.
//!
//! Every test owns a `tenant_id` derived from its own name, for the reason the
//! store suites give: the suite runs in parallel against one shared database,
//! so isolation cannot come from truncating shared tables. Tenancy is the
//! isolation mechanism the schema already has.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};

use mundu::auth::Principal;
use mundu::embed::DeterministicEmbedder;
use mundu::mcp::tools::{
    CreateRelationshipInput, NodeSpec, PrecisionRequirement, RateMemoryUtilityInput,
    RatingRejectionReason, RatingReport, RejectionReason, RelationshipReport, RelationshipSpec,
    SearchInput, SearchMethod, Tools, UpsertNodeInput,
};
use mundu::store::Store;

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
    let tenant = format!("test-mcp-{name}");

    for statement in [
        "DELETE FROM edge_rating WHERE tenant_id = $1",
        "DELETE FROM edge WHERE tenant_id = $1",
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

fn node(display_name: &str) -> NodeSpec {
    NodeSpec {
        display_name: display_name.to_string(),
        aliases: Vec::new(),
    }
}

fn relationship(
    source_node_id: &str,
    target_node_id: &str,
    relationship_type: &str,
    proof_excerpt: &str,
) -> RelationshipSpec {
    RelationshipSpec {
        source_node_id: source_node_id.to_string(),
        target_node_id: target_node_id.to_string(),
        relationship_type: relationship_type.to_string(),
        proof_excerpt: proof_excerpt.to_string(),
        valid_from: None,
    }
}

/// Mints the entities a relationship test needs through the real tool path, so
/// the ids it then quotes are exactly the ones an agent would have been handed.
async fn seed_nodes(tools: &Tools, caller: &Principal, display_names: &[&str]) {
    tools
        .upsert_node(
            caller,
            Some("test-harness/1.0"),
            UpsertNodeInput {
                nodes: display_names.iter().copied().map(node).collect(),
            },
        )
        .await
        .expect("seeding nodes should succeed");
}

/// The edge id a report carries, or a failure naming what came back instead.
fn edge_id_of(report: &RelationshipReport) -> uuid::Uuid {
    match report {
        RelationshipReport::Created { edge_id, .. }
        | RelationshipReport::Backfilled { edge_id, .. } => *edge_id,
        other => panic!("expected a written relationship, got {other:#?}"),
    }
}

/// The first operation, end to end: names in, server-minted ids out, and the
/// caller told for each one whether it resolved to something that already
/// existed or minted a node.
///
/// `display_name` is echoed back deliberately. The store's outcome carries only
/// an id, and an agent that sent twenty names needs to know which answer
/// belongs to which — an ordering it should not have to trust silently.
#[tokio::test]
async fn upsert_node_mints_ids_and_reports_what_resolved() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "upsert-basics").await;
    let caller = principal(&tenant, "agent-a");

    let created = tools
        .upsert_node(
            &caller,
            Some("test-harness/1.0"),
            UpsertNodeInput {
                nodes: vec![
                    NodeSpec {
                        display_name: "Alice Whitfield".to_string(),
                        aliases: vec!["Alice W".to_string()],
                    },
                    node("Project Phoenix"),
                ],
            },
        )
        .await
        .expect("upserting two fresh nodes should succeed");

    assert_eq!(created.len(), 2, "one report per input, in input order");
    assert_eq!(created[0].node_id, "alice-whitfield");
    assert_eq!(created[0].display_name, "Alice Whitfield");
    assert!(
        !created[0].matched,
        "a name nothing resolved to is a new node, not a match: {created:#?}"
    );
    assert!(
        created[0].candidates.is_none(),
        "an unambiguous creation has no near misses to report: {created:#?}"
    );
    assert_eq!(created[1].node_id, "project-phoenix");

    // The alias is what makes the second call a match rather than a second
    // creation: it names the same entity by a name that is not its id.
    let matched = tools
        .upsert_node(
            &caller,
            Some("test-harness/1.0"),
            UpsertNodeInput {
                nodes: vec![node("Alice W")],
            },
        )
        .await
        .expect("upserting a recorded alias should succeed");

    assert!(
        matched[0].matched,
        "an alias of an existing node must resolve to it: {matched:#?}"
    );
    assert_eq!(matched[0].node_id, "alice-whitfield");
    assert_eq!(
        matched[0].display_name, "Alice W",
        "the report echoes the name the caller sent, not the node's"
    );
}

/// Provenance is assembled by the tool layer from the request's client info and
/// the credential's subject, and by nothing else. It is asserted against the
/// stored row rather than the return value, because the row is what an operator
/// auditing a poisoned fact will actually read.
#[tokio::test]
async fn upsert_node_records_the_client_and_principal_as_provenance() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "upsert-provenance").await;
    let caller = principal(&tenant, "agent-a");

    tools
        .upsert_node(
            &caller,
            Some("acme-agent/2.1"),
            UpsertNodeInput {
                nodes: vec![node("Alice Whitfield")],
            },
        )
        .await
        .expect("upserting should succeed");

    let (client, written_by): (Option<String>, Option<String>) = sqlx::query_as(
        "SELECT writer_client, writer_principal FROM node \
         WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(&tenant)
    .bind("alice-whitfield")
    .fetch_one(tools.store().pool())
    .await
    .expect("the node should have been written");

    assert_eq!(client.as_deref(), Some("acme-agent/2.1"));
    assert_eq!(written_by.as_deref(), Some("agent-a"));
}

/// The second write tool, end to end: node ids and proof text in, edge ids out,
/// and the caller's `valid_from` honoured rather than quietly replaced by the
/// arrival time.
#[tokio::test]
async fn create_provenance_relationship_writes_edges_and_hands_back_their_ids() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "relationship-basics").await;
    let caller = principal(&tenant, "agent-a");
    seed_nodes(
        &tools,
        &caller,
        &["Alice Whitfield", "Bob Nkemelu", "Kafka"],
    )
    .await;

    let asserted_at = Utc.with_ymd_and_hms(2019, 3, 1, 12, 0, 0).single().unwrap();

    let reports = tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![
                    RelationshipSpec {
                        valid_from: Some(asserted_at),
                        ..relationship(
                            "alice-whitfield",
                            "bob-nkemelu",
                            "REPORTS_TO",
                            "Alice Whitfield reports to Bob Nkemelu, per the org chart",
                        )
                    },
                    relationship(
                        "alice-whitfield",
                        "kafka",
                        "USES",
                        "Alice Whitfield's team publishes settlement events to Kafka",
                    ),
                ],
            },
        )
        .await
        .expect("writing two well-formed relationships should succeed");

    assert_eq!(reports.len(), 2, "one report per input, in input order");
    assert!(
        matches!(reports[0], RelationshipReport::Created { .. }),
        "a first fact about a pair is current, not history: {reports:#?}"
    );

    let (stored_type, valid_from, client, written_by): (
        String,
        DateTime<Utc>,
        Option<String>,
        Option<String>,
    ) = sqlx::query_as(
        "SELECT relationship_type, valid_from, writer_client, writer_principal FROM edge \
         WHERE tenant_id = $1 AND edge_id = $2",
    )
    .bind(&tenant)
    .bind(edge_id_of(&reports[0]))
    .fetch_one(tools.store().pool())
    .await
    .expect("the edge should have been written");

    assert_eq!(stored_type, "REPORTS_TO");
    assert_eq!(
        valid_from, asserted_at,
        "the caller's valid_from must reach the store, not be replaced by now()"
    );
    assert_eq!(client.as_deref(), Some("acme-agent/2.1"));
    assert_eq!(written_by.as_deref(), Some("agent-a"));
}

/// One conversation's worth of memory should cost two calls, not thirty — which
/// only holds if a single bad item does not sink the other twenty-nine.
///
/// The batch below misspells one endpoint. The rejection has to name what it
/// could not find and offer the ids it might have meant, and every other item
/// has to be committed and readable afterwards.
#[tokio::test]
async fn one_unknown_endpoint_fails_alone_and_its_siblings_are_committed() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "relationship-partial").await;
    let caller = principal(&tenant, "agent-a");
    seed_nodes(
        &tools,
        &caller,
        &["Alice Whitfield", "Bob Nkemelu", "Kafka"],
    )
    .await;

    let reports = tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![
                    relationship(
                        "alice-whitfield",
                        "kafka",
                        "USES",
                        "Alice Whitfield's team publishes settlement events to Kafka",
                    ),
                    // The id an agent would produce from a half-remembered name.
                    relationship(
                        "alice-whitfield",
                        "bob-nkemelou",
                        "REPORTS_TO",
                        "Alice Whitfield reports to Bob Nkemelu, per the org chart",
                    ),
                    relationship(
                        "bob-nkemelu",
                        "kafka",
                        "USES",
                        "Bob Nkemelu's team consumes the settlement topic",
                    ),
                ],
            },
        )
        .await
        .expect("a batch with one bad item is still a successful call");

    assert_eq!(reports.len(), 3);
    assert!(matches!(reports[0], RelationshipReport::Created { .. }));
    assert!(matches!(reports[2], RelationshipReport::Created { .. }));

    let RelationshipReport::Rejected {
        reason,
        node_id,
        suggestions,
        ..
    } = &reports[1]
    else {
        panic!("the misspelled endpoint should have been rejected: {reports:#?}");
    };

    assert_eq!(*reason, RejectionReason::UnknownTargetNode);
    assert_eq!(
        node_id, "bob-nkemelou",
        "the rejection quotes the id exactly as the caller wrote it"
    );
    assert!(
        suggestions
            .iter()
            .any(|suggestion| suggestion.node_id == "bob-nkemelu"),
        "the near match the store found must reach the caller: {suggestions:#?}"
    );

    // The point of the whole exercise: the good items are in the database.
    let (written,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM edge WHERE tenant_id = $1 AND status = 'active'")
            .bind(&tenant)
            .fetch_one(tools.store().pool())
            .await
            .expect("counting the tenant's edges should succeed");

    assert_eq!(
        written, 2,
        "the two well-formed relationships must survive their bad sibling"
    );
}

/// Every hit carries the handle the rating loop needs.
///
/// This is the one field the whole feedback loop hangs off: `rate_memory_utility`
/// consumes `edge_id`, and a stateless server has no session in which to
/// remember what it just showed. A hit without one is a fact an agent can read
/// and can never tell the server was useful.
#[tokio::test]
async fn every_search_hit_carries_an_edge_id_and_the_columns_around_it() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "search-shape").await;
    let caller = principal(&tenant, "agent-a");
    seed_nodes(&tools, &caller, &["Alice Whitfield", "Bob Nkemelu"]).await;

    let written = tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![relationship(
                    "alice-whitfield",
                    "bob-nkemelu",
                    "REPORTS_TO",
                    "Alice Whitfield reports to Bob Nkemelu, per the org chart",
                )],
            },
        )
        .await
        .expect("seeding a relationship should succeed");
    let edge_id = edge_id_of(&written[0]);

    // Nothing but a query: search_method and precision_requirement are
    // optional, and a model should not have to make a retrieval-engineering
    // decision to ask a question.
    let hits = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: "Alice Whitfield".to_string(),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");

    assert!(!hits.is_empty(), "the seeded fact should be findable");
    assert!(
        hits.iter().all(|hit| !hit.edge_id.is_nil()),
        "every hit must carry an edge_id: {hits:#?}"
    );

    let hit = hits
        .iter()
        .find(|hit| hit.edge_id == edge_id)
        .unwrap_or_else(|| panic!("the seeded edge should be among the hits: {hits:#?}"));

    assert_eq!(hit.source_node_id, "alice-whitfield");
    assert_eq!(hit.target_node_id, "bob-nkemelu");
    assert_eq!(hit.relationship_type, "REPORTS_TO");
    assert!(
        (0.0..=1.0).contains(&hit.effective_salience),
        "salience is a fraction: {hit:#?}"
    );
    assert!(hit.score > 0.0, "a returned hit scored above the cutoff");
    assert!(hit.valid_until.is_none(), "a current fact has no end");
    assert!(!hit.disputed);
    assert_eq!(hit.writer_client.as_deref(), Some("acme-agent/2.1"));
    assert_eq!(hit.writer_principal.as_deref(), Some("agent-a"));
    assert!(
        hit.proof.contains("reports to Bob Nkemelu"),
        "the proof excerpt comes back with the fact: {hit:#?}"
    );
    assert!(!hit.proof_truncated);
}

/// The optional arguments are optional *and* connected: each one has to reach
/// the store rather than be accepted and dropped.
#[tokio::test]
async fn the_optional_search_arguments_reach_the_store() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "search-options").await;
    let caller = principal(&tenant, "agent-a");
    seed_nodes(
        &tools,
        &caller,
        &[
            "Alice Whitfield",
            "Bob Nkemelu",
            "Zebra Logistics",
            "Quarterly Report",
            "Harbour Crane",
            "Night Shift",
        ],
    )
    .await;

    // Long enough to be snipped by default, so include_full_proof has something
    // to be observed doing.
    let long_proof = format!(
        "Alice Whitfield reports to Bob Nkemelu, per the org chart. {}",
        "The reorganisation memo restates the reporting line at length. ".repeat(5)
    );

    tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![
                    relationship("alice-whitfield", "bob-nkemelu", "REPORTS_TO", &long_proof),
                    relationship(
                        "zebra-logistics",
                        "quarterly-report",
                        "AUTHORED_BY",
                        "the freight volumes appendix was compiled by the logistics desk",
                    ),
                    relationship(
                        "harbour-crane",
                        "night-shift",
                        "USES",
                        "the harbour crane is booked for the night shift on Thursdays",
                    ),
                    relationship(
                        "zebra-logistics",
                        "harbour-crane",
                        "OWNS",
                        "the crane on berth four belongs to the logistics operator",
                    ),
                ],
            },
        )
        .await
        .expect("seeding relationships should succeed");

    let query = "Alice Whitfield".to_string();

    let snipped = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: query.clone(),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");
    assert!(
        snipped[0].proof_truncated && snipped[0].proof.ends_with('…'),
        "a long proof is snipped by default: {:#?}",
        snipped[0]
    );

    let whole = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: query.clone(),
                include_full_proof: true,
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");
    assert_eq!(
        whole[0].proof, long_proof,
        "include_full_proof must reach the store"
    );
    assert!(!whole[0].proof_truncated);

    let limited = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: query.clone(),
                limit: Some(1),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");
    assert_eq!(limited.len(), 1, "limit must reach the store");

    // The lexical half matches node names only, so a query naming no node
    // returns nothing through it — while the default hybrid search above
    // returned several. Anything less than that would pass on a search_method
    // that was accepted and ignored.
    let lexical_only = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: "settlement events pipeline".to_string(),
                search_method: Some(SearchMethod::Lexical),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");
    assert!(
        lexical_only.is_empty(),
        "search_method must reach the store: {lexical_only:#?}"
    );

    // One edge hangs off the node the query names and is therefore in both
    // candidate lists; the other three are only ever vector neighbours.
    let strict = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: query.clone(),
                precision_requirement: Some(PrecisionRequirement::Strict),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");
    let relaxed = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query,
                precision_requirement: Some(PrecisionRequirement::Relaxed),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");

    assert_eq!(strict.len(), 1, "at Strict the two lists must agree");
    assert_eq!(
        relaxed.len(),
        4,
        "at Relaxed the caller would rather sift than miss something"
    );
}

/// The loop the whole server exists to close: write a fact, find it again, and
/// tell the server the answer was worth having — with nothing carried between
/// the three calls but the `edge_id` the search handed back.
///
/// There is no session here on purpose. A stateless server cannot remember what
/// it just showed, so the handle has to travel through the model, which is why
/// `search_knowledge_graph` mints one for every hit.
#[tokio::test]
async fn the_write_search_rate_loop_closes_on_the_edge_ids_search_returned() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "full-loop").await;
    let caller = principal(&tenant, "agent-a");

    // 1. Write.
    tools
        .upsert_node(
            &caller,
            Some("acme-agent/2.1"),
            UpsertNodeInput {
                nodes: vec![node("Alice Whitfield"), node("Bob Nkemelu")],
            },
        )
        .await
        .expect("upserting should succeed");

    tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![relationship(
                    "alice-whitfield",
                    "bob-nkemelu",
                    "REPORTS_TO",
                    "Alice Whitfield reports to Bob Nkemelu, per the org chart",
                )],
            },
        )
        .await
        .expect("writing the relationship should succeed");

    // 2. Search. Everything the next step needs comes out of here.
    let hits = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: "who does Alice Whitfield report to".to_string(),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");
    assert!(!hits.is_empty(), "the fact just written should be findable");

    // Salience starts at the top, so the rating that has somewhere to go is a
    // negative one.
    let edge_ids: Vec<uuid::Uuid> = hits.iter().map(|hit| hit.edge_id).collect();

    // 3. Rate, using nothing but the ids the search returned.
    let reports = tools
        .rate_memory_utility(
            &caller,
            RateMemoryUtilityInput {
                edge_ids: edge_ids.clone(),
                utility_score: -1.0,
            },
        )
        .await
        .expect("rating the ids a search returned should succeed");

    assert_eq!(reports.len(), edge_ids.len(), "one report per id");
    for report in &reports {
        let RatingReport::Rated { edge_id, salience } = report else {
            panic!("an id search just returned must be ratable: {reports:#?}");
        };
        assert!(
            edge_ids.contains(edge_id),
            "the report names the id it was given"
        );
        assert!(
            (*salience - 0.75).abs() < 1e-6,
            "a full point of utility moves a quarter of the scale, got {salience}"
        );
    }

    // The signal is durable, not just reported.
    let (logged,): (i64,) = sqlx::query_as("SELECT count(*) FROM edge_rating WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(tools.store().pool())
        .await
        .expect("counting ratings should succeed");
    assert_eq!(logged as usize, edge_ids.len());

    // And a later search sees the lowered salience, which is the point of the
    // loop: the feedback changes what the next lookup ranks first.
    let after = tools
        .search_knowledge_graph(
            &caller,
            SearchInput {
                query: "who does Alice Whitfield report to".to_string(),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching again should succeed");
    let rated = after
        .iter()
        .find(|hit| hit.edge_id == edge_ids[0])
        .expect("the rated fact should still be findable");
    assert!(
        rated.effective_salience < 0.8,
        "the rating should be visible to the next search: {rated:#?}"
    );
}

/// A stale handle in a batch of five costs that one rating, not the other four.
///
/// Agents carry `edge_id`s across calls, so one of them being retired or simply
/// wrong is the ordinary case rather than an exceptional one.
#[tokio::test]
async fn an_unknown_edge_id_is_rejected_alone() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "rating-partial").await;
    let caller = principal(&tenant, "agent-a");
    seed_nodes(&tools, &caller, &["Alice Whitfield", "Bob Nkemelu"]).await;

    let written = tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![relationship(
                    "alice-whitfield",
                    "bob-nkemelu",
                    "REPORTS_TO",
                    "Alice Whitfield reports to Bob Nkemelu, per the org chart",
                )],
            },
        )
        .await
        .expect("seeding a relationship should succeed");
    let known = edge_id_of(&written[0]);
    let stale = uuid::Uuid::new_v4();

    let reports = tools
        .rate_memory_utility(
            &caller,
            RateMemoryUtilityInput {
                edge_ids: vec![stale, known],
                utility_score: -0.5,
            },
        )
        .await
        .expect("a batch with one stale id is still a successful call");

    assert_eq!(reports.len(), 2);
    assert!(
        matches!(
            &reports[0],
            RatingReport::Rejected { edge_id, reason, .. }
                if *edge_id == stale && *reason == RatingRejectionReason::UnknownEdge
        ),
        "an id this memory does not hold is rejected on its own: {reports:#?}"
    );
    assert!(
        matches!(&reports[1], RatingReport::Rated { edge_id, .. } if *edge_id == known),
        "its sibling must still be rated: {reports:#?}"
    );

    let (logged,): (i64,) = sqlx::query_as("SELECT count(*) FROM edge_rating WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(tools.store().pool())
        .await
        .expect("counting ratings should succeed");
    assert_eq!(logged, 1, "exactly the good rating should be on the record");
}

/// A score off the scale is a property of the whole call, not of one id, so it
/// is refused before anything is written rather than reported per item.
#[tokio::test]
async fn a_utility_score_off_the_scale_is_refused() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "rating-scale").await;
    let caller = principal(&tenant, "agent-a");

    let error = tools
        .rate_memory_utility(
            &caller,
            RateMemoryUtilityInput {
                edge_ids: vec![uuid::Uuid::new_v4()],
                utility_score: 7.0,
            },
        )
        .await
        .expect_err("7.0 is not on the utility scale");

    assert!(
        error.to_string().contains("utility_score"),
        "the error should name what was wrong with the call, got {error}"
    );
}

/// The foundational boundary, exercised through the tool layer rather than the
/// store: two callers whose credentials name different tenants share nothing.
///
/// Both tenants below hold the same entity names and the same proof text, so
/// the vectors are identical — a query that reached past the boundary would
/// rank the neighbour's fact exactly as highly as its own and hand an agent
/// someone else's memory with a proof excerpt that reads perfectly.
///
/// The tenant is not an argument to any of these calls. It cannot be: there is
/// no field to put it in. That is what makes the boundary hold even when the
/// text a model is reasoning over was written by somebody else.
#[tokio::test]
async fn two_principals_in_different_tenants_share_nothing() {
    let tools = tools().await;
    let acme = fresh_tenant(&tools, "isolation-acme").await;
    let globex = fresh_tenant(&tools, "isolation-globex").await;
    let alice = principal(&acme, "alice@acme");
    let bob = principal(&globex, "bob@globex");

    let shared_proof = "Alice Whitfield reports to Bob Nkemelu, per the org chart";

    let mut written = Vec::new();
    for (caller, extra) in [(&alice, "Acme Only"), (&bob, "Globex Only")] {
        seed_nodes(&tools, caller, &["Alice Whitfield", "Bob Nkemelu", extra]).await;

        let reports = tools
            .create_provenance_relationship(
                caller,
                Some("acme-agent/2.1"),
                CreateRelationshipInput {
                    relationships: vec![relationship(
                        "alice-whitfield",
                        "bob-nkemelu",
                        "REPORTS_TO",
                        shared_proof,
                    )],
                },
            )
            .await
            .expect("each tenant should be able to record its own copy");

        written.push(edge_id_of(&reports[0]));
    }
    let (acme_edge, globex_edge) = (written[0], written[1]);
    assert_ne!(acme_edge, globex_edge, "two tenants, two distinct facts");

    // Search sees only its own tenant's copy, though both are equally similar.
    let hits = tools
        .search_knowledge_graph(
            &alice,
            SearchInput {
                query: "who does Alice Whitfield report to".to_string(),
                precision_requirement: Some(PrecisionRequirement::Relaxed),
                limit: Some(50),
                ..SearchInput::default()
            },
        )
        .await
        .expect("searching should succeed");

    assert_eq!(
        hits.iter().map(|hit| hit.edge_id).collect::<Vec<_>>(),
        vec![acme_edge],
        "a search must return this tenant's fact and only this tenant's fact"
    );

    // A handle from the other tenant reads exactly like an id that exists
    // nowhere — which is all a caller may learn.
    let ratings = tools
        .rate_memory_utility(
            &alice,
            RateMemoryUtilityInput {
                edge_ids: vec![globex_edge],
                utility_score: 1.0,
            },
        )
        .await
        .expect("rating an id from another tenant is a rejection, not an error");

    assert!(
        matches!(
            &ratings[0],
            RatingReport::Rejected { reason, .. } if *reason == RatingRejectionReason::UnknownEdge
        ),
        "another tenant's edge_id must not be ratable: {ratings:#?}"
    );

    let (moved,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM edge_rating WHERE tenant_id = $1 OR edge_id = $2")
            .bind(&globex)
            .bind(globex_edge)
            .fetch_one(tools.store().pool())
            .await
            .expect("counting the neighbour's ratings should succeed");
    assert_eq!(moved, 0, "nothing may be written into the other tenant");

    // Nor can a write bind to an entity only the other tenant holds.
    let reports = tools
        .create_provenance_relationship(
            &alice,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![relationship(
                    "alice-whitfield",
                    "globex-only",
                    "RELATED_TO",
                    "a fact that would straddle two customers",
                )],
            },
        )
        .await
        .expect("the call succeeds; the item does not");

    assert!(
        matches!(
            &reports[0],
            RelationshipReport::Rejected { reason, .. }
                if *reason == RejectionReason::UnknownTargetNode
        ),
        "a node only the other tenant holds must not be reachable: {reports:#?}"
    );
}

// ---------------------------------------------------------------------------
// Input guards at the tool boundary
// ---------------------------------------------------------------------------

/// The most items any array argument may carry, spelled out here rather than
/// imported: the number is part of the contract a caller plans against, so a
/// change to it should break a test rather than quietly re-fit one.
const BATCH_LIMIT: usize = 64;

/// The most characters an entity name may carry.
const DISPLAY_NAME_LIMIT: usize = 200;

/// The most characters a proof excerpt may carry.
const PROOF_LIMIT: usize = 2_000;

/// Counts the nodes a tenant holds, so "nothing was written" can be asserted
/// rather than assumed.
async fn node_count(tools: &Tools, tenant: &str) -> i64 {
    let (count,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(tenant)
        .fetch_one(tools.store().pool())
        .await
        .expect("counting rows should succeed");
    count
}

/// The same, for edges.
async fn edge_count(tools: &Tools, tenant: &str) -> i64 {
    let (count,): (i64,) = sqlx::query_as("SELECT count(*) FROM edge WHERE tenant_id = $1")
        .bind(tenant)
        .fetch_one(tools.store().pool())
        .await
        .expect("counting rows should succeed");
    count
}

/// **The availability guard.**
///
/// Every array argument is embedded in a single forward pass, so the memory one
/// call costs is the product of its length and its longest text. Nothing capped
/// that length, and the server is stateless behind a load balancer — so a
/// client that retried an over-large call would take down the next instance
/// too.
///
/// The error has to name both numbers. A caller told only "too many" cannot
/// decide how to split; a caller told the limit and its own count can.
#[tokio::test]
async fn an_over_large_batch_is_refused_by_every_tool_that_takes_one() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "batch-cap").await;
    let caller = principal(&tenant, "agent-a");

    let over = BATCH_LIMIT + 1;

    let refusals = [
        tools
            .upsert_node(
                &caller,
                Some("acme-agent/2.1"),
                UpsertNodeInput {
                    nodes: (0..over)
                        .map(|index| node(&format!("Entity Number {index}")))
                        .collect(),
                },
            )
            .await
            .map(|reports| format!("{reports:#?}"))
            .expect_err("an over-large upsert_node batch must be refused")
            .to_string(),
        tools
            .create_provenance_relationship(
                &caller,
                Some("acme-agent/2.1"),
                CreateRelationshipInput {
                    relationships: (0..over)
                        .map(|index| {
                            relationship(
                                "alice-whitfield",
                                "bob-nkemelu",
                                "RELATED_TO",
                                &format!("the memo restates the reporting line, point {index}"),
                            )
                        })
                        .collect(),
                },
            )
            .await
            .map(|reports| format!("{reports:#?}"))
            .expect_err("an over-large create_provenance_relationship batch must be refused")
            .to_string(),
        tools
            .rate_memory_utility(
                &caller,
                RateMemoryUtilityInput {
                    edge_ids: (0..over).map(|_| uuid::Uuid::new_v4()).collect(),
                    utility_score: 1.0,
                },
            )
            .await
            .map(|reports| format!("{reports:#?}"))
            .expect_err("an over-large rate_memory_utility batch must be refused")
            .to_string(),
    ];

    for (argument, message) in ["nodes", "relationships", "edge_ids"]
        .into_iter()
        .zip(&refusals)
    {
        assert!(
            message.contains(argument),
            "the refusal should name the argument that was too long, got: {message}"
        );
        assert!(
            message.contains(&over.to_string()),
            "the refusal should say how many items arrived, got: {message}"
        );
        assert!(
            message.contains(&BATCH_LIMIT.to_string()),
            "the refusal should name the limit so a caller can split, got: {message}"
        );
    }

    assert_eq!(
        node_count(&tools, &tenant).await,
        0,
        "an over-large batch is decidable from the argument, so nothing may be written"
    );
}

/// The off-by-one guard. A cap that refuses the batch it advertises is a cap
/// that costs a caller a round trip to discover.
#[tokio::test]
async fn a_batch_at_exactly_the_limit_is_accepted() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "batch-cap-exact").await;
    let caller = principal(&tenant, "agent-a");

    let names: Vec<String> = (0..BATCH_LIMIT)
        .map(|index| format!("Exact Entity Number {index}"))
        .collect();

    let reports = tools
        .upsert_node(
            &caller,
            Some("acme-agent/2.1"),
            UpsertNodeInput {
                nodes: names.iter().map(|name| node(name)).collect(),
            },
        )
        .await
        .expect("a batch at exactly the limit must be accepted");
    assert_eq!(reports.len(), BATCH_LIMIT);

    let relationships: Vec<RelationshipSpec> = (0..BATCH_LIMIT)
        .map(|index| {
            relationship(
                &reports[index].node_id,
                &reports[(index + 1) % BATCH_LIMIT].node_id,
                "RELATED_TO",
                &format!("the roster lists them side by side at position {index}"),
            )
        })
        .collect();

    let written = tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput { relationships },
        )
        .await
        .expect("a batch at exactly the limit must be accepted");
    assert_eq!(written.len(), BATCH_LIMIT);

    let edge_ids: Vec<uuid::Uuid> = written.iter().map(edge_id_of).collect();
    let rated = tools
        .rate_memory_utility(
            &caller,
            RateMemoryUtilityInput {
                edge_ids,
                utility_score: 1.0,
            },
        )
        .await
        .expect("a batch at exactly the limit must be accepted");
    assert_eq!(rated.len(), BATCH_LIMIT);
    assert!(
        rated
            .iter()
            .all(|report| matches!(report, RatingReport::Rated { .. })),
        "every id came from the write above, so every one should have been rated: {rated:#?}"
    );
}

/// The other term in the product. One item carrying a megabyte of prose is the
/// same availability failure as a thousand items carrying a sentence each, and
/// the batch cap alone does not see it.
///
/// Counted in characters rather than bytes, and asserted on a multibyte name:
/// a byte budget would give an entity named in Japanese a third of the name an
/// English one gets, and it is the token count the model cares about, which
/// tracks characters far better than bytes.
#[tokio::test]
async fn an_over_long_display_name_or_proof_excerpt_is_refused() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "text-cap").await;
    let caller = principal(&tenant, "agent-a");

    let over_long_name = "N".repeat(DISPLAY_NAME_LIMIT + 1);
    let error = tools
        .upsert_node(
            &caller,
            Some("acme-agent/2.1"),
            UpsertNodeInput {
                nodes: vec![node("Alice Whitfield"), node(&over_long_name)],
            },
        )
        .await
        .map(|reports| format!("{reports:#?}"))
        .expect_err("a display_name past the cap must be refused")
        .to_string();

    assert!(
        error.contains("display_name"),
        "the refusal should name the field, got: {error}"
    );
    assert!(
        error.contains(&(DISPLAY_NAME_LIMIT + 1).to_string())
            && error.contains(&DISPLAY_NAME_LIMIT.to_string()),
        "the refusal should carry both the length received and the limit, got: {error}"
    );
    assert!(
        error.contains('1'),
        "the refusal should name the item it is about, got: {error}"
    );
    assert_eq!(
        node_count(&tools, &tenant).await,
        0,
        "the sibling must not have committed under a call that reports failure"
    );

    let over_long_proof = "p".repeat(PROOF_LIMIT + 1);
    let error = tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![relationship(
                    "alice-whitfield",
                    "bob-nkemelu",
                    "RELATED_TO",
                    &over_long_proof,
                )],
            },
        )
        .await
        .map(|reports| format!("{reports:#?}"))
        .expect_err("a proof_excerpt past the cap must be refused")
        .to_string();

    assert!(
        error.contains("proof_excerpt"),
        "the refusal should name the field, got: {error}"
    );
    assert!(
        error.contains(&(PROOF_LIMIT + 1).to_string()) && error.contains(&PROOF_LIMIT.to_string()),
        "the refusal should carry both the length received and the limit, got: {error}"
    );

    // Exactly at the cap, in a script where one character is three bytes: the
    // budget is characters, so this is a name at the limit and not past it.
    let at_the_cap: String = "東".repeat(DISPLAY_NAME_LIMIT);
    let reports = tools
        .upsert_node(
            &caller,
            Some("acme-agent/2.1"),
            UpsertNodeInput {
                nodes: vec![node(&at_the_cap)],
            },
        )
        .await
        .expect("a name of exactly the limit, in any script, must be accepted");
    assert_eq!(reports.len(), 1);

    let proof_at_the_cap = "q".repeat(PROOF_LIMIT);
    let written = tools
        .create_provenance_relationship(
            &caller,
            Some("acme-agent/2.1"),
            CreateRelationshipInput {
                relationships: vec![relationship(
                    &reports[0].node_id,
                    &reports[0].node_id,
                    "RELATED_TO",
                    &proof_at_the_cap,
                )],
            },
        )
        .await
        .expect("a proof of exactly the limit must be accepted");
    assert!(
        matches!(written[0], RelationshipReport::Created { .. }),
        "a proof at exactly the limit is written: {written:#?}"
    );

    // An alias is stored text on the same node and gets the same budget.
    let error = tools
        .upsert_node(
            &caller,
            Some("acme-agent/2.1"),
            UpsertNodeInput {
                nodes: vec![NodeSpec {
                    display_name: "Bob Nkemelu".to_string(),
                    aliases: vec!["B. Nkemelu".to_string(), "A".repeat(DISPLAY_NAME_LIMIT + 1)],
                }],
            },
        )
        .await
        .map(|reports| format!("{reports:#?}"))
        .expect_err("an alias past the cap must be refused")
        .to_string();
    assert!(
        error.contains("alias"),
        "the refusal should name the field, got: {error}"
    );
}

/// **The data-quality guard.**
///
/// The tool's whole premise is that a fact must be justified — the brief says
/// "you MUST supply verbatim proof" — and the excerpt is also what lets a
/// reading agent decide whether to believe a fact it did not write. A proofless
/// edge is a bare assertion wearing the costume of an evidenced one, and it is
/// embedded into a `proof_vector` of essentially nothing, so it pollutes every
/// semantic search that follows.
///
/// The floor is three characters *after trimming*, and it is deliberately a
/// floor that cannot bite rather than a judgement about evidence: the shortest
/// real quotation anyone can construct is longer than that in every script,
/// including the ones this server's tokenizer reads at one token per character.
#[tokio::test]
async fn a_proof_excerpt_that_is_not_evidence_is_refused() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "blank-proof").await;
    let caller = principal(&tenant, "agent-a");
    seed_nodes(&tools, &caller, &["Alice Whitfield", "Bob Nkemelu"]).await;

    for (label, proof) in [
        ("empty", ""),
        ("a single space", " "),
        ("whitespace only", "  \t \n  "),
        ("an ideographic space", "\u{3000}\u{3000}\u{3000}\u{3000}"),
        ("too short to be a quotation", "ok"),
        ("punctuation standing in for a quotation", ". "),
    ] {
        let outcome = tools
            .create_provenance_relationship(
                &caller,
                Some("acme-agent/2.1"),
                CreateRelationshipInput {
                    relationships: vec![relationship(
                        "alice-whitfield",
                        "bob-nkemelu",
                        "REPORTS_TO",
                        proof,
                    )],
                },
            )
            .await;

        let error = match outcome {
            Ok(reports) => panic!("{label} is not proof, but it was written: {reports:#?}"),
            Err(error) => error.to_string(),
        };

        assert!(
            error.contains("proof_excerpt"),
            "the refusal for {label} should name the field, got: {error}"
        );
    }

    assert_eq!(
        edge_count(&tools, &tenant).await,
        0,
        "an unevidenced fact is decidable from the argument, so nothing may be written"
    );
}

/// The other half of the same guard, and the one that is easy to get wrong.
/// A short quotation is still a quotation, and a bar set to filter weak
/// evidence would refuse honest writes in every language that says more per
/// character than English does.
#[tokio::test]
async fn a_legitimately_short_proof_excerpt_is_accepted() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "short-proof").await;
    let caller = principal(&tenant, "agent-a");
    seed_nodes(&tools, &caller, &["Alice Whitfield", "Bob Nkemelu"]).await;

    for proof in [
        "Yes",
        "彼はRustを使う",
        "Alice: Bob is my manager",
        "  I report to Bob.  ",
    ] {
        let written = tools
            .create_provenance_relationship(
                &caller,
                Some("acme-agent/2.1"),
                CreateRelationshipInput {
                    relationships: vec![relationship(
                        "alice-whitfield",
                        "bob-nkemelu",
                        "RELATED_TO",
                        proof,
                    )],
                },
            )
            .await
            .unwrap_or_else(|error| panic!("`{proof}` is a real quotation, got: {error}"));

        assert!(
            matches!(written[0], RelationshipReport::Created { .. }),
            "`{proof}` should have been written: {written:#?}"
        );
    }
}
