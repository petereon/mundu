//! Integration tests for search, against a real Postgres.
//!
//! Every test owns a `tenant_id` derived from its own name, for the reason the
//! node and edge suites give: the suite runs in parallel against one shared
//! database, so isolation cannot come from truncating shared tables. Tenancy is
//! the isolation mechanism the schema already has.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use chrono::{DateTime, Utc};
use mundu::embed::DeterministicEmbedder;
use mundu::store::edge::{EdgeInput, EdgeOutcome};
use mundu::store::node::{NodeInput, canonical_node_id};
use mundu::store::search::{
    CANDIDATE_DEPTH, MAX_LIMIT, Precision, SearchHit, SearchMethod, SearchOptions,
};
use mundu::store::{Provenance, Store};
use uuid::Uuid;

/// One [`Store`] per test, deliberately not a shared `static`, for the reason
/// the node suite documents: a sqlx pool is bound to the reactor that created
/// it, and every `#[tokio::test]` builds and tears down its own.
async fn store() -> Store {
    common::store().await
}

/// Names a tenant after the test that owns it and clears anything a previous
/// run left behind, so "search returned two hits" means the same thing on the
/// first run and the hundredth.
async fn fresh_tenant(store: &Store, name: &str) -> String {
    let tenant = format!("test-search-{name}");

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

/// Mints the nodes a search test needs through the real upsert path, so the
/// display names and aliases the lexical half matches on are exactly what the
/// server would have stored.
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

/// Writes every input through the real edge path and returns the ids in input
/// order, so a test can name the edge it expects to rank first.
async fn create_edges(store: &Store, tenant: &str, inputs: &[EdgeInput]) -> Vec<Uuid> {
    create_edges_as(store, tenant, inputs, &anonymous()).await
}

async fn create_edges_as(
    store: &Store,
    tenant: &str,
    inputs: &[EdgeInput],
    provenance: &Provenance,
) -> Vec<Uuid> {
    let outcomes = store
        .create_edges(tenant, inputs, &embedder(), provenance)
        .await
        .expect("seeding edges should succeed");

    outcomes
        .iter()
        .map(|outcome| match outcome {
            EdgeOutcome::Created { edge_id, .. } => *edge_id,
            other => panic!("expected the seeded edge to be current, got {other:?}"),
        })
        .collect()
}

/// Marks an edge disputed, or clears it. Task 8 owns the quorum that decides
/// this; a search only has to rank and report it.
async fn set_disputed(store: &Store, tenant: &str, edge_id: Uuid, disputed: bool) {
    let disputed_at = disputed.then(Utc::now);

    sqlx::query("UPDATE edge SET disputed_at = $3 WHERE tenant_id = $1 AND edge_id = $2")
        .bind(tenant)
        .bind(edge_id)
        .bind(disputed_at)
        .execute(store.pool())
        .await
        .expect("disputing the edge should succeed");
}

/// Runs a search with the default options.
async fn search(store: &Store, tenant: &str, query: &str) -> Vec<SearchHit> {
    search_with(store, tenant, query, &SearchOptions::default()).await
}

async fn search_with(
    store: &Store,
    tenant: &str,
    query: &str,
    options: &SearchOptions,
) -> Vec<SearchHit> {
    store
        .search(tenant, query, options, &embedder())
        .await
        .expect("searching should succeed")
}

/// The ids one search returns, in the order it ranked them.
///
/// Comparing whole id lists rather than lengths is the point: two settings that
/// happen to return the same *number* of hits and a different set of them are
/// not the same answer.
async fn ids_from(
    store: &Store,
    tenant: &str,
    query: &str,
    method: SearchMethod,
    precision: Precision,
    limit: Option<usize>,
) -> Vec<Uuid> {
    search_with(
        store,
        tenant,
        query,
        &SearchOptions {
            search_method: Some(method),
            precision: Some(precision),
            limit,
            ..SearchOptions::default()
        },
    )
    .await
    .iter()
    .map(|hit| hit.edge_id)
    .collect()
}

/// Where `edge_id` came back in the ranking, or a failure naming what did.
fn rank_of(hits: &[SearchHit], edge_id: Uuid) -> usize {
    hits.iter()
        .position(|hit| hit.edge_id == edge_id)
        .unwrap_or_else(|| {
            panic!(
                "edge {edge_id} is missing from the results: {:#?}",
                hits.iter()
                    .map(|hit| (hit.edge_id, hit.score))
                    .collect::<Vec<_>>()
            )
        })
}

/// Sets an edge's stored salience directly.
///
/// Ratings are Task 8, so there is no path through the server that moves
/// `salience_base` yet, and the decay half-life is long enough that waiting for
/// it is not a test. Both columns move together because effective salience is a
/// function of the pair: a base with no `updated_at` decays from whenever the
/// edge happened to be written.
async fn set_salience(
    store: &Store,
    tenant: &str,
    edge_id: Uuid,
    base: f32,
    updated_at: DateTime<Utc>,
) {
    let affected =
        sqlx::query("UPDATE edge SET salience_base = $3, salience_updated_at = $4 WHERE tenant_id = $1 AND edge_id = $2")
            .bind(tenant)
            .bind(edge_id)
            .bind(base)
            .bind(updated_at)
            .execute(store.pool())
            .await
            .expect("setting salience should succeed")
            .rows_affected();

    assert_eq!(affected, 1, "expected to update exactly one edge");
}

/// Moves an edge to a status no path in the server produces yet.
///
/// Archival is an operator action by design — an agent-reachable one would hand
/// injected proof text an erasure primitive — so there is nothing to call here
/// and the row is moved directly, which is what an operator would do.
async fn set_status(store: &Store, tenant: &str, edge_id: Uuid, status: &str) {
    let affected = sqlx::query(
        "UPDATE edge SET status = $3::edge_status WHERE tenant_id = $1 AND edge_id = $2",
    )
    .bind(tenant)
    .bind(edge_id)
    .bind(status)
    .execute(store.pool())
    .await
    .expect("setting the status should succeed")
    .rows_affected();

    assert_eq!(affected, 1, "expected to update exactly one edge");
}

/// A fixed instant, so a test that talks about "2019" is not quietly comparing
/// against the clock of whatever machine it runs on.
fn instant(year: i32, month: u32, day: u32) -> DateTime<Utc> {
    use chrono::TimeZone;

    Utc.with_ymd_and_hms(year, month, day, 12, 0, 0)
        .single()
        .expect("a valid civil timestamp")
}

/// **The recall bound has to be a property of this server, not of the plan the
/// planner happened to pick.**
///
/// The semantic half asks the vector index for [`CANDIDATE_DEPTH`] rows and
/// then narrows them by `tenant_id`, which is a predicate no vector index
/// carries. pgvector's HNSW scan hands back at most `hnsw.ef_search` tuples for
/// the whole index before any filter is applied, and the shipped default is 40
/// — so on the defaults the tenant filter is a *post*-filter over forty rows
/// drawn from every customer's data, and a tenant's own facts are crowded out
/// of its own search results by rows it is not allowed to see.
///
/// That was not a hypothesis. On a wiped database this file reported ten of
/// fifteen tests failing, and the plans the server actually ran show why: a
/// tenant holding 55 active edges was handed 14 of them, tenants holding two or
/// five were handed none, and every one of those scans read the same ~40 index
/// tuples and threw away the 32–53 that belonged to somebody else. It looked
/// like a flake rather than a defect because the planner stops choosing the
/// HNSW index once the table grows enough for a b-tree scan to cost less, so
/// the same corpus answered correctly as soon as it was warm.
///
/// It is asserted on the settings rather than on a search because the fault
/// only appears when the planner picks the vector index, and there is no honest
/// way for a test to insist that it does — a test that bullies the planner into
/// one plan is a test about the planner, and would pass while the real query
/// took the other one. What can be pinned is the thing the answer must not
/// depend on: whichever plan is chosen, a connection out of this pool can read
/// a full candidate list through the index.
#[tokio::test]
async fn every_pooled_connection_can_read_a_full_candidate_list_through_the_vector_index() {
    let store = &store().await;

    let (ef_search,): (String,) = sqlx::query_as("SHOW hnsw.ef_search")
        .fetch_one(store.pool())
        .await
        .expect("reading the candidate-list size should succeed");
    let ef_search: i64 = ef_search
        .parse()
        .expect("hnsw.ef_search is a number of tuples");

    assert!(
        ef_search >= CANDIDATE_DEPTH,
        "a scan that stops after {ef_search} tuples cannot answer a query \
         asking for {CANDIDATE_DEPTH} of them"
    );

    let (iterative_scan,): (String,) = sqlx::query_as("SHOW hnsw.iterative_scan")
        .fetch_one(store.pool())
        .await
        .expect("reading the scan mode should succeed");

    assert_eq!(
        iterative_scan, "strict_order",
        "the tenant filter is applied after the index has already chosen its \
         rows, so the scan has to be able to keep going — and to keep going in \
         distance order, because the fusion downstream reads rank and cannot \
         tell that a list arrived slightly out of order"
    );
}

/// Two facts the query cannot tell apart, so salience has to.
///
/// Both edges carry the *same* proof text, which is the only way two rows are
/// genuinely equally similar to a query: identical text embeds to an identical
/// vector, so the distance ordering between them is a tie that some arbitrary
/// tiebreak resolves. That is precisely where salience is supposed to speak —
/// of two facts a query cannot distinguish, the one the graph has more reason
/// to believe goes first.
///
/// The pair is asserted in both directions, and the second half is what makes
/// this a test. Whichever edge the tiebreak puts first is a property of a
/// random uuid, so a ranking that ignored salience entirely would satisfy one
/// direction by luck; it cannot satisfy both, because the relevance order does
/// not change between the two searches and the expected answer does.
#[tokio::test]
async fn two_equally_similar_edges_order_by_salience() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "salience-orders-a-tie").await;
    seed_nodes(store, &tenant, &["Alice Whitfield", "Kafka", "Postgres"]).await;

    // No node in this tenant is named anything like the query, so the lexical
    // half stays out of it and the two edges reach fusion on similarity alone.
    let shared_proof = "the settlement events pipeline was signed off in March";

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new("alice-whitfield", "kafka", "USES", shared_proof),
            EdgeInput::new("alice-whitfield", "postgres", "USES", shared_proof),
        ],
    )
    .await;
    let (kafka_edge, postgres_edge) = (ids[0], ids[1]);

    for (salient, faded) in [(kafka_edge, postgres_edge), (postgres_edge, kafka_edge)] {
        set_salience(store, &tenant, salient, 1.0, Utc::now()).await;
        set_salience(store, &tenant, faded, 0.05, Utc::now()).await;

        let hits = search(store, &tenant, "settlement events pipeline").await;

        assert!(
            rank_of(&hits, salient) < rank_of(&hits, faded),
            "the more salient of two equally similar facts must come first: {hits:#?}"
        );
    }
}

/// The property the whole re-rank is shaped around: salience breaks ties, and
/// can never override relevance.
///
/// The multiplier is `0.7 + 0.3 * effective_salience`, so the most a fact's age
/// can cost it is 30% of its relevance. The obvious alternative — multiplying
/// by salience directly — is what this test exists to rule out, and the pair
/// below is built so that the obvious version gets it *wrong*: an almost-dead
/// but exactly-right answer against a brand-new edge about something else
/// entirely.
///
/// The arithmetic, with the numbers this tenant produces. The right answer is
/// first in the lexical list and present in the semantic one, so its fused
/// relevance is about `1/13 + 1/14 = 0.1484`; the irrelevant edge is in the
/// semantic list alone, at best `1/13 = 0.0769`.
///
/// * bounded: `0.1484 * (0.7 + 0.3*0.1) = 0.1083` beats `0.0769 * 1.0`.
/// * unbounded: `0.1484 * 0.1 = 0.0148` loses to `0.0769 * 1.0` — by five to
///   one. The junk wins, and an agent asking who Alice reports to is told about
///   freight volumes.
///
/// The margin here comes from corroboration — two lists against one — and not
/// from the band, which is why this passes at any `RRF_K`. Its unit-test
/// counterpart, `the_fusion_outspreads_the_band_across_the_default_page`, is
/// the case this cannot see: two hits the *same* list ranks ten apart, where
/// the band alone decides and, at `RRF_K = 60`, decided wrongly.
#[tokio::test]
async fn a_faded_but_relevant_edge_outranks_a_fresh_irrelevant_one() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "bounded-band").await;
    seed_nodes(
        store,
        &tenant,
        &[
            "Alice Whitfield",
            "Bob Nkemelu",
            "Zebra Logistics",
            "Quarterly Report",
        ],
    )
    .await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "alice-whitfield",
                "bob-nkemelu",
                "REPORTS_TO",
                "Alice Whitfield reports to Bob Nkemelu, per the org chart",
            ),
            EdgeInput::new(
                "zebra-logistics",
                "quarterly-report",
                "AUTHORED_BY",
                "the freight volumes appendix was compiled by the logistics desk",
            ),
        ],
    )
    .await;
    let (relevant, irrelevant) = (ids[0], ids[1]);

    // Nearly forgotten, and exactly what was asked for.
    set_salience(store, &tenant, relevant, 0.1, Utc::now()).await;
    // Written this morning, and about something else.
    set_salience(store, &tenant, irrelevant, 1.0, Utc::now()).await;

    let hits = search(store, &tenant, "Alice Whitfield").await;

    assert!(
        rank_of(&hits, relevant) < rank_of(&hits, irrelevant),
        "salience must not be able to buy a place above a better match: {hits:#?}"
    );
}

/// The tail has to be cut somewhere, and where is the caller's to say.
///
/// A vector index always returns its `LIMIT`, however little any of it has to
/// do with the query — "nearest" is not "near". So the tail is dropped relative
/// to the best hit rather than at an absolute score: RRF values are a function
/// of list depth and have no meaning on their own, so an absolute floor would
/// be a number tuned to one corpus and wrong on the next.
///
/// Here one edge hangs off the node the query names and is therefore in both
/// candidate lists; the other three are only ever vector neighbours. At
/// `Strict` the two lists have to agree, so only the first survives. At
/// `Relaxed` a caller that would rather sift than miss something gets all four.
#[tokio::test]
async fn the_relative_cutoff_tightens_with_the_precision_the_caller_asks_for() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "relative-cutoff").await;
    seed_nodes(
        store,
        &tenant,
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

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "alice-whitfield",
                "bob-nkemelu",
                "REPORTS_TO",
                "Alice Whitfield reports to Bob Nkemelu, per the org chart",
            ),
            EdgeInput::new(
                "zebra-logistics",
                "quarterly-report",
                "AUTHORED_BY",
                "the freight volumes appendix was compiled by the logistics desk",
            ),
            EdgeInput::new(
                "harbour-crane",
                "night-shift",
                "USES",
                "the harbour crane is booked for the night shift on Thursdays",
            ),
            EdgeInput::new(
                "zebra-logistics",
                "harbour-crane",
                "OWNS",
                "the crane on berth four belongs to the logistics operator",
            ),
        ],
    )
    .await;

    let strict = search_with(
        store,
        &tenant,
        "Alice Whitfield",
        &SearchOptions {
            precision: Some(Precision::Strict),
            ..SearchOptions::default()
        },
    )
    .await;

    assert_eq!(
        strict.iter().map(|hit| hit.edge_id).collect::<Vec<_>>(),
        vec![ids[0]],
        "at Strict, an edge only one list has heard of is not an answer"
    );

    let relaxed = search_with(
        store,
        &tenant,
        "Alice Whitfield",
        &SearchOptions {
            precision: Some(Precision::Relaxed),
            ..SearchOptions::default()
        },
    )
    .await;

    assert_eq!(
        relaxed.len(),
        4,
        "at Relaxed the caller would rather sift than miss something: {relaxed:#?}"
    );
    assert_eq!(
        relaxed[0].edge_id, ids[0],
        "loosening the cutoff must not change the order, only the depth"
    );
}

/// **Precision only bites where two lists can disagree.**
///
/// The cutoff is a fraction of the best *fused* score, and fusion is the whole
/// source of the spread it reads. In a single-list search every candidate
/// appears in exactly one list, so every score is `1/(RRF_K + rank)`, a
/// sequence that is the same for every query and every corpus — and a fraction
/// of its first term is therefore a bound on *rank* rather than a statement
/// about relevance. Search declines to apply the cutoff at all in that case, so
/// `Relaxed`, `Standard` and `Strict` return the same edges in the same order.
///
/// It is the guard doing that, and not the depth of the bounds. When this was
/// written all three bounds sat far below anything a caller reads back and the
/// equalities would have held either way; at `RRF_K = 12` the `Strict` bound is
/// rank 9 and the corpus below is five edges, so a cutoff that *was* applied
/// would still not show — which is why the deeper corpus in
/// [`a_single_list_search_keeps_a_tail_no_fusion_ever_measured`] exists.
///
/// A characterisation test: it passed the first time it was run, against the
/// behaviour that was already there. It is written down because the property is
/// invisible from the code — three named settings, a cutoff plainly applied to
/// all of them — and because the tool description now tells a model this is so.
/// A promise in a description with nothing holding it to the code is a promise
/// that goes stale on the next change.
///
/// The hybrid half is what stops it being vacuous: the same corpus and the same
/// query, with two lists to disagree, answer differently.
#[tokio::test]
async fn precision_changes_nothing_a_single_list_search_can_show() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "single-list-precision").await;
    seed_nodes(
        store,
        &tenant,
        &[
            "Alice Whitfield",
            "Bob Nkemelu",
            "Kafka",
            "Zebra Logistics",
            "Quarterly Report",
            "Harbour Crane",
            "Night Shift",
        ],
    )
    .await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "alice-whitfield",
                "bob-nkemelu",
                "REPORTS_TO",
                "Alice Whitfield reports to Bob Nkemelu, per the org chart",
            ),
            EdgeInput::new(
                "alice-whitfield",
                "kafka",
                "USES",
                "Alice Whitfield's team publishes settlement events to Kafka",
            ),
            EdgeInput::new(
                "zebra-logistics",
                "quarterly-report",
                "AUTHORED_BY",
                "the freight volumes appendix was compiled by the logistics desk",
            ),
            EdgeInput::new(
                "harbour-crane",
                "night-shift",
                "USES",
                "the harbour crane is booked for the night shift on Thursdays",
            ),
            EdgeInput::new(
                "zebra-logistics",
                "harbour-crane",
                "OWNS",
                "the crane on berth four belongs to the logistics operator",
            ),
        ],
    )
    .await;

    // Both of these hang off `alice-whitfield`, so the lexical half ties them
    // on every key it reads and hands them identical relevance — which is the
    // honest answer and leaves salience to order them. Salience is not what
    // this test is about, and left alone it is not even stable: the two edges
    // were written milliseconds apart, `effective_salience` is a `REAL`, and
    // one ULP at 0.99999 is about a fifth of a second of decay, so for a few
    // milliseconds in every fifth of a second the pair rounds to adjacent
    // floats and swaps. Pinning both to one instant removes the variable. The
    // quantisation itself is a separate, older thing and is written up in
    // `docs/plans/follow-ups.md`.
    let pinned = Utc::now();
    for edge_id in [ids[0], ids[1]] {
        set_salience(store, &tenant, edge_id, 1.0, pinned).await;
    }

    let query = "Alice Whitfield";

    // The lexical half matches the one node the query names and returns the two
    // edges touching it; the semantic half ranks every edge in the tenant.
    for (method, expected) in [(SearchMethod::Semantic, 5), (SearchMethod::Lexical, 2)] {
        let relaxed = ids_from(store, &tenant, query, method, Precision::Relaxed, None).await;
        let standard = ids_from(store, &tenant, query, method, Precision::Standard, None).await;
        let strict = ids_from(store, &tenant, query, method, Precision::Strict, None).await;

        // A search that returned one hit, or none, would satisfy the equalities
        // below without exercising a cutoff at all.
        assert_eq!(
            strict.len(),
            expected,
            "{method:?} returned {strict:?}, which is not the list this test is about"
        );

        assert_eq!(
            relaxed, standard,
            "{method:?} answers Relaxed and Standard differently"
        );
        assert_eq!(
            standard, strict,
            "{method:?} answers Standard and Strict differently"
        );
    }

    let hybrid_relaxed = ids_from(
        store,
        &tenant,
        query,
        SearchMethod::Hybrid,
        Precision::Relaxed,
        None,
    )
    .await;
    let hybrid_strict = ids_from(
        store,
        &tenant,
        query,
        SearchMethod::Hybrid,
        Precision::Strict,
        None,
    )
    .await;

    assert_eq!(
        (hybrid_strict.len(), hybrid_relaxed.len()),
        (2, 5),
        "the same corpus and the same query must answer differently once there \
         are two lists to disagree, or the equalities above prove nothing: \
         strict {hybrid_strict:?}, relaxed {hybrid_relaxed:?}"
    );
}

/// **Where the single-list cutoff is not inert, it is arbitrary — so it is not
/// applied.**
///
/// `Strict` is 0.6 of the best score, and in a single list the best score is
/// `1/(RRF_K + 1)`. Solve it and the setting means "keep the top 9", 9 being
/// `13 / 0.6 - 12` and nothing else: it is a property of the fusion constant,
/// not of the query, the corpus, or how good the tenth hit was. Change `RRF_K`
/// and the same word names a different depth on the same data — which is not a
/// hypothetical, because `RRF_K` did change. At 60 the same fraction solved to
/// 41.
///
/// That move is what turned this from a tail-end concern into a default one.
/// At 41 the truncation was invisible until a caller asked for more than 41
/// hits; at 9 it is *inside* the default limit of ten, so a semantic search
/// would come back a hit short for a caller who asked for nothing in
/// particular. Either way the relative cutoff is skipped outright when there is
/// only one list: "precision applies to hybrid search" is a sentence a model
/// can act on, and "applies to hybrid search, and also truncates a single-index
/// search at 9" is not.
///
/// The corpus is deliberately deeper than [`MAX_LIMIT`], because that is where
/// the two settings differed most; they now differ from the tenth hit onward.
#[tokio::test]
async fn a_single_list_search_keeps_a_tail_no_fusion_ever_measured() {
    /// Past the rank bound `Strict` used to impose, and past `MAX_LIMIT`, so a
    /// caller asking for the cap can be served it.
    const DEPOTS: usize = 55;

    let store = &store().await;
    let tenant = fresh_tenant(store, "single-list-rank-bound").await;

    let depots: Vec<String> = (0..DEPOTS)
        .map(|index| format!("Depot {index:02}"))
        .collect();
    let mut display_names: Vec<&str> = vec!["Night Shift"];
    display_names.extend(depots.iter().map(String::as_str));
    seed_nodes(store, &tenant, &display_names).await;

    let inputs: Vec<EdgeInput> = depots
        .iter()
        .map(|depot| {
            EdgeInput::new(
                canonical_node_id("Night Shift"),
                canonical_node_id(depot),
                "USES",
                format!("the night shift signed for pallets at {depot} on Thursday"),
            )
        })
        .collect();
    create_edges(store, &tenant, &inputs).await;

    let query = "who signed for the overnight pallets";
    let relaxed = ids_from(
        store,
        &tenant,
        query,
        SearchMethod::Semantic,
        Precision::Relaxed,
        Some(MAX_LIMIT),
    )
    .await;
    let strict = ids_from(
        store,
        &tenant,
        query,
        SearchMethod::Semantic,
        Precision::Strict,
        Some(MAX_LIMIT),
    )
    .await;

    assert_eq!(
        relaxed.len(),
        MAX_LIMIT,
        "the corpus has to run deeper than the cap or there is no tail to lose"
    );
    assert_eq!(
        strict,
        relaxed,
        "a semantic search has one ranking and nothing to corroborate, so \
         Strict must not cut {} hits off the end of it",
        relaxed.len() - strict.len()
    );
}

/// Salience ranks. It never gates.
///
/// The edge below has decayed to about seven billionths of its original weight
/// — twenty-seven half-lives — and it still comes back, at the tightest
/// precision the caller can ask for. That is a deliberate one-way-door
/// argument, not a tolerance: an edge search will not return cannot be read,
/// therefore cannot be rated, therefore can never climb back. Excluding a fact
/// for being old is the one decay outcome that is irreversible, so the cutoff
/// is taken on relevance before any salience is even read.
#[tokio::test]
async fn an_all_but_forgotten_edge_is_still_returned() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "never-gated").await;
    seed_nodes(store, &tenant, &["Alice Whitfield", "Kafka", "Postgres"]).await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "alice-whitfield",
                "kafka",
                "USES",
                "Alice Whitfield's team publishes settlement events to Kafka",
            ),
            EdgeInput::new(
                "alice-whitfield",
                "postgres",
                "USES",
                "Alice Whitfield's team stored invoices in Postgres years ago",
            ),
        ],
    )
    .await;
    let (fresh, forgotten) = (ids[0], ids[1]);

    set_salience(store, &tenant, fresh, 1.0, Utc::now()).await;
    set_salience(
        store,
        &tenant,
        forgotten,
        1.0,
        Utc::now() - chrono::Duration::days(730),
    )
    .await;

    for precision in [Precision::Relaxed, Precision::Standard, Precision::Strict] {
        let hits = search_with(
            store,
            &tenant,
            "Alice Whitfield",
            &SearchOptions {
                precision: Some(precision),
                ..SearchOptions::default()
            },
        )
        .await;

        let hit = &hits[rank_of(&hits, forgotten)];
        assert!(
            hit.effective_salience < 0.01,
            "at {precision:?} the edge came back, but with salience {} — \
             two years of decay should have left it near zero, so this test \
             is not exercising what it claims to",
            hit.effective_salience
        );
        assert!(
            rank_of(&hits, fresh) < rank_of(&hits, forgotten),
            "at {precision:?} the decayed edge should sink, not vanish: {hits:#?}"
        );
    }
}

/// A caller asking for five hundred results does not know what it is asking
/// for.
///
/// Every hit is injected into an agent's context on every lookup, and keeping
/// that context clean is the reason this server exists; five hundred proofs is
/// most of a context window spent on a single tool call. So the limit is
/// clamped rather than honoured, and clamped rather than rejected — an error
/// here teaches a model nothing it can act on, while fifty good hits answer the
/// question it actually asked.
#[tokio::test]
async fn the_limit_is_capped_however_much_a_caller_asks_for() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "limit-cap").await;

    let services: Vec<String> = (1..=60)
        .map(|index| format!("Service {index:02}"))
        .collect();
    let mut names: Vec<&str> = vec!["Kafka"];
    names.extend(services.iter().map(String::as_str));
    seed_nodes(store, &tenant, &names).await;

    let inputs: Vec<EdgeInput> = (1..=60)
        .map(|index| {
            EdgeInput::new(
                "kafka",
                format!("service-{index:02}"),
                "USES",
                format!("service {index:02} consumes the settlement topic from Kafka"),
            )
        })
        .collect();
    let ids = create_edges(store, &tenant, &inputs).await;
    assert_eq!(ids.len(), 60, "every seeded edge should have been written");

    let capped = search_with(
        store,
        &tenant,
        "Kafka",
        &SearchOptions {
            limit: Some(500),
            ..SearchOptions::default()
        },
    )
    .await;
    assert_eq!(capped.len(), 50, "the hard cap is fifty, not five hundred");

    let defaulted = search(store, &tenant, "Kafka").await;
    assert_eq!(
        defaulted.len(),
        10,
        "a caller that says nothing gets a page it can read"
    );
}

/// `tenant_id` is the whole isolation boundary, and search is where a hole in
/// it would be a disclosure rather than a corruption.
///
/// Two customers naming a person or a project the same thing is the ordinary
/// case, so both tenants below hold the same names and the same proof text —
/// the vectors are identical, which means a query that reaches past the
/// boundary would rank the neighbour's fact exactly as highly as its own and
/// hand an agent someone else's memory with a proof excerpt that reads
/// perfectly.
#[tokio::test]
async fn results_never_reach_across_tenants() {
    let store = &store().await;
    let acme = fresh_tenant(store, "isolation-acme").await;
    let globex = fresh_tenant(store, "isolation-globex").await;

    let mut written = Vec::new();
    for tenant in [&acme, &globex] {
        seed_nodes(store, tenant, &["Alice Whitfield", "Bob Nkemelu"]).await;
        written.push(
            create_edges(
                store,
                tenant,
                &[EdgeInput::new(
                    "alice-whitfield",
                    "bob-nkemelu",
                    "REPORTS_TO",
                    "Alice Whitfield reports to Bob Nkemelu, per the org chart",
                )],
            )
            .await[0],
        );
    }
    let (mine, theirs) = (written[0], written[1]);

    let hits = search(store, &acme, "Alice Whitfield").await;

    assert_eq!(
        hits.iter().map(|hit| hit.edge_id).collect::<Vec<_>>(),
        vec![mine],
        "a search may only ever see its own tenant's edges; {theirs} belongs to another customer"
    );
}

/// Proof is the expensive part of a hit, so it is a snippet unless the caller
/// says otherwise.
///
/// Ten full excerpts is on the order of three thousand tokens of an agent's
/// context per lookup, and this server's whole claim is that it keeps that
/// context clean. A snippet is enough for a model to decide whether the fact is
/// the one it wanted; `include_full_proof` is there for when it has decided.
///
/// The excerpt below is Japanese, which is the case a naive `&proof[..200]`
/// gets wrong: byte 200 lands in the middle of a three-byte character and Rust
/// panics on the slice. Proof text is quoted verbatim from whatever an agent
/// read, so it is exactly as likely to be Japanese as English, and a truncation
/// that panics turns a search into a 500.
#[tokio::test]
async fn proof_is_snipped_by_default_and_returned_whole_on_request() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "proof-snippet").await;
    seed_nodes(store, &tenant, &["Alice Whitfield", "Kafka", "Postgres"]).await;

    // 320 characters, 960 bytes: every boundary the snippet could take is a
    // character boundary that is not a byte boundary.
    let multibyte = "東京の倉庫の在庫は".repeat(36);
    assert_eq!(multibyte.chars().count(), 324);
    assert!(
        !multibyte.is_char_boundary(200),
        "byte 200 must be mid-character, or this test proves nothing"
    );

    let short = "Alice Whitfield's team publishes settlement events to Kafka";

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new("alice-whitfield", "kafka", "USES", &multibyte),
            EdgeInput::new("alice-whitfield", "postgres", "USES", short),
        ],
    )
    .await;
    let (long_edge, short_edge) = (ids[0], ids[1]);

    let hits = search(store, &tenant, "Alice Whitfield").await;

    let snipped = &hits[rank_of(&hits, long_edge)];
    assert!(
        snipped.proof_truncated,
        "the caller has to be told there is more: {snipped:#?}"
    );
    assert!(
        snipped.proof.chars().count() <= 210,
        "expected about 200 characters, got {}",
        snipped.proof.chars().count()
    );
    assert!(
        multibyte.starts_with(snipped.proof.trim_end_matches('…')),
        "a snippet has to be a prefix of the excerpt, not a re-rendering of it"
    );

    let whole = &hits[rank_of(&hits, short_edge)];
    assert_eq!(
        whole.proof, short,
        "an excerpt that already fits is returned as it was written"
    );
    assert!(!whole.proof_truncated);

    let full = search_with(
        store,
        &tenant,
        "Alice Whitfield",
        &SearchOptions {
            include_full_proof: true,
            ..SearchOptions::default()
        },
    )
    .await;

    let complete = &full[rank_of(&full, long_edge)];
    assert_eq!(
        complete.proof, multibyte,
        "include_full_proof means all of it"
    );
    assert!(!complete.proof_truncated);
}

/// Search answers "what is true now", and only the active edges are that.
///
/// A superseded edge is a fact that stopped holding, and its `proof_excerpt`
/// still reads perfectly — "Alice Whitfield reports to Bob Nkemelu" is a
/// sentence an agent has no way to date. Returning it beside the current fact
/// would put two contradictory answers in front of a model with nothing to
/// choose between them. An archived edge is an operator's explicit decision
/// that this should not be read at all, which is a weaker statement than
/// deletion but not a weaker one than this.
#[tokio::test]
async fn superseded_and_archived_edges_never_appear() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "only-active").await;
    seed_nodes(
        store,
        &tenant,
        &["Alice Whitfield", "Bob Nkemelu", "Carol Ibarra", "Kafka"],
    )
    .await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::with_valid_from(
                "alice-whitfield",
                "bob-nkemelu",
                "REPORTS_TO",
                "Alice Whitfield reports to Bob Nkemelu, per the 2019 org chart",
                instant(2019, 3, 1),
            ),
            EdgeInput::with_valid_from(
                "alice-whitfield",
                "carol-ibarra",
                "REPORTS_TO",
                "Alice Whitfield reports to Carol Ibarra after the June reorg",
                instant(2024, 6, 1),
            ),
            EdgeInput::new(
                "alice-whitfield",
                "kafka",
                "USES",
                "Alice Whitfield's team publishes settlement events to Kafka",
            ),
        ],
    )
    .await;
    let (superseded, current, archived) = (ids[0], ids[1], ids[2]);

    set_status(store, &tenant, archived, "archived").await;

    let hits = search(store, &tenant, "Alice Whitfield").await;

    assert_eq!(
        hits.iter().map(|hit| hit.edge_id).collect::<Vec<_>>(),
        vec![current],
        "expected only the current fact; {superseded} is history and {archived} is archived"
    );
}

/// A hit that cannot be named is a hit that cannot be rated.
///
/// `edge_id` is the handle the whole feedback loop turns on: an agent reads a
/// result, finds it useful, and says so by naming it. Without the id there is
/// nothing to name, salience never moves, and the ranking above is a static
/// function of whatever the writers happened to say. So the id is asserted to
/// be the one the write path minted, and asserted again against the table —
/// an id that does not resolve is not a handle.
///
/// The rest of a hit is checked here too, because nothing else does. A disputed
/// edge is deliberately among them: it is returned, flagged, and left to the
/// agent to judge from the proof and the writer, because hiding it would make
/// the dispute invisible rather than the fact harmless.
#[tokio::test]
async fn every_hit_carries_a_usable_edge_id_and_the_columns_around_it() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "hit-shape").await;
    seed_nodes(store, &tenant, &["Alice Whitfield", "Bob Nkemelu", "Kafka"]).await;

    let provenance = Provenance {
        client: Some("claude-desktop/1.4".to_string()),
        principal: Some("agent-7@example.test".to_string()),
    };

    let ids = create_edges_as(
        store,
        &tenant,
        &[
            EdgeInput::with_valid_from(
                "alice-whitfield",
                "bob-nkemelu",
                "REPORTS_TO",
                "Alice Whitfield reports to Bob Nkemelu, per the org chart",
                instant(2024, 6, 1),
            ),
            EdgeInput::new(
                "alice-whitfield",
                "kafka",
                "USES",
                "Alice Whitfield's team publishes settlement events to Kafka",
            ),
        ],
        &provenance,
    )
    .await;
    let (reports_to, contested) = (ids[0], ids[1]);
    set_disputed(store, &tenant, contested, true).await;

    let hits = search(store, &tenant, "Alice Whitfield").await;
    assert_eq!(hits.len(), 2, "both edges should have come back: {hits:#?}");

    let returned: Vec<Uuid> = hits.iter().map(|hit| hit.edge_id).collect();
    let (resolvable,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM edge WHERE tenant_id = $1 AND edge_id = ANY($2)")
            .bind(&tenant)
            .bind(&returned)
            .fetch_one(store.pool())
            .await
            .expect("resolving the returned ids should succeed");
    assert_eq!(
        resolvable, 2,
        "every id a hit carries has to name a row the rating loop can find"
    );

    let hit = &hits[rank_of(&hits, reports_to)];
    assert_eq!(hit.source_node_id, "alice-whitfield");
    assert_eq!(hit.target_node_id, "bob-nkemelu");
    assert_eq!(hit.relationship_type, "REPORTS_TO");
    assert_eq!(hit.valid_from, instant(2024, 6, 1));
    assert_eq!(hit.valid_until, None, "an active fact is still open");
    assert!(!hit.disputed);
    assert_eq!(hit.writer_client.as_deref(), Some("claude-desktop/1.4"));
    assert_eq!(
        hit.writer_principal.as_deref(),
        Some("agent-7@example.test"),
        "who asserted a fact is half of what makes it judgeable"
    );
    assert!(
        hit.effective_salience > 0.99,
        "a fact written moments ago has not decayed: {hit:#?}"
    );
    assert!(hit.score > 0.0, "a returned hit scored something");

    assert!(
        hits[rank_of(&hits, contested)].disputed,
        "a disputed edge is returned and flagged, not hidden: {hits:#?}"
    );
}

/// The two halves answer different questions, and a caller may ask for one.
///
/// `Lexical` is "what do we know about this entity", and it can only ever
/// return edges hanging off a node whose name matched — an edge whose proof
/// text merely reads like the query is not an answer to that question.
/// `Semantic` is "what reads like this", and it reaches edges no name points
/// at. The default is neither, because an agent is not well placed to know
/// which one holds its answer.
#[tokio::test]
async fn each_search_method_uses_only_its_own_candidate_list() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "search-method").await;
    seed_nodes(
        store,
        &tenant,
        &[
            "Alice Whitfield",
            "Bob Nkemelu",
            "Zebra Logistics",
            "Quarterly Report",
        ],
    )
    .await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "alice-whitfield",
                "bob-nkemelu",
                "REPORTS_TO",
                "Alice Whitfield reports to Bob Nkemelu, per the org chart",
            ),
            EdgeInput::new(
                "zebra-logistics",
                "quarterly-report",
                "AUTHORED_BY",
                "the freight volumes appendix was compiled by the logistics desk",
            ),
        ],
    )
    .await;
    let (named, unnamed) = (ids[0], ids[1]);

    let lexical = search_with(
        store,
        &tenant,
        "Alice Whitfield",
        &SearchOptions {
            search_method: Some(SearchMethod::Lexical),
            ..SearchOptions::default()
        },
    )
    .await;
    assert_eq!(
        lexical.iter().map(|hit| hit.edge_id).collect::<Vec<_>>(),
        vec![named],
        "no node in this tenant is named anything like {unnamed}'s endpoints"
    );

    let semantic = search_with(
        store,
        &tenant,
        "Alice Whitfield",
        &SearchOptions {
            search_method: Some(SearchMethod::Semantic),
            ..SearchOptions::default()
        },
    )
    .await;
    assert_eq!(
        semantic.len(),
        2,
        "the vector half ranks every active edge, named or not: {semantic:#?}"
    );
}

/// A query with nothing in it has no answer, and the answer it would otherwise
/// get is the worst kind: plausible.
///
/// The vector half will happily rank every edge in the tenant by distance to
/// the embedding of an empty string and hand back ten of them, which is not a
/// failure a caller can see — the hits look exactly like hits. The lexical half
/// is worse: `strpos(anything, '')` is 1, so every node in the tenant matches.
/// Both are junk in an agent's context, so a blank query returns nothing.
#[tokio::test]
async fn a_blank_query_returns_nothing_rather_than_an_arbitrary_page() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "blank-query").await;
    seed_nodes(store, &tenant, &["Alice Whitfield", "Kafka"]).await;

    create_edges(
        store,
        &tenant,
        &[EdgeInput::new(
            "alice-whitfield",
            "kafka",
            "USES",
            "Alice Whitfield's team publishes settlement events to Kafka",
        )],
    )
    .await;

    for query in ["", "   ", "\n\t"] {
        let hits = search(store, &tenant, query).await;
        assert!(
            hits.is_empty(),
            "a query of {query:?} has nothing to be relevant to: {hits:#?}"
        );
    }
}

/// A disputed fact sinks below the ordinary band, and is still returned.
///
/// The schema states this outright — "a disputed edge is still returned by
/// search, demoted below the normal salience band and flagged in output, so the
/// agent can judge it from proof and provenance" — and both halves of that
/// matter. Hiding it would make the *dispute* invisible rather than the fact
/// harmless: the edge stops being retrievable, therefore stops being ratable,
/// and the quorum that flagged it can never be revised. Leaving it at its
/// ordinary rank would hand an agent a fact two principals have said is wrong,
/// above facts nobody has objected to.
///
/// Both directions again, and for the same reason as the salience tie: the two
/// edges below carry the same proof text, so which one relevance puts first is
/// a random uuid, and only asserting both ways rules out passing by luck.
#[tokio::test]
async fn a_disputed_edge_sinks_below_the_band_but_is_still_returned() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "disputed-demotion").await;
    seed_nodes(store, &tenant, &["Alice Whitfield", "Kafka", "Postgres"]).await;

    let shared_proof = "the settlement events pipeline was signed off in March";
    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new("alice-whitfield", "kafka", "USES", shared_proof),
            EdgeInput::new("alice-whitfield", "postgres", "USES", shared_proof),
        ],
    )
    .await;

    for (contested, trusted) in [(ids[0], ids[1]), (ids[1], ids[0])] {
        set_disputed(store, &tenant, contested, true).await;
        set_disputed(store, &tenant, trusted, false).await;

        let hits = search(store, &tenant, "settlement events pipeline").await;

        assert!(
            rank_of(&hits, trusted) < rank_of(&hits, contested),
            "a contested fact must not sit above an uncontested one: {hits:#?}"
        );
        assert!(
            hits[rank_of(&hits, contested)].disputed,
            "and it comes back flagged, not quietly demoted"
        );
    }
}

/// The plainest thing an agent asks for: a name it has in front of it.
///
/// The lexical half exists for exactly this. A vector search over proof text
/// answers "what reads like this sentence", which is a different question from
/// "what do we know about Alice Whitfield" — and the second question is the one
/// an agent asks when it is holding a name. Fusing the two by rank means the
/// edges hanging off the named node come back above edges that merely happen to
/// sit somewhere nearby in embedding space.
#[tokio::test]
async fn a_query_naming_a_node_ranks_that_nodes_edges_above_unrelated_ones() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "name-match").await;
    seed_nodes(
        store,
        &tenant,
        &[
            "Alice Whitfield",
            "Bob Nkemelu",
            "Kafka",
            "Zebra Logistics",
            "Quarterly Report",
        ],
    )
    .await;

    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "alice-whitfield",
                "bob-nkemelu",
                "REPORTS_TO",
                "Alice Whitfield reports to Bob Nkemelu, per the org chart",
            ),
            EdgeInput::new(
                "alice-whitfield",
                "kafka",
                "USES",
                "Alice Whitfield's team publishes settlement events to Kafka",
            ),
            EdgeInput::new(
                "zebra-logistics",
                "quarterly-report",
                "AUTHORED_BY",
                "the freight volumes appendix was compiled by the logistics desk",
            ),
        ],
    )
    .await;
    let (reports_to, uses, unrelated) = (ids[0], ids[1], ids[2]);

    let hits = search(store, &tenant, "Alice Whitfield").await;

    let unrelated_rank = rank_of(&hits, unrelated);
    assert!(
        rank_of(&hits, reports_to) < unrelated_rank && rank_of(&hits, uses) < unrelated_rank,
        "both of the named node's edges must outrank the unrelated one: {hits:#?}"
    );
}

/// **A list that cannot separate its rows must not pretend it can.**
///
/// Every edge here hangs off `alice-whitfield`, which is the ordinary shape of
/// a query that names one person. The lexical half orders on `exact`, on the
/// count of query terms in the name, and on the name's length — all three
/// properties of the matched *node*, so all three are identical across every
/// candidate. Nothing about the edges themselves reaches that ordering.
///
/// So the lexical list has one honest thing to say — "these are equally good" —
/// and the only ranking left in the fused result is the semantic one. Breaking
/// the tie on `edge_id` instead hands fusion an arbitrary order and it scores
/// it as a second opinion: an edge comes back *corroborated* by two lists when
/// one of them ranked nothing at all.
///
/// The assertion is therefore that the fused ranking of these edges is exactly
/// the semantic ranking of them. The semantic order is read from the server
/// rather than written down, because it is a property of the embedder and not
/// of this test; what the test pins is that fusion did not overturn it with a
/// UUID.
#[tokio::test]
async fn edges_sharing_one_matched_node_keep_the_order_the_semantic_list_gave_them() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "lexical-tie").await;
    seed_nodes(
        store,
        &tenant,
        &[
            "Alice Whitfield",
            "Zebra Logistics",
            "Quarterly Report",
            "Kafka",
            "Helsinki",
            "Bob Nkemelu",
        ],
    )
    .await;

    // Five edges, one source, five `many`-cardinality types, and not one target
    // whose name contains "alice" or "whitfield" — so `alice-whitfield` is the
    // only node the query matches and every edge ties on every key the lexical
    // ordering reads.
    let ids = create_edges(
        store,
        &tenant,
        &[
            EdgeInput::new(
                "alice-whitfield",
                "zebra-logistics",
                "MEMBER_OF",
                "Alice Whitfield has been on the Zebra Logistics payroll since 2019",
            ),
            EdgeInput::new(
                "alice-whitfield",
                "kafka",
                "USES",
                "Alice Whitfield's team publishes settlement events to Kafka",
            ),
            EdgeInput::new(
                "alice-whitfield",
                "quarterly-report",
                "AUTHORED_BY",
                "the freight volumes appendix was compiled by Alice Whitfield",
            ),
            EdgeInput::new(
                "alice-whitfield",
                "helsinki",
                "PREFERS",
                "Alice Whitfield asked to be based out of Helsinki next year",
            ),
            EdgeInput::new(
                "alice-whitfield",
                "bob-nkemelu",
                "RELATED_TO",
                "Alice Whitfield and Bob Nkemelu ran the settlement migration together",
            ),
        ],
    )
    .await;

    let semantic = ids_from(
        store,
        &tenant,
        "Alice Whitfield",
        SearchMethod::Semantic,
        Precision::Relaxed,
        None,
    )
    .await;
    let lexical = ids_from(
        store,
        &tenant,
        "Alice Whitfield",
        SearchMethod::Lexical,
        Precision::Relaxed,
        None,
    )
    .await;
    let hybrid = ids_from(
        store,
        &tenant,
        "Alice Whitfield",
        SearchMethod::Hybrid,
        Precision::Relaxed,
        None,
    )
    .await;

    assert_eq!(
        semantic.len(),
        ids.len(),
        "the fixture only tests anything if both halves see every edge: {semantic:#?}"
    );
    assert_eq!(lexical.len(), ids.len(), "{lexical:#?}");
    assert_ne!(
        lexical, semantic,
        "the fixture only tests anything if the two lists disagree — otherwise \
         a UUID sort and a ranking are indistinguishable here"
    );

    assert_eq!(
        hybrid, semantic,
        "the lexical half tied on every key it reads, so it had nothing to \
         contribute to the order and the semantic ranking should have survived \
         intact; instead fusion scored the lexical list's edge-id order as a \
         second opinion.\n  semantic: {semantic:#?}\n  lexical (edge-id order): \
         {lexical:#?}\n  fused: {hybrid:#?}"
    );
}
