//! Integration tests for edge ratings and the dispute quorum, against a real
//! Postgres.
//!
//! Every test owns a `tenant_id` derived from its own name, for the reason the
//! node, edge and search suites give: the suite runs in parallel against one
//! shared database, so isolation cannot come from truncating shared tables.
//! Tenancy is the isolation mechanism the schema already has.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use chrono::{DateTime, Duration, Utc};
use mundu::embed::DeterministicEmbedder;
use mundu::error::StoreError;
use mundu::store::edge::{EdgeInput, EdgeOutcome};
use mundu::store::node::NodeInput;
use mundu::store::rating::{RatingOutcome, RatingRejection};
use mundu::store::search::SearchOptions;
use mundu::store::{Provenance, Store};
use sqlx::Row;
use uuid::Uuid;

/// The half-life the store decays salience over, restated here because a test
/// cannot see the crate's internal constant. A rating rebases the stored value
/// to the present before it applies its delta, and the only way to write a test
/// about that is to age an edge by a known multiple of this number.
const SALIENCE_HALF_LIFE_DAYS: i64 = 27;

/// One [`Store`] per test, deliberately not a shared `static`, for the reason
/// the node suite documents: a sqlx pool is bound to the reactor that created
/// it, and every `#[tokio::test]` builds and tears down its own.
async fn store() -> Store {
    common::store().await
}

/// Names a tenant after the test that owns it and clears anything a previous
/// run left behind, so "one rating was appended" means the same thing on the
/// first run and the hundredth.
///
/// Ratings go first: they reference edges, which reference nodes and each other.
async fn fresh_tenant(store: &Store, name: &str) -> String {
    let tenant = format!("test-rating-{name}");

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

/// Mints the nodes a rating test needs through the real upsert path, so the ids
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

/// Writes one edge through the real edge path and returns the id the store
/// minted for it.
async fn create_edge(store: &Store, tenant: &str, input: EdgeInput) -> Uuid {
    let outcomes = store
        .create_edges(tenant, &[input], &embedder(), &anonymous())
        .await
        .expect("seeding the edge should succeed");

    match &outcomes[0] {
        EdgeOutcome::Created { edge_id, .. } => *edge_id,
        other => panic!("expected the seeded edge to be current, got {other:?}"),
    }
}

/// The edge every test that only needs "some edge" rates.
async fn seed_one_edge(store: &Store, tenant: &str) -> Uuid {
    seed_nodes(store, tenant, &["Alice Whitfield", "Kafka"]).await;

    create_edge(
        store,
        tenant,
        EdgeInput::new(
            "alice-whitfield",
            "kafka",
            "USES",
            "Alice Whitfield's team publishes settlement events to Kafka",
        ),
    )
    .await
}

/// Sets an edge's stored salience directly, both columns together, because
/// effective salience is a function of the pair: a base with no `updated_at`
/// decays from whenever the edge happened to be written.
///
/// This is arranging a starting point, not exercising a path. A freshly written
/// edge sits at 1.0, where nothing positive can be observed to raise it.
async fn set_salience(
    store: &Store,
    tenant: &str,
    edge_id: Uuid,
    base: f32,
    updated_at: DateTime<Utc>,
) {
    let affected = sqlx::query(
        "UPDATE edge SET salience_base = $3, salience_updated_at = $4 \
         WHERE tenant_id = $1 AND edge_id = $2",
    )
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

/// The salience the database actually holds, which is the only value a later
/// search will rank on.
async fn stored_salience(store: &Store, tenant: &str, edge_id: Uuid) -> f32 {
    let (base,): (f32,) =
        sqlx::query_as("SELECT salience_base FROM edge WHERE tenant_id = $1 AND edge_id = $2")
            .bind(tenant)
            .bind(edge_id)
            .fetch_one(store.pool())
            .await
            .expect("reading the edge's salience should succeed");

    base
}

/// One appended rating, as the log holds it.
#[derive(Debug, Clone, PartialEq)]
struct RatingRow {
    edge_id: Uuid,
    utility_score: f32,
    rated_by: Option<String>,
}

/// Every rating in a tenant, oldest first.
async fn ratings_of(store: &Store, tenant: &str) -> Vec<RatingRow> {
    let rows = sqlx::query(
        "SELECT edge_id, utility_score, rated_by FROM edge_rating \
         WHERE tenant_id = $1 ORDER BY rating_id",
    )
    .bind(tenant)
    .fetch_all(store.pool())
    .await
    .expect("reading the tenant's ratings should succeed");

    rows.into_iter()
        .map(|row| RatingRow {
            edge_id: row.get("edge_id"),
            utility_score: row.get("utility_score"),
            rated_by: row.get("rated_by"),
        })
        .collect()
}

/// When the edge was flagged as disputed, if it has been.
async fn disputed_at(store: &Store, tenant: &str, edge_id: Uuid) -> Option<DateTime<Utc>> {
    let (disputed_at,): (Option<DateTime<Utc>>,) =
        sqlx::query_as("SELECT disputed_at FROM edge WHERE tenant_id = $1 AND edge_id = $2")
            .bind(tenant)
            .bind(edge_id)
            .fetch_one(store.pool())
            .await
            .expect("reading the edge's dispute flag should succeed");

    disputed_at
}

/// The edge's lifecycle status, which rating must never touch.
async fn status_of(store: &Store, tenant: &str, edge_id: Uuid) -> String {
    let (status,): (String,) =
        sqlx::query_as("SELECT status::text FROM edge WHERE tenant_id = $1 AND edge_id = $2")
            .bind(tenant)
            .bind(edge_id)
            .fetch_one(store.pool())
            .await
            .expect("reading the edge's status should succeed");

    status
}

/// Rates one edge and returns the single outcome, so a test can talk about
/// "the rating" without indexing into a batch.
async fn rate_one(
    store: &Store,
    tenant: &str,
    edge_id: Uuid,
    utility_score: f32,
    rated_by: Option<&str>,
) -> RatingOutcome {
    let mut outcomes = store
        .rate_edges(tenant, &[edge_id], utility_score, rated_by)
        .await
        .expect("rating an edge this tenant holds should succeed");

    assert_eq!(outcomes.len(), 1, "one edge id, one outcome");
    outcomes.remove(0)
}

/// The salience an outcome reports, or a failure naming what came back instead.
fn rated_salience(outcome: &RatingOutcome) -> f32 {
    match outcome {
        RatingOutcome::Rated { salience, .. } => *salience,
        other => panic!("expected the edge to be rated, got {other:?}"),
    }
}

/// Salience is a `REAL`, so equality is asserted with a tolerance — wide enough
/// to absorb `f32` rounding and the fraction of a half-life a test takes to
/// run, narrow enough that 0.65 and 0.90 are different answers.
fn assert_close(actual: f32, expected: f32, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-3,
        "{what}: expected {expected}, got {actual}"
    );
}

/// Positive ratings raise salience, and the raise stops at 1.0.
///
/// The ceiling is not decoration: `salience_base` is `CHECK`-constrained to
/// `[0, 1]`, so an unclamped increment does not merely overshoot, it aborts the
/// write — and the rating that carried it would be lost along with it. Clamping
/// also keeps the scale meaningful: if enthusiastic agents could pile salience
/// up to 4.0, an edge would need sixteen negative ratings before it fell back
/// into the band search actually ranks on.
#[tokio::test]
async fn a_positive_rating_raises_salience_and_stops_at_one() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "positive-caps").await;
    let edge = seed_one_edge(store, &tenant).await;

    set_salience(store, &tenant, edge, 0.4, Utc::now()).await;

    // +1.0 of utility is worth a quarter of the scale, so the first two
    // ratings are ordinary arithmetic and the third would overshoot.
    for expected in [0.65, 0.90, 1.0, 1.0] {
        let outcome = rate_one(store, &tenant, edge, 1.0, Some("agent-a")).await;

        assert_close(rated_salience(&outcome), expected, "the reported salience");
        assert_close(
            stored_salience(store, &tenant, edge).await,
            expected,
            "the stored salience",
        );
    }
}

/// A score outside `[-1, 1]` is refused, before anything is written.
///
/// The database would refuse it too, as `edge_rating_range` — but only per
/// statement, after earlier edges in the batch have already committed their
/// ratings and moved their salience. The caller would then hold an error and a
/// half-applied batch, with nothing to say where it stopped, and the only safe
/// retry duplicates every rating that did land.
///
/// A bad score is decidable from the argument alone, the way a blank display
/// name and a future-dated `valid_from` are, so it is decided there: one error
/// naming the score, and a table nobody has touched.
#[tokio::test]
async fn a_score_outside_the_scale_is_refused_before_anything_is_written() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "score-range").await;
    let edge = seed_one_edge(store, &tenant).await;

    set_salience(store, &tenant, edge, 0.5, Utc::now()).await;

    // Both ends, and a NaN: a comparison against NaN is false whichever way it
    // is written, so a range check assembled from `>` and `<` lets it through
    // to poison `salience_base` with a value no later rating can recover from.
    for score in [1.5_f32, -1.5, 42.0, f32::NAN, f32::INFINITY] {
        let error = store
            .rate_edges(&tenant, &[edge], score, Some("agent-a"))
            .await
            .expect_err("a score off the scale must be refused");

        assert!(
            matches!(error, StoreError::UtilityScoreOutOfRange { .. }),
            "expected a range error for {score}, got {error:?}"
        );
    }

    assert!(
        ratings_of(store, &tenant).await.is_empty(),
        "a refused call must not have appended anything"
    );
    assert_close(
        stored_salience(store, &tenant, edge).await,
        0.5,
        "a refused call must not have moved salience",
    );
}

/// A rating rebases: it decays the stored value to the present *before* it
/// applies its delta.
///
/// `salience_base` is not the edge's salience, it is the salience at
/// `salience_updated_at`; every reader decays it forward. So adding to the base
/// directly reads the value as though no time had passed since it was written,
/// and quietly hands back everything decay took away — an edge nobody has
/// touched for a year would jump from an effective 0.0001 to 0.25 on one
/// lukewarm rating, and a stale fact would outrank a fresh one because it was
/// once popular.
///
/// The edge below sits one half-life behind, so its effective salience is 0.50
/// and a full positive rating must leave 0.75. Reading the base instead gives
/// 1.0, which is also what a clamp would produce from 1.25 — hence the aged
/// base of exactly 1.0, where the wrong answer is visibly wrong.
#[tokio::test]
async fn a_rating_decays_the_stored_salience_before_it_adds_its_delta() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "rebases-on-decay").await;
    let edge = seed_one_edge(store, &tenant).await;

    let one_half_life_ago = Utc::now() - Duration::days(SALIENCE_HALF_LIFE_DAYS);
    set_salience(store, &tenant, edge, 1.0, one_half_life_ago).await;

    let outcome = rate_one(store, &tenant, edge, 1.0, Some("agent-a")).await;

    // 27 days rather than the 27.1 the decay actually uses, because a test
    // cannot see that constant: the remaining tenth of a day is worth ~0.0013
    // of salience, so the tolerance is widened just past it.
    let expected = 0.5 * 0.5_f32.powf(-0.1 / 27.1) + 0.25;
    assert!(
        (rated_salience(&outcome) - expected).abs() < 2e-3,
        "expected the aged edge to be rated up from its decayed 0.5, not its \
         stored 1.0: expected about {expected}, got {}",
        rated_salience(&outcome)
    );

    assert_close(
        stored_salience(store, &tenant, edge).await,
        rated_salience(&outcome),
        "the stored salience is what was reported",
    );
}

/// One principal cannot dispute an edge by rating it twice, or twenty times.
///
/// The quorum counts *principals*, not ratings, and the difference is the whole
/// of the protection. An agent runs in a loop: whatever made it rate a fact
/// badly once will make it rate the same fact badly on every pass, so a count
/// of ratings is a count of retries, and one agent stuck in a loop would flag
/// as "the graph disagrees with this". Worse, the loop is drivable — an
/// injected line in some other agent's proof text can ask a reader to rate a
/// competing fact down, and a per-rating threshold would let that one reader
/// carry out the sentence alone.
#[tokio::test]
async fn negatives_from_one_principal_never_reach_the_quorum() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "same-principal").await;
    let edge = seed_one_edge(store, &tenant).await;

    for _ in 0..5 {
        rate_one(store, &tenant, edge, -1.0, Some("agent-a")).await;
    }

    assert_eq!(
        disputed_at(store, &tenant, edge).await,
        None,
        "one principal repeating itself is one objection, however loudly"
    );
}

/// Two principals objecting is a dispute, and the first one to reach the
/// quorum is when the dispute began.
///
/// Two independent agents both finding a fact unhelpful is a signal about the
/// fact rather than about either agent, which is exactly what one agent
/// repeating itself is not. The flag is soft: search still returns the edge,
/// demoted and marked, so an agent can judge it from proof and provenance.
///
/// `disputed_at` is set once and left. It records when the graph first came to
/// doubt the fact, and refreshing it on every later negative would turn an
/// audit trail into a "most recently complained about" clock — and hand anyone
/// who can drive a rating loop a way to keep a fact looking freshly contested.
#[tokio::test]
async fn negatives_from_two_principals_flag_the_edge_as_disputed() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "two-principals").await;
    let edge = seed_one_edge(store, &tenant).await;

    rate_one(store, &tenant, edge, -1.0, Some("agent-a")).await;
    assert_eq!(
        disputed_at(store, &tenant, edge).await,
        None,
        "one objection is not yet a dispute"
    );

    rate_one(store, &tenant, edge, -0.5, Some("agent-b")).await;
    let flagged = disputed_at(store, &tenant, edge)
        .await
        .expect("a second principal objecting is a dispute");

    rate_one(store, &tenant, edge, -1.0, Some("agent-c")).await;
    assert_eq!(
        disputed_at(store, &tenant, edge).await,
        Some(flagged),
        "a third objection must not restamp when the dispute began"
    );
}

/// A blank principal is nobody, and two spellings of nobody are not two
/// principals.
///
/// `rated_by` reaches this store from the validated credential, and an auth
/// mode with no principal to report may hand over an empty string rather than
/// nothing at all — the two are the same fact about attribution, and only one
/// of them is written as such. Left as given, `""` and `"   "` are two distinct
/// strings and therefore a quorum: an anonymous caller disputing any fact it
/// can name, twice, on its own.
///
/// So a blank is stored as NULL, which is what the log means by "we could not
/// tie this to anybody", and the quorum's one rule about attribution stays in
/// one place.
#[tokio::test]
async fn a_blank_principal_is_recorded_as_nobody_and_cannot_vote() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "blank-principal").await;
    let edge = seed_one_edge(store, &tenant).await;

    rate_one(store, &tenant, edge, -1.0, Some("")).await;
    rate_one(store, &tenant, edge, -1.0, Some("   ")).await;
    rate_one(store, &tenant, edge, -1.0, Some("\t\n")).await;

    assert_eq!(
        disputed_at(store, &tenant, edge).await,
        None,
        "three spellings of nobody are still nobody"
    );
    assert!(
        ratings_of(store, &tenant)
            .await
            .iter()
            .all(|rating| rating.rated_by.is_none()),
        "and the log says so, rather than keeping a name that names no one"
    );
}

/// Two agents rating the same edge at the same moment both count.
///
/// A rating is a read-modify-write: read the decayed salience, add a delta,
/// write it back. Two of them overlapping under read-committed both read the
/// same value and both write their own answer over it, so one rating is
/// silently discarded — no error, no log line, nothing in the table to say it
/// happened. That is precisely the failure this store must not have, because
/// the whole feedback loop is agents rating what they read, in parallel, and a
/// rating that vanishes looks exactly like a rating that was applied.
///
/// The overlap is arranged rather than hoped for. The transaction below holds a
/// row lock on the edge until both raters are queued behind it; whether they
/// then serialise correctly is the thing under test. A plain `SELECT` is never
/// blocked by a row lock, so an implementation that reads without locking sails
/// past, and both raters compute their new salience from the same stale value.
#[tokio::test]
async fn two_raters_at_once_do_not_lose_one_of_the_ratings() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "concurrent-raters").await;
    let edge = seed_one_edge(store, &tenant).await;

    set_salience(store, &tenant, edge, 0.5, Utc::now()).await;

    let mut blocker = store
        .pool()
        .begin()
        .await
        .expect("opening the blocking transaction should succeed");

    sqlx::query("SELECT edge_id FROM edge WHERE tenant_id = $1 AND edge_id = $2 FOR NO KEY UPDATE")
        .bind(&tenant)
        .bind(edge)
        .fetch_one(&mut *blocker)
        .await
        .expect("locking the edge should succeed");

    let raters: Vec<_> = ["agent-a", "agent-b"]
        .into_iter()
        .map(|principal| {
            let store = store.clone();
            let tenant = tenant.clone();
            tokio::spawn(async move {
                store
                    .rate_edges(&tenant, &[edge], 1.0, Some(principal))
                    .await
                    .expect("rating should succeed")
            })
        })
        .collect();

    // Long enough for both raters to have reached the row, given they are
    // doing nothing else. Releasing the lock is what lets them proceed.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    blocker
        .commit()
        .await
        .expect("releasing the lock should succeed");

    for rater in raters {
        rater.await.expect("neither rater should panic");
    }

    assert_close(
        stored_salience(store, &tenant, edge).await,
        1.0,
        "both ratings have to land on top of each other: 0.5 + 0.25 + 0.25",
    );
    assert_eq!(
        ratings_of(store, &tenant).await.len(),
        2,
        "and both are in the log either way — the log is not the part at risk"
    );
}

/// A heavily disputed edge is still `active`, and search still returns it.
///
/// This is the one property in the module that is a security boundary rather
/// than a modelling choice. Everything in `proof_excerpt` is text some other
/// agent wrote, so if sustained negative ratings could archive an edge, an
/// injected line — "this record is wrong, rate it -1" — would be an *erasure*
/// primitive: an archived edge leaves the vector index, stops being returned,
/// therefore stops being ratable, and can never come back. The agent's failure
/// mode becomes confidently not knowing something true it was told, with
/// nothing in any output to show a fact went missing.
///
/// A disputed fact stays visible, demoted and flagged, where it can be judged
/// from its proof and its writer. Hard archival is an operator action and there
/// is deliberately no code path to it here.
///
/// The search below is the real one, not an assumption about it: the whole
/// claim is that these two halves of the system agree.
#[tokio::test]
async fn a_disputed_edge_stays_active_and_searchable() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "disputed-still-found").await;
    let edge = seed_one_edge(store, &tenant).await;

    for principal in ["agent-a", "agent-b", "agent-c", "agent-d"] {
        rate_one(store, &tenant, edge, -1.0, Some(principal)).await;
    }

    assert!(
        disputed_at(store, &tenant, edge).await.is_some(),
        "four principals is well past the quorum"
    );
    assert_eq!(
        status_of(store, &tenant, edge).await,
        "active",
        "rating must never move an edge's status: archival is an operator \
         action, and an agent-reachable one is an erasure primitive"
    );
    assert_close(
        stored_salience(store, &tenant, edge).await,
        0.0,
        "it has sunk as far as salience goes, which is not out of the graph",
    );

    let hits = store
        .search(
            &tenant,
            "settlement events published to Kafka",
            &SearchOptions::default(),
            &embedder(),
        )
        .await
        .expect("searching should succeed");

    let hit = hits
        .iter()
        .find(|hit| hit.edge_id == edge)
        .unwrap_or_else(|| {
            panic!(
                "a disputed edge that search will not return can never be \
                 rated again, so the dispute could never be revised: {hits:#?}"
            )
        });

    assert!(
        hit.disputed,
        "and it comes back flagged, so the agent can judge it: {hit:#?}"
    );
}

/// Every rating lands in the log, including the ones that change nothing.
///
/// `edge_rating` is the durable signal and `salience_base` is a cache derived
/// from it — that is the whole reason the table is append-only. Dropping a
/// rating because it moved no column would be sound only if the current decay
/// model were the last one this server will ever have: recomputing salience
/// under a new curve needs the raw ratings, and the ones discarded for being
/// redundant under the old model are exactly the ones that cannot be recovered.
///
/// The quorum needs the same history for a different reason. It counts distinct
/// principals over all of an edge's negative ratings, so a rating that is the
/// second from one principal still has to be there when a third principal
/// arrives.
#[tokio::test]
async fn every_rating_is_appended_even_when_it_moves_nothing() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "append-only").await;
    let edge = seed_one_edge(store, &tenant).await;

    // A fresh edge sits at 1.0, so both of these are no-ops for salience.
    rate_one(store, &tenant, edge, 1.0, Some("agent-a")).await;
    rate_one(store, &tenant, edge, 1.0, Some("agent-a")).await;
    rate_one(store, &tenant, edge, -0.5, Some("agent-b")).await;
    rate_one(store, &tenant, edge, -0.5, None).await;

    // The same id twice in one batch is two ratings, not one deduplicated
    // opinion: the caller said it twice.
    let outcomes = store
        .rate_edges(&tenant, &[edge, edge], 0.5, Some("agent-c"))
        .await
        .expect("rating the same edge twice in one batch should succeed");
    assert_eq!(outcomes.len(), 2, "one outcome per id, repeats included");

    assert_eq!(
        ratings_of(store, &tenant).await,
        vec![
            RatingRow {
                edge_id: edge,
                utility_score: 1.0,
                rated_by: Some("agent-a".to_string()),
            },
            RatingRow {
                edge_id: edge,
                utility_score: 1.0,
                rated_by: Some("agent-a".to_string()),
            },
            RatingRow {
                edge_id: edge,
                utility_score: -0.5,
                rated_by: Some("agent-b".to_string()),
            },
            RatingRow {
                edge_id: edge,
                utility_score: -0.5,
                rated_by: None,
            },
            RatingRow {
                edge_id: edge,
                utility_score: 0.5,
                rated_by: Some("agent-c".to_string()),
            },
            RatingRow {
                edge_id: edge,
                utility_score: 0.5,
                rated_by: Some("agent-c".to_string()),
            },
        ],
        "the log keeps all of them, in the order they were given"
    );
}

/// An unattributed rating cannot be part of a crowd, however many arrive.
///
/// `rated_by` is the principal from the validated credential, so a NULL means
/// this server cannot tie the objection to anybody. Counting those towards the
/// quorum would let a single anonymous caller — one loop, one token, one
/// script — dispute every fact it can name, and the dispute would carry no
/// record of who asked for it. Ratings are still appended: an unattributed
/// objection is worth keeping and still moves salience, it just cannot vote.
///
/// The named principal at the end is the control. Without it the test would
/// pass just as well against an implementation that never disputes anything.
#[tokio::test]
async fn anonymous_negatives_never_reach_the_quorum_however_many_arrive() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "anonymous-quorum").await;
    let edge = seed_one_edge(store, &tenant).await;

    for _ in 0..6 {
        rate_one(store, &tenant, edge, -1.0, None).await;
    }
    assert_eq!(
        disputed_at(store, &tenant, edge).await,
        None,
        "nobody in particular, six times over, is still nobody in particular"
    );

    rate_one(store, &tenant, edge, -1.0, Some("agent-a")).await;
    assert_eq!(
        disputed_at(store, &tenant, edge).await,
        None,
        "and a crowd of anonymous ratings must not top one real principal up \
         to a quorum"
    );

    rate_one(store, &tenant, edge, -1.0, Some("agent-b")).await;
    assert!(
        disputed_at(store, &tenant, edge).await.is_some(),
        "two attributable principals is the quorum, so this fixture can \
         actually reach it"
    );

    assert_eq!(
        ratings_of(store, &tenant).await.len(),
        8,
        "every rating is in the log, votes and non-votes alike"
    );
}

/// An edge another tenant owns fails its own item and nothing else — and fails
/// it exactly the way a made-up uuid does.
///
/// `tenant_id` is the whole isolation boundary and comes from a validated
/// credential rather than a tool argument, so an id from outside it is not a
/// rating this server may apply. Two properties matter beyond refusing it:
///
/// The refusal is per item. An agent that carries one stale `edge_id` out of
/// five has still read four facts worth rating, and sinking the call teaches it
/// nothing about which id was wrong.
///
/// And the refusal says the same thing either way. If a foreign id came back
/// differently from an unknown one, this call would answer "does edge X exist
/// somewhere in this database" for any uuid a caller cared to try — a
/// cross-tenant oracle built out of an error type.
#[tokio::test]
async fn an_edge_from_another_tenant_fails_its_own_item_and_says_nothing_more() {
    let store = &store().await;
    let owner = fresh_tenant(store, "foreign-owner").await;
    let rater = fresh_tenant(store, "foreign-rater").await;

    let theirs = seed_one_edge(store, &owner).await;
    let ours = seed_one_edge(store, &rater).await;
    let invented = Uuid::new_v4();

    set_salience(store, &owner, theirs, 0.5, Utc::now()).await;
    set_salience(store, &rater, ours, 0.5, Utc::now()).await;

    let outcomes = store
        .rate_edges(&rater, &[theirs, ours, invented], -1.0, Some("agent-a"))
        .await
        .expect("a batch naming a foreign edge is not a failed call");

    assert_eq!(outcomes.len(), 3, "one outcome per id, in input order");
    assert_eq!(
        outcomes[0],
        RatingOutcome::Rejected {
            edge_id: theirs,
            reason: RatingRejection::UnknownEdge,
        },
        "another tenant's edge is simply not an edge here"
    );
    assert_eq!(
        outcomes[2],
        RatingOutcome::Rejected {
            edge_id: invented,
            reason: RatingRejection::UnknownEdge,
        },
        "and an id that exists nowhere is refused identically, or the \
         difference between the two answers is a cross-tenant oracle"
    );

    assert_close(
        rated_salience(&outcomes[1]),
        0.25,
        "the sibling in the same batch still committed",
    );
    assert_close(
        stored_salience(store, &rater, ours).await,
        0.25,
        "and it committed to the table, not just to the reply",
    );

    assert_close(
        stored_salience(store, &owner, theirs).await,
        0.5,
        "the other tenant's edge must be untouched",
    );
    assert!(
        ratings_of(store, &owner).await.is_empty(),
        "and nothing may be appended against it either"
    );
}

/// Negative ratings lower salience, and the fall stops at 0.0.
///
/// The floor has to be applied in this code rather than left to the column's
/// `CHECK`: the constraint does not clamp, it aborts the statement, so an
/// unclamped decrement would roll back the whole rating — the append-only log
/// entry included — and the agent reporting that a fact misled it would get an
/// error instead of a recorded objection.
///
/// Nor is zero a tombstone. Search never filters on salience, so an edge at 0.0
/// is still returned, still ratable, and still recoverable by a single positive
/// rating. Sinking is not deleting.
#[tokio::test]
async fn a_negative_rating_lowers_salience_and_stops_at_zero() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "negative-floors").await;
    let edge = seed_one_edge(store, &tenant).await;

    set_salience(store, &tenant, edge, 0.3, Utc::now()).await;

    for expected in [0.05, 0.0, 0.0] {
        // One principal throughout: this test is about arithmetic, and two
        // negatives from one agent are deliberately not a dispute.
        let outcome = rate_one(store, &tenant, edge, -1.0, Some("agent-a")).await;

        assert_close(rated_salience(&outcome), expected, "the reported salience");
        assert_close(
            stored_salience(store, &tenant, edge).await,
            expected,
            "the stored salience",
        );
    }
}
