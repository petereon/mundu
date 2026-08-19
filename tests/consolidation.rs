//! Integration tests for the consolidation worker, against a real Postgres.
//!
//! Every test owns a `tenant_id` derived from its own name, for the reason the
//! node, edge and merge suites give: the suite runs in parallel against one
//! shared database, so isolation cannot come from truncating shared tables.
//! Tenancy is the isolation mechanism the schema already has — and it is also
//! what the worker's own scoping is under test here, so a shared tenant would
//! make the interesting failure invisible.
//!
//! **Time is injected, not waited for.** Every entry point takes the instant it
//! is running at, and the tests hand it one. Nothing here sleeps: the four-hour
//! bound is exercised by moving the clock, not by moving the wall.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use chrono::{DateTime, Duration, TimeZone, Utc};
use mundu::embed::{DeterministicEmbedder, Embedder};
use mundu::store::consolidate::{
    ConsolidationAttempt, ConsolidationPolicy, LexicalVerifier, Trigger, advisory_lock_key,
};
use mundu::store::node::NodeInput;
use mundu::store::{Provenance, Store};

/// One [`Store`] per test, deliberately not a shared `static`, for the reason
/// the node suite documents: a sqlx pool is bound to the reactor that created
/// it, and every `#[tokio::test]` builds and tears down its own.
async fn store() -> Store {
    common::store().await
}

/// Names a tenant after the test that owns it and clears anything a previous
/// run left behind.
///
/// The order is the reverse of the reference graph: the worker's own log points
/// at its runs, the merge audit points at edges and nodes, edges point at nodes
/// and at each other, so the leaves come off first.
async fn fresh_tenant(store: &Store, name: &str) -> String {
    let tenant = format!("test-consolidation-{name}");

    for statement in [
        "DELETE FROM consolidation_decision WHERE tenant_id = $1",
        "DELETE FROM consolidation_run WHERE tenant_id = $1",
        "DELETE FROM consolidation_state WHERE tenant_id = $1",
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

/// A fixed instant to measure every injected clock from, so a test that moves
/// time reads as an offset rather than as two absolute dates.
fn epoch() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0)
        .single()
        .expect("a fixed valid instant")
}

/// The bounds under test, spelled out rather than defaulted: a test that
/// depended on the shipped four hours would silently change meaning the day
/// somebody tuned it.
fn policy(growth_percent: f64, max_interval: Duration) -> ConsolidationPolicy {
    ConsolidationPolicy {
        growth_percent,
        max_interval,
    }
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

/// Distinct, mutually unrelated names, for tests that care about *how many*
/// nodes a tenant holds rather than what they are called. The deterministic
/// embedder maps different text to near-orthogonal vectors, so nothing in here
/// is ever a merge candidate.
async fn seed_unrelated_nodes(store: &Store, tenant: &str, count: usize, tag: &str) {
    let names: Vec<String> = (0..count)
        .map(|index| format!("Unrelated {tag} Entity Number {index}"))
        .collect();
    let borrowed: Vec<&str> = names.iter().map(String::as_str).collect();
    seed_nodes(store, tenant, &borrowed).await;
}

/// Both bounds must hold, and whichever fires first wins. This is the growth
/// half: a tenant whose graph has grown by the configured percentage is
/// consolidated without waiting out the interval, and a tenant that has grown
/// by less than that is not.
///
/// The "and not before" half is the one worth having. A trigger that fires on
/// any growth at all is a trigger that runs continuously on a busy tenant,
/// which is how a background job starves the request path it shares a pool
/// with.
#[tokio::test]
async fn the_growth_trigger_fires_at_the_configured_percentage_and_not_before() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "growth").await;
    let policy = policy(50.0, Duration::hours(4));

    seed_unrelated_nodes(store, &tenant, 10, "growth").await;

    // A tenant nobody has consolidated before has no baseline to have grown
    // from, so the first sight of it runs and records one.
    let first = store
        .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(&first, ConsolidationAttempt::Ran(report) if report.triggered_by == Trigger::FirstRun),
        "the first sight of a tenant should run and establish a baseline, got {first:?}"
    );

    // Four more nodes against a baseline of ten is 40%, under the configured
    // 50%, and two minutes is nowhere near the four-hour bound.
    seed_unrelated_nodes(store, &tenant, 4, "growth-second-wave").await;
    let under = store
        .consolidate_tenant(
            &tenant,
            &policy,
            &LexicalVerifier,
            epoch() + Duration::minutes(2),
        )
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(under, ConsolidationAttempt::NotDue),
        "40% growth must not fire a 50% trigger, got {under:?}"
    );

    // The fifteenth node is exactly 50%.
    seed_unrelated_nodes(store, &tenant, 1, "growth-third-wave").await;
    let at = store
        .consolidate_tenant(
            &tenant,
            &policy,
            &LexicalVerifier,
            epoch() + Duration::minutes(4),
        )
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(&at, ConsolidationAttempt::Ran(report) if report.triggered_by == Trigger::Growth),
        "exactly the configured percentage is the configured percentage, got {at:?}"
    );

    // And the run it just did reset the baseline, so the same graph does not
    // trigger again.
    let again = store
        .consolidate_tenant(
            &tenant,
            &policy,
            &LexicalVerifier,
            epoch() + Duration::minutes(6),
        )
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(again, ConsolidationAttempt::NotDue),
        "a completed run resets the baseline it is measured against, got {again:?}"
    );
}

/// The other bound, and the one that makes the schedule a guarantee rather
/// than a hope. A tenant that writes nothing at all still has duplicates from
/// whatever it wrote before, and a purely growth-driven schedule would never
/// look at it again.
///
/// No sleeping. The interval is four hours and the clock is handed to the call,
/// so the whole bound is exercised in milliseconds.
#[tokio::test]
async fn the_interval_trigger_fires_after_the_configured_interval_with_no_growth() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "interval").await;
    let policy = policy(50.0, Duration::hours(4));

    seed_unrelated_nodes(store, &tenant, 3, "interval").await;

    let first = store
        .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(first, ConsolidationAttempt::Ran(_)),
        "the first sight of a tenant should run, got {first:?}"
    );

    // One minute short of the bound, with not a single node written since.
    let just_short = store
        .consolidate_tenant(
            &tenant,
            &policy,
            &LexicalVerifier,
            epoch() + Duration::hours(4) - Duration::minutes(1),
        )
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(just_short, ConsolidationAttempt::NotDue),
        "three hours fifty-nine is not four hours, got {just_short:?}"
    );

    let elapsed = store
        .consolidate_tenant(
            &tenant,
            &policy,
            &LexicalVerifier,
            epoch() + Duration::hours(4),
        )
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(&elapsed, ConsolidationAttempt::Ran(report)
            if report.triggered_by == Trigger::Interval),
        "exactly the configured interval is the configured interval, and zero growth is \
         no excuse, got {elapsed:?}"
    );

    // The run moved `last_run_at`, so the clock starts again from it rather
    // than from the first run.
    let soon_after = store
        .consolidate_tenant(
            &tenant,
            &policy,
            &LexicalVerifier,
            epoch() + Duration::hours(4) + Duration::minutes(1),
        )
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(soon_after, ConsolidationAttempt::NotDue),
        "the interval is measured from the last run, not from the first, got {soon_after:?}"
    );
}

/// One run, as the database holds it.
#[derive(Debug)]
struct RunRow {
    run_id: uuid::Uuid,
    triggered_by: String,
    verifier: String,
    started_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
    node_count: i64,
    edge_count: i64,
    pairs_considered: i32,
    pairs_merged: i32,
}

/// Every run this tenant has had, oldest first — the operator's query, which is
/// the point of the table.
async fn runs_of(store: &Store, tenant: &str) -> Vec<RunRow> {
    sqlx::query_as(
        "SELECT run_id, triggered_by::text, verifier, started_at, finished_at, \
                node_count, edge_count, pairs_considered, pairs_merged \
         FROM consolidation_run WHERE tenant_id = $1 ORDER BY started_at",
    )
    .bind(tenant)
    .fetch_all(store.pool())
    .await
    .expect("reading the run log should succeed")
    .into_iter()
    .map(
        |(
            run_id,
            triggered_by,
            verifier,
            started_at,
            finished_at,
            node_count,
            edge_count,
            pairs_considered,
            pairs_merged,
        )| RunRow {
            run_id,
            triggered_by,
            verifier,
            started_at,
            finished_at,
            node_count,
            edge_count,
            pairs_considered,
            pairs_merged,
        },
    )
    .collect()
}

/// A worker that merges entities and leaves only a log line behind is a worker
/// whose behaviour an operator can reconstruct only from whatever log retention
/// happens to be. "What did the worker do last night" has to be answerable from
/// the database.
#[tokio::test]
async fn a_run_records_itself_and_the_baseline_it_leaves_behind() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "run-record").await;
    let policy = policy(50.0, Duration::hours(4));

    seed_unrelated_nodes(store, &tenant, 4, "record").await;

    let attempt = store
        .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a consolidation attempt should not fail");
    let ConsolidationAttempt::Ran(report) = attempt else {
        panic!("the first sight of a tenant should run, got {attempt:?}");
    };

    let runs = runs_of(store, &tenant).await;
    assert_eq!(runs.len(), 1, "one run should have written one row");

    let run = &runs[0];
    assert_eq!(
        run.run_id, report.run_id,
        "the handle the caller got back must be the row in the table, or the two \
         records of one run cannot be joined"
    );
    assert_eq!(run.triggered_by, "first_run");
    assert_eq!(
        run.verifier, "lexical",
        "which verifier judged a run has to survive the run; a decision is only \
         attributable if the thing that made it is named"
    );
    assert_eq!(run.started_at, epoch(), "the clock the run was given");
    assert!(run.finished_at >= run.started_at);
    assert_eq!(
        (run.node_count, run.edge_count),
        (4, 0),
        "the counts this run measured, so the next run's growth arithmetic is auditable"
    );
    assert_eq!((run.pairs_considered, run.pairs_merged), (0, 0));

    // And the baseline the next run is measured against.
    let (last_run_at, nodes, edges): (DateTime<Utc>, i64, i64) = sqlx::query_as(
        "SELECT last_run_at, node_count, edge_count FROM consolidation_state \
         WHERE tenant_id = $1",
    )
    .bind(&tenant)
    .fetch_one(store.pool())
    .await
    .expect("a completed run should leave a baseline behind");

    assert_eq!(last_run_at, epoch());
    assert_eq!((nodes, edges), (4, 0));

    // A second run appends rather than replaces: the log is history.
    store
        .consolidate_tenant(
            &tenant,
            &policy,
            &LexicalVerifier,
            epoch() + Duration::hours(4),
        )
        .await
        .expect("a consolidation attempt should not fail");

    let runs = runs_of(store, &tenant).await;
    assert_eq!(runs.len(), 2, "the run log is append-only");
    assert_eq!(runs[1].triggered_by, "interval");
}

/// A vector at an exact cosine from `base`, in the plane `base` and `other`
/// span. The same construction the node suite uses: the deterministic embedder
/// maps different text to near-orthogonal vectors, so a controlled similarity
/// is the only way to put a real number in front of the resolver.
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

async fn embed(text: &str) -> Vec<f32> {
    embedder()
        .embed_documents(&[text.to_string()])
        .await
        .expect("the deterministic embedder cannot fail")
        .remove(0)
}

/// Writes a node whose `name_vector` is not the embedding of its own display
/// name, so a pair can be put in front of the worker at a chosen cosine.
async fn seed_node(store: &Store, tenant: &str, display_name: &str, vector: &[f32]) -> String {
    let node_id = display_name
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");

    sqlx::query(
        "INSERT INTO node (tenant_id, node_id, display_name, name_vector, embedding_model) \
         VALUES ($1, $2, $3, $4::real[]::vector, $5)",
    )
    .bind(tenant)
    .bind(&node_id)
    .bind(display_name)
    .bind(vector)
    .bind(embedder().model_id())
    .execute(store.pool())
    .await
    .expect("seeding a node should succeed");

    node_id
}

/// Seeds two nodes whose vectors sit at exactly `cosine`, the first written
/// first so its `created_at` is genuinely older.
async fn seed_pair(
    store: &Store,
    tenant: &str,
    first: &str,
    second: &str,
    cosine: f64,
) -> (String, String) {
    let anchor = embed(first).await;
    let older = seed_node(store, tenant, first, &anchor).await;
    let newer = seed_node(
        store,
        tenant,
        second,
        &at_cosine(&anchor, &embed(second).await, cosine),
    )
    .await;

    (older, newer)
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

/// The point of the whole exercise. Two nodes for one entity — the shape two
/// agents writing at the same moment leave behind, each reading a graph without
/// the other's uncommitted row — are fused by a run, through the audited,
/// reversible `merge_nodes` and not by writing `merged_into` directly.
///
/// 0.95 is comfortably over the 0.92 both these long names are held to, and
/// `Quarterly Business Review` against `Quarterly Business Reviews` is a pair
/// the lexical guard is silent about: one name contains the other, which is
/// elaboration rather than substitution.
#[tokio::test]
async fn two_genuine_duplicates_are_merged_by_a_run() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "duplicates").await;
    let policy = policy(50.0, Duration::hours(4));

    let (older, newer) = seed_pair(
        store,
        &tenant,
        "Quarterly Business Review",
        "Quarterly Business Reviews",
        0.95,
    )
    .await;

    let attempt = store
        .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a consolidation attempt should not fail");
    let ConsolidationAttempt::Ran(report) = attempt else {
        panic!("the first sight of a tenant should run, got {attempt:?}");
    };

    assert_eq!(
        report.merged_count(),
        1,
        "one duplicate pair should have produced one merge, got {:?}",
        report.decisions
    );

    assert_eq!(
        merged_into(store, &tenant, &newer).await.as_deref(),
        Some(older.as_str()),
        "the newer node is the one retired: the older id has been visible to agents \
         for longer and is the one they are holding"
    );
    assert_eq!(
        merged_into(store, &tenant, &older).await,
        None,
        "the winner must stay live"
    );

    // Through `merge_nodes`, which means the reversible audit exists. The
    // worker's entire safety argument is that its merges can be undone.
    let (audited,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM node_merge \
         WHERE tenant_id = $1 AND loser_node_id = $2 AND winner_node_id = $3",
    )
    .bind(&tenant)
    .bind(&newer)
    .bind(&older)
    .fetch_one(store.pool())
    .await
    .expect("counting should succeed");
    assert_eq!(
        audited, 1,
        "an automatic merge with no audit row is exactly the irreversible operation \
         this worker is only defensible for not being"
    );

    let (client, principal): (Option<String>, Option<String>) = sqlx::query_as(
        "SELECT merged_by_client, merged_by_principal FROM node_merge WHERE tenant_id = $1",
    )
    .bind(&tenant)
    .fetch_one(store.pool())
    .await
    .expect("the audit row should be readable");
    assert_eq!(
        client.as_deref(),
        Some("mundu-consolidation-worker"),
        "a merge nobody asked for has to be attributable to the thing that made it"
    );
    assert_eq!(
        principal, None,
        "there is no principal behind an automatic merge, and inventing one would \
         attribute it to a person"
    );
}

/// One considered pair, as the database holds it.
#[derive(Debug)]
struct DecisionRow {
    node_a_id: String,
    node_b_id: String,
    similarity: f64,
    bar: f64,
    verdict: String,
    reason: String,
    merge_id: Option<uuid::Uuid>,
}

/// Every pair this tenant's runs considered, most similar first.
async fn decisions_of(store: &Store, tenant: &str) -> Vec<DecisionRow> {
    sqlx::query_as(
        "SELECT node_a_id, node_b_id, similarity, bar, verdict::text, reason, merge_id \
         FROM consolidation_decision WHERE tenant_id = $1 ORDER BY similarity DESC, node_a_id",
    )
    .bind(tenant)
    .fetch_all(store.pool())
    .await
    .expect("reading the decision log should succeed")
    .into_iter()
    .map(
        |(node_a_id, node_b_id, similarity, bar, verdict, reason, merge_id)| DecisionRow {
            node_a_id,
            node_b_id,
            similarity,
            bar,
            verdict,
            reason,
            merge_id,
        },
    )
    .collect()
}

/// The guard that makes the whole thing defensible. Measured on the real model,
/// `Acme Corporation` against `Acme Holdings` — two companies — sits at 0.9451
/// while `billing-service-v2` against `the v2 billing service` — one service —
/// sits at 0.9295. The populations interleave, so no cosine separates them, and
/// a worker deciding on the cosine alone would fuse two companies on a timer.
///
/// 0.99 here, deliberately absurd: the veto is not a tie-break the vector can
/// outvote. Both names carry a word the other lacks, which is a substitution,
/// and a substitution is how sibling entities are named.
#[tokio::test]
async fn a_pair_the_lexical_guard_vetoes_is_never_merged_however_close_its_vectors() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "veto").await;
    let policy = policy(50.0, Duration::hours(4));

    let (first, second) =
        seed_pair(store, &tenant, "Acme Corporation", "Acme Holdings", 0.99).await;

    let attempt = store
        .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a consolidation attempt should not fail");
    let ConsolidationAttempt::Ran(report) = attempt else {
        panic!("the first sight of a tenant should run, got {attempt:?}");
    };

    assert_eq!(
        report.merged_count(),
        0,
        "two companies must not be fused because their names embed alike, got {:?}",
        report.decisions
    );
    assert_eq!(merged_into(store, &tenant, &first).await, None);
    assert_eq!(merged_into(store, &tenant, &second).await, None);

    // And the refusal is on the record, with its reason. An operator has to be
    // able to find the merges the worker is deliberately not making.
    let decisions = decisions_of(store, &tenant).await;
    assert_eq!(
        decisions.len(),
        1,
        "the pair was considered, so it belongs in the log, got {decisions:?}"
    );

    let decision = &decisions[0];
    assert_eq!(
        (decision.node_a_id.as_str(), decision.node_b_id.as_str()),
        ("acme-corporation", "acme-holdings"),
        "one pair, one row, ordered by id so the scan's direction cannot double it"
    );
    assert_eq!(decision.verdict, "declined");
    assert_eq!(decision.reason, "names_contradict");
    assert_eq!(
        decision.merge_id, None,
        "a declined pair has no merge to point at"
    );
    assert!(
        (decision.similarity - 0.99).abs() < 1e-4,
        "the cosine the run measured should be recorded as measured, got {}",
        decision.similarity
    );
    assert!(
        (decision.bar - 0.92).abs() < 1e-9,
        "the bar it had to clear should be recorded beside it, got {}",
        decision.bar
    );
}

/// The inline resolver attaches to a confident neighbour only when there is
/// exactly one of them: two nodes answering to one name is ambiguity however
/// high the scores are. The worker holds itself to that rule from **both**
/// sides, which is strictly stronger — the write path only ever asks the
/// question from the incoming name's side, because that is the only side it
/// has.
///
/// Three nodes, one of which is confidently close to two others. Nothing may
/// move: which entity that node is remains an open question, and a merge would
/// be answering it by picking the larger cosine.
#[tokio::test]
async fn a_pair_whose_end_is_confident_about_somebody_else_too_is_declined() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "contested").await;
    let policy = policy(50.0, Duration::hours(4));

    let hub = "Northwind Trading Company";
    let anchor = embed(hub).await;
    let hub_id = seed_node(store, &tenant, hub, &anchor).await;

    let near = "Northwind Trading";
    let near_id = seed_node(
        store,
        &tenant,
        near,
        &at_cosine(&anchor, &embed(near).await, 0.96),
    )
    .await;

    let also = "The Northwind Trading Company";
    let also_id = seed_node(
        store,
        &tenant,
        also,
        &at_cosine(&anchor, &embed(also).await, 0.95),
    )
    .await;

    let attempt = store
        .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a consolidation attempt should not fail");
    let ConsolidationAttempt::Ran(report) = attempt else {
        panic!("the first sight of a tenant should run, got {attempt:?}");
    };

    assert_eq!(
        report.merged_count(),
        0,
        "a node equally confident about two others is a question, not an answer, got {:?}",
        report.decisions
    );
    for node_id in [&hub_id, &near_id, &also_id] {
        assert_eq!(
            merged_into(store, &tenant, node_id).await,
            None,
            "{node_id} should still be live"
        );
    }

    let decisions = decisions_of(store, &tenant).await;
    let reason_for = |left: &str, right: &str| {
        let (a, b) = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        decisions
            .iter()
            .find(|row| row.node_a_id == a && row.node_b_id == b)
            .unwrap_or_else(|| panic!("({a}, {b}) should have been considered, got {decisions:?}"))
            .reason
            .clone()
    };

    assert_eq!(
        reason_for(&hub_id, &near_id),
        "contested",
        "the closer of the two is still contested, because its other end is not exclusive"
    );
    assert_eq!(reason_for(&hub_id, &also_id), "contested");
}

/// Every query a run makes is scoped to one tenant, and this is the test that
/// says so from the outside.
///
/// The other tenant's node is placed at 0.97 from this tenant's anchor —
/// *closer* than the genuine duplicate at 0.95. A scan that reached across
/// would not merge the wrong pair, because `merge_nodes` would refuse an id it
/// cannot see; it would do something quieter and worse. The anchor would have
/// two confident neighbours instead of one, the contest gate would fire, and
/// the tenant's real duplicate would silently never be consolidated — a
/// customer's graph degraded by another customer's data, with nothing in any
/// log to say why.
#[tokio::test]
async fn a_run_is_scoped_to_one_tenant_and_never_considers_anothers_nodes() {
    let store = &store().await;
    let mine = fresh_tenant(store, "scope-mine").await;
    let theirs = fresh_tenant(store, "scope-theirs").await;
    let policy = policy(50.0, Duration::hours(4));

    let anchor = embed("Helios Data Platform").await;
    let winner = seed_node(store, &mine, "Helios Data Platform", &anchor).await;
    let loser = seed_node(
        store,
        &mine,
        "Helios Data Platforms",
        &at_cosine(&anchor, &embed("Helios Data Platforms").await, 0.95),
    )
    .await;

    let neighbour = "Helios Data Warehouse";
    let intruder = seed_node(
        store,
        &theirs,
        neighbour,
        &at_cosine(&anchor, &embed(neighbour).await, 0.97),
    )
    .await;

    let attempt = store
        .consolidate_tenant(&mine, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a consolidation attempt should not fail");
    let ConsolidationAttempt::Ran(report) = attempt else {
        panic!("the first sight of a tenant should run, got {attempt:?}");
    };

    assert_eq!(
        report.nodes_scanned, 2,
        "the scan should have seen this tenant's two nodes and nobody else's"
    );
    assert_eq!(
        report.merged_count(),
        1,
        "the tenant's own duplicate should merge; a leak would make its anchor \
         contested instead, got {:?}",
        report.decisions
    );
    assert_eq!(
        merged_into(store, &mine, &loser).await.as_deref(),
        Some(winner.as_str())
    );

    for decision in decisions_of(store, &mine).await {
        assert!(
            decision.node_a_id != intruder && decision.node_b_id != intruder,
            "another tenant's node reached this tenant's decision log: {decision:?}"
        );
    }

    // Nothing of theirs moved, and nothing of theirs was scheduled.
    assert_eq!(
        merged_into(store, &theirs, &intruder).await,
        None,
        "another tenant's node must be untouched"
    );
    assert!(
        runs_of(store, &theirs).await.is_empty(),
        "consolidating one tenant must not run another"
    );
    let (their_state,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM consolidation_state WHERE tenant_id = $1")
            .bind(&theirs)
            .fetch_one(store.pool())
            .await
            .expect("counting should succeed");
    assert_eq!(
        their_state, 0,
        "one tenant's run must not reset another tenant's growth baseline"
    );
}

/// Every instance runs this worker, so N instances is N concurrent
/// consolidations over the same rows unless something prevents it. Two stores,
/// two pools, two sets of connections — as close to two processes as one test
/// can get — asked to consolidate one tenant at the same instant.
///
/// Exactly one run may happen. The loser has to skip: not queue behind the
/// winner and redo the work it just did, and not fail, because a background
/// job that errors on the ordinary case teaches an operator to ignore it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_workers_racing_over_one_tenant_produce_exactly_one_run() {
    let one = store().await;
    let two = store().await;
    let tenant = fresh_tenant(&one, "race").await;
    let policy = policy(50.0, Duration::hours(4));

    // Enough nodes that a run takes long enough to overlap, and one genuine
    // duplicate pair so the two runs would have something to fight over.
    seed_unrelated_nodes(&one, &tenant, 12, "race").await;
    let (winner, loser) = seed_pair(
        &one,
        &tenant,
        "Meridian Logistics Group",
        "Meridian Logistics Groups",
        0.96,
    )
    .await;

    // Separate tasks on a multi-threaded runtime, released together. Joining
    // two futures on one task is not a race: it interleaves them at await
    // points, and the first one through can finish and write its baseline
    // before the second is ever polled — which looks exactly like the lock
    // working.
    let gate = std::sync::Arc::new(tokio::sync::Barrier::new(2));
    let workers: Vec<_> = [one.clone(), two.clone()]
        .into_iter()
        .map(|store| {
            let tenant = tenant.clone();
            let policy = policy.clone();
            let gate = gate.clone();
            tokio::spawn(async move {
                gate.wait().await;
                store
                    .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
                    .await
            })
        })
        .collect();

    let mut attempts = Vec::new();
    for worker in workers {
        attempts.push(worker.await.expect("neither worker should panic"));
    }
    let mut attempts = attempts.into_iter();
    let first = attempts
        .next()
        .expect("two workers")
        .expect("the winner of the race must not fail");
    let second = attempts
        .next()
        .expect("two workers")
        .expect("the loser of the race must skip cleanly, not error");

    let ran = [&first, &second]
        .iter()
        .filter(|attempt| matches!(attempt, ConsolidationAttempt::Ran(_)))
        .count();
    assert_eq!(
        ran, 1,
        "exactly one of two racing workers may consolidate, got {first:?} and {second:?}"
    );

    for attempt in [&first, &second] {
        assert!(
            matches!(
                attempt,
                ConsolidationAttempt::Ran(_)
                    | ConsolidationAttempt::Skipped
                    | ConsolidationAttempt::NotDue
            ),
            "the loser must skip or find nothing to do, got {attempt:?}"
        );
    }

    let runs = runs_of(&one, &tenant).await;
    assert_eq!(
        runs.len(),
        1,
        "two racing workers wrote {} runs; the database is the record and it has to \
         say what actually happened",
        runs.len()
    );

    // And the one run still did its job.
    assert_eq!(
        merged_into(&one, &tenant, &loser).await.as_deref(),
        Some(winner.as_str()),
        "the duplicate should still have been consolidated by whichever worker won"
    );
    assert_eq!(runs[0].pairs_merged, 1);
}

/// The deterministic half of the same guarantee, with the race taken out of it.
///
/// The lock is held by somebody else — an instance mid-run, or one that has
/// died and not yet been noticed — and the worker has to come straight back
/// having done nothing. Not block: a run can take as long as it takes, and a
/// worker queued behind one would wake up holding a stale reason to run and do
/// the work again. Not error: this is the ordinary case on every instance but
/// one.
#[tokio::test]
async fn a_worker_that_loses_the_lock_skips_immediately_and_writes_nothing() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "lock-held").await;
    let policy = policy(50.0, Duration::hours(4));

    seed_pair(
        store,
        &tenant,
        "Calder Freight Systems",
        "Calder Freight System",
        0.96,
    )
    .await;

    // Somebody else's session, holding this tenant's lock.
    let (namespace, key) = advisory_lock_key(&tenant);
    let mut holder = store
        .pool()
        .acquire()
        .await
        .expect("a connection should be available");
    let (held,): (bool,) = sqlx::query_as("SELECT pg_try_advisory_lock($1, $2)")
        .bind(namespace)
        .bind(key)
        .fetch_one(&mut *holder)
        .await
        .expect("taking the lock should succeed");
    assert!(
        held,
        "the test should have the lock before the worker tries"
    );

    let started = std::time::Instant::now();
    let attempt = store
        .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a worker that loses the lock must not error");
    let waited = started.elapsed();

    assert!(
        matches!(attempt, ConsolidationAttempt::Skipped),
        "a tenant somebody else is consolidating must be skipped, got {attempt:?}"
    );
    assert!(
        waited < std::time::Duration::from_secs(5),
        "the loser waited {waited:?}; a losing instance must not queue behind the winner"
    );

    assert!(
        runs_of(store, &tenant).await.is_empty(),
        "a skipped attempt is not a run and must not be recorded as one"
    );
    let (state_rows,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM consolidation_state WHERE tenant_id = $1")
            .bind(&tenant)
            .fetch_one(store.pool())
            .await
            .expect("counting should succeed");
    assert_eq!(
        state_rows, 0,
        "a skipped attempt must not move the baseline; the tenant is still due"
    );
    assert_eq!(
        merged_into(store, &tenant, "calder-freight-system").await,
        None,
        "a skipped attempt must merge nothing"
    );

    // Released — by this session, and, had the holder been a dead instance, by
    // its socket closing. The tenant is still due, so the next attempt runs.
    sqlx::query("SELECT pg_advisory_unlock($1, $2)")
        .bind(namespace)
        .bind(key)
        .execute(&mut *holder)
        .await
        .expect("releasing the lock should succeed");
    drop(holder);

    let attempt = store
        .consolidate_tenant(&tenant, &policy, &LexicalVerifier, epoch())
        .await
        .expect("a consolidation attempt should not fail");
    assert!(
        matches!(&attempt, ConsolidationAttempt::Ran(report) if report.merged_count() == 1),
        "once the lock is free the tenant is still due and still has its duplicate, \
         got {attempt:?}"
    );
}

/// The worker's tick asks the database which tenants exist, because there is no
/// tenant table: a tenant is whatever a validated credential claims, and the
/// first node it writes is what makes it exist. The listing is a loose index
/// scan over the primary key rather than a `SELECT DISTINCT`, so this pins that
/// it still returns every tenant.
///
/// **Deliberately not a test of `tick_once`.** A tick consolidates every tenant
/// in the database, and this suite shares one database with every other suite —
/// running one here would merge other tests' nodes underneath them.
#[tokio::test]
async fn the_worker_can_enumerate_every_tenant_that_has_written_a_node() {
    let store = &store().await;
    let mine = fresh_tenant(store, "listing-mine").await;
    let theirs = fresh_tenant(store, "listing-theirs").await;

    seed_unrelated_nodes(store, &mine, 2, "listing").await;
    seed_unrelated_nodes(store, &theirs, 1, "listing").await;

    let tenants = store
        .tenants_with_nodes()
        .await
        .expect("listing tenants should succeed");

    assert!(tenants.contains(&mine), "got {tenants:?}");
    assert!(tenants.contains(&theirs), "got {tenants:?}");

    let mut sorted = tenants.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        tenants.len(),
        "a tenant listed twice would be consolidated twice in one tick"
    );
    assert!(
        !tenants.iter().any(String::is_empty),
        "the empty tenant is the curated vocabulary, which owns no nodes and must \
         never be scheduled"
    );
}
