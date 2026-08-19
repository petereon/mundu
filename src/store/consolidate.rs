//! The consolidation worker: the thing that calls [`Store::merge_nodes`]
//! without being asked.
//!
//! # Why this is delicate
//!
//! Entity resolution refuses to fuse two entities in three separate places —
//! the ambiguous band mints a node and reports its near misses, the lexical
//! guard demotes a confident vector match whose names contradict, and the
//! short-name bar sits above the whole band it cannot separate. All three were
//! justified by one sentence: fragmentation is recoverable by a later merge,
//! over-merge is not. A worker that merged, on a timer, what the inline
//! resolver had just declined to merge — on the same evidence — would undo
//! those three decisions at scale and without anybody deciding to.
//!
//! Two things make this defensible, and neither is optional:
//!
//! 1. **Merges are reversible and audited.** `node_merge` and
//!    `node_merge_edge` record enough to reconstruct an unmerge. The worker's
//!    entire safety argument rests on that audit staying trustworthy, so this
//!    module goes through `merge_nodes` and never writes `merged_into` itself.
//! 2. **The automatic bar is stricter than the inline one, never looser.** See
//!    [`bar_for`] and [`CandidatePair::gate`].
//!
//! # What that means the worker can actually catch
//!
//! Being held to the inline resolver's own standard, and then held above it,
//! the worker cannot merge anything the inline path would refuse today. So it
//! only ever catches pairs that were **created when neither could see the
//! other**, or that the graph has since made unambiguous:
//!
//! * two agents writing near-identical names at the same moment, each reading a
//!   graph without the other's uncommitted row — `ON CONFLICT DO NOTHING`
//!   catches an exact id collision, but two names a cosine apart genuinely
//!   race;
//! * a name that was ambiguous when it arrived because two nodes were confident
//!   neighbours of it, one of which has since been merged away, leaving exactly
//!   one;
//! * nodes written before this server's thresholds, or its lexical guard, were
//!   what they are now.
//!
//! That is a narrow catch, and it is the honest one. A worker with a wider
//! catch would be a worker overruling the resolver.
//!
//! # The answer nobody gave it
//!
//! One case is worth spelling out, because it is the one where an automatic
//! merge would be overruling a *person*: a node minted because a client was
//! asked which entity a contested name meant and answered "create a new one".
//!
//! Nothing records that answer, so the worker cannot read it — but it does not
//! need to. A client is only ever asked when the ladder was uncertain, and the
//! ladder is uncertain in exactly three shapes: two or more confident
//! neighbours, one confident neighbour whose name contradicts, or nothing
//! confident at all. The contest gate refuses the first, the lexical veto
//! refuses the second, and the bar refuses the third. So a node created by a
//! refusal can only be merged later if the graph itself has changed into one
//! the inline resolver would resolve without asking — which is the same
//! standard as everything else here, and not the worker second-guessing an
//! answer it was not given.
//!
//! # What it deliberately does not do
//!
//! There is no MCP tool in front of this. Consolidation is an operator and
//! server concern, for the reason `rating.rs` gives about archival and 0003
//! gives about merging: everything in this graph is text some other agent
//! wrote, and a semi-destructive operation an agent can trigger is an erasure
//! primitive handed to prompt injection.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use sha2::{Digest, Sha256};
use sqlx::{Connection, PgConnection, PgPool};
use uuid::Uuid;

use super::node::{
    AMBIGUOUS_SIMILARITY, CANDIDATE_LIMIT, CONFIDENT_SIMILARITY, CONFIDENT_SIMILARITY_SHORT,
    SHORT_NAME_CHARS, names_contradict,
};
use super::{Provenance, Store};
use crate::error::StoreError;

/// The `writer_client` an automatic merge is recorded under.
///
/// `merged_by_principal` is left NULL beside it, deliberately. There is no
/// principal behind a merge nobody asked for, and putting one there would
/// attribute an automatic decision to a person who did not make it.
pub const CONSOLIDATION_CLIENT: &str = "mundu-consolidation-worker";

/// The `classid` half of every advisory lock this module takes: the ASCII bytes
/// of `mund`.
///
/// Advisory locks share one namespace across the whole database, and this
/// server is not guaranteed to be the only thing in it. A fixed, recognisable
/// class means a lock this worker holds is identifiable in `pg_locks` rather
/// than being an unexplained integer, and that a collision with somebody else's
/// scheme takes deliberate effort.
pub const CONSOLIDATION_LOCK_CLASS: i32 = 0x6d75_6e64;

/// The advisory-lock key this tenant is consolidated under.
///
/// **Per tenant, not global.** A global lock would serialise every tenant's
/// consolidation behind every other tenant's, which on a shared deployment is
/// the difference between "four hours" and "four hours times the number of
/// customers". Two tenants whose digests collide in 32 bits will take turns
/// instead of running together; that costs one of them a skipped tick and no
/// correctness, which is why 32 bits is enough.
///
/// Derived from a digest rather than from a counter or a table, because it has
/// to be the same number in every instance of a stateless server with nothing
/// shared but the database.
pub fn advisory_lock_key(tenant: &str) -> (i32, i32) {
    let digest = Sha256::digest(tenant.as_bytes());
    let head: [u8; 4] = digest[..4]
        .try_into()
        .expect("a SHA-256 digest is longer than four bytes");

    (CONSOLIDATION_LOCK_CLASS, u32::from_be_bytes(head) as i32)
}

/// The most live nodes one run will pull a neighbour list for.
///
/// A bound on the work, not a judgement about tenant size. Each scanned node
/// costs one HNSW probe under `hnsw.iterative_scan`, and the worker shares its
/// pool with the request path — an unbounded scan on a large tenant is how a
/// background job turns into an outage.
///
/// Ordered newest first, which is also what the original specification asked
/// for: a run looks for duplicates among *new* nodes. An old node is not
/// skipped by that ordering, only reached from the other side — it appears in a
/// new node's neighbour list, and [`Store::consolidate_tenant`] completes its
/// list with a second pass precisely so a new duplicate of an old entity is
/// judged on the same evidence as any other pair.
const MAX_SCANNED_NODES: usize = 1_000;

/// The most candidate pairs one run will consider, most similar first.
///
/// Whatever a run leaves behind is not lost: the next run is at most
/// [`ConsolidationPolicy::max_interval`] away, and a tenant producing more than
/// this many near-duplicate pairs between two runs has a bigger problem than
/// the tail of one scan.
const MAX_PAIRS_PER_RUN: usize = 200;

// ---------------------------------------------------------------------------
// Triggering
// ---------------------------------------------------------------------------

/// The two bounds a run has to sit inside.
///
/// Both hold at once and whichever fires first wins, which is what the
/// deployment asked for: consolidate when the graph has grown enough to be
/// worth consolidating, and in any case not less often than every four hours.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsolidationPolicy {
    /// How much a tenant's node or edge count has to have risen, in percent of
    /// what it was at the last completed run, before growth alone fires a run.
    ///
    /// Node **or** edge, not both: a conversation that adds fifty facts about
    /// nodes that already existed has changed nothing about which entities are
    /// duplicated, but a conversation that adds fifty facts has usually also
    /// named some entities, and requiring both would let a tenant grow one
    /// dimension without bound.
    pub growth_percent: f64,
    /// The longest a tenant may go unconsolidated whatever its growth.
    pub max_interval: Duration,
}

/// Ten percent and four hours.
///
/// Four hours is what the deployment specified. Ten percent is chosen against
/// it rather than independently: on a tenant quiet enough that growth never
/// fires, the interval bound is the whole schedule, and on a tenant busy enough
/// to grow a tenth in under four hours, a run is cheap relative to what
/// produced the growth.
impl Default for ConsolidationPolicy {
    fn default() -> Self {
        Self {
            growth_percent: 10.0,
            max_interval: Duration::hours(4),
        }
    }
}

/// Which bound fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// No baseline exists, so there is nothing to have grown from and nothing
    /// to have elapsed since. Recorded distinctly rather than as growth,
    /// because claiming a measurement that was never taken is how a log stops
    /// being evidence.
    FirstRun,
    /// The node or edge count rose by at least [`ConsolidationPolicy::growth_percent`].
    Growth,
    /// [`ConsolidationPolicy::max_interval`] has passed since the last run.
    Interval,
}

impl Trigger {
    /// The label the `consolidation_trigger` enum carries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FirstRun => "first_run",
            Self::Growth => "growth",
            Self::Interval => "interval",
        }
    }
}

/// What a tenant measured, as the trigger arithmetic reads it.
///
/// Every row, tombstones and superseded edges included. Nothing in this schema
/// is ever deleted, so counted this way both numbers are monotonically
/// non-decreasing: growth can never come out negative, and a run — which
/// retires nodes without removing their rows — cannot reset the baseline it is
/// about to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantCounts {
    pub nodes: i64,
    pub edges: i64,
}

/// What the last completed run left behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TenantState {
    last_run_at: DateTime<Utc>,
    counts: TenantCounts,
}

/// Whether this tenant is due, and why.
///
/// Pure, and separated from everything that touches the database, because it is
/// the one part of the schedule that can be reasoned about exhaustively: three
/// inputs, four outcomes, no I/O.
fn trigger_for(
    state: Option<TenantState>,
    counts: TenantCounts,
    policy: &ConsolidationPolicy,
    now: DateTime<Utc>,
) -> Option<Trigger> {
    let Some(state) = state else {
        return Some(Trigger::FirstRun);
    };

    if grown(state.counts.nodes, counts.nodes, policy.growth_percent)
        || grown(state.counts.edges, counts.edges, policy.growth_percent)
    {
        return Some(Trigger::Growth);
    }

    // `>=`, so a bound of exactly four hours is four hours and not four hours
    // plus one tick. Clock skew is not a factor: both ends of this comparison
    // come from the clock the run was given, never from `now()` inside
    // Postgres, which is a different machine — the merge suite has already had
    // to be corrected once for measuring a server-set instant against the host.
    if now.signed_duration_since(state.last_run_at) >= policy.max_interval {
        return Some(Trigger::Interval);
    }

    None
}

/// Whether `now` is at least `percent` above `before`.
///
/// `before == 0` is growth as soon as anything exists at all: there is no
/// percentage of nothing, and a tenant whose first node has just arrived has
/// unambiguously grown. `now == before` is not growth at any percentage,
/// including zero — otherwise a policy of `0.0` would fire on every tick
/// forever, on an empty tenant most of all.
fn grown(before: i64, now: i64, percent: f64) -> bool {
    if now <= before {
        return false;
    }

    now as f64 - before as f64 >= before as f64 * percent / 100.0
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// A candidate pair, as the scan found it, already oriented.
///
/// One pair is one [`CandidatePair`] however the scan reached it: the
/// neighbour relation is symmetric and the scan sees both directions, so pairs
/// are deduplicated on the unordered pair of ids.
///
/// **Which of the two would win is decided by the scan, not by the merge.** The
/// older node by `created_at` wins, ties broken by the lexicographically
/// smaller `node_id`. Two reasons, and neither is about the entities: the older
/// id has been visible to agents for longer, so it is the one more of them are
/// holding in a context this server cannot reach or correct; and the rule has
/// to be a *function of the pair* rather than of the order the scan happened to
/// walk it, or two runs over the same graph would disagree about which
/// tombstone to leave.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidatePair {
    /// The node a merge would keep.
    pub winner_node_id: String,
    pub winner_display_name: String,
    /// The node a merge would retire into the winner. Never deleted: it stays
    /// as a tombstone that still resolves.
    pub loser_node_id: String,
    pub loser_display_name: String,
    /// The measured cosine between the two `name_vector`s. Both were embedded
    /// by the same model — the scan refuses to compare across models, because a
    /// cosine between two embedding spaces is a number with no meaning that
    /// still sorts.
    pub similarity: f64,
    /// The bar this pair had to clear, from [`bar_for`].
    pub bar: f64,
}

impl CandidatePair {
    /// The two ids in the order the decision log records them, smaller first,
    /// so one pair is one row whichever of the two the scan reached first.
    pub fn ordered_ids(&self) -> (&str, &str) {
        if self.winner_node_id <= self.loser_node_id {
            (&self.winner_node_id, &self.loser_node_id)
        } else {
            (&self.loser_node_id, &self.winner_node_id)
        }
    }
}

/// Why a pair was not merged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclineReason {
    /// Inside the band worth reporting, below the band worth acting on. This is
    /// the ambiguous tier the inline resolver hands back as near misses, and
    /// the worker treats it identically: recorded, never merged.
    BelowConfidentBar,
    /// [`names_contradict`] vetoed it. Both names carry a word the other lacks,
    /// which is a substitution rather than an elaboration, and a substitution is
    /// how sibling entities are named.
    NamesContradict,
    /// One side or the other has a second neighbour it is equally confident
    /// about, so which entity this is remains an open question.
    Contested,
    /// The pair passed everything and [`Store::merge_nodes`] still refused it.
    /// Most often benign — another writer merged one of the two in between —
    /// and always worth writing down, because the alternative is a pair that
    /// silently never merges.
    MergeRefused(String),
}

impl DeclineReason {
    /// The stable prefix the `reason` column carries. An operator's query is
    /// `reason = 'names_contradict'`, or `reason LIKE 'merge_refused:%'` for
    /// the one variant that carries detail.
    pub fn code(&self) -> &'static str {
        match self {
            Self::BelowConfidentBar => "below_confident_bar",
            Self::NamesContradict => "names_contradict",
            Self::Contested => "contested",
            Self::MergeRefused(_) => "merge_refused",
        }
    }
}

impl std::fmt::Display for DeclineReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MergeRefused(detail) => write!(formatter, "{}: {detail}", self.code()),
            other => formatter.write_str(other.code()),
        }
    }
}

/// A verifier's answer about one pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Merge,
    Decline(DeclineReason),
}

/// The seam an LLM verifier would slot into.
///
/// **Veto-only, by construction.** A verifier is asked only about pairs that
/// have already cleared the vector bar, the same-model check and the contest
/// check in [`CandidatePair::gate`]. It is never asked about a pair those gates
/// refused, so returning [`Verdict::Merge`] cannot promote anything — the most
/// a verifier can ever do is decline. That is what makes the seam safe to
/// extend: a future verifier can be wrong in the direction of leaving
/// duplicates behind, which is recoverable, and cannot be wrong in the
/// direction of fusing two entities, which is what the bar exists to prevent.
///
/// The original specification called for a batch-inference LLM here. There is
/// none available to this process, and MCP sampling is both deprecated in
/// `2026-07-28` and unreachable from a background worker with no request
/// context to sample against. So the shipped verifier is [`LexicalVerifier`],
/// which uses the evidence the codebase already has.
#[async_trait]
pub trait MergeVerifier: Send + Sync {
    /// Recorded on every run, so a decision stays attributable to the thing
    /// that actually made it.
    fn name(&self) -> &str;

    /// Judges one gated pair.
    async fn verify(&self, pair: &CandidatePair) -> Result<Verdict, StoreError>;
}

/// The verifier this server ships: the lexical guard from the inline resolver,
/// and nothing else.
///
/// [`names_contradict`] is not a score and does not pretend to be one. It says
/// only that two names disagree about *which* entity they name — both carrying
/// a word the other lacks, or disagreeing about a version or an ordinal — and
/// it is silent where two names have nothing to disagree about. Applying it
/// here is applying exactly the check the write path applies at the same
/// decision, on exactly the same text: `display_name` is what the vector was
/// computed from, so this checks the claim the cosine made rather than a
/// second, later reading of it.
#[derive(Debug, Clone, Copy, Default)]
pub struct LexicalVerifier;

#[async_trait]
impl MergeVerifier for LexicalVerifier {
    fn name(&self) -> &str {
        "lexical"
    }

    async fn verify(&self, pair: &CandidatePair) -> Result<Verdict, StoreError> {
        if names_contradict(&pair.winner_display_name, &pair.loser_display_name) {
            return Ok(Verdict::Decline(DeclineReason::NamesContradict));
        }

        Ok(Verdict::Merge)
    }
}

// ---------------------------------------------------------------------------
// What a run reports
// ---------------------------------------------------------------------------

/// What happened to one candidate pair.
#[derive(Debug, Clone, PartialEq)]
pub struct PairDecision {
    pub pair: CandidatePair,
    pub outcome: PairOutcome,
}

/// Merged, or declined and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairOutcome {
    Merged {
        /// The handle into `node_merge`, which is where an unmerge starts.
        merge_id: Uuid,
        winner_node_id: String,
        loser_node_id: String,
    },
    Declined(DeclineReason),
}

/// One run, as it happened.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsolidationReport {
    pub run_id: Uuid,
    pub tenant: String,
    pub triggered_by: Trigger,
    pub counts: TenantCounts,
    /// How many live nodes the scan actually looked at.
    pub nodes_scanned: usize,
    /// Whether that was all of them. A run is bounded so it cannot starve the
    /// request path, so on a large tenant it sees the newest slice; this is how
    /// "found nothing" is told apart from "did not look everywhere".
    pub scan_truncated: bool,
    pub decisions: Vec<PairDecision>,
}

impl ConsolidationReport {
    /// Only the pairs this run actually merged.
    pub fn merges(&self) -> impl Iterator<Item = &PairDecision> {
        self.decisions
            .iter()
            .filter(|decision| matches!(decision.outcome, PairOutcome::Merged { .. }))
    }

    pub fn merged_count(&self) -> usize {
        self.merges().count()
    }
}

/// What a single attempt came to.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsolidationAttempt {
    /// Neither bound had been reached. The overwhelmingly common answer.
    NotDue,
    /// Another instance is consolidating this tenant right now. Skipped, not
    /// queued and not failed: see [`Store::consolidate_tenant`].
    Skipped,
    Ran(ConsolidationReport),
}

// ---------------------------------------------------------------------------
// The bar
// ---------------------------------------------------------------------------

/// The cosine a pair has to clear before anything else is asked about it.
///
/// **The stricter of the two names' bars, always.** The inline resolver picks a
/// threshold from the *incoming* name's length — [`CONFIDENT_SIMILARITY_SHORT`]
/// for a short name, [`CONFIDENT_SIMILARITY`] for a long one — because at that
/// moment there is an incoming name and an existing one. Here there is no
/// incoming name; there are two nodes that both already exist, so "the
/// incoming one" is whichever happened to be written second, which is a fact
/// about arrival order rather than about the entities.
///
/// Taking the maximum makes the worker's bar independent of that order, and
/// makes it *at least* the inline bar in both directions: a pair the inline
/// path would have merged at 0.925 because the long name arrived second is
/// refused here if the other name is short. That is the intended asymmetry.
/// The cost is a duplicate that survives; the alternative cost is a fusion.
///
/// Neither constant is redefined here. They are imported, so the two bars
/// cannot drift apart, and a change to the inline threshold moves this one with
/// it.
pub fn bar_for(left_display_name: &str, right_display_name: &str) -> f64 {
    own_bar(left_display_name).max(own_bar(right_display_name))
}

/// The bar the inline resolver would set from *this* name alone.
///
/// `Thresholds::for_name` in the same words, over the same two constants, and
/// it exists separately from [`bar_for`] because the two are used for opposite
/// purposes and must err in opposite directions.
///
/// [`bar_for`] decides whether a pair may be merged, so it takes the **maximum**
/// and refuses more. This decides whether some third node counts as a rival
/// claim on one end of that pair, so it takes each node's **own** bar and
/// admits more — and every extra rival admitted is a pair declined. Using the
/// pair rule for both would let a long name quietly stop noticing a short rival
/// at 0.925, which is the one direction that ends in a fusion.
fn own_bar(display_name: &str) -> f64 {
    if display_name.trim().chars().count() <= SHORT_NAME_CHARS {
        CONFIDENT_SIMILARITY_SHORT
    } else {
        CONFIDENT_SIMILARITY
    }
}

// ---------------------------------------------------------------------------
// A run
// ---------------------------------------------------------------------------

impl Store {
    /// Consolidates one tenant, if it is due and if no other instance is
    /// already doing it.
    ///
    /// `now` is passed in rather than read, so the schedule is testable without
    /// waiting out a four-hour bound and so every instant this writes comes
    /// from one clock. Production hands it `Utc::now()`.
    pub async fn consolidate_tenant(
        &self,
        tenant: &str,
        policy: &ConsolidationPolicy,
        verifier: &dyn MergeVerifier,
        now: DateTime<Utc>,
    ) -> Result<ConsolidationAttempt, StoreError> {
        // Asked once cheaply before any lock is taken, because "no" is the
        // answer almost every time and the worker shares its pool with the
        // request path: a tick over a hundred tenants must not acquire a
        // hundred connections to discover there is nothing to do.
        //
        // It is not the answer that counts. Between this read and the lock,
        // another instance can complete the very run this one is about to
        // start, so the question is asked again under the lock and *that*
        // answer is the one acted on — the same resolve, lock, resolve-again
        // shape `merge_nodes` uses, for the same reason.
        if trigger_for(
            read_state(&self.pool, tenant).await?,
            tenant_counts(&self.pool, tenant).await?,
            policy,
            now,
        )
        .is_none()
        {
            return Ok(ConsolidationAttempt::NotDue);
        }

        let Some(mut lock) = TenantLock::try_acquire(&self.pool, tenant).await? else {
            tracing::debug!(
                tenant,
                "another instance is consolidating this tenant; skipping"
            );
            return Ok(ConsolidationAttempt::Skipped);
        };

        let outcome = self
            .consolidate_locked(tenant, policy, verifier, now, &mut lock)
            .await;

        // Before the answer goes back, whatever the answer is. The baseline
        // this run wrote has to be visible to the next instance that takes the
        // lock, and an error path that kept the lock would wedge the tenant
        // until the connection was noticed.
        lock.release().await;

        outcome
    }

    /// One run, with this tenant's advisory lock held.
    async fn consolidate_locked(
        &self,
        tenant: &str,
        policy: &ConsolidationPolicy,
        verifier: &dyn MergeVerifier,
        now: DateTime<Utc>,
        lock: &mut TenantLock,
    ) -> Result<ConsolidationAttempt, StoreError> {
        // Wall-clock, measured against a monotonic source and added to the
        // instant the run was given. `finished_at` has to be a real duration —
        // an operator asking how long last night's run took is asking whether
        // it is still cheap — and it has to be on the same clock as
        // `started_at`, because the two are subtracted from each other.
        let began = std::time::Instant::now();

        // The authoritative reading, on the locked connection. Nothing else can
        // be consolidating this tenant now, so a baseline read here is a
        // baseline no other run is about to move.
        let counts = tenant_counts(lock.connection(), tenant).await?;
        let state = read_state(lock.connection(), tenant).await?;

        let Some(triggered_by) = trigger_for(state, counts, policy, now) else {
            return Ok(ConsolidationAttempt::NotDue);
        };

        let run_id = Uuid::new_v4();
        let scan = self.scan(tenant).await?;
        let mut decisions = Vec::new();

        for pair in scan.pairs() {
            let outcome = self.judge_and_merge(tenant, &pair, &scan, verifier).await?;

            match &outcome {
                PairOutcome::Merged {
                    winner_node_id,
                    loser_node_id,
                    merge_id,
                } => tracing::info!(
                    %run_id,
                    tenant,
                    winner = %winner_node_id,
                    loser = %loser_node_id,
                    %merge_id,
                    similarity = pair.similarity,
                    bar = pair.bar,
                    "consolidation merged a duplicate pair"
                ),
                // Everything the gates threw out is logged, but the ambiguous
                // band is logged quietly: it is most of the volume and the
                // least surprising thing a run can say. It is still written to
                // `consolidation_decision` in full, which is where an operator
                // reads it from.
                PairOutcome::Declined(DeclineReason::BelowConfidentBar) => tracing::debug!(
                    %run_id,
                    tenant,
                    node_a = pair.ordered_ids().0,
                    node_b = pair.ordered_ids().1,
                    similarity = pair.similarity,
                    bar = pair.bar,
                    "consolidation considered a pair below the confident bar"
                ),
                PairOutcome::Declined(reason) => tracing::info!(
                    %run_id,
                    tenant,
                    node_a = pair.ordered_ids().0,
                    node_b = pair.ordered_ids().1,
                    similarity = pair.similarity,
                    bar = pair.bar,
                    reason = %reason,
                    "consolidation declined to merge a pair"
                ),
            }

            decisions.push(PairDecision { pair, outcome });
        }

        let report = ConsolidationReport {
            run_id,
            tenant: tenant.to_string(),
            triggered_by,
            counts,
            nodes_scanned: scan.nodes_scanned,
            scan_truncated: scan.truncated,
            decisions,
        };

        let finished_at = now + Duration::from_std(began.elapsed()).unwrap_or_default();
        record_run(&self.pool, &report, verifier.name(), now, finished_at).await?;
        write_state(&self.pool, tenant, now, counts).await?;

        tracing::info!(
            %run_id,
            tenant,
            trigger = triggered_by.as_str(),
            verifier = verifier.name(),
            nodes = counts.nodes,
            edges = counts.edges,
            nodes_scanned = scan.nodes_scanned,
            scan_truncated = scan.truncated,
            considered = report.decisions.len(),
            merged = report.merged_count(),
            "consolidation run finished"
        );

        Ok(ConsolidationAttempt::Ran(report))
    }

    /// Decides one pair and, if everything agrees, merges it.
    ///
    /// The gates run here and the verifier runs after them, which is the whole
    /// of the safety argument: a verifier is never asked about a pair the
    /// arithmetic refused, so no verifier — this one or a later one — can
    /// promote anything.
    async fn judge_and_merge(
        &self,
        tenant: &str,
        pair: &CandidatePair,
        scan: &Scan,
        verifier: &dyn MergeVerifier,
    ) -> Result<PairOutcome, StoreError> {
        if pair.similarity < pair.bar {
            return Ok(PairOutcome::Declined(DeclineReason::BelowConfidentBar));
        }

        if scan.contested(pair) {
            return Ok(PairOutcome::Declined(DeclineReason::Contested));
        }

        if let Verdict::Decline(reason) = verifier.verify(pair).await? {
            return Ok(PairOutcome::Declined(reason));
        }

        let (winner, loser) = (&pair.winner_node_id, &pair.loser_node_id);
        let provenance = Provenance {
            client: Some(CONSOLIDATION_CLIENT.to_string()),
            principal: None,
        };

        match self.merge_nodes(tenant, loser, winner, &provenance).await {
            Ok(outcome) => Ok(PairOutcome::Merged {
                merge_id: outcome.merge_id,
                winner_node_id: outcome.winner_node_id,
                loser_node_id: outcome.loser_node_id,
            }),
            // A refusal here is usually a race the worker lost — another writer
            // merged one of the two between the scan and now — and losing it is
            // the correct outcome. It is recorded rather than propagated,
            // because one contested pair must not abandon the rest of the run.
            Err(error) => Ok(PairOutcome::Declined(DeclineReason::MergeRefused(
                error.to_string(),
            ))),
        }
    }
}

/// Writes the run and every decision it made, in one transaction.
///
/// One transaction because the two halves are one record: a run row with no
/// decisions under it reads as "considered nothing", which is a different claim
/// from "the decisions did not get written", and nothing downstream could tell
/// them apart.
///
/// Not in the same transaction as the merges, though, and deliberately.
/// [`Store::merge_nodes`] commits its own audit with its own write — that is
/// what makes a merge reversible — and holding those open until the end of a
/// run would put every merge in a batch at the mercy of the last one. A run
/// that dies halfway leaves the merges it completed committed and audited, and
/// loses only its own summary of them.
async fn record_run(
    pool: &sqlx::PgPool,
    report: &ConsolidationReport,
    verifier: &str,
    started_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
) -> Result<(), StoreError> {
    let mut transaction = pool.begin().await?;

    sqlx::query(
        "INSERT INTO consolidation_run \
         (run_id, tenant_id, triggered_by, verifier, started_at, finished_at, \
          node_count, edge_count, nodes_scanned, scan_truncated, \
          pairs_considered, pairs_merged) \
         VALUES ($1, $2, $3::consolidation_trigger, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
    )
    .bind(report.run_id)
    .bind(&report.tenant)
    .bind(report.triggered_by.as_str())
    .bind(verifier)
    .bind(started_at)
    .bind(finished_at)
    .bind(report.counts.nodes)
    .bind(report.counts.edges)
    .bind(i32::try_from(report.nodes_scanned).unwrap_or(i32::MAX))
    .bind(report.scan_truncated)
    .bind(i32::try_from(report.decisions.len()).unwrap_or(i32::MAX))
    .bind(i32::try_from(report.merged_count()).unwrap_or(i32::MAX))
    .execute(&mut *transaction)
    .await?;

    // Seven parallel arrays stitched back together by `unnest`, the same shape
    // `record_merge` uses for the edges it changed: a run that considered two
    // hundred pairs is still one round trip.
    let mut node_a = Vec::with_capacity(report.decisions.len());
    let mut node_b = Vec::with_capacity(report.decisions.len());
    let mut similarity = Vec::with_capacity(report.decisions.len());
    let mut bar = Vec::with_capacity(report.decisions.len());
    let mut verdict = Vec::with_capacity(report.decisions.len());
    let mut reason = Vec::with_capacity(report.decisions.len());
    let mut merge_id: Vec<Option<Uuid>> = Vec::with_capacity(report.decisions.len());

    for decision in &report.decisions {
        let (left, right) = decision.pair.ordered_ids();
        node_a.push(left.to_string());
        node_b.push(right.to_string());
        similarity.push(decision.pair.similarity);
        bar.push(decision.pair.bar);

        match &decision.outcome {
            PairOutcome::Merged { merge_id: id, .. } => {
                verdict.push("merged".to_string());
                // What the merge did is in `node_merge`; this is why the worker
                // chose to make it. Every gate passed and this verifier had the
                // last word, which is the whole of the reason and is worth
                // naming here because a later verifier will give a different
                // one for the same arithmetic.
                reason.push(format!("verified:{verifier}"));
                merge_id.push(Some(*id));
            }
            PairOutcome::Declined(declined) => {
                verdict.push("declined".to_string());
                reason.push(declined.to_string());
                merge_id.push(None);
            }
        }
    }

    sqlx::query(
        "INSERT INTO consolidation_decision \
         (run_id, tenant_id, node_a_id, node_b_id, similarity, bar, verdict, reason, merge_id) \
         SELECT $1, $2, a, b, similarity, bar, verdict::consolidation_verdict, reason, merge_id \
         FROM unnest($3::text[], $4::text[], $5::double precision[], $6::double precision[], \
                     $7::text[], $8::text[], $9::uuid[]) \
              AS considered(a, b, similarity, bar, verdict, reason, merge_id)",
    )
    .bind(report.run_id)
    .bind(&report.tenant)
    .bind(&node_a)
    .bind(&node_b)
    .bind(&similarity)
    .bind(&bar)
    .bind(&verdict)
    .bind(&reason)
    .bind(&merge_id)
    .execute(&mut *transaction)
    .await?;

    transaction.commit().await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

/// What one pass over every tenant came to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TickSummary {
    pub tenants: usize,
    pub ran: usize,
    pub skipped: usize,
    pub not_due: usize,
    pub failed: usize,
    pub merged: usize,
}

/// The background task every instance runs.
///
/// It wakes on a fixed tick and asks each tenant whether it is due. The tick is
/// not the schedule — [`ConsolidationPolicy`] is — it is only how finely the
/// schedule can be observed, so it has to be shorter than
/// [`ConsolidationPolicy::max_interval`] or the bound the deployment asked for
/// cannot be honoured.
///
/// **What keeps this off the request path.** A tenant that is not due costs two
/// short reads and no lock. A tenant that is costs one connection detached from
/// the pool for the length of its run, plus whatever one merge at a time holds
/// — two of sixteen at the peak. The run itself is bounded by
/// [`MAX_SCANNED_NODES`] and [`MAX_PAIRS_PER_RUN`], so the work one tick can
/// ask of the database does not scale with how large a tenant has become.
pub struct ConsolidationWorker {
    store: Store,
    policy: ConsolidationPolicy,
    verifier: Arc<dyn MergeVerifier>,
    tick: std::time::Duration,
}

impl ConsolidationWorker {
    pub fn new(store: Store, policy: ConsolidationPolicy, tick: std::time::Duration) -> Self {
        Self {
            store,
            policy,
            verifier: Arc::new(LexicalVerifier),
            tick,
        }
    }

    /// Replaces the shipped verifier. The seam an LLM verifier arrives through;
    /// see [`MergeVerifier`] for why it can only ever narrow what is merged.
    pub fn with_verifier(mut self, verifier: Arc<dyn MergeVerifier>) -> Self {
        self.verifier = verifier;
        self
    }

    /// One pass over every tenant that has written a node.
    ///
    /// A tenant whose consolidation fails is counted and logged, never
    /// propagated: one tenant's broken graph must not stop every other tenant
    /// from being consolidated, and the tick that returned early would leave
    /// the tail of the tenant list permanently unvisited.
    pub async fn tick_once(&self, now: DateTime<Utc>) -> Result<TickSummary, StoreError> {
        let tenants = self.store.tenants_with_nodes().await?;
        let mut summary = TickSummary {
            tenants: tenants.len(),
            ..TickSummary::default()
        };

        for tenant in tenants {
            match self
                .store
                .consolidate_tenant(&tenant, &self.policy, self.verifier.as_ref(), now)
                .await
            {
                Ok(ConsolidationAttempt::NotDue) => summary.not_due += 1,
                Ok(ConsolidationAttempt::Skipped) => summary.skipped += 1,
                Ok(ConsolidationAttempt::Ran(report)) => {
                    summary.ran += 1;
                    summary.merged += report.merged_count();
                }
                Err(error) => {
                    summary.failed += 1;
                    tracing::error!(tenant, %error, "consolidating a tenant failed");
                }
            }
        }

        Ok(summary)
    }

    /// Ticks forever. Returns only if the process is going away.
    pub async fn run(self) {
        tracing::info!(
            growth_percent = self.policy.growth_percent,
            max_interval_seconds = self.policy.max_interval.num_seconds(),
            tick_seconds = self.tick.as_secs(),
            verifier = self.verifier.name(),
            "consolidation worker started; duplicate entities will be merged \
             automatically, reversibly, and only above the inline resolver's own bar"
        );

        let mut ticker = tokio::time::interval(self.tick);
        // A tick missed because the previous one ran long is skipped rather
        // than fired immediately afterwards: catching up would run a burst of
        // passes back to back, which is the opposite of what a bounded
        // background job is for.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            ticker.tick().await;

            match self.tick_once(Utc::now()).await {
                Ok(summary) if summary.ran > 0 || summary.failed > 0 => {
                    tracing::info!(
                        tenants = summary.tenants,
                        ran = summary.ran,
                        skipped = summary.skipped,
                        not_due = summary.not_due,
                        failed = summary.failed,
                        merged = summary.merged,
                        "consolidation tick finished"
                    );
                }
                Ok(summary) => tracing::debug!(
                    tenants = summary.tenants,
                    skipped = summary.skipped,
                    "consolidation tick found nothing due"
                ),
                // Only the tenant *listing* can fail this far out; a per-tenant
                // failure is counted above. So this is the database being
                // unreachable, which the next tick will retry.
                Err(error) => tracing::error!(%error, "a consolidation tick could not start"),
            }
        }
    }

    /// Starts [`ConsolidationWorker::run`] on the runtime.
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(self.run())
    }
}

impl Store {
    /// Every tenant that has written a node.
    ///
    /// A loose index scan rather than `SELECT DISTINCT`, which would read every
    /// row of `node` to answer a question whose answer is the number of
    /// customers. This walks the primary key one tenant at a time — one index
    /// probe per tenant — so a tick over a large database costs what the
    /// tenant list costs and not what the graph costs.
    ///
    /// `node` rather than a tenant table, because there is no tenant table: a
    /// tenant is whatever a validated credential claims, and the first thing it
    /// writes is what makes it exist here.
    pub async fn tenants_with_nodes(&self) -> Result<Vec<String>, StoreError> {
        let rows: Vec<(Option<String>,)> = sqlx::query_as(
            "WITH RECURSIVE walk AS ( \
                 (SELECT tenant_id FROM node ORDER BY tenant_id LIMIT 1) \
                 UNION ALL \
                 SELECT (SELECT n.tenant_id FROM node AS n \
                         WHERE n.tenant_id > walk.tenant_id \
                         ORDER BY n.tenant_id LIMIT 1) \
                 FROM walk WHERE walk.tenant_id IS NOT NULL \
             ) \
             SELECT tenant_id FROM walk WHERE tenant_id IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().flat_map(|(tenant,)| tenant).collect())
    }
}

// ---------------------------------------------------------------------------
// Running one at a time behind a load balancer
// ---------------------------------------------------------------------------

/// This tenant's consolidation slot, held for the length of one run.
///
/// **Why an advisory lock and not leader election.** Every instance runs this
/// worker, so N instances is N concurrent consolidations over the same rows
/// unless something prevents it. The thing that prevents it has to live where
/// the instances can all see it, and the only thing they share is the database
/// — this architecture is stateless precisely so that nothing else is shared,
/// and a leader elected in application memory would be a leader per process.
///
/// **Session-scoped, not transaction-scoped.** `pg_try_advisory_xact_lock`
/// would release itself, but only at the end of a transaction, and holding one
/// open for the length of a run would pin a snapshot, block vacuum, and put
/// every merge in the run at the mercy of the last one. A session lock is held
/// by the backend rather than by a transaction, which is what allows the merges
/// underneath it to commit one at a time.
///
/// **What happens when an instance dies holding it.** Postgres releases every
/// session-level advisory lock when the session ends, and a session ends when
/// its socket closes — so a killed process, a severed network, or an OOM
/// releases this without anybody intervening. That is the whole failure story,
/// and it is only true because of [`TenantLock::connection`]: the connection is
/// **detached from the pool** the moment the lock is taken. A pooled connection
/// that went back to the pool still holding a session lock would keep holding
/// it, because the session is still open — and no later attempt would ever get
/// it. Detached, the connection is owned by this struct, and dropping it, on
/// any path including a panic, ends the session and frees the lock.
struct TenantLock {
    connection: PgConnection,
    class: i32,
    key: i32,
}

impl TenantLock {
    /// Takes the lock, or reports that somebody else has it.
    ///
    /// `pg_try_advisory_lock`, never `pg_advisory_lock`: a losing instance has
    /// to come back immediately having done nothing. Blocking would queue it
    /// behind a run that is doing the work it wanted to do, and it would wake
    /// up to do that work again — with a reason to run that the winner has
    /// since invalidated.
    async fn try_acquire(pool: &PgPool, tenant: &str) -> Result<Option<Self>, StoreError> {
        let (class, key) = advisory_lock_key(tenant);
        let mut pooled = pool.acquire().await?;

        // A lock can only be stranded on a pooled connection in one way: this
        // future being dropped between Postgres taking the lock and the row
        // reaching Rust. Then the `PoolConnection` goes back to the pool with
        // its session still holding the lock, and the tenant is wedged until
        // that connection is recycled. Clearing this session's own locks first
        // costs one round trip on a path that is already acquiring a
        // connection, and it is safe to be indiscriminate because nothing else
        // in this server takes an advisory lock, and because a lock held by a
        // *running* consolidation is on a detached connection this call can
        // never be handed.
        sqlx::query("SELECT pg_advisory_unlock_all()")
            .execute(&mut *pooled)
            .await?;

        let (acquired,): (bool,) = sqlx::query_as("SELECT pg_try_advisory_lock($1, $2)")
            .bind(class)
            .bind(key)
            .fetch_one(&mut *pooled)
            .await?;

        if !acquired {
            return Ok(None);
        }

        Ok(Some(Self {
            // Out of the pool, for the reason the type documents. The pool
            // opens a replacement when it needs one; that is one connection per
            // run, which is at most one per tenant per interval.
            connection: pooled.detach(),
            class,
            key,
        }))
    }

    /// The locked session, for the reads whose answer the lock is protecting.
    fn connection(&mut self) -> &mut PgConnection {
        &mut self.connection
    }

    /// Releases the lock and ends the session.
    ///
    /// Both, and in that order. The unlock is the orderly path; closing is what
    /// makes the outcome the same if the unlock failed, because a closed
    /// session holds nothing. Neither failure is propagated: the run has
    /// already happened and already been recorded, and turning a tidy-up error
    /// into the caller's error would report a successful consolidation as a
    /// failed one.
    async fn release(self) {
        let Self {
            mut connection,
            class,
            key,
        } = self;

        if let Err(error) = sqlx::query("SELECT pg_advisory_unlock($1, $2)")
            .bind(class)
            .bind(key)
            .execute(&mut connection)
            .await
        {
            tracing::warn!(
                %error,
                "could not release the consolidation lock; closing the session, which \
                 releases it"
            );
        }

        if let Err(error) = connection.close().await {
            tracing::debug!(%error, "the consolidation lock's session did not close cleanly");
        }
    }
}

// ---------------------------------------------------------------------------
// The scan
// ---------------------------------------------------------------------------

/// One live node, as the scan needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ScannedNode {
    display_name: String,
    created_at: DateTime<Utc>,
}

/// One row of a neighbour list: a node, one of its nearest same-model
/// neighbours, and the cosine between them.
type NeighbourRow = (
    String,
    String,
    DateTime<Utc>,
    String,
    String,
    DateTime<Utc>,
    f64,
);

/// What one run looked at.
#[derive(Debug, Default)]
struct Scan {
    /// Every node either side of every row, so a pair can be oriented and
    /// named without going back to the database.
    nodes: BTreeMap<String, ScannedNode>,
    /// The full neighbour list of every node the scan probed, in both
    /// directions. The counting in [`Scan::contested`] needs a *complete* list
    /// for both ends of a pair, which is the only reason the second pass in
    /// [`Store::scan`] exists.
    neighbours: BTreeMap<String, BTreeMap<String, f64>>,
    /// The unordered pairs worth considering, keyed on the smaller id.
    candidates: BTreeMap<(String, String), f64>,
    nodes_scanned: usize,
    truncated: bool,
}

impl Scan {
    /// Records a neighbour row, in both directions.
    ///
    /// Both, because cosine is symmetric but a top-`k` neighbour list is not:
    /// a node with many close neighbours can crowd one out of its own list
    /// while still appearing in that one's. Unioning the two directions can
    /// only ever make a node look like it has *more* confident neighbours, and
    /// more is the safe direction — it declines pairs rather than merging them.
    fn record(&mut self, row: NeighbourRow, is_candidate: bool) {
        let (left, left_name, left_created, right, right_name, right_created, similarity) = row;

        self.nodes.entry(left.clone()).or_insert(ScannedNode {
            display_name: left_name,
            created_at: left_created,
        });
        self.nodes.entry(right.clone()).or_insert(ScannedNode {
            display_name: right_name,
            created_at: right_created,
        });

        self.neighbours
            .entry(left.clone())
            .or_default()
            .insert(right.clone(), similarity);
        self.neighbours
            .entry(right.clone())
            .or_default()
            .insert(left.clone(), similarity);

        if is_candidate {
            let key = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            self.candidates.insert(key, similarity);
        }
    }

    /// Whether either end of this pair has a rival claim on it.
    ///
    /// The inline resolver attaches to a confident neighbour only when there is
    /// **exactly one**: two nodes answering to one name is ambiguity however
    /// high the scores are, and taking the higher of two would be an over-merge
    /// dressed up as certainty. This is that rule, asked from both ends rather
    /// than from one.
    ///
    /// Asking from both ends is strictly stronger than the write path, and not
    /// gratuitously: the write path has an incoming name and an existing graph,
    /// so there is only one side to ask from. Here both nodes already exist, so
    /// the question "is this the only thing that could be the same entity" has
    /// two answers and both have to be yes.
    fn contested(&self, pair: &CandidatePair) -> bool {
        !self.exclusive_to(&pair.winner_node_id, &pair.loser_node_id)
            || !self.exclusive_to(&pair.loser_node_id, &pair.winner_node_id)
    }

    /// Whether `expected` is the *only* node `node_id` is confident about.
    ///
    /// Unknown either way counts as contested. A node the scan has no list for
    /// is a node nothing has been established about, and the safe reading of an
    /// absent answer is that the question is open.
    fn exclusive_to(&self, node_id: &str, expected: &str) -> bool {
        let (Some(node), Some(neighbours)) =
            (self.nodes.get(node_id), self.neighbours.get(node_id))
        else {
            return false;
        };

        let bar = own_bar(&node.display_name);
        let mut confident = neighbours
            .iter()
            .filter(|(_, similarity)| **similarity >= bar)
            .map(|(other, _)| other.as_str());

        confident.next() == Some(expected) && confident.next().is_none()
    }

    /// The candidate pairs, most similar first, oriented and capped.
    fn pairs(&self) -> Vec<CandidatePair> {
        let mut pairs: Vec<CandidatePair> = self
            .candidates
            .iter()
            .filter_map(|((left, right), similarity)| {
                let left_node = self.nodes.get(left)?;
                let right_node = self.nodes.get(right)?;

                // Older first, ties broken by the smaller id — see
                // `CandidatePair`.
                let (winner, loser) =
                    if (left_node.created_at, left) <= (right_node.created_at, right) {
                        (left, right)
                    } else {
                        (right, left)
                    };

                Some(CandidatePair {
                    winner_node_id: winner.clone(),
                    winner_display_name: self.nodes.get(winner)?.display_name.clone(),
                    loser_node_id: loser.clone(),
                    loser_display_name: self.nodes.get(loser)?.display_name.clone(),
                    similarity: *similarity,
                    bar: bar_for(&left_node.display_name, &right_node.display_name),
                })
            })
            .collect();

        // Most similar first, then by id, so a capped run keeps the pairs it is
        // most confident about and two runs over one graph agree.
        pairs.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.ordered_ids().cmp(&right.ordered_ids()))
        });
        pairs.truncate(MAX_PAIRS_PER_RUN);
        pairs
    }
}

impl Store {
    /// Pulls the neighbour lists this run will decide from.
    ///
    /// Two passes, and the second is not an optimisation. The first probes the
    /// newest live nodes and yields the candidate pairs. The second completes
    /// the neighbour list of every node that turned up only as somebody else's
    /// neighbour — an older node a new duplicate landed next to. Without it,
    /// [`Scan::contested`] would be counting an incomplete list for one side of
    /// exactly the pairs this worker exists to catch, and an undercount there
    /// merges a contested pair.
    async fn scan(&self, tenant: &str) -> Result<Scan, StoreError> {
        let mut scan = Scan::default();

        let (probed, truncated) = live_node_ids(&self.pool, tenant, MAX_SCANNED_NODES).await?;
        scan.nodes_scanned = probed.len();
        scan.truncated = truncated;

        let probed_set: BTreeSet<&String> = probed.iter().collect();
        let rows = neighbours_of(&self.pool, tenant, &probed).await?;

        let unprobed: Vec<String> = rows
            .iter()
            .map(|row| row.3.clone())
            .filter(|id| !probed_set.contains(id))
            .collect::<BTreeSet<String>>()
            .into_iter()
            .take(MAX_SCANNED_NODES)
            .collect();

        for row in rows {
            scan.record(row, true);
        }

        for row in neighbours_of(&self.pool, tenant, &unprobed).await? {
            // Not candidates. A pair between two nodes that both turned up only
            // as neighbours is a pair this run did not go looking for; it will
            // be a candidate on the run that scans one of them. Recording it
            // still sharpens the contest count, which is the point.
            scan.record(row, false);
        }

        Ok(scan)
    }
}

/// The live nodes a run will probe, newest first, and whether there were more.
///
/// `merged_into IS NULL` because a tombstone is not a candidate for anything —
/// and because the partial HNSW index the neighbour query rides on is defined
/// `WHERE merged_into IS NULL`. `embedding_model IS NOT NULL` because a vector
/// whose provenance is unknown cannot be compared to anything: the schema keeps
/// that column precisely so a mixed-model graph is detectable rather than
/// silently producing cosines between two different spaces.
async fn live_node_ids(
    executor: impl sqlx::PgExecutor<'_>,
    tenant: &str,
    limit: usize,
) -> Result<(Vec<String>, bool), StoreError> {
    // One more than the cap, so "there were more" is answered by the same query
    // rather than by a second count that could disagree with it.
    let probe = i64::try_from(limit).unwrap_or(i64::MAX).saturating_add(1);

    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT node_id FROM node \
         WHERE tenant_id = $1 \
           AND merged_into IS NULL \
           AND name_vector IS NOT NULL \
           AND embedding_model IS NOT NULL \
         ORDER BY created_at DESC, node_id \
         LIMIT $2",
    )
    .bind(tenant)
    .bind(probe)
    .fetch_all(executor)
    .await?;

    let truncated = rows.len() > limit;
    Ok((
        rows.into_iter().map(|(id,)| id).take(limit).collect(),
        truncated,
    ))
}

/// Each named node's nearest same-model live neighbours, down to the ambiguous
/// floor.
///
/// The same three restrictions the inline resolver's tier 3 applies, for the
/// same three reasons, and the same [`CANDIDATE_LIMIT`] — imported rather than
/// restated, so the worker cannot come to see a wider or narrower slice of the
/// graph than the write path does.
///
/// [`AMBIGUOUS_SIMILARITY`] is the floor rather than the confident bar because
/// what is *considered* is deliberately wider than what is acted on. That band
/// is exactly the one the resolver hands back to an agent as near misses, and
/// writing it down is what lets an operator see the merges the worker is
/// declining to make.
async fn neighbours_of(
    executor: impl sqlx::PgExecutor<'_>,
    tenant: &str,
    node_ids: &[String],
) -> Result<Vec<NeighbourRow>, StoreError> {
    if node_ids.is_empty() {
        return Ok(Vec::new());
    }

    Ok(sqlx::query_as(
        "SELECT s.node_id, s.display_name, s.created_at, \
                nb.node_id, nb.display_name, nb.created_at, \
                1 - (s.name_vector <=> nb.name_vector) \
         FROM node AS s \
         CROSS JOIN LATERAL ( \
             SELECT n.node_id, n.display_name, n.created_at, n.name_vector \
             FROM node AS n \
             WHERE n.tenant_id = $1 \
               AND n.merged_into IS NULL \
               AND n.name_vector IS NOT NULL \
               AND n.embedding_model = s.embedding_model \
               AND n.node_id <> s.node_id \
             ORDER BY n.name_vector <=> s.name_vector \
             LIMIT $3 \
         ) AS nb \
         WHERE s.tenant_id = $1 \
           AND s.node_id = ANY($2::text[]) \
           AND s.merged_into IS NULL \
           AND s.name_vector IS NOT NULL \
           AND s.embedding_model IS NOT NULL \
           AND 1 - (s.name_vector <=> nb.name_vector) >= $4",
    )
    .bind(tenant)
    .bind(node_ids)
    .bind(CANDIDATE_LIMIT)
    .bind(AMBIGUOUS_SIMILARITY)
    .fetch_all(executor)
    .await?)
}

/// Every row, for the reason [`TenantCounts`] gives.
async fn tenant_counts(
    executor: impl sqlx::PgExecutor<'_>,
    tenant: &str,
) -> Result<TenantCounts, StoreError> {
    let (nodes, edges): (i64, i64) = sqlx::query_as(
        "SELECT (SELECT count(*) FROM node WHERE tenant_id = $1), \
                (SELECT count(*) FROM edge WHERE tenant_id = $1)",
    )
    .bind(tenant)
    .fetch_one(executor)
    .await?;

    Ok(TenantCounts { nodes, edges })
}

async fn read_state(
    executor: impl sqlx::PgExecutor<'_>,
    tenant: &str,
) -> Result<Option<TenantState>, StoreError> {
    let row: Option<(DateTime<Utc>, i64, i64)> = sqlx::query_as(
        "SELECT last_run_at, node_count, edge_count FROM consolidation_state \
         WHERE tenant_id = $1",
    )
    .bind(tenant)
    .fetch_optional(executor)
    .await?;

    Ok(row.map(|(last_run_at, nodes, edges)| TenantState {
        last_run_at,
        counts: TenantCounts { nodes, edges },
    }))
}

/// The baseline the next run is measured against.
///
/// The counts written are the ones the run *started* from, not a fresh reading:
/// anything written while the run was working has not been consolidated, so it
/// belongs to the next run's growth rather than being absorbed by the one that
/// could not see it.
async fn write_state(
    executor: impl sqlx::PgExecutor<'_>,
    tenant: &str,
    finished_at: DateTime<Utc>,
    counts: TenantCounts,
) -> Result<(), StoreError> {
    sqlx::query(
        "INSERT INTO consolidation_state (tenant_id, last_run_at, node_count, edge_count) \
         VALUES ($1, $2, $3, $4) \
         ON CONFLICT (tenant_id) DO UPDATE \
             SET last_run_at = EXCLUDED.last_run_at, \
                 node_count  = EXCLUDED.node_count, \
                 edge_count  = EXCLUDED.edge_count",
    )
    .bind(tenant)
    .bind(finished_at)
    .bind(counts.nodes)
    .bind(counts.edges)
    .execute(executor)
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(hours: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(hours * 3600, 0).expect("a valid instant")
    }

    fn state(hours: i64, nodes: i64, edges: i64) -> TenantState {
        TenantState {
            last_run_at: at(hours),
            counts: TenantCounts { nodes, edges },
        }
    }

    fn counts(nodes: i64, edges: i64) -> TenantCounts {
        TenantCounts { nodes, edges }
    }

    /// The claim the whole worker rests on, checked rather than asserted in
    /// prose: for every pair of names, the bar an automatic merge must clear is
    /// at least the bar the inline resolver would have applied — from *either*
    /// side, because which name is "incoming" is an accident of arrival order
    /// that the worker does not have.
    ///
    /// A worker that merged what the write path declined, on the same evidence,
    /// would undo three deliberate design decisions at scale. This is the line
    /// that stops it, so it is checked over the whole space the length rule can
    /// produce.
    #[test]
    fn the_automatic_bar_is_never_below_the_inline_bar_from_either_side() {
        let short = "Phoenix";
        let long = "Quarterly Business Review";
        assert!(short.chars().count() <= SHORT_NAME_CHARS);
        assert!(long.chars().count() > SHORT_NAME_CHARS);

        for left in [short, long] {
            for right in [short, long] {
                let automatic = bar_for(left, right);

                assert!(
                    automatic >= own_bar(left) && automatic >= own_bar(right),
                    "{left:?} against {right:?}: the worker's bar {automatic} is below \
                     an inline bar of {} or {}",
                    own_bar(left),
                    own_bar(right)
                );
            }
        }

        // And it is *strictly* stricter in the one case where arrival order
        // would otherwise decide: a long name resolving against a short one is
        // held to 0.92 inline, and to 0.93 here.
        assert!(
            (bar_for(long, short) - CONFIDENT_SIMILARITY_SHORT).abs() < f64::EPSILON,
            "a pair with a short name in it takes the short bar whichever side it is on"
        );
        assert!(own_bar(long) < bar_for(long, short));
    }

    /// Whitespace and multi-byte scripts, because the length rule counts
    /// characters of the trimmed name — the same rule `Thresholds::for_name`
    /// uses, and a name in a non-Latin script is not long merely for being
    /// multi-byte.
    #[test]
    fn the_length_rule_counts_trimmed_characters_rather_than_bytes() {
        assert!((own_bar("  Phoenix  ") - CONFIDENT_SIMILARITY_SHORT).abs() < f64::EPSILON);
        // Twelve characters, thirty-six bytes.
        let twelve = "東".repeat(SHORT_NAME_CHARS);
        assert_eq!(twelve.len(), SHORT_NAME_CHARS * 3);
        assert!((own_bar(&twelve) - CONFIDENT_SIMILARITY_SHORT).abs() < f64::EPSILON);
        assert!(
            (own_bar(&"東".repeat(SHORT_NAME_CHARS + 1)) - CONFIDENT_SIMILARITY).abs()
                < f64::EPSILON
        );
    }

    /// A tenant nobody has consolidated has no baseline to have grown from and
    /// no interval to have elapsed. It runs, and the run is what gives it one.
    #[test]
    fn a_tenant_with_no_baseline_is_always_due() {
        let policy = ConsolidationPolicy::default();

        assert_eq!(
            trigger_for(None, counts(0, 0), &policy, at(0)),
            Some(Trigger::FirstRun)
        );
        assert_eq!(
            trigger_for(None, counts(9_000, 40_000), &policy, at(0)),
            Some(Trigger::FirstRun)
        );
    }

    /// Both bounds hold and whichever fires first wins. Growth is checked
    /// first, so a tenant that is due for both reports the reason that is
    /// actually informative.
    #[test]
    fn growth_and_interval_are_both_bounds_and_growth_is_named_first() {
        let policy = ConsolidationPolicy {
            growth_percent: 10.0,
            max_interval: Duration::hours(4),
        };
        let before = Some(state(0, 100, 100));

        assert_eq!(
            trigger_for(before, counts(109, 100), &policy, at(1)),
            None,
            "nine percent is not ten"
        );
        assert_eq!(
            trigger_for(before, counts(110, 100), &policy, at(1)),
            Some(Trigger::Growth),
            "exactly the configured percentage is the configured percentage"
        );
        assert_eq!(
            trigger_for(before, counts(100, 110), &policy, at(1)),
            Some(Trigger::Growth),
            "edges count too: a tenant can grow one dimension without the other"
        );
        assert_eq!(
            trigger_for(before, counts(100, 100), &policy, at(4)),
            Some(Trigger::Interval),
            "zero growth is no excuse once the interval has passed"
        );
        assert_eq!(
            trigger_for(before, counts(200, 100), &policy, at(9)),
            Some(Trigger::Growth),
            "due for both reasons, and growth is the one worth reporting"
        );
    }

    /// The degenerate ends of the growth arithmetic, which are where a
    /// scheduler either never fires or never stops.
    #[test]
    fn growth_handles_an_empty_baseline_and_a_motionless_tenant() {
        assert!(grown(0, 1, 10.0), "there is no percentage of nothing");
        assert!(
            !grown(0, 0, 10.0),
            "an empty tenant that is still empty has not grown"
        );
        assert!(
            !grown(5, 5, 0.0),
            "a policy of zero percent must not fire on a tenant that wrote nothing, \
             or the worker never stops"
        );
        assert!(grown(5, 6, 0.0), "but any write at all clears zero percent");
        assert!(
            !grown(100, 99, 10.0),
            "counts cannot fall, and if one ever did it is not growth"
        );
    }

    /// The lock key has to be the same number in every instance of a stateless
    /// server, and a different number for tenants that are not the same tenant.
    #[test]
    fn the_lock_key_is_stable_per_tenant_and_namespaced() {
        assert_eq!(advisory_lock_key("acme"), advisory_lock_key("acme"));
        assert_ne!(advisory_lock_key("acme"), advisory_lock_key("acme "));
        assert_ne!(advisory_lock_key("acme"), advisory_lock_key("initech"));

        let (class, _) = advisory_lock_key("acme");
        assert_eq!(
            class, CONSOLIDATION_LOCK_CLASS,
            "the class is fixed so a lock this worker holds is identifiable in pg_locks"
        );
    }

    fn pair(left: &str, right: &str) -> CandidatePair {
        CandidatePair {
            winner_node_id: "left".to_string(),
            winner_display_name: left.to_string(),
            loser_node_id: "right".to_string(),
            loser_display_name: right.to_string(),
            similarity: 0.99,
            bar: bar_for(left, right),
        }
    }

    /// The shipped verifier is the write path's own guard and nothing else. It
    /// says only that two names disagree; it never says they agree, so its
    /// `Merge` is the absence of a veto rather than a positive judgement — and
    /// that is exactly what the gates in front of it require it to be.
    #[tokio::test]
    async fn the_lexical_verifier_vetoes_a_substitution_and_allows_an_elaboration() {
        let verifier = LexicalVerifier;

        assert_eq!(
            verifier
                .verify(&pair("Acme Corporation", "Acme Holdings"))
                .await
                .expect("the lexical verifier cannot fail"),
            Verdict::Decline(DeclineReason::NamesContradict),
            "two companies"
        );
        assert_eq!(
            verifier
                .verify(&pair("billing-service-v2", "billing-service-v1"))
                .await
                .expect("the lexical verifier cannot fail"),
            Verdict::Decline(DeclineReason::NamesContradict),
            "a version decides an entity by itself"
        );
        assert_eq!(
            verifier
                .verify(&pair(
                    "Quarterly Business Review",
                    "Quarterly Business Reviews"
                ))
                .await
                .expect("the lexical verifier cannot fail"),
            Verdict::Merge,
            "one name elaborating on another is not a disagreement"
        );
        assert_eq!(verifier.name(), "lexical");
    }

    /// The `reason` column is queried by an operator, so the code in front of
    /// it has to be stable and the one variant that carries detail has to stay
    /// prefix-matchable.
    #[test]
    fn a_decline_reason_reads_as_a_stable_code() {
        assert_eq!(DeclineReason::Contested.to_string(), "contested");
        assert_eq!(
            DeclineReason::BelowConfidentBar.to_string(),
            "below_confident_bar"
        );
        assert_eq!(
            DeclineReason::NamesContradict.to_string(),
            "names_contradict"
        );

        let refused = DeclineReason::MergeRefused("node `x` was already merged".to_string());
        assert_eq!(refused.code(), "merge_refused");
        assert!(
            refused.to_string().starts_with("merge_refused:"),
            "a reason with detail must still start with its code, got {refused}"
        );
    }

    /// Four hours and ten percent, said out loud. A default nobody wrote down
    /// is a default that drifts.
    #[test]
    fn the_shipped_policy_is_four_hours_and_ten_percent() {
        let policy = ConsolidationPolicy::default();

        assert_eq!(policy.max_interval, Duration::hours(4));
        assert!((policy.growth_percent - 10.0).abs() < f64::EPSILON);
    }
}
