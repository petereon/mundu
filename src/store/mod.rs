//! Postgres-backed persistence.
//!
//! Every query in this module is a runtime query — `sqlx::query`, `bind`, and a
//! hand-written row decode — rather than one of the `query!` macros. The macros
//! type-check against a live database *at compile time*, which would make the
//! crate unbuildable on any machine without one, including CI and a fresh
//! checkout. The compile-time checking is worth less than being able to build.

pub mod consolidate;
pub mod edge;
pub mod explore;
pub mod merge;
pub mod node;
pub mod rating;
pub mod relationship;
pub mod search;

use std::time::Duration;

use sqlx::postgres::PgPoolOptions;
use sqlx::{Executor, PgPool};

use crate::error::StoreError;

/// The most items any one array argument may carry.
///
/// **Measured, not guessed.** Every batch this server accepts is embedded by
/// `bge-base-en-v1.5`, and every batch used to be embedded in a single forward
/// pass, which is what made a batched write affordable and also what made an
/// unbounded one fatal: the cost was linear in the number of items and
/// quadratic in the longest sequence, and the tokenizer truncates every input
/// at the model's 512 positions, so the per-item worst case was a constant. On
/// this machine, at that ceiling, it measured **77 MB of peak resident set per
/// item** (64 items: 5.8 GB peak against a 0.9 GB idle process, 88 s of CPU). A
/// thousand items — which nothing refused before this constant existed — is
/// 77 GB, and the server is stateless behind a load balancer, so the client
/// that retried would take the next instance down too.
///
/// 64 is where three unrelated arguments happen to meet:
///
/// - a conversation yields a few dozen entities and a few dozen facts, not
///   thousands, so a cap here refuses no honest write;
/// - it is above [`search::MAX_LIMIT`], the most hits one search can return, so
///   `rate_memory_utility` can still rate everything a lookup returned in one
///   call — which its tool description promises;
/// - it bounds the total work one request can ask for.
///
/// What it no longer has to do is bound peak memory. `embed_documents` now runs
/// a batch as however many forward passes it takes, each inside its own budget
/// of token positions, so the figure above is a property of one pass rather
/// than of one request: the same 64 items at the ceiling measure 1174 MiB peak
/// instead of 5675 MiB. The two caps are complementary and both stay — chunking
/// bounds the memory inside a request, this bounds how much request there is.
///
/// Applied to `nodes`, `relationships`, `edge_ids` and to a single node's
/// `aliases` — an entity with more than 64 names is not an entity, and
/// `surface_forms` deduplicates them against each other pairwise.
pub const MAX_BATCH_ITEMS: usize = 64;

/// Refuses an array argument longer than [`MAX_BATCH_ITEMS`].
///
/// Whole-call, like every other guard decidable from the argument alone
/// ([`StoreError::BlankDisplayName`], [`StoreError::BlankRelationshipType`],
/// [`StoreError::FutureValidFrom`], [`StoreError::UtilityScoreOutOfRange`]) and
/// unlike the per-item outcomes, which are answers about the *database* and can
/// be an honest race with another agent. Refusing the call is also the entire
/// point here: a cap that let the first 64 items through would have done the
/// work the cap exists to avoid.
///
/// `argument` is the name the caller used, so the message points at the field
/// to split rather than at "the batch".
pub(crate) fn reject_oversized_batch(
    argument: &'static str,
    received: usize,
) -> Result<(), StoreError> {
    if received > MAX_BATCH_ITEMS {
        return Err(StoreError::BatchTooLarge {
            argument,
            received,
            limit: MAX_BATCH_ITEMS,
        });
    }

    Ok(())
}

/// The most characters one entity name — a `display_name` or an alias — may
/// carry.
///
/// A name is a name, not prose. Twelve characters is already where the resolver
/// stops treating a name as short ([`node::SHORT_NAME_CHARS`]); the longest
/// registered company names in the world run to about 160. Past 200 the string
/// is a paragraph in the wrong field, and it is also about to become a
/// `node_id` — the primary key an agent quotes back on every later write.
pub const MAX_DISPLAY_NAME_CHARS: usize = 200;

/// The most characters one `proof_excerpt` may carry.
///
/// Placed against the model rather than against a sense of what looks long.
/// Measured on `bge-base-en-v1.5`'s own tokenizer, ordinary English prose runs
/// at **5.4 characters per wordpiece token**, so 2000 characters is about 370
/// tokens — inside the 512 positions the model has, which means a proof written
/// at this cap is embedded whole rather than silently half-embedded. It is also
/// ten times [`search`]'s 200-character snippet, so it cannot refuse anything a
/// caller would reasonably quote.
///
/// It does not bound the forward pass: the tokenizer truncates at 512 positions
/// regardless, and a script that tokenizes at one token per character — CJK
/// measured at 1.00 — reaches that ceiling in 512 characters. What this bounds
/// is everything linear: the copies of the batch made before tokenisation, the
/// row, and the text search hands back. The embedder's own budget over token
/// positions is what bounds the pass.
pub const MAX_PROOF_EXCERPT_CHARS: usize = 2_000;

/// The fewest characters of `proof_excerpt`, after trimming, that count as
/// evidence.
///
/// This is a floor chosen so that it *cannot* bite, not a judgement about how
/// good a proof is. "Is this actually evidence?" is not a question a length can
/// answer, and a bar set high enough to filter weak evidence would refuse
/// honest writes in every language that says more per character than English
/// does. So the number is placed below the shortest real quotation that can be
/// constructed in the densest script this server's tokenizer reads — CJK, at a
/// measured 1.00 characters per token, where three characters is already three
/// tokens and enough for a subject, a verb and an object.
///
/// What it does rule out is the degenerate case the blank check alone misses: a
/// single character, or a full stop, standing in for a quotation.
///
/// Sits beside [`node::NodeInput`]'s blank-display-name rule and the blank
/// relationship type in [`relationship`]: same shape of guard, same layer, same
/// whole-call refusal.
pub const MIN_PROOF_EXCERPT_CHARS: usize = 3;

/// Counted in `chars`, never in bytes.
///
/// A byte budget would give a proof written in Japanese a third of the words an
/// English one gets, for a limit that exists because of tokens — and tokens
/// track characters far better than they track bytes.
fn text_length(text: &str) -> usize {
    text.chars().count()
}

/// Refuses a single text longer than `limit`.
///
/// Whole-call and before any write, for the reason
/// [`reject_oversized_batch`] gives: it is decidable from the argument, so
/// finding it halfway through a batch would leave the caller unable to tell
/// what landed.
pub(crate) fn reject_overlong_text(
    field: &'static str,
    index: usize,
    text: &str,
    limit: usize,
) -> Result<(), StoreError> {
    let chars = text_length(text);

    if chars > limit {
        return Err(StoreError::TextTooLong {
            index,
            field,
            chars,
            limit,
        });
    }

    Ok(())
}

/// Refuses a `proof_excerpt` that is not evidence.
///
/// Trimmed before it is measured, because a proof of four spaces is exactly as
/// unevidenced as a proof of none, and the store has no other way to tell them
/// apart — the schema can carry a `minLength`, but a `minLength` counts spaces.
pub(crate) fn reject_unevidenced_proof(index: usize, proof: &str) -> Result<(), StoreError> {
    let chars = text_length(proof.trim());

    if chars < MIN_PROOF_EXCERPT_CHARS {
        return Err(StoreError::UnevidencedProofExcerpt {
            index,
            chars,
            minimum: MIN_PROOF_EXCERPT_CHARS,
        });
    }

    Ok(())
}

/// Bounded so an unreachable or wedged database surfaces as an error instead of
/// a request that never returns. sqlx's default is 30s, which is long enough
/// that a test suite looks hung rather than failed.
const ACQUIRE_TIMEOUT: Duration = Duration::from_secs(10);

/// Comfortably above the number of concurrent writers a single instance is
/// expected to serve, and well under Postgres' default 100 connection slots so
/// several instances can share one database.
const MAX_CONNECTIONS: u32 = 16;

/// What every connection in the pool has to be told before it runs a vector
/// query, and why it is not tuning.
///
/// pgvector's HNSW scan is not merely approximate, it is *bounded*: with
/// iterative scan off it hands back at most `hnsw.ef_search` tuples for the
/// whole index and then stops, and the shipped default is 40. Every vector
/// query here asks for [`search::CANDIDATE_DEPTH`] rows and narrows them with
/// `tenant_id` — a predicate no vector index carries — so the tenant filter is
/// applied *after* the index has already decided which forty rows exist in the
/// world. A tenant's own facts are then crowded out of its own results by rows
/// belonging to other customers, and which ones survive is a function of what
/// those customers happened to write.
///
/// That is not a tolerance to be widened, it is a wrong answer: measured on
/// this suite's corpus of 270 edges, a tenant holding 55 active edges got 14 of
/// them back, and several tenants holding two or five got none at all. The
/// caller cannot see it, because a short candidate list looks exactly like a
/// small graph. It only surfaced as a flake because the planner stops choosing
/// the HNSW index once the table grows enough for a b-tree scan to look
/// cheaper — so recall silently depended on how warm the index was.
///
/// * `hnsw.ef_search` is raised to [`search::CANDIDATE_DEPTH`], because a query
///   that asks for 200 candidates from an index configured to return 40 is
///   asking for something the database will not do, and the search module
///   documents that depth as "the real recall bound".
/// * `hnsw.iterative_scan` is what makes the *filter* safe. The scan keeps
///   pulling from the index until the `LIMIT` is satisfied or
///   `hnsw.max_scan_tuples` (20,000, left at its default) is reached, instead
///   of stopping at the first `ef_search` rows and letting the filter eat them.
///   `strict_order`, not `relaxed_order`, because the fusion downstream reads
///   *rank*: a list returned slightly out of distance order is a list whose
///   ranks are wrong, and RRF has no way to notice.
///
/// **The cost.** A tenant with fewer active edges than
/// [`search::CANDIDATE_DEPTH`] can no longer be answered out of forty index
/// tuples; its search walks the index until it is exhausted or 20,000 tuples
/// have been examined. That is a bounded amount of extra work, bought in
/// exchange for a recall bound that does not depend on other tenants' data —
/// and the thing it replaces was not cheaper, it was wrong.
///
/// Set per session rather than per statement because it has to hold for every
/// vector query the store runs, `node_name_vector_idx` as much as
/// `edge_proof_vector_idx`. `SET` on a not-yet-loaded extension's GUC is
/// accepted as a placeholder and applied when pgvector's library loads, so this
/// does not depend on the connection having touched a vector first.
const VECTOR_RECALL_SESSION_SETUP: &str =
    "SET hnsw.ef_search = 200; SET hnsw.iterative_scan = strict_order";

/// The `200` above is [`search::CANDIDATE_DEPTH`], and the two drifting apart
/// is a silent recall regression — the query would go on asking for a depth the
/// index had quietly stopped being able to supply. So it does not compile.
const _: () = assert!(
    search::CANDIDATE_DEPTH == 200,
    "hnsw.ef_search in VECTOR_RECALL_SESSION_SETUP must be search::CANDIDATE_DEPTH"
);

/// The database handle every store operation hangs off.
#[derive(Debug, Clone)]
pub struct Store {
    pool: PgPool,
}

impl Store {
    /// Opens a pool against `database_url`, verifying the connection eagerly:
    /// a misconfigured URL should fail at startup, not on the first tool call.
    pub async fn connect(database_url: &str) -> Result<Self, StoreError> {
        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .acquire_timeout(ACQUIRE_TIMEOUT)
            // Applied to every connection the pool opens, not just the first.
            // A connection that missed it would answer searches with the
            // truncated candidate list [`VECTOR_RECALL_SESSION_SETUP`]
            // describes, and would answer them without complaining.
            .after_connect(|connection, _metadata| {
                Box::pin(async move {
                    connection.execute(VECTOR_RECALL_SESSION_SETUP).await?;
                    Ok(())
                })
            })
            .connect(database_url)
            .await
            .map_err(|source| StoreError::Connect { source })?;

        Ok(Self { pool })
    }

    /// Applies any migrations the database has not seen.
    ///
    /// The migration set is embedded in the binary by `sqlx::migrate!`, so a
    /// deployment cannot drift from the schema its code was built against.
    pub async fn migrate(&self) -> Result<(), StoreError> {
        sqlx::migrate!("./migrations").run(&self.pool).await?;
        Ok(())
    }

    /// Escape hatch for tests and for later store modules that need to compose
    /// their own queries.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

/// Who asked for a write.
///
/// Taken from the originating request's `_meta` clientInfo and the validated
/// credential, never from tool arguments. Both halves are optional because a
/// client may decline to identify itself, and a write from an anonymous client
/// is still a write worth keeping — recorded as unattributed rather than
/// refused or, worse, attributed to whoever wrote last.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Provenance {
    pub client: Option<String>,
    pub principal: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The boundary, from both sides. A cap that refuses the batch it
    /// advertises costs a caller a round trip to discover.
    #[test]
    fn the_batch_cap_admits_its_own_limit_and_refuses_one_more() {
        assert!(reject_oversized_batch("nodes", MAX_BATCH_ITEMS).is_ok());
        assert!(reject_oversized_batch("nodes", 0).is_ok());

        let error = reject_oversized_batch("nodes", MAX_BATCH_ITEMS + 1)
            .expect_err("one past the cap is past the cap");

        assert!(
            matches!(
                error,
                StoreError::BatchTooLarge { argument: "nodes", received, limit }
                    if received == MAX_BATCH_ITEMS + 1 && limit == MAX_BATCH_ITEMS
            ),
            "a caller told only `too many` cannot decide how to split, got {error:?}"
        );
    }

    /// Characters, not bytes. A byte budget would give a name written in
    /// Japanese a third of the name an English one gets, for a limit that
    /// exists because of tokens.
    #[test]
    fn a_text_budget_is_spent_in_characters_rather_than_bytes() {
        let at_the_cap = "東".repeat(MAX_DISPLAY_NAME_CHARS);
        assert_eq!(
            at_the_cap.len(),
            MAX_DISPLAY_NAME_CHARS * 3,
            "three bytes each, so a byte budget would have refused this"
        );
        assert!(
            reject_overlong_text("display_name", 0, &at_the_cap, MAX_DISPLAY_NAME_CHARS).is_ok()
        );

        let error = reject_overlong_text(
            "display_name",
            2,
            &"東".repeat(MAX_DISPLAY_NAME_CHARS + 1),
            MAX_DISPLAY_NAME_CHARS,
        )
        .expect_err("one character past the cap is past the cap");

        assert!(
            matches!(
                error,
                StoreError::TextTooLong { index: 2, field: "display_name", chars, limit }
                    if chars == MAX_DISPLAY_NAME_CHARS + 1 && limit == MAX_DISPLAY_NAME_CHARS
            ),
            "the refusal names the item, the field and both numbers, got {error:?}"
        );
    }

    /// Trimmed before it is counted, so whitespace cannot stand in for
    /// evidence — including the whitespace a `minLength` in the schema would
    /// have happily counted.
    #[test]
    fn a_proof_is_measured_on_what_is_left_after_trimming() {
        for empty in ["", " ", "  \t \n ", "\u{3000}\u{3000}\u{3000}\u{3000}"] {
            assert!(
                matches!(
                    reject_unevidenced_proof(0, empty),
                    Err(StoreError::UnevidencedProofExcerpt { chars: 0, .. })
                ),
                "{empty:?} is not evidence"
            );
        }

        assert!(reject_unevidenced_proof(0, "ok").is_err(), "two characters");
        assert!(
            reject_unevidenced_proof(0, "Yes").is_ok(),
            "three characters"
        );
        assert!(
            reject_unevidenced_proof(0, "  Yes  ").is_ok(),
            "padding neither helps nor hurts a real quotation"
        );
    }
}
