//! Agent feedback on the facts search returned: ratings, the salience they
//! move, and the dispute quorum.
//!
//! Two things about this path are structural rather than incidental.
//!
//! **`edge_rating` is the signal; `salience_base` is a cache.** Every rating is
//! appended, including the ones that move no column, because salience is
//! derived from the raw ratings under a decay model this server may well
//! change — and a rating discarded for being redundant under today's model
//! cannot be recovered under tomorrow's. The quorum reads the same log for the
//! same reason: it is a question about the whole history of objections.
//!
//! **Nothing here archives, supersedes or deletes.** `status` appears in no
//! statement in this module. Everything in `proof_excerpt` is text written by
//! other agents, so a path from sustained negative ratings to archival would
//! hand prompt injection an *erasure* primitive: an archived edge leaves the
//! vector index, so it stops being returned, therefore stops being ratable, and
//! can never recover — while the agent's failure mode becomes confidently not
//! knowing something true it was told. A poisoned fact is at least visible and
//! carries provenance an operator can audit. So disputes are soft and
//! advisory, and hard archival is an operator action with no code path here.

use uuid::Uuid;

use super::search::SALIENCE_HALF_LIFE_DAYS;
use super::{Store, reject_oversized_batch};
use crate::error::StoreError;

/// How much of the salience scale one full point of utility is worth.
const SALIENCE_DELTA_PER_POINT: f32 = 0.25;

/// What happened to one edge id.
#[derive(Debug, Clone, PartialEq)]
pub enum RatingOutcome {
    /// The rating was appended and the edge's salience moved to `salience`.
    Rated { edge_id: Uuid, salience: f32 },
    /// Nothing was recorded for this id. Its siblings in the batch are
    /// unaffected.
    Rejected {
        edge_id: Uuid,
        reason: RatingRejection,
    },
}

/// Why one id was not rated.
///
/// One variant, deliberately. This tenant does not have that edge is the whole
/// of what a caller may learn: a separate "belongs to someone else" would
/// answer "does this uuid exist anywhere in the database" for any id a caller
/// cared to try, which is a cross-tenant oracle assembled out of an error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatingRejection {
    /// No edge with this id in this tenant.
    UnknownEdge,
}

impl Store {
    /// Records `utility_score` against each edge, in input order.
    ///
    /// Each id gets its own transaction, the way the edge writer's batch does:
    /// an id this tenant does not have fails that item and leaves its siblings
    /// committed, because an agent that carried one stale `edge_id` out of five
    /// has still read four facts worth rating.
    ///
    /// `rated_by` is the principal from the validated credential, never a tool
    /// argument. It is the only thing the dispute quorum counts, so a caller
    /// that could name its own principal could name two and dispute any fact it
    /// can see.
    ///
    /// Salience is *rebased* before the delta is applied — decayed to the
    /// present, then moved — because `salience_base` is the value as of
    /// `salience_updated_at` rather than the value now. Adding to it directly
    /// would hand back everything decay had taken, and a fact nobody has
    /// touched in a year would outrank this morning's on one lukewarm rating.
    pub async fn rate_edges(
        &self,
        tenant: &str,
        edge_ids: &[Uuid],
        utility_score: f32,
        rated_by: Option<&str>,
    ) -> Result<Vec<RatingOutcome>, StoreError> {
        // Checked before anything is written, the way the node upsert checks a
        // blank display name: the score applies to the whole batch, so a bad
        // one is decidable from the argument and finding it halfway through
        // would leave earlier edges rated under a call that reports failure.
        //
        // Nothing here is embedded, so the cost of an over-long `edge_ids` is
        // not the model but the loop below: one transaction and one row lock
        // per id, all of them on a single pooled connection this request holds
        // for the duration. Sixteen such calls is the whole pool.
        reject_oversized_batch("edge_ids", edge_ids.len())?;
        reject_score_off_the_scale(utility_score)?;

        let rated_by = attributable(rated_by);

        let mut outcomes = Vec::with_capacity(edge_ids.len());

        for &edge_id in edge_ids {
            let mut transaction = self.pool.begin().await?;

            // `FOR NO KEY UPDATE` is what makes the read-modify-write below
            // atomic. Without it, two agents rating the same edge at the same
            // moment both read the same salience under read-committed and both
            // write their own answer over it, so one rating is discarded with
            // no error and nothing in any table to say it happened — which, for
            // a feedback loop whose whole input is concurrent agents, is the
            // one failure mode that cannot be detected after the fact.
            //
            // `NO KEY` rather than the stronger `FOR UPDATE`, for the reason
            // the edge writer's source-node lock gives: `FOR UPDATE` conflicts
            // with the `FOR KEY SHARE` Postgres takes on referenced rows, and
            // this row is the target of `superseded_by_edge_id`.
            let current: Option<(f32,)> = sqlx::query_as(
                "SELECT effective_salience(salience_base, salience_updated_at, $3, now()) \
                 FROM edge WHERE tenant_id = $1 AND edge_id = $2 \
                 FOR NO KEY UPDATE",
            )
            .bind(tenant)
            .bind(edge_id)
            .bind(SALIENCE_HALF_LIFE_DAYS)
            .fetch_optional(&mut *transaction)
            .await?;

            // Scoped by `tenant_id`, so an edge another tenant owns reads
            // exactly like an id that exists nowhere — which is all a caller
            // may learn, and all this reports.
            let Some((current,)) = current else {
                transaction.rollback().await?;
                outcomes.push(RatingOutcome::Rejected {
                    edge_id,
                    reason: RatingRejection::UnknownEdge,
                });
                continue;
            };

            sqlx::query(
                "INSERT INTO edge_rating (tenant_id, edge_id, utility_score, rated_by) \
                 VALUES ($1, $2, $3, $4)",
            )
            .bind(tenant)
            .bind(edge_id)
            .bind(utility_score)
            .bind(rated_by)
            .execute(&mut *transaction)
            .await?;

            let salience = adjusted_salience(current, utility_score);

            sqlx::query(
                "UPDATE edge SET salience_base = $3, salience_updated_at = now() \
                 WHERE tenant_id = $1 AND edge_id = $2",
            )
            .bind(tenant)
            .bind(edge_id)
            .bind(salience)
            .execute(&mut *transaction)
            .await?;

            // Only a negative rating can carry an edge over the quorum, and
            // nothing here ever clears a dispute: a positive rating raises
            // salience, which is how a contested fact recovers its rank, but
            // the objections two principals raised are still on the record.
            if utility_score < 0.0 && quorum_reached(&mut transaction, tenant, edge_id).await? {
                dispute(&mut transaction, tenant, edge_id).await?;
            }

            transaction.commit().await?;

            outcomes.push(RatingOutcome::Rated { edge_id, salience });
        }

        Ok(outcomes)
    }
}

/// The principal a rating can be attributed to, if there is one.
///
/// A blank is not a name, so it is recorded as NULL — the column's own way of
/// saying nobody. An auth mode with no principal to report may hand over an
/// empty string rather than nothing at all, and both mean the same thing about
/// attribution; kept as given, `""` and `"   "` are two distinct strings and
/// therefore a quorum of two, which is one anonymous caller disputing whatever
/// it likes on its own say-so.
///
/// Whitespace is only *inspected*, never stripped: `rated_by` is an identifier
/// from a credential, and rewriting it would break the exact match an operator
/// auditing "everything this principal rated" depends on.
fn attributable(rated_by: Option<&str>) -> Option<&str> {
    rated_by.filter(|principal| !principal.trim().is_empty())
}

/// How many distinct principals have to object before an edge is disputed.
///
/// Two, and they have to be *distinct*: an agent runs in a loop, so whatever
/// made it rate a fact badly once will make it rate the same fact badly on
/// every pass, and a count of ratings would be a count of retries. Two
/// independent principals is the smallest number that says something about the
/// fact rather than about one agent.
const DISPUTE_QUORUM: i64 = 2;

/// Whether enough distinct principals have rated this edge negatively.
///
/// Counted from `edge_rating` rather than from a tally kept on the edge, which
/// is the reason that table is append-only: the quorum is a question about the
/// whole history of objections, and the rating just inserted is part of it.
///
/// `count(DISTINCT rated_by)` ignores NULLs — Postgres never counts a NULL in a
/// column count — and the `IS NOT NULL` beside it is there to say so out loud.
/// It is load-bearing: an unattributed rating is one this server cannot tie to
/// anybody, so however many of them arrive, they are not a crowd. Without it an
/// anonymous caller could dispute any fact it could name.
async fn quorum_reached(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    edge_id: Uuid,
) -> Result<bool, StoreError> {
    let (principals,): (i64,) = sqlx::query_as(
        "SELECT count(DISTINCT rated_by) FROM edge_rating \
         WHERE tenant_id = $1 AND edge_id = $2 \
           AND utility_score < 0 AND rated_by IS NOT NULL",
    )
    .bind(tenant)
    .bind(edge_id)
    .fetch_one(connection)
    .await?;

    Ok(principals >= DISPUTE_QUORUM)
}

/// Flags an edge as disputed, if it is not flagged already.
///
/// `disputed_at IS NULL` in the predicate rather than a blind `SET`: the column
/// records when the graph first came to doubt the fact, so restamping it on
/// every later objection would turn an audit trail into a "most recently
/// complained about" clock.
///
/// `status` is deliberately absent from this statement, and from every other
/// statement in this module. Archiving on sustained negative ratings is the one
/// change that would make this path dangerous: proof text is untrusted, so an
/// agent-reachable archive is an erasure primitive for whoever wrote it, and an
/// erased fact is invisible in search output while a disputed one is returned,
/// flagged, and auditable. Hard archival is an operator action.
async fn dispute(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    edge_id: Uuid,
) -> Result<(), StoreError> {
    sqlx::query(
        "UPDATE edge SET disputed_at = now() \
         WHERE tenant_id = $1 AND edge_id = $2 AND disputed_at IS NULL",
    )
    .bind(tenant)
    .bind(edge_id)
    .execute(connection)
    .await?;

    Ok(())
}

/// Refuses a score that is not on the utility scale.
///
/// Written as a range containment rather than as `score < -1.0 || score > 1.0`,
/// which is the same test for every value except NaN: both comparisons against
/// NaN are false, so the hand-written form lets it through, and a NaN reaching
/// `adjusted_salience` clamps to nothing — `f32::clamp` returns NaN for a NaN
/// input — leaving a column no later rating can move back.
fn reject_score_off_the_scale(utility_score: f32) -> Result<(), StoreError> {
    if !(-1.0..=1.0).contains(&utility_score) {
        return Err(StoreError::UtilityScoreOutOfRange { utility_score });
    }

    Ok(())
}

/// The salience an edge holding `current` takes after one rating.
///
/// Clamped here rather than left to the column's `CHECK`, which does not clamp:
/// it aborts the statement, taking the append-only rating down with it.
fn adjusted_salience(current: f32, utility_score: f32) -> f32 {
    (current + utility_score * SALIENCE_DELTA_PER_POINT).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both ends of the scale are on it. `-1.0` is the score an agent gives a
    /// fact that misled it and `1.0` the one it gives a fact it could not have
    /// answered without, so an exclusive bound would refuse the two ratings the
    /// scale exists to express.
    #[test]
    fn the_ends_of_the_scale_are_valid_scores() {
        assert!(reject_score_off_the_scale(-1.0).is_ok());
        assert!(reject_score_off_the_scale(1.0).is_ok());
        assert!(reject_score_off_the_scale(0.0).is_ok());
    }

    /// Anything off the scale is refused, and the error carries the score so a
    /// caller is told what it sent rather than that something was wrong.
    #[test]
    fn a_score_off_the_scale_is_refused_and_named() {
        let error = reject_score_off_the_scale(1.5).expect_err("1.5 is off the scale");

        assert!(
            matches!(
                error,
                StoreError::UtilityScoreOutOfRange { utility_score } if utility_score == 1.5
            ),
            "expected the score to be named, got {error:?}"
        );
        assert!(reject_score_off_the_scale(-1.5).is_err());
    }

    /// NaN is the case the obvious range check gets wrong: `score < -1.0` and
    /// `score > 1.0` are *both* false for it, so a hand-written check passes it
    /// through — and NaN survives `clamp` untouched, so it would land in
    /// `salience_base` as a value no later rating could move.
    #[test]
    fn a_score_that_is_not_a_number_is_refused() {
        assert!(reject_score_off_the_scale(f32::NAN).is_err());
        assert!(reject_score_off_the_scale(f32::INFINITY).is_err());
        assert!(reject_score_off_the_scale(f32::NEG_INFINITY).is_err());
    }

    /// A full point of utility is worth a quarter of the scale, in both
    /// directions.
    #[test]
    fn a_full_point_of_utility_moves_a_quarter_of_the_scale() {
        assert!((adjusted_salience(0.5, 1.0) - 0.75).abs() < f32::EPSILON);
        assert!((adjusted_salience(0.5, -1.0) - 0.25).abs() < f32::EPSILON);
        assert!((adjusted_salience(0.5, 0.4) - 0.6).abs() < 1e-6);
    }

    /// The column is `CHECK`-constrained to `[0, 1]`, and the constraint does
    /// not clamp — it aborts the statement, taking the rating that caused it
    /// down with it. So both ends are clamped here.
    #[test]
    fn an_adjustment_never_leaves_the_columns_range() {
        assert!((adjusted_salience(0.9, 1.0) - 1.0).abs() < f32::EPSILON);
        assert!(adjusted_salience(0.1, -1.0).abs() < f32::EPSILON);
    }

    /// Zero is a floor, not a tombstone: search never filters on salience, so
    /// an edge at 0.0 is still returned and one positive rating brings it back.
    /// Decay and disputes must not be one-way doors.
    #[test]
    fn an_edge_at_the_floor_can_still_be_rated_back_up() {
        assert!((adjusted_salience(0.0, 1.0) - 0.25).abs() < f32::EPSILON);
    }

    /// A blank is not a name. Every spelling of one collapses to the same
    /// nobody, or an anonymous caller could reach the dispute quorum alone.
    #[test]
    fn every_spelling_of_a_blank_principal_is_nobody() {
        assert_eq!(attributable(None), None);
        assert_eq!(attributable(Some("")), None);
        assert_eq!(attributable(Some("   ")), None);
        assert_eq!(attributable(Some("\t\n")), None);
    }

    /// A real principal is passed through exactly as the credential gave it:
    /// it is an identifier, and rewriting it would break the exact match an
    /// operator auditing one principal's ratings depends on.
    #[test]
    fn a_real_principal_is_kept_verbatim() {
        assert_eq!(attributable(Some("agent-a")), Some("agent-a"));
        assert_eq!(attributable(Some(" agent-a ")), Some(" agent-a "));
    }
}
