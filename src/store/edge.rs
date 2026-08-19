//! Edge creation.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::node::{NodeSuggestion, follow_merges, suggest_nodes};
use super::relationship::{CoercedType, coerce_relationship_type};
use super::{
    MAX_PROOF_EXCERPT_CHARS, Provenance, Store, reject_overlong_text, reject_oversized_batch,
    reject_unevidenced_proof,
};
use crate::embed::Embedder;
use crate::error::StoreError;

/// One fact as an agent asserted it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeInput {
    pub source_node_id: String,
    pub target_node_id: String,
    /// Exactly as the caller wrote it. Coercion happens server-side.
    pub relationship_type: String,
    pub proof_excerpt: String,
    /// When the fact began to hold in the world. Defaults to now.
    pub valid_from: Option<DateTime<Utc>>,
}

impl EdgeInput {
    pub fn new(
        source_node_id: impl Into<String>,
        target_node_id: impl Into<String>,
        relationship_type: impl Into<String>,
        proof_excerpt: impl Into<String>,
    ) -> Self {
        Self {
            source_node_id: source_node_id.into(),
            target_node_id: target_node_id.into(),
            relationship_type: relationship_type.into(),
            proof_excerpt: proof_excerpt.into(),
            valid_from: None,
        }
    }

    pub fn with_valid_from(
        source_node_id: impl Into<String>,
        target_node_id: impl Into<String>,
        relationship_type: impl Into<String>,
        proof_excerpt: impl Into<String>,
        valid_from: DateTime<Utc>,
    ) -> Self {
        Self {
            valid_from: Some(valid_from),
            ..Self::new(
                source_node_id,
                target_node_id,
                relationship_type,
                proof_excerpt,
            )
        }
    }
}

/// What happened to one input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeOutcome {
    /// Written, and current.
    Created {
        edge_id: Uuid,
        /// The edge this one retired, when the type is `one`-cardinality.
        superseded_edge_id: Option<Uuid>,
    },
    /// Written as history that is already closed: an older fact that arrived
    /// after the fact holding from a later instant.
    Backfilled {
        edge_id: Uuid,
        /// The still-active edge that closes this one.
        superseded_by_edge_id: Uuid,
        valid_until: DateTime<Utc>,
    },
    /// Nothing was written for this input. Its siblings in the batch are
    /// unaffected.
    Rejected { reason: EdgeRejection },
}

/// Why one input was not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeRejection {
    /// One end of the edge names a node this tenant does not have.
    UnknownEndpoint {
        endpoint: Endpoint,
        /// The id exactly as the caller wrote it.
        node_id: String,
        /// Ids the caller may have meant, so a model can correct itself instead
        /// of sending the same batch again.
        suggestions: Vec<NodeSuggestion>,
    },
}

/// Which end of an edge a rejection is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    Source,
    Target,
}

impl Store {
    /// Writes each input as an edge, in input order.
    pub async fn create_edges(
        &self,
        tenant: &str,
        inputs: &[EdgeInput],
        embedder: &dyn Embedder,
        provenance: &Provenance,
    ) -> Result<Vec<EdgeOutcome>, StoreError> {
        // Before the clock is even read: this one needs nothing but the
        // argument's length, and the whole point of a batch cap is to refuse
        // the call before it costs anything. Everything below — a round trip
        // for the clock, a type resolution per item, one forward pass over
        // every proof — scales with `inputs`.
        reject_oversized_batch("relationships", inputs.len())?;

        // Both halves of the proof contract, for the whole batch, before
        // anything is embedded or written — the way this module already checks
        // `valid_from` and the way the node upsert checks a blank display name.
        // Neither needs the database, so neither may be discovered halfway
        // through a batch that has already committed some of its siblings.
        for (index, input) in inputs.iter().enumerate() {
            reject_overlong_text(
                "proof_excerpt",
                index,
                &input.proof_excerpt,
                MAX_PROOF_EXCERPT_CHARS,
            )?;
            reject_unevidenced_proof(index, &input.proof_excerpt)?;
        }

        // The database clock, not this process's. `valid_from` orders
        // supersession across every instance behind the load balancer, so the
        // comparison has to be against one shared clock: two instances whose
        // NTP has drifted apart must not disagree about which of two facts is
        // the later one.
        let server_now = server_now(&self.pool).await?;

        // Checked for the whole batch before anything is embedded or written,
        // the way the node upsert checks its display names. It needs no more
        // than the argument and the clock, so a batch containing one is refused
        // whole rather than half-committed — and unlike an unknown endpoint,
        // which can be an honest race with another agent's node write, a
        // future-dated fact is never a legitimate write this server lost a race
        // to.
        reject_future_valid_from(inputs, server_now)?;

        // Every type in the batch, resolved before any edge is written, for the
        // same reason: a type that folds to nothing is a caller bug decidable
        // from the argument, and finding it halfway through would leave the
        // earlier items committed under a call that reports failure.
        let types = resolve_types(&self.pool, tenant, inputs, embedder).await?;

        let proofs: Vec<String> = inputs
            .iter()
            .map(|input| input.proof_excerpt.clone())
            .collect();

        // One forward pass for the whole batch. A proof excerpt is stored text,
        // so it goes through the document side of the embedder; `embed_query`
        // is for search strings and applies the model's query instruction.
        let vectors = embedder.embed_documents(&proofs).await?;

        let mut outcomes = Vec::with_capacity(inputs.len());

        for ((input, vector), resolved) in inputs.iter().zip(vectors).zip(&types) {
            // Resolved before the write rather than left to the column default,
            // because the placement decision below compares it against the
            // incumbent's and cannot wait for the row to exist.
            let valid_from = input.valid_from.unwrap_or(server_now);

            // One transaction per item, not one for the batch. An unknown
            // endpoint has to fail its own item and leave its siblings
            // committed, and a rolled-back savepoint inside a shared
            // transaction would still be lost if a later item failed outright.
            // It also bounds how long the source-node lock below is held to the
            // one edge that needs it, and means a writer never holds two node
            // locks at once — so two batches touching the same pair of nodes in
            // opposite orders cannot deadlock against each other.
            let mut transaction = self.pool.begin().await?;

            let (source_node_id, target_node_id) =
                match resolve_endpoints(&mut transaction, tenant, input).await? {
                    Ok(endpoints) => endpoints,
                    Err(reason) => {
                        transaction.rollback().await?;
                        outcomes.push(EdgeOutcome::Rejected { reason });
                        continue;
                    }
                };

            let edge_id = Uuid::new_v4();

            // Only a `one`-cardinality type has an incumbent to displace. A
            // `many` type never supersedes: "A uses B" and "A uses C" are both
            // legitimately true at once.
            let incumbent = if resolved.single_valued {
                // Taken *before* the incumbent is read, and released only by
                // the commit below. Two agents asserting Alice's manager at the
                // same moment otherwise both read an empty incumbent — under
                // read-committed neither can see the other's uncommitted row —
                // and both insert as active, leaving the graph asserting two
                // managers for one person with no error anywhere.
                //
                // The database cannot express this as a constraint: a unique
                // index over `(tenant, source, type) WHERE status = 'active'`
                // would be wrong for every `many` type, and cardinality lives
                // in `relationship_type`, which an index on `edge` cannot
                // consult. So the exclusion is taken explicitly, on the one row
                // every writer for this source must pass through.
                lock_source_node(&mut transaction, tenant, &source_node_id).await?;

                active_edge(
                    &mut transaction,
                    tenant,
                    &source_node_id,
                    &resolved.coerced.canonical,
                )
                .await?
            } else {
                None
            };

            let placement = place(valid_from, &target_node_id, incumbent.as_ref());

            // `recorded_at` is deliberately absent from the column list: it is
            // when the server came to believe the fact, so it is the column
            // default and nothing a caller can reach.
            sqlx::query(
                "INSERT INTO edge \
                 (tenant_id, edge_id, source_node_id, target_node_id, relationship_type, \
                  relationship_type_tenant, raw_relationship_type, proof_excerpt, proof_vector, \
                  embedding_model, valid_from, valid_until, status, superseded_by_edge_id, \
                  writer_client, writer_principal) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::real[]::vector, $10, $11, $12, \
                         $13::edge_status, $14, $15, $16)",
            )
            .bind(tenant)
            .bind(edge_id)
            .bind(&source_node_id)
            .bind(&target_node_id)
            .bind(&resolved.coerced.canonical)
            .bind(&resolved.coerced.scope)
            .bind(&resolved.coerced.raw)
            .bind(&input.proof_excerpt)
            .bind(&vector)
            .bind(embedder.model_id())
            .bind(valid_from)
            .bind(placement.valid_until())
            .bind(placement.status())
            .bind(placement.successor())
            // From the originating request, never from tool arguments: a model
            // that could name its own writer could also name someone else's.
            .bind(provenance.client.as_deref())
            .bind(provenance.principal.as_deref())
            .execute(&mut *transaction)
            .await?;

            // After the insert, never before: `superseded_by_edge_id` is a
            // foreign key into this same table, so the row it points at has to
            // exist first.
            if let Placement::Current {
                displaced: Some(displaced),
            } = placement
            {
                supersede(&mut transaction, tenant, displaced, valid_from, edge_id).await?;
            }

            transaction.commit().await?;

            outcomes.push(placement.into_outcome(edge_id));
        }

        Ok(outcomes)
    }
}

/// Resolves both ends of an edge to nodes this tenant actually holds.
///
/// The outer `Result` is the database; the inner one is the caller. An unknown
/// endpoint is not a failure of this call, it is an answer about this item, and
/// the caller's siblings in the batch are unaffected by it.
///
/// The source is checked first, and one rejection comes back rather than two.
/// An agent that got both ends wrong has a bigger problem than the round trip.
async fn resolve_endpoints(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    input: &EdgeInput,
) -> Result<Result<(String, String), EdgeRejection>, StoreError> {
    let source = match resolve_endpoint(connection, tenant, Endpoint::Source, &input.source_node_id)
        .await?
    {
        Ok(source) => source,
        Err(rejection) => return Ok(Err(rejection)),
    };

    let target = match resolve_endpoint(connection, tenant, Endpoint::Target, &input.target_node_id)
        .await?
    {
        Ok(target) => target,
        Err(rejection) => return Ok(Err(rejection)),
    };

    Ok(Ok((source, target)))
}

/// The live node one endpoint names, or what to tell the caller instead.
///
/// Merges are followed rather than merely tolerated. Agents hold `node_id`s
/// across stateless calls, so an id this server handed out months ago may since
/// have been merged away; writing the edge against the tombstone would hang the
/// fact off a node that entity resolution has already retired, splitting one
/// entity's facts across two rows with nothing to say they belong together.
async fn resolve_endpoint(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    endpoint: Endpoint,
    named: &str,
) -> Result<Result<String, EdgeRejection>, StoreError> {
    if let Some(resolved) = follow_merges(connection, tenant, named).await? {
        return Ok(Ok(resolved));
    }

    Ok(Err(EdgeRejection::UnknownEndpoint {
        endpoint,
        node_id: named.to_string(),
        suggestions: suggest_nodes(connection, tenant, named).await?,
    }))
}

/// How far ahead of the server's clock a caller's `valid_from` may sit.
///
/// This is a tolerance for clock skew between an agent's host and this
/// database, not a window in which post-dating a fact is allowed: a minute of
/// drift between two machines is ordinary, and refusing it would fail honest
/// writes for a reason no agent could diagnose. Anything beyond it is refused
/// outright rather than clamped — see [`reject_future_valid_from`].
const FUTURE_VALID_FROM_TOLERANCE_SECONDS: i64 = 60;

/// Refuses any input dated meaningfully ahead of the server's clock.
///
/// Refusal rather than clamping, because `valid_from` is the key supersession
/// orders on. A fact dated next year outranks every fact written between now
/// and then, so it does not merely arrive wrong, it stays current and cannot be
/// displaced by anything true. And the argument is filled in by a model that has
/// just read `proof_excerpt` text some other agent wrote, which makes it
/// attacker-influenced input rather than a field an operator typed.
///
/// Clamping would accept the write and quietly rewrite it, leaving the attempt
/// indistinguishable from an honest write in every column of the table.
/// Refusing puts it in front of the caller, who is the only party that can tell
/// a mistyped year from a deliberate one.
fn reject_future_valid_from(
    inputs: &[EdgeInput],
    server_now: DateTime<Utc>,
) -> Result<(), StoreError> {
    let horizon = server_now + chrono::Duration::seconds(FUTURE_VALID_FROM_TOLERANCE_SECONDS);

    for (index, input) in inputs.iter().enumerate() {
        if let Some(valid_from) = input.valid_from
            && valid_from > horizon
        {
            return Err(StoreError::FutureValidFrom {
                index,
                valid_from,
                tolerance_seconds: FUTURE_VALID_FROM_TOLERANCE_SECONDS,
            });
        }
    }

    Ok(())
}

/// The current fact for one `(source, type)`, as far as supersession cares.
///
/// Visible to `merge`, which has to re-run this module's contest over a set of
/// edges a merge has just landed on one source node. See [`place`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ActiveEdge {
    pub(super) edge_id: Uuid,
    pub(super) valid_from: DateTime<Utc>,
    pub(super) target_node_id: String,
}

/// Where an incoming fact lands relative to the active edge it contends with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Placement {
    /// It is the current fact. `displaced` is the edge it retired, if any.
    Current { displaced: Option<Uuid> },
    /// It is history that was already closed when it arrived, because a fact
    /// holding from a later instant is already recorded.
    Closed {
        successor: Uuid,
        valid_until: DateTime<Utc>,
    },
}

impl Placement {
    fn status(self) -> &'static str {
        match self {
            Self::Current { .. } => "active",
            Self::Closed { .. } => "superseded",
        }
    }

    fn valid_until(self) -> Option<DateTime<Utc>> {
        match self {
            Self::Current { .. } => None,
            Self::Closed { valid_until, .. } => Some(valid_until),
        }
    }

    fn successor(self) -> Option<Uuid> {
        match self {
            Self::Current { .. } => None,
            Self::Closed { successor, .. } => Some(successor),
        }
    }

    fn into_outcome(self, edge_id: Uuid) -> EdgeOutcome {
        match self {
            Self::Current { displaced } => EdgeOutcome::Created {
                edge_id,
                superseded_edge_id: displaced,
            },
            Self::Closed {
                successor,
                valid_until,
            } => EdgeOutcome::Backfilled {
                edge_id,
                superseded_by_edge_id: successor,
                valid_until,
            },
        }
    }
}

/// Decides where an incoming `one`-cardinality fact belongs, by `valid_from`
/// and — only when that ties — by target.
///
/// Arrival order is deliberately not an input. Writes reach this server out of
/// order — it is stateless behind a round-robin load balancer with many agents
/// writing at once — so an agent backfilling "Alice reported to Bob in 2019" can
/// arrive long after the fact that Alice now reports to Carol. Ordering by
/// arrival would let that backfill retire current truth and leave the graph
/// asserting a 2019 fact as the present one, with a `proof_excerpt` that reads
/// perfectly and survives an audit.
///
/// **The tie.** Two facts claiming the same instant cannot both be current, and
/// `valid_from` has nothing left to say about them. The tie is broken on
/// `target_node_id`, greatest first. Which target that favours is arbitrary —
/// there is no fact of the matter — but it is a pure function of the two rows,
/// so every replica and every replay of the same two writes reaches the same
/// answer. Breaking the tie by arrival instead would make the graph's current
/// truth a property of which replica the load balancer picked.
///
/// A tie on the target too is the same fact asserted twice: the incumbent
/// stands, and the newcomer is recorded as a zero-length interval, keeping its
/// proof and its writer without claiming it was ever the current fact.
///
/// **Known limit.** A closed row is closed against the *active* edge, so two
/// backfills into the same gap — 2019 and then 2021, under a 2024 incumbent —
/// both record `valid_until = 2024` and their intervals overlap. Only the
/// history chain is affected; the current fact is right in every case. Closing
/// each backfill against its true successor means rewriting rows that are
/// already history, which is a repair pass over an interval chain rather than
/// something a single write should attempt on its way past.
pub(super) fn place(
    valid_from: DateTime<Utc>,
    target_node_id: &str,
    incumbent: Option<&ActiveEdge>,
) -> Placement {
    let Some(incumbent) = incumbent else {
        return Placement::Current { displaced: None };
    };

    let displaces =
        (valid_from, target_node_id) > (incumbent.valid_from, incumbent.target_node_id.as_str());

    if displaces {
        return Placement::Current {
            displaced: Some(incumbent.edge_id),
        };
    }

    Placement::Closed {
        successor: incumbent.edge_id,
        valid_until: incumbent.valid_from,
    }
}

/// The database's clock.
///
/// Read once per batch rather than per row, so every item in one call defaults
/// to the same instant and two items that both default cannot be ordered by how
/// long the batch took to process.
async fn server_now(pool: &sqlx::PgPool) -> Result<DateTime<Utc>, StoreError> {
    let (now,): (DateTime<Utc>,) = sqlx::query_as("SELECT now()").fetch_one(pool).await?;

    Ok(now)
}

/// One input's relationship type, resolved against the vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedType {
    coerced: CoercedType,
    /// `one` cardinality: at most one active edge per `(source, type)`.
    single_valued: bool,
}

/// Resolves every input's relationship type, in input order, before any edge is
/// written.
///
/// Two reasons this is a phase of its own rather than a step inside each item's
/// transaction. It refuses a type that folds to nothing — a caller bug, and
/// decidable from the argument — while the batch is still untouched, so a bad
/// item costs the caller a rejection instead of leaving half its batch
/// committed under a call that reported failure. And coercion is the expensive
/// part of an edge write: a batch that says `USES` five times embeds and looks
/// it up once here.
///
/// The vocabulary it may register into is the caller's own; coercion enforces
/// that. Registering a type for an item that is later rejected for an unknown
/// endpoint is harmless — the type was coined by intent, and the retry would
/// register it anyway.
async fn resolve_types(
    pool: &sqlx::PgPool,
    tenant: &str,
    inputs: &[EdgeInput],
    embedder: &dyn Embedder,
) -> Result<Vec<ResolvedType>, StoreError> {
    let mut transaction = pool.begin().await?;
    let mut resolved: Vec<ResolvedType> = Vec::with_capacity(inputs.len());
    // Keyed on the raw string, so the cache can only ever hand back what
    // coercion would have returned for that exact input. A batch is small
    // enough that a linear scan beats a hash map.
    let mut seen: Vec<(&str, ResolvedType)> = Vec::new();

    for input in inputs {
        let raw = input.relationship_type.as_str();

        if let Some((_, hit)) = seen.iter().find(|(cached, _)| *cached == raw) {
            resolved.push(hit.clone());
            continue;
        }

        let coerced = coerce_relationship_type(&mut transaction, tenant, raw, embedder).await?;
        let single_valued = is_single_valued(&mut transaction, &coerced).await?;
        let entry = ResolvedType {
            coerced,
            single_valued,
        };

        seen.push((raw, entry.clone()));
        resolved.push(entry);
    }

    transaction.commit().await?;

    Ok(resolved)
}

/// Whether this type admits at most one active edge per `(source, type)`.
///
/// Cardinality lives in `relationship_type`, keyed by `(scope, name)` — a name
/// alone stopped identifying a row when the vocabulary gained a tenant scope,
/// so the scope coercion resolved to is part of the lookup.
async fn is_single_valued(
    connection: &mut sqlx::PgConnection,
    coerced: &CoercedType,
) -> Result<bool, StoreError> {
    let (cardinality,): (String,) = sqlx::query_as(
        "SELECT cardinality::text FROM relationship_type WHERE tenant_id = $1 AND name = $2",
    )
    .bind(&coerced.scope)
    .bind(&coerced.canonical)
    .fetch_one(connection)
    .await?;

    Ok(cardinality == "one")
}

/// Serialises every `one`-cardinality writer for one source node.
///
/// `FOR NO KEY UPDATE` rather than `FOR UPDATE`, which is the stronger lock and
/// the wrong one here. Two writers taking `FOR NO KEY UPDATE` on the same row
/// still exclude each other, which is the whole requirement. But `FOR UPDATE`
/// also conflicts with `FOR KEY SHARE`, and `FOR KEY SHARE` is exactly what
/// Postgres takes on both endpoint rows when it checks the foreign keys of an
/// `edge` insert. So under `FOR UPDATE`, one agent writing `alice -> bob` while
/// another writes `bob -> alice` — reciprocal facts, an ordinary thing for two
/// agents to learn at once — would each hold the lock the other's insert needs,
/// and Postgres would break the deadlock by killing one of them. The weaker
/// lock keeps the mutual exclusion and removes the deadlock.
///
/// The row is only read; nothing about the node changes. The lock is a rendezvous
/// point, and the node row is the one every writer for this source must pass.
async fn lock_source_node(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    source_node_id: &str,
) -> Result<(), StoreError> {
    sqlx::query(
        "SELECT node_id FROM node \
         WHERE tenant_id = $1 AND node_id = $2 \
         FOR NO KEY UPDATE",
    )
    .bind(tenant)
    .bind(source_node_id)
    .fetch_optional(connection)
    .await?;

    Ok(())
}

/// The active edge for one `(source, type)`, if there is one.
///
/// Keyed on the type's *name*, not on `(scope, name)`. A tenant that coins a
/// name the curated vocabulary later adopts would otherwise hold two types that
/// read identically to every agent and every operator, and two active edges
/// under them would be two contradictory answers to one question — "who does
/// Alice report to" — with nothing on screen to tell them apart. The name is
/// what a reader sees, so the name is what may only hold once.
async fn active_edge(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    source_node_id: &str,
    relationship_type: &str,
) -> Result<Option<ActiveEdge>, StoreError> {
    // Ordered by the same key `place` compares on, so that if a tenant somehow
    // holds two active edges for one `(source, type)` — a state this path never
    // creates, but an operator or a future migration could — the one that
    // answers here is the one that would have won, rather than whichever the
    // planner reached first.
    let row: Option<(Uuid, DateTime<Utc>, String)> = sqlx::query_as(
        "SELECT edge_id, valid_from, target_node_id FROM edge \
         WHERE tenant_id = $1 AND source_node_id = $2 AND relationship_type = $3 \
           AND status = 'active' \
         ORDER BY valid_from DESC, target_node_id DESC, edge_id \
         LIMIT 1",
    )
    .bind(tenant)
    .bind(source_node_id)
    .bind(relationship_type)
    .fetch_optional(connection)
    .await?;

    Ok(row.map(|(edge_id, valid_from, target_node_id)| ActiveEdge {
        edge_id,
        valid_from,
        target_node_id,
    }))
}

/// Retires `edge_id`, recording when it stopped holding and what replaced it.
///
/// All three columns move together. An edge marked `superseded` with no
/// `valid_until` is a fact that was true until an unspecified moment, which is
/// not a bi-temporal record of anything, and one with no `superseded_by_edge_id`
/// cannot be audited back to the write that retired it.
pub(super) async fn supersede(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    edge_id: Uuid,
    valid_until: DateTime<Utc>,
    successor: Uuid,
) -> Result<(), StoreError> {
    sqlx::query(
        "UPDATE edge \
         SET status = 'superseded', valid_until = $3, superseded_by_edge_id = $4 \
         WHERE tenant_id = $1 AND edge_id = $2",
    )
    .bind(tenant)
    .bind(edge_id)
    .bind(valid_until)
    .bind(successor)
    .execute(connection)
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instant(year: i32, month: u32, day: u32) -> DateTime<Utc> {
        use chrono::TimeZone;

        Utc.with_ymd_and_hms(year, month, day, 12, 0, 0)
            .single()
            .expect("a valid civil timestamp")
    }

    fn incumbent(valid_from: DateTime<Utc>, target: &str) -> ActiveEdge {
        ActiveEdge {
            edge_id: Uuid::nil(),
            valid_from,
            target_node_id: target.to_string(),
        }
    }

    /// Nothing to contend with is the ordinary case, and it must not invent a
    /// supersession to report.
    #[test]
    fn a_fact_with_no_incumbent_is_simply_current() {
        assert_eq!(
            place(instant(2024, 6, 1), "carol", None),
            Placement::Current { displaced: None }
        );
    }

    /// The ordering key is `valid_from`, so a fact holding from a later instant
    /// displaces one holding from an earlier one.
    #[test]
    fn a_later_fact_displaces_the_incumbent() {
        assert_eq!(
            place(
                instant(2024, 6, 1),
                "carol",
                Some(&incumbent(instant(2019, 3, 1), "bob"))
            ),
            Placement::Current {
                displaced: Some(Uuid::nil())
            }
        );
    }

    /// The late-arriving backfill. It is history the moment it arrives, and the
    /// incumbent is not touched — see the module's integration tests, which
    /// assert that against a real table.
    #[test]
    fn an_earlier_fact_arrives_already_closed() {
        assert_eq!(
            place(
                instant(2019, 3, 1),
                "bob",
                Some(&incumbent(instant(2024, 6, 1), "carol"))
            ),
            Placement::Closed {
                successor: Uuid::nil(),
                valid_until: instant(2024, 6, 1),
            }
        );
    }

    /// The tie rule, stated in one place: same instant, greater target wins,
    /// and it wins whichever of the two arrived first. Both directions are
    /// asserted, because a rule that only holds one way round is arrival order
    /// wearing a comparison.
    #[test]
    fn an_exact_tie_is_broken_on_the_target_in_both_directions() {
        let claimed = instant(2024, 6, 1);

        assert_eq!(
            place(claimed, "carol", Some(&incumbent(claimed, "bob"))),
            Placement::Current {
                displaced: Some(Uuid::nil())
            },
            "carol > bob, so carol takes the current slot"
        );

        assert_eq!(
            place(claimed, "bob", Some(&incumbent(claimed, "carol"))),
            Placement::Closed {
                successor: Uuid::nil(),
                valid_until: claimed,
            },
            "and the same pair the other way round reaches the same answer"
        );
    }

    /// The same fact asserted twice. There is nothing to choose between them,
    /// so the newcomer becomes a zero-length interval rather than churning
    /// which row is current.
    #[test]
    fn a_tie_on_the_target_too_keeps_the_incumbent() {
        let claimed = instant(2024, 6, 1);

        assert_eq!(
            place(claimed, "bob", Some(&incumbent(claimed, "bob"))),
            Placement::Closed {
                successor: Uuid::nil(),
                valid_until: claimed,
            }
        );
    }

    /// The tolerance is for clock skew between an agent's host and this
    /// database, so an instant a few seconds ahead is an ordinary write.
    #[test]
    fn a_valid_from_inside_the_skew_tolerance_is_accepted() {
        let now = instant(2024, 6, 1);
        let inputs = [EdgeInput::with_valid_from(
            "alice",
            "bob",
            "REPORTS_TO",
            "proof",
            now + chrono::Duration::seconds(FUTURE_VALID_FROM_TOLERANCE_SECONDS),
        )];

        assert!(reject_future_valid_from(&inputs, now).is_ok());
    }

    /// One second past the tolerance is refused, and the error names the item
    /// so a caller with a batch knows which one to fix.
    #[test]
    fn a_valid_from_past_the_skew_tolerance_names_the_item_it_refuses() {
        let now = instant(2024, 6, 1);
        let inputs = [
            EdgeInput::new("alice", "bob", "REPORTS_TO", "proof"),
            EdgeInput::with_valid_from(
                "alice",
                "carol",
                "REPORTS_TO",
                "proof",
                now + chrono::Duration::seconds(FUTURE_VALID_FROM_TOLERANCE_SECONDS + 1),
            ),
        ];

        let error = reject_future_valid_from(&inputs, now)
            .expect_err("a future-dated fact must be refused");

        assert!(
            matches!(error, StoreError::FutureValidFrom { index: 1, .. }),
            "expected item 1 to be named, got {error:?}"
        );
    }
}
