//! Merging one node into another.
//!
//! Entity resolution is deliberately biased toward creating a duplicate rather
//! than fusing two entities: the ambiguous band mints a node and reports its
//! near misses, the lexical guard demotes a confident vector match whose names
//! contradict, and the short-name bar is placed above the whole band it cannot
//! separate. Every one of those was justified by the same sentence —
//! fragmentation is recoverable by a later merge, over-merge is not. This is
//! the later merge. Until it existed the recoverable error was not, in fact,
//! recoverable, because `merged_into` was read in three places and written in
//! none.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::edge::{ActiveEdge, Placement, place, supersede};
use super::node::{merge_chain, record_surface_forms};
use super::{Provenance, Store};
use crate::error::StoreError;

/// What a merge did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeOutcome {
    /// The audit row this merge wrote. The handle an undo starts from.
    pub merge_id: Uuid,
    /// The node that was tombstoned. Still resolvable; never deleted.
    pub loser_node_id: String,
    /// The node everything landed on.
    pub winner_node_id: String,
    /// Every edge this merge changed, with the state it held beforehand.
    pub edges: Vec<MergedEdge>,
    /// Exactly the surface forms this merge appended to the winner — the
    /// loser's names minus whatever the winner already answered to. What an
    /// undo would have to take back off, and no more.
    pub aliases_added: Vec<String>,
    /// The edges a `one`-cardinality contest on the winner retired, in the
    /// order they were closed.
    pub superseded_edge_ids: Vec<Uuid>,
    /// The edges this merge turned into a claim about a single entity, and so
    /// archived. See [`archive_self_edges`].
    pub archived_edge_ids: Vec<Uuid>,
}

/// One edge's prior state as Postgres hands it back: the id, then every column
/// a merge can overwrite, in the order the queries below select them.
type PriorEdgeRow = (
    Uuid,
    String,
    String,
    String,
    Option<DateTime<Utc>>,
    Option<Uuid>,
);

/// One edge as it stood before the merge touched it.
///
/// The prior values, not the new ones: the new ones are readable from the
/// table, and these are the only copy of what was there before.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergedEdge {
    pub edge_id: Uuid,
    pub previous_source_node_id: String,
    pub previous_target_node_id: String,
    pub previous_status: String,
    pub previous_valid_until: Option<DateTime<Utc>>,
    pub previous_superseded_by_edge_id: Option<Uuid>,
}

impl From<PriorEdgeRow> for MergedEdge {
    fn from(row: PriorEdgeRow) -> Self {
        let (
            edge_id,
            previous_source_node_id,
            previous_target_node_id,
            previous_status,
            previous_valid_until,
            previous_superseded_by_edge_id,
        ) = row;

        Self {
            edge_id,
            previous_source_node_id,
            previous_target_node_id,
            previous_status,
            previous_valid_until,
            previous_superseded_by_edge_id,
        }
    }
}

impl Store {
    /// Retires `loser_node_id` into `winner_node_id`, moving everything the
    /// loser carried onto the winner.
    ///
    /// Nothing is deleted. The loser stays in `node` as a tombstone whose
    /// `merged_into` points at the winner, because agents hold `node_id`s
    /// across stateless calls and a deleted id would dangle in a context window
    /// this server cannot reach or correct.
    ///
    /// One transaction for the whole merge, unlike the per-item transactions
    /// the node and edge batches use. Those exist so one bad item does not take
    /// its siblings down; this is one operation, and half of it — edges moved,
    /// tombstone not written — is a graph with facts hanging off a node nothing
    /// redirects to.
    ///
    /// **Order matters, and it is not the order the requirements are listed
    /// in.** Edges are re-pointed first, because everything after reads the set
    /// they landed in. Then the self-edges the merge just created are archived,
    /// *before* the `one`-cardinality contest, because a `loser -> winner` edge
    /// under such a type is now a tautology and one dated later than the truth
    /// would take the current slot from it. Then the names are unioned. Then
    /// the contest. Then the tombstone, and then the audit — which reads the
    /// loser's row as the merge left it.
    ///
    /// **What this does not serialise against.** [`lock_nodes`] holds both node
    /// rows, which excludes every other merge and every `one`-cardinality edge
    /// writer for these sources. It does not exclude an edge being written
    /// *at* the loser as a target: that writer takes only the `FOR KEY SHARE`
    /// Postgres uses for a foreign key check, which by design does not conflict
    /// with `FOR NO KEY UPDATE`, and taking the lock that would conflict is
    /// what `create_edges` explains would deadlock ordinary reciprocal writes.
    /// Such an edge can land pointing at the tombstone. It is not lost and not
    /// corrupt — `follow_merges` resolves the endpoint for any later reader —
    /// but until something repairs it, that one edge names a retired node.
    pub async fn merge_nodes(
        &self,
        tenant: &str,
        loser_node_id: &str,
        winner_node_id: &str,
        provenance: &Provenance,
    ) -> Result<MergeOutcome, StoreError> {
        let mut transaction = self.pool.begin().await?;

        // Resolve, lock, resolve again — and the second answer is the one that
        // counts.
        //
        // The first pass has to run unlocked, because the rows to lock are its
        // output, so by the time the locks are held its answer may be a merge
        // out of date. That is not a cosmetic staleness: two callers merging
        // `X` into `Y` and `Y` into `X` at the same moment would each validate
        // against a graph without the other's write, and the second to commit
        // would close the `merged_into` loop that `follow_merges` guards
        // against on the read side.
        //
        // Re-resolving under the locks closes it. If the pair comes back
        // unchanged, nothing moved and nothing can: both ends are held. If it
        // comes back different, a merge committed in between — so the locks are
        // now on the wrong rows, and the loop takes them on the right ones. It
        // terminates because a transaction never gives a lock back, and each
        // pass resolves strictly further along a `merged_into` chain that is
        // finite and, since `resolve_pair` refuses a loop, acyclic.
        let (loser, winner) = loop {
            let pair =
                resolve_pair(&mut transaction, tenant, loser_node_id, winner_node_id).await?;
            lock_nodes(&mut transaction, tenant, &pair.0, &pair.1).await?;

            let settled =
                resolve_pair(&mut transaction, tenant, loser_node_id, winner_node_id).await?;
            if settled == pair {
                break settled;
            }
        };

        let mut edges = repoint_edges(&mut transaction, tenant, &loser, &winner).await?;
        // Before the contest below, not after: a `loser -> winner` edge under a
        // `one`-cardinality type is now a tautology, and a tautology dated later
        // than the truth would take the current slot from it.
        let archived_edge_ids = archive_self_edges(&mut transaction, tenant, &edges).await?;

        let aliases_added = union_names(&mut transaction, tenant, &loser, &winner).await?;
        let superseded_edge_ids =
            resolve_cardinality(&mut transaction, tenant, &winner, &mut edges).await?;

        sqlx::query("UPDATE node SET merged_into = $3 WHERE tenant_id = $1 AND node_id = $2")
            .bind(tenant)
            .bind(&loser)
            .bind(&winner)
            .execute(&mut *transaction)
            .await?;

        let merge_id = record_merge(
            &mut transaction,
            tenant,
            &loser,
            &winner,
            &aliases_added,
            &edges,
            provenance,
        )
        .await?;

        transaction.commit().await?;

        Ok(MergeOutcome {
            merge_id,
            loser_node_id: loser,
            winner_node_id: winner,
            edges,
            aliases_added,
            superseded_edge_ids,
            archived_edge_ids,
        })
    }
}

/// Serialises this merge against every other writer that passes through either
/// node row.
///
/// The same rendezvous point `create_edges` uses, taken on both ends rather
/// than on one: this call changes which node an edge hangs off, so a
/// `one`-cardinality writer choosing an incumbent for the winner at the same
/// moment would choose it from a set this call is halfway through changing.
///
/// Sorted, so two merges touching the same pair in opposite directions queue
/// rather than deadlock. `FOR NO KEY UPDATE` rather than `FOR UPDATE`, for the
/// reason `lock_source_node` gives: the stronger lock conflicts with the
/// `FOR KEY SHARE` Postgres takes on an edge's endpoints while checking its
/// foreign keys, so it would deadlock against ordinary edge writes.
async fn lock_nodes(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    loser: &str,
    winner: &str,
) -> Result<(), StoreError> {
    let mut ordered = [loser, winner];
    ordered.sort_unstable();

    for node_id in ordered {
        sqlx::query(
            "SELECT node_id FROM node \
             WHERE tenant_id = $1 AND node_id = $2 \
             FOR NO KEY UPDATE",
        )
        .bind(tenant)
        .bind(node_id)
        .fetch_optional(&mut *connection)
        .await?;
    }

    Ok(())
}

/// Moves every edge with the loser at either end onto the winner, reporting
/// what each row held first.
///
/// One statement for both ends. An edge may have the loser at *both* — a fact
/// the loser asserted about itself — and two statements would have to decide
/// which of them saw it first; `RETURNING` the prior values from a single
/// `UPDATE` means the recorded `previous_*` pair is the state the row actually
/// had, whichever ends moved.
async fn repoint_edges(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    loser: &str,
    winner: &str,
) -> Result<Vec<MergedEdge>, StoreError> {
    let rows: Vec<PriorEdgeRow> = sqlx::query_as(
        "UPDATE edge AS e \
         SET source_node_id = CASE WHEN e.source_node_id = $2 THEN $3 \
                                   ELSE e.source_node_id END, \
             target_node_id = CASE WHEN e.target_node_id = $2 THEN $3 \
                                   ELSE e.target_node_id END \
         FROM edge AS before \
         WHERE e.tenant_id = $1 AND before.tenant_id = e.tenant_id \
           AND before.edge_id = e.edge_id \
           AND $2 IN (e.source_node_id, e.target_node_id) \
         RETURNING e.edge_id, before.source_node_id, before.target_node_id, \
                   before.status::text, before.valid_until, before.superseded_by_edge_id",
    )
    .bind(tenant)
    .bind(loser)
    .bind(winner)
    .fetch_all(&mut *connection)
    .await?;

    Ok(rows.into_iter().map(MergedEdge::from).collect())
}

/// Moves every name the loser answered to onto the winner, and reports exactly
/// which ones were new.
///
/// The display name goes across with the aliases. After the merge it is a name
/// for the winner and for nothing else — it is also, usually, the name that
/// made the duplicate reachable in the first place, so the next agent to say it
/// has to land on the winner.
///
/// The append itself is `record_surface_forms`, the same function the upsert
/// path uses, rather than a second implementation of "union without recasing".
/// Two rules for what counts as a name a node already has would diverge, and
/// the one here would diverge silently: nothing reads it except a later merge.
///
/// The delta is measured by reading the array on both sides of that call rather
/// than by predicting it. `record_surface_forms` filters inside its `UPDATE`
/// precisely so a concurrent writer cannot lose an append, and re-deriving the
/// filter here would be checking a prediction instead of recording a fact. The
/// winner's row is held under [`lock_nodes`], so nothing can slip between the
/// two reads: `aliases` is not a key column, so an ordinary `UPDATE` of it
/// takes the row lock this call already holds.
async fn union_names(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    loser: &str,
    winner: &str,
) -> Result<Vec<String>, StoreError> {
    let (display_name, aliases): (String, Vec<String>) = sqlx::query_as(
        "SELECT display_name, aliases FROM node WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(tenant)
    .bind(loser)
    .fetch_one(&mut *connection)
    .await?;

    let forms: Vec<String> = std::iter::once(display_name).chain(aliases).collect();

    let before = alias_count(&mut *connection, tenant, winner).await?;
    record_surface_forms(&mut *connection, tenant, winner, &forms).await?;
    let after = winner_aliases(&mut *connection, tenant, winner).await?;

    Ok(after.into_iter().skip(before).collect())
}

/// How many aliases the winner carries. Only the count is needed on the way in:
/// `record_surface_forms` appends, so everything past this index afterwards is
/// what this call added.
async fn alias_count(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_id: &str,
) -> Result<usize, StoreError> {
    let (count,): (i32,) = sqlx::query_as(
        "SELECT coalesce(array_length(aliases, 1), 0) FROM node \
         WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(tenant)
    .bind(node_id)
    .fetch_one(connection)
    .await?;

    Ok(count.max(0) as usize)
}

/// The winner's alias array in full.
async fn winner_aliases(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_id: &str,
) -> Result<Vec<String>, StoreError> {
    let (aliases,): (Vec<String>,) =
        sqlx::query_as("SELECT aliases FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(tenant)
            .bind(node_id)
            .fetch_one(connection)
            .await?;

    Ok(aliases)
}

/// Restores the `one`-cardinality invariant on the winner, by replaying
/// `create_edges`' own contest over the edges that now share a source.
///
/// **What breaks.** `create_edges` allows at most one active edge per
/// `(tenant, source_node_id, relationship_type)` when the type is
/// `one`-cardinality. Two edges under that type may nonetheless both be active
/// at once, legally, as long as their sources differ — and re-pointing has just
/// made their sources the same. Nothing on the write path will ever notice:
/// `create_edges` only ever compares an *incoming* fact against the incumbent,
/// and no fact is incoming here.
///
/// **Why it is not resolved with a second rule.** [`place`] is imported from
/// `edge`, not reimplemented. So is [`supersede`], which moves `status`,
/// `valid_until` and `superseded_by_edge_id` together. A merge that closed an
/// interval by a rule one edit away from the write path's would produce history
/// that reads correctly and orders differently, and the divergence would only
/// ever surface as two contradictory answers to one question.
///
/// The shape is the one difference. `create_edges` offers one fact to one
/// incumbent; this offers a whole set, each in turn, to the incumbent the
/// previous round left standing. That is the same algorithm run to a fixed
/// point rather than once, so the row left active is the row a replay of these
/// writes would have left active — and the intervals close pairwise, each
/// against its immediate successor rather than all against the last, which is
/// strictly better than the backfill limit `place` documents.
///
/// Only *active* edges contend. History that was already closed keeps the
/// successor it was closed against, because it lost that contest to a fact that
/// is still on the table.
async fn resolve_cardinality(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    winner: &str,
    changed: &mut Vec<MergedEdge>,
) -> Result<Vec<Uuid>, StoreError> {
    let mut retired = Vec::new();

    for relationship_type in contested_types(&mut *connection, tenant, winner).await? {
        // Ascending, by exactly the key `place` compares on and `active_edge`
        // orders by. Offered in this order, the incumbent only ever moves
        // forwards, and the row left standing is the maximum on
        // `(valid_from, target_node_id)` with the lowest `edge_id` breaking a
        // total tie — which is the row `active_edge` would have picked.
        let contenders: Vec<(Uuid, DateTime<Utc>, String)> = sqlx::query_as(
            "SELECT edge_id, valid_from, target_node_id FROM edge \
             WHERE tenant_id = $1 AND source_node_id = $2 AND relationship_type = $3 \
               AND status = 'active' \
             ORDER BY valid_from, target_node_id, edge_id",
        )
        .bind(tenant)
        .bind(winner)
        .bind(&relationship_type)
        .fetch_all(&mut *connection)
        .await?;

        let mut incumbent: Option<ActiveEdge> = None;

        for (edge_id, valid_from, target_node_id) in contenders {
            match place(valid_from, &target_node_id, incumbent.as_ref()) {
                Placement::Current { displaced } => {
                    if let Some(displaced) = displaced {
                        record_prior_state(&mut *connection, tenant, displaced, changed).await?;
                        supersede(&mut *connection, tenant, displaced, valid_from, edge_id).await?;
                        retired.push(displaced);
                    }

                    incumbent = Some(ActiveEdge {
                        edge_id,
                        valid_from,
                        target_node_id,
                    });
                }
                Placement::Closed {
                    successor,
                    valid_until,
                } => {
                    // Reachable only on an exact tie of `(valid_from, target)`,
                    // because the ordering above hands them over ascending: the
                    // same fact asserted twice, which `place` records as a
                    // zero-length interval rather than churning what is current.
                    record_prior_state(&mut *connection, tenant, edge_id, changed).await?;
                    supersede(&mut *connection, tenant, edge_id, valid_until, successor).await?;
                    retired.push(edge_id);
                }
            }
        }
    }

    Ok(retired)
}

/// The relationship types under which the winner now holds more than one active
/// edge, and should not.
///
/// Keyed on the type's *name*, not on `(scope, name)`, for the reason
/// `active_edge` gives: a tenant that coins a name the curated vocabulary also
/// carries would otherwise hold two types that read identically to every agent
/// and every operator, and two active edges under them would be two
/// contradictory answers to one question. A name is single-valued here if any
/// vocabulary this tenant can see says it is — the curated scope or its own.
async fn contested_types(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    winner: &str,
) -> Result<Vec<String>, StoreError> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT e.relationship_type FROM edge AS e \
         WHERE e.tenant_id = $1 AND e.source_node_id = $2 AND e.status = 'active' \
           AND EXISTS (SELECT 1 FROM relationship_type AS rt \
                       WHERE rt.name = e.relationship_type \
                         AND rt.tenant_id IN ('', $1) \
                         AND rt.cardinality = 'one') \
         GROUP BY e.relationship_type \
         HAVING count(*) > 1 \
         ORDER BY e.relationship_type",
    )
    .bind(tenant)
    .bind(winner)
    .fetch_all(connection)
    .await?;

    Ok(rows.into_iter().map(|(name,)| name).collect())
}

/// Archives every edge this merge collapsed onto a single entity.
///
/// An edge `loser -> winner` becomes `winner -> winner`, and so does
/// `loser -> loser`. The fact did not become false; it stopped being a fact
/// about a relationship, because a relationship needs two ends and it now has
/// one. Leaving it active is not the neutral option: under a
/// `one`-cardinality type it contends for the current slot on `valid_from` like
/// anything else, so a self-reference dated after the truth would displace the
/// truth, and "who does this report to" would answer "itself".
///
/// Archived rather than deleted, and archived rather than superseded. The row
/// keeps its proof, its writer and its dates, so the merge stays reversible —
/// which is the precondition for anything automating it. And `superseded` would
/// be a lie of a specific kind: it means *a later fact replaced this one*, and
/// would demand a `valid_until` and a successor that do not exist. Nothing
/// replaced this edge. It stopped being sayable.
///
/// This is not the archival `rating.rs` refuses to automate. That refusal is
/// about `proof_excerpt` — untrusted text driving erasure — and nothing here
/// reads the proof. The trigger is that two endpoint ids became equal, which is
/// arithmetic on a merge an operator asked for.
///
/// Scoped to the edges this call re-pointed, so an edge that already pointed at
/// itself before the merge is left exactly as it was found. And to the active
/// ones: history that is already closed is not competing for anything.
async fn archive_self_edges(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    edges: &[MergedEdge],
) -> Result<Vec<Uuid>, StoreError> {
    let touched: Vec<Uuid> = edges.iter().map(|edge| edge.edge_id).collect();

    let rows: Vec<(Uuid,)> = sqlx::query_as(
        "UPDATE edge SET status = 'archived' \
         WHERE tenant_id = $1 AND edge_id = ANY($2::uuid[]) \
           AND source_node_id = target_node_id \
           AND status = 'active' \
         RETURNING edge_id",
    )
    .bind(tenant)
    .bind(&touched)
    .fetch_all(connection)
    .await?;

    Ok(rows.into_iter().map(|(edge_id,)| edge_id).collect())
}

/// The two live nodes a merge is actually between, or the reason there are not
/// two.
///
/// Four refusals, and the order they are checked in is the order a caller can
/// act on them.
///
/// **The named ids are the same.** Incoherent on its face. The schema's
/// `node_merge_not_self` CHECK would refuse the write too, but a constraint
/// violation arrives as a database error and tells the caller nothing about
/// what it asked for.
///
/// **Either id is not in this tenant.** Reported as unknown, with the role
/// named so a caller can tell a mistyped loser from a mistyped winner. This is
/// also what a cross-tenant merge produces, deliberately: `tenant` comes from a
/// validated credential rather than an argument, so the other tenant's node is
/// simply absent, and a distinct "that belongs to someone else" would answer
/// "does this id exist anywhere in this database" for any id a caller cared to
/// try.
///
/// **The loser is already a tombstone.** Refused rather than followed, which is
/// the one asymmetry here. The winner *is* followed to its live head, because
/// the edges are about to be re-pointed at it and an edge hung off a tombstone
/// is the fragmentation this call exists to repair — the same reason
/// `resolve_endpoint` follows an endpoint an agent supplied. But naming a loser
/// is naming a specific row to retire, and quietly retiring a different row
/// because the named one had moved is a semi-destructive call doing something
/// larger than it was asked. It would also rewrite the existing redirect and
/// strand the facts that already moved along it.
///
/// **The winner's chain leads back to the loser.** Then `loser -> winner` and
/// the existing `winner -> ... -> loser` close a loop, and `follow_merges`
/// would walk it forever — an unbounded loop inside an open transaction, which
/// does not fail a request so much as take the process with it. That guard
/// exists on the read side; this is the write that would have armed it.
///
/// With those four settled, a cycle is not merely unlikely, it is
/// unconstructible: `merged_into` is only ever written pointing at a node whose
/// own `merged_into` is NULL, so the edge this call adds has nowhere to lead.
async fn resolve_pair(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    loser_node_id: &str,
    winner_node_id: &str,
) -> Result<(String, String), StoreError> {
    if loser_node_id == winner_node_id {
        return Err(StoreError::SelfMerge {
            tenant: tenant.to_string(),
            node_id: loser_node_id.to_string(),
        });
    }

    let loser_chain = merge_chain(&mut *connection, tenant, loser_node_id)
        .await?
        .ok_or_else(|| StoreError::UnknownMergeNode {
            tenant: tenant.to_string(),
            node_id: loser_node_id.to_string(),
            role: "loser",
        })?;

    if loser_chain.len() > 1 {
        return Err(StoreError::AlreadyMerged {
            tenant: tenant.to_string(),
            node_id: loser_node_id.to_string(),
            merged_into: loser_chain[1].clone(),
        });
    }

    let winner_chain = merge_chain(&mut *connection, tenant, winner_node_id)
        .await?
        .ok_or_else(|| StoreError::UnknownMergeNode {
            tenant: tenant.to_string(),
            node_id: winner_node_id.to_string(),
            role: "winner",
        })?;

    if winner_chain.iter().any(|hop| hop == loser_node_id) {
        return Err(StoreError::MergeCycle {
            tenant: tenant.to_string(),
            node_id: loser_node_id.to_string(),
        });
    }

    let winner = winner_chain
        .into_iter()
        .next_back()
        .expect("a chain always holds at least the node it started from");

    Ok((loser_node_id.to_string(), winner))
}

/// Makes sure the change log holds this edge's state from *before* the merge.
///
/// [`repoint_edges`] already recorded every edge the loser was an endpoint of.
/// The `one`-cardinality contest can retire one it did not: an edge that was
/// already hanging off the winner, and lost the current slot to a fact that
/// arrived with the loser. That edge is changed by the merge and was never
/// moved by it, and it is the row an audit is most likely to be missing —
/// nothing about it mentions the loser.
///
/// Only the first record for an edge is kept. A second would be the state some
/// earlier step of this same merge left behind, which is not what the merge
/// found.
async fn record_prior_state(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    edge_id: Uuid,
    changed: &mut Vec<MergedEdge>,
) -> Result<(), StoreError> {
    if changed.iter().any(|edge| edge.edge_id == edge_id) {
        return Ok(());
    }

    let row: PriorEdgeRow = sqlx::query_as(
        "SELECT edge_id, source_node_id, target_node_id, status::text, valid_until, \
                superseded_by_edge_id \
         FROM edge WHERE tenant_id = $1 AND edge_id = $2",
    )
    .bind(tenant)
    .bind(edge_id)
    .fetch_one(connection)
    .await?;

    changed.push(MergedEdge::from(row));

    Ok(())
}

/// Writes the audit trail, inside the same transaction as the merge itself.
///
/// Not afterwards, and not best-effort. A merge that committed without its
/// audit row would be exactly the irreversible operation this whole design
/// exists to avoid, and it would be irreversible silently — the graph would
/// look merged and nothing would say what it had been.
///
/// What is stored is what an undo needs and no more: which node lost, which
/// won, when, on whose authority, precisely which surface forms were appended
/// to the winner, and for every edge the merge changed, the value of every
/// column the merge can write.
async fn record_merge(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    loser: &str,
    winner: &str,
    aliases_added: &[String],
    edges: &[MergedEdge],
    provenance: &Provenance,
) -> Result<Uuid, StoreError> {
    let merge_id = Uuid::new_v4();

    // Read back off the tombstone rather than carried down from the union
    // above: this is the loser as the merge left it, which is the state an undo
    // is undoing from.
    let (display_name, aliases): (String, Vec<String>) = sqlx::query_as(
        "SELECT display_name, aliases FROM node WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(tenant)
    .bind(loser)
    .fetch_one(&mut *connection)
    .await?;

    // `merged_at` is deliberately absent from the column list, the way
    // `recorded_at` is on an edge insert: it is when the server did this, so it
    // is the column default and nothing a caller can reach.
    sqlx::query(
        "INSERT INTO node_merge \
         (merge_id, tenant_id, loser_node_id, winner_node_id, loser_display_name, \
          loser_aliases, aliases_added, merged_by_client, merged_by_principal) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
    )
    .bind(merge_id)
    .bind(tenant)
    .bind(loser)
    .bind(winner)
    .bind(&display_name)
    .bind(&aliases)
    .bind(aliases_added)
    // From the originating request, never from tool arguments, for the reason
    // every other writer column in this schema gives.
    .bind(provenance.client.as_deref())
    .bind(provenance.principal.as_deref())
    .execute(&mut *connection)
    .await?;

    // One statement rather than one per edge: the arrays go down as six
    // parallel columns and are stitched back together by `unnest`, so a merge
    // touching forty edges is still a single round trip.
    let edge_ids: Vec<Uuid> = edges.iter().map(|edge| edge.edge_id).collect();
    let sources: Vec<String> = edges
        .iter()
        .map(|edge| edge.previous_source_node_id.clone())
        .collect();
    let targets: Vec<String> = edges
        .iter()
        .map(|edge| edge.previous_target_node_id.clone())
        .collect();
    let statuses: Vec<String> = edges
        .iter()
        .map(|edge| edge.previous_status.clone())
        .collect();
    let valid_untils: Vec<Option<DateTime<Utc>>> =
        edges.iter().map(|edge| edge.previous_valid_until).collect();
    let successors: Vec<Option<Uuid>> = edges
        .iter()
        .map(|edge| edge.previous_superseded_by_edge_id)
        .collect();

    sqlx::query(
        "INSERT INTO node_merge_edge \
         (merge_id, tenant_id, edge_id, previous_source_node_id, previous_target_node_id, \
          previous_status, previous_valid_until, previous_superseded_by_edge_id) \
         SELECT $1, $2, edge_id, source, target, status::edge_status, valid_until, successor \
         FROM unnest($3::uuid[], $4::text[], $5::text[], $6::text[], $7::timestamptz[], \
                     $8::uuid[]) \
              AS moved(edge_id, source, target, status, valid_until, successor)",
    )
    .bind(merge_id)
    .bind(tenant)
    .bind(&edge_ids)
    .bind(&sources)
    .bind(&targets)
    .bind(&statuses)
    .bind(&valid_untils)
    .bind(&successors)
    .execute(&mut *connection)
    .await?;

    Ok(merge_id)
}
