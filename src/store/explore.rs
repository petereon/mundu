//! Traversal: what is connected to this entity.
//!
//! [`search`](super::search) ranks *proof text* and hands back edges, which
//! answers "what does this memory know about X" and cannot answer "what is X
//! attached to". A neighbourhood is a different question with a different
//! shape: it starts from an entity rather than from a sentence, it has no
//! ranking to fall out of, and — the part that decides the whole design — its
//! cost is not predictable from the entity you start at.
//!
//! # Why the second hop is summarised rather than capped
//!
//! Measured on a real graph of 124 nodes and 265 edges:
//!
//! ```text
//! starting node                      degree  depth-2 edges  proof text  % of graph
//! gateway-go-agent-facing-service      49        202          100 KB       76%
//! commsdk-marble-agent-communicator    40        191           92 KB       72%
//! nginx-mgmt-container                  3         67           42 KB       25%
//! alice-whitfield                       3          5          0.3 KB        2%
//! ```
//!
//! Read the third row. A node of **degree 3** reaches a quarter of the graph at
//! depth 2, because one of its three neighbours is the hub — so a cap derived
//! from the starting node's own degree would have let it through. 100 KB of
//! proof text is roughly 25,000 tokens arriving in one tool result, which is
//! not a slow answer, it is an answer that displaces the conversation it was
//! supposed to help.
//!
//! So the second hop returns no proof text at all. Each neighbour comes back as
//! [`NeighbourSummary`] — its name, its total degree, and its edges counted by
//! relationship type — which is enough for a model to decide *which* neighbour
//! is worth a second call and cheap enough that the decision costs a few dozen
//! tokens instead of the graph.
//!
//! # What is in scope
//!
//! Active edges of live nodes, in the caller's tenant. Tombstoned nodes
//! (`merged_into IS NOT NULL`) are not entities any more, they are redirects,
//! and the resolution ladder already follows them to whatever is live. Non-
//! active edges are history: real data, deliberately kept, and deliberately not
//! what a caller asking "what is connected to this" is asking for. There is no
//! flag to switch either back on — see [`Store::explore`].

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use sqlx::Row;
use uuid::Uuid;

use super::Store;
use super::node::{CANDIDATE_LIMIT, MatchedBy, NodeLookup};
use super::relationship::registered_type_for_lookup;
use super::search::{MAX_LIMIT, SALIENCE_HALF_LIFE_DAYS, snippet};
use crate::embed::Embedder;
use crate::error::StoreError;

/// The most entities one exploration reports on.
///
/// Not a new number: it is [`CANDIDATE_LIMIT`], the number of near misses the
/// resolution ladder already hands back for a contested name, and the only way
/// this tool ever sees more than one entity is by being handed that list. A
/// second, larger cap here would be a cap on something that cannot be that
/// large.
pub const MAX_MATCHES: usize = CANDIDATE_LIMIT as usize;

/// The most edges one exploration returns, across every match in it.
///
/// [`MAX_LIMIT`] is what `search_knowledge_graph` already spends at its
/// ceiling: fifty proof snippets, which is the largest amount of stored text
/// this memory has ever put in one tool result. Traversal is not a reason to
/// raise that — the measurement in this module's header is a reason to be more
/// careful with it, not less — so one exploration is held to exactly the budget
/// one search is, and the budget is shared out between the matches rather than
/// granted to each of them. One match reads its whole neighbourhood up to
/// fifty edges; five contested candidates get ten each.
///
/// What is *not* returned is counted and reported. See
/// [`ExploredNode::edges_withheld`]: a caller that cannot tell a small
/// neighbourhood from a truncated one will read the first as the second.
pub const EDGE_BUDGET: usize = MAX_LIMIT;

/// How far a caller may look. Two hops, and the second one is summarised.
///
/// Deeper is not offered rather than refused: the numbers in this module's
/// header are for depth 2 on a graph of 265 edges, and depth 3 on that same
/// graph is the graph.
pub const MAX_DEPTH: u8 = 2;

/// Everything about an exploration that is not the entity it starts from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExploreOptions {
    /// Keep only edges of one type, as the caller spelled it. Coerced onto the
    /// vocabulary by [`registered_type_for_lookup`], so `"depends on"` finds
    /// `DEPENDS_ON`.
    pub relationship_type: Option<String>,
    /// 1 or 2, defaulting to 1. Anything larger is clamped rather than
    /// refused: a caller that asked for more than this memory will serve has
    /// asked a reasonable question badly, and an error would cost it a round
    /// trip to learn a number the description already gives it.
    pub depth: Option<u8>,
}

/// One entity the caller's `node` argument resolved to, and its neighbourhood.
#[derive(Debug, Clone, PartialEq)]
pub struct ExploredNode {
    pub node_id: String,
    pub display_name: String,
    pub aliases: Vec<String>,
    /// Every active edge touching this node, whichever end it is on, and
    /// **before** any `relationship_type` filter or edge cap. It is the number
    /// that says how much of the neighbourhood the caller is not looking at.
    pub degree: i64,
    /// Which tier of the resolution ladder found this entity.
    pub matched_by: MatchedBy,
    /// The cosine, when the match was a similarity and never otherwise.
    pub similarity: Option<f64>,
    /// The first hop, best-salience first, up to this match's share of
    /// [`EDGE_BUDGET`].
    pub edges: Vec<ExploredEdge>,
    /// How many edges matched the same filter and did not fit in the budget.
    pub edges_withheld: i64,
    /// The second hop, summarised, or `None` when depth 1 was asked for.
    pub neighbours: Option<Vec<NeighbourSummary>>,
}

/// One edge of the first hop.
#[derive(Debug, Clone, PartialEq)]
pub struct ExploredEdge {
    pub edge_id: Uuid,
    pub source_node_id: String,
    pub target_node_id: String,
    pub relationship_type: String,
    /// The leading [`snippet`] of the proof, on the same budget search uses.
    pub proof: String,
    pub proof_truncated: bool,
    /// Salience as of this call, after decay — the key the edges were ordered
    /// and capped on.
    pub salience: f64,
    pub disputed: bool,
    pub valid_from: DateTime<Utc>,
}

/// One entity at the far end of a first-hop edge, described without its edges.
///
/// The whole answer to "is it worth another call": how big this neighbour's own
/// neighbourhood is, and what kind of edges it is made of. No proof text, no
/// edge ids, nothing that scales with the size of the graph — see this module's
/// header for the measurement that decided it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighbourSummary {
    pub node_id: String,
    pub display_name: String,
    /// Every active edge touching this neighbour, which is the sum of the
    /// counts below.
    pub degree: i64,
    /// `{"DEPENDS_ON": 12, "USES": 9}`, over the neighbour's whole active
    /// neighbourhood rather than the filtered view the caller asked for: the
    /// point of the summary is to say what is over there, including the part a
    /// filter would have hidden.
    ///
    /// Sorted by name rather than by count, so two calls about the same
    /// neighbour read identically.
    pub relationship_types: BTreeMap<String, i64>,
}

impl Store {
    /// Resolves `node` to one or more entities and reports what each is
    /// connected to.
    ///
    /// `node` is a `node_id` **or** a name: it goes through
    /// [`Store::look_up_node`], which is the same ladder `upsert_node` runs, so
    /// an id the server minted, a name some agent recorded and a paraphrase of
    /// one are all admissible and the answer says which of the three happened.
    /// A contested name resolves to nothing in particular, and its near misses
    /// come back as several matches rather than a guess at one.
    ///
    /// **Only the current picture.** Every query below reads
    /// `status = 'active'` and live nodes, and there is deliberately no flag to
    /// widen that. The graph is bi-temporal and its history is real data, but
    /// exposing it here would be the wrong door: history is a question about a
    /// time window (`valid_from`, `valid_until`, what was believed when), and a
    /// boolean cannot ask it. What a boolean *would* do is multiply an answer
    /// whose size this module works hard to bound — a superseded edge carries a
    /// full proof excerpt exactly like a live one — and hand a model a mixture
    /// of current and retired facts distinguishable only by a timestamp it has
    /// to read. Every edge returned here carries its `valid_from`, so the
    /// dating of the current picture is visible; a history tool, if one is ever
    /// wanted, is a tool with a time argument, not a flag on this one.
    ///
    /// **Nothing here reads outside `tenant`**, which comes from a validated
    /// credential rather than an argument. That is the invariant of the whole
    /// server and this is a traversal, which is the one operation that could
    /// walk out of a tenant one edge at a time if a query forgot it.
    pub async fn explore(
        &self,
        tenant: &str,
        node: &str,
        options: &ExploreOptions,
        embedder: &dyn Embedder,
    ) -> Result<Vec<ExploredNode>, StoreError> {
        let matches = self.matches_for(tenant, node, embedder).await?;

        if matches.is_empty() {
            return Ok(Vec::new());
        }

        let depth = options.depth.unwrap_or(1).clamp(1, MAX_DEPTH);

        let mut connection = self.pool.acquire().await?;

        // Resolved once for the whole call rather than per match: it is a
        // question about the vocabulary, and the vocabulary does not change
        // between two matches of one exploration.
        //
        // A type this vocabulary does not hold matches no edge, which is the
        // honest answer. The alternative is dropping the filter, and a caller
        // that asked for one type and silently got all of them cannot tell —
        // whereas an empty edge list beside a non-zero `degree` says exactly
        // what happened.
        let filter: Option<String> = match options.relationship_type.as_deref() {
            None => None,
            Some(raw) => Some(
                registered_type_for_lookup(&mut connection, tenant, raw)
                    .await?
                    .unwrap_or_else(|| NO_SUCH_TYPE.to_string()),
            ),
        };

        let per_match = share_of_budget(matches.len());

        let mut explored = Vec::with_capacity(matches.len());

        for candidate in matches {
            // A row that is not here, or is a tombstone, is not an entity to
            // report on. Unreachable through the ladder — tiers 1 and 2 follow
            // merges and tier 3 excludes them — and cheap to be sure of.
            let Some(described) =
                describe_node(&mut connection, tenant, &candidate.node_id).await?
            else {
                continue;
            };

            let (edges, matching) = first_hop(
                &mut connection,
                tenant,
                &candidate.node_id,
                filter.as_deref(),
                per_match,
            )
            .await?;

            let neighbours = match depth {
                1 => None,
                _ => Some(
                    summarise_neighbours(
                        &mut connection,
                        tenant,
                        &candidate.node_id,
                        &neighbours_of(&candidate.node_id, &edges),
                    )
                    .await?,
                ),
            };

            explored.push(ExploredNode {
                node_id: candidate.node_id,
                display_name: described.display_name,
                aliases: described.aliases,
                degree: described.degree,
                matched_by: candidate.matched_by,
                similarity: candidate.similarity,
                edges_withheld: matching - edges.len() as i64,
                edges,
                neighbours,
            });
        }

        Ok(explored)
    }

    /// The entities `node` resolves to, in the order the ladder ranked them.
    ///
    /// One when the ladder was certain. When it was not, its near misses — up
    /// to [`MAX_MATCHES`], which is the ladder's own [`CANDIDATE_LIMIT`] —
    /// each carrying the cosine that put it in the list, so a model can tell a
    /// hit from a shortlist without counting the entries.
    async fn matches_for(
        &self,
        tenant: &str,
        node: &str,
        embedder: &dyn Embedder,
    ) -> Result<Vec<Candidate>, StoreError> {
        Ok(match self.look_up_node(tenant, node, embedder).await? {
            NodeLookup::Matched {
                node_id,
                matched_by,
                similarity,
            } => vec![Candidate {
                node_id,
                matched_by,
                similarity,
            }],
            NodeLookup::Unresolved { candidates } => candidates
                .into_iter()
                .take(MAX_MATCHES)
                .map(|candidate| Candidate {
                    node_id: candidate.node_id,
                    matched_by: MatchedBy::Similarity,
                    similarity: Some(candidate.similarity),
                })
                .collect(),
        })
    }
}

/// Stands in for a relationship type the vocabulary does not hold.
///
/// `relationship_type` is a primary key in `relationship_type` and a foreign
/// key from `edge`, and it is [`normalise`](super::relationship)d to upper case
/// with `_` between tokens — so a name carrying a space can never be one, and
/// this cannot collide with a type some tenant coined.
const NO_SUCH_TYPE: &str = "no such relationship type";

/// How many edges each of `matches` entities may spend.
///
/// [`EDGE_BUDGET`] divided between them, floored, and never zero: one match
/// reads a whole neighbourhood, five contested candidates get a fifth of one
/// each, and a match that divided down to nothing would come back as an empty
/// edge list — which reads exactly like an isolated entity and is the one
/// answer this tool must not give by accident.
fn share_of_budget(matches: usize) -> usize {
    EDGE_BUDGET
        .checked_div(matches)
        .unwrap_or(EDGE_BUDGET)
        .max(1)
}

/// One entity the ladder offered, before its neighbourhood is read.
struct Candidate {
    node_id: String,
    matched_by: MatchedBy,
    similarity: Option<f64>,
}

/// A node's own columns, and how much of the graph touches it.
struct DescribedNode {
    display_name: String,
    aliases: Vec<String>,
    degree: i64,
}

/// Reads one live node, with its total active degree.
///
/// `merged_into IS NULL` is in the predicate rather than checked afterwards: a
/// tombstone still has a display name and still has edges hanging off it, so a
/// query that found one would answer confidently about an entity that no longer
/// exists.
async fn describe_node(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_id: &str,
) -> Result<Option<DescribedNode>, StoreError> {
    let row: Option<(String, Vec<String>, i64)> = sqlx::query_as(
        "SELECT display_name, aliases, \
                (SELECT count(*) FROM edge \
                  WHERE edge.tenant_id = node.tenant_id \
                    AND edge.status = 'active' \
                    AND node.node_id IN (edge.source_node_id, edge.target_node_id)) \
         FROM node \
         WHERE tenant_id = $1 AND node_id = $2 AND merged_into IS NULL",
    )
    .bind(tenant)
    .bind(node_id)
    .fetch_optional(connection)
    .await?;

    Ok(row.map(|(display_name, aliases, degree)| DescribedNode {
        display_name,
        aliases,
        degree,
    }))
}

/// The edges touching `node_id`, best-salience first, and how many there were
/// before the cap.
///
/// **The order is the cap.** Something has to decide which edges survive a
/// budget, and salience is the ranking this store already keeps: it is what
/// ratings move, what decay lowers, and what `search_knowledge_graph` re-weights
/// its results by. Ordering by `valid_from` would return the newest facts
/// rather than the ones anybody found useful, and ordering by `edge_id` would
/// return a UUID sort while looking like a judgement. Ties fall to `edge_id`
/// so that one dataset is one answer on every replica.
///
/// The count is taken with a window over the filtered set, before the `LIMIT`,
/// which is what makes [`ExploredNode::edges_withheld`] a real number rather
/// than an inference from the page being full.
async fn first_hop(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_id: &str,
    relationship_type: Option<&str>,
    limit: usize,
) -> Result<(Vec<ExploredEdge>, i64), StoreError> {
    let rows = sqlx::query(
        "SELECT edge_id, source_node_id, target_node_id, relationship_type, proof_excerpt, \
                effective_salience(salience_base, salience_updated_at, $3, now()) \
                    ::double precision AS salience, \
                (disputed_at IS NOT NULL) AS disputed, valid_from, \
                count(*) OVER () AS matching \
         FROM edge \
         WHERE tenant_id = $1 \
           AND status = 'active' \
           AND $2 IN (source_node_id, target_node_id) \
           AND ($4::text IS NULL OR relationship_type = $4) \
         ORDER BY salience DESC, edge_id \
         LIMIT $5",
    )
    .bind(tenant)
    .bind(node_id)
    .bind(SALIENCE_HALF_LIFE_DAYS)
    .bind(relationship_type)
    .bind(limit as i64)
    .fetch_all(connection)
    .await?;

    // Zero rows is zero matches: nothing was withheld from an empty
    // neighbourhood, and there is no window value to read.
    let matching: i64 = rows.first().map_or(0, |row| row.get("matching"));

    let edges = rows
        .into_iter()
        .map(|row| {
            let excerpt: String = row.get("proof_excerpt");
            let (proof, proof_truncated) = snippet(&excerpt);

            ExploredEdge {
                edge_id: row.get("edge_id"),
                source_node_id: row.get("source_node_id"),
                target_node_id: row.get("target_node_id"),
                relationship_type: row.get("relationship_type"),
                proof,
                proof_truncated,
                salience: row.get("salience"),
                disputed: row.get("disputed"),
                valid_from: row.get("valid_from"),
            }
        })
        .collect();

    Ok((edges, matching))
}

/// The far end of each returned edge, deduplicated, in the order the edges came
/// back.
///
/// Read off the edges the caller was actually shown rather than off the whole
/// neighbourhood, so the second hop explains the first: a neighbour whose edge
/// was withheld by the budget is not summarised, because there is nothing in
/// the answer for the summary to be about. It also bounds the second hop for
/// free — at most one neighbour per returned edge.
///
/// A self-edge names the starting node at both ends. It is not a second hop, so
/// it contributes no neighbour.
fn neighbours_of(node_id: &str, edges: &[ExploredEdge]) -> Vec<String> {
    let mut neighbours: Vec<String> = Vec::new();

    for edge in edges {
        let far = match edge.source_node_id == node_id {
            true => &edge.target_node_id,
            false => &edge.source_node_id,
        };

        if far != node_id && !neighbours.iter().any(|seen| seen == far) {
            neighbours.push(far.clone());
        }
    }

    neighbours
}

/// Counts each neighbour's active edges by relationship type.
///
/// One query for the whole hop rather than one per neighbour: a hub's
/// neighbourhood is dozens of nodes, and this is the query that would otherwise
/// turn "one more hop" into dozens of round trips.
///
/// Neighbours are ordered as `neighbours_of` gave them — by the salience of the
/// edge that reached them — rather than by id, so the most strongly held
/// connections are read first by a model that stops reading.
async fn summarise_neighbours(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_id: &str,
    neighbours: &[String],
) -> Result<Vec<NeighbourSummary>, StoreError> {
    if neighbours.is_empty() {
        return Ok(Vec::new());
    }

    let rows: Vec<(String, String, String, i64)> = sqlx::query_as(
        "SELECT node.node_id, node.display_name, edge.relationship_type, count(*) \
         FROM node \
         JOIN edge \
           ON edge.tenant_id = node.tenant_id \
          AND edge.status = 'active' \
          AND node.node_id IN (edge.source_node_id, edge.target_node_id) \
         WHERE node.tenant_id = $1 \
           AND node.node_id = ANY($2::text[]) \
           AND node.merged_into IS NULL \
         GROUP BY node.node_id, node.display_name, edge.relationship_type",
    )
    .bind(tenant)
    .bind(neighbours)
    .fetch_all(connection)
    .await?;

    debug_assert!(
        !neighbours.iter().any(|neighbour| neighbour == node_id),
        "the starting node is not its own second hop"
    );

    Ok(neighbours
        .iter()
        .filter_map(|neighbour| {
            let counted: Vec<&(String, String, String, i64)> = rows
                .iter()
                .filter(|(candidate, ..)| candidate == neighbour)
                .collect();

            // Nothing to summarise means the node was merged away between the
            // first hop and this query. Dropping it is the same choice
            // `describe_node` makes about a tombstone, for the same reason.
            let display_name = counted.first()?.1.clone();

            Some(NeighbourSummary {
                node_id: neighbour.clone(),
                display_name,
                degree: counted.iter().map(|(.., count)| count).sum(),
                relationship_types: counted
                    .iter()
                    .map(|(_, _, relationship_type, count)| (relationship_type.clone(), *count))
                    .collect::<BTreeMap<String, i64>>(),
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(source: &str, target: &str) -> ExploredEdge {
        ExploredEdge {
            edge_id: Uuid::nil(),
            source_node_id: source.to_string(),
            target_node_id: target.to_string(),
            relationship_type: "USES".to_string(),
            proof: "because".to_string(),
            proof_truncated: false,
            salience: 1.0,
            disputed: false,
            valid_from: DateTime::<Utc>::MIN_UTC,
        }
    }

    /// The far end, whichever end that is, once each. An entity connected to
    /// the same neighbour three ways is one neighbour to visit, not three.
    #[test]
    fn the_second_hop_is_the_far_end_of_each_edge_deduplicated() {
        let edges = [
            edge("gateway", "postgres"),
            edge("marble", "gateway"),
            edge("gateway", "postgres"),
        ];

        assert_eq!(
            neighbours_of("gateway", &edges),
            vec!["postgres".to_string(), "marble".to_string()],
            "in the order the edges came back, which is salience order"
        );
    }

    /// A self-edge is a fact about one entity, not a hop to another one.
    /// Following it would summarise the starting node as its own neighbour.
    #[test]
    fn a_self_edge_contributes_no_neighbour() {
        assert!(neighbours_of("gateway", &[edge("gateway", "gateway")]).is_empty());
    }

    /// The budget is shared, not granted per match: five contested candidates
    /// must not cost five times what one certain answer costs.
    #[test]
    fn the_edge_budget_is_divided_between_the_matches_and_never_reaches_zero() {
        assert_eq!(
            share_of_budget(1),
            EDGE_BUDGET,
            "one match reads a whole neighbourhood"
        );
        assert_eq!(
            share_of_budget(MAX_MATCHES) * MAX_MATCHES,
            EDGE_BUDGET,
            "and the most contested answer spends the same in total, not five times it"
        );

        for matches in [0, EDGE_BUDGET + 1, usize::MAX] {
            assert!(
                share_of_budget(matches) >= 1,
                "a share of zero would be an empty edge list, which reads exactly \
                 like an isolated entity"
            );
        }
    }
}
