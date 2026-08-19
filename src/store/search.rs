//! Search over the graph's edges.
//!
//! Two candidate lists are retrieved independently and fused by *rank*, not by
//! score. A cosine and a lexical hit are not on a comparable scale — a weighted
//! sum of them is a number whose weights nobody can justify and whose behaviour
//! changes with the query — so the fusion is Reciprocal Rank Fusion, which only
//! ever reads where a list placed an edge.
//!
//! *Where a list placed it*, and not where it happened to arrive: each list
//! states its own ranks, ties and all. A list that cannot separate two edges
//! has to be able to say so, or fusion reads whatever order fell out of the
//! query as a second opinion and scores agreement it never got. See
//! [`RankedEdge`].

use chrono::{DateTime, Utc};
use sqlx::Row;
use uuid::Uuid;

use super::Store;
use crate::embed::Embedder;
use crate::error::StoreError;

// ---------------------------------------------------------------------------
// Tuning
//
// Every number search ranks with lives in this block. Scattering them through
// the query bodies is what would make reasoning about them impossible.
//
// They are not all of a kind. A cosine threshold is a claim about a corpus and
// Task 12 measures it. `RRF_K` and the cutoff fractions are not: an RRF score
// is `1/(k + rank)` and never reads the corpus, so no measurement can calibrate
// them and they are *derived* instead — see [`RRF_K`], which is solved from
// [`DEFAULT_LIMIT`] and [`SALIENCE_BAND_FLOOR`] rather than chosen. They are
// also coupled, and the couplings are the easy thing to miss: the band's
// promise depends on `RRF_K`, and what each cutoff fraction *means* depends on
// it too. Move one and re-run the unit tests at the bottom of this file, which
// exist to say so.
// ---------------------------------------------------------------------------

/// The rank offset in Reciprocal Rank Fusion, `score = Σ 1/(k + rank)`.
///
/// **Derived, not inherited.** An RRF score is a pure function of rank —
/// `1/(k + rank)` never reads the corpus — so this constant does not need
/// measuring against a tenant's data. It can be solved, and two inequalities
/// solve it.
///
/// Write `L` for [`DEFAULT_LIMIT`] and `F` for [`SALIENCE_BAND_FLOOR`]. The
/// band multiplies a relevance score by at most `1/F`, so:
///
/// 1. **Relevance must outspread the band across a page.** The flattest thing
///    fusion can produce over a window is one consistent ordering — a single
///    list, or both lists agreeing — where rank 1 to rank `L` spans
///    `(k + L)/(k + 1)`. Salience must not be able to invert the ends of that,
///    so `(k + L)/(k + 1) > 1/F`, which is `k < (L·F − 1)/(1 − F)` — at
///    `L = 10, F = 0.7`, `k < 20`.
/// 2. **Agreement must still beat being first in one list.** An edge both lists
///    rank `r` scores `2/(k + r)` against `1/(k + 1)` for an edge only one list
///    has, and agreement wins exactly while `r < k + 2`. Requiring that
///    anywhere on the page a caller reads gives `k > L − 2` — at `L = 10`,
///    `k > 8`.
///
/// **Both inequalities read a rank, and ranks now skip — so both were
/// re-checked rather than assumed.** Each list states its own ranks and ties
/// share one, so a list runs 1, 1, 3, 4, 4, 6 rather than 1, 2, 3, 4, 5, 6. A
/// skipping sequence reads as though it might run off the end of its list. It
/// cannot: a row's competition rank is one plus the number of rows strictly
/// ahead of it, at most every row above it is strictly ahead, so no row ranks
/// deeper than its own position and the deepest rank in a list of `n` is still
/// `n` — which is exactly what `L` means above.
///
/// Inequality 2 can only get easier: every contribution `1/(k + rank)` is at
/// least the `1/(k + position)` it was solved with. Inequality 1 is untouched,
/// because it is stated over one *consistent ordering of distinct hits* and a
/// tie is neither — a tie compresses the spread by making two hits equally
/// relevant, and ordering equally relevant hits is the band's job rather than
/// an inversion of relevance. That is the same reading that already covers two
/// lists disagreeing, which flattens a page further than any tie does. So the
/// derivation stands and `k` does not move; the unit test
/// `competition_ranking_never_ranks_a_row_deeper_than_its_position` checks it
/// over every tie pattern a default page can hold.
///
/// So `8 < k < 20`, and **12** is the integer maximising the smaller of the two
/// relative margins: the page outspreads the band by 1.18x and agreement
/// outscores a list-leader by 1.18x, the two coming within 0.3% of each other
/// at the real-valued optimum of 12.04. Either bound is 18% away, which is the
/// most either can be.
///
/// The window is `DEFAULT_LIMIT` rather than [`MAX_LIMIT`] because
/// `(k + n)/(k + 1)` grows with `n`: the short page is the binding constraint
/// and satisfying it satisfies every longer one. At the cap the spread is
/// 4.77x.
///
/// **Why not 60.** That is the value from the original RRF paper (Cormack,
/// Clarke & Buettcher, 2009), tuned for fusing TREC runs of thousands of
/// documents, and it was inherited rather than derived. On a tenant's memory
/// graph it fails inequality 1 outright: rank 1 to rank 10 spanned 1.148x
/// against a band 1.429x wide, so a fresh edge at the bottom of the default
/// page outscored a faded one at the top and the band inverted the very
/// ordering it exists to protect. It also flattened the reported `score` into
/// noise — a live query returned 0.0328/0.0323/0.0317 for its top three, which
/// is `2/61, 2/62, 2/63`, a 1.6% step per rank telling a model nothing. At 12
/// the step is 7.7%.
///
/// Fusion still rewards agreement over confidence, which is the property the
/// paper's large `k` was buying; it now does so to rank `k + 2 = 14` rather
/// than to rank 62, which is past the page either way.
pub const RRF_K: f64 = 12.0;

/// How deep each candidate list is read before fusion.
///
/// Four times the hard cap on results. RRF can only rank what it is given, so
/// this is the real recall bound; it is not a page size, and shrinking it to
/// the caller's limit would drop edges that rank badly in one list and first in
/// the other, which are exactly the ones fusion exists to recover.
pub const CANDIDATE_DEPTH: i64 = 200;

/// How many results a caller gets when it does not say.
pub const DEFAULT_LIMIT: usize = 10;

/// The most any caller gets, however large a limit it asks for.
///
/// Ten proofs is already on the order of three thousand tokens of an agent's
/// context, spent on one tool call. A caller asking for five hundred has not
/// thought about that, so the number is clamped rather than honoured — and
/// clamped rather than refused, because an error teaches a model nothing it can
/// act on while fifty good hits answer the question it asked.
pub const MAX_LIMIT: usize = 50;

/// The least of its relevance an edge keeps after the salience re-rank, and
/// the span salience is mapped across: `0.7 + 0.3 * effective_salience`.
///
/// The band is the design. Multiplying relevance by raw salience reads as the
/// natural thing and is wrong in a way that is easy to miss until it is in
/// front of an agent: a perfect match at salience 0.10 scores 0.095 while a
/// barely-related edge written this morning scores 0.50, so the junk outranks
/// the right answer five to one. Bounded into `[0.7, 1.0]`, the most age can
/// cost a fact is 30% of its relevance — enough to order two facts a query
/// cannot tell apart, never enough to promote one the query did not ask for.
///
/// The floor also has to stay above zero for a reason beyond ranking: an edge
/// scored at zero is an edge that never appears, therefore never gets rated,
/// therefore can never recover. Decay must not be a one-way door.
///
/// **The band is half of a pair, and the other half is [`RRF_K`].** "Never
/// enough to promote one the query did not ask for" is not a property of these
/// two numbers alone — it holds only while fusion spreads relevance across a
/// page by more than `1/0.7 = 1.429x`, and at `RRF_K = 60` it spread by 1.148x
/// and the sentence above was false. Narrowing the band to `[0.9, 1.0]` was the
/// other way to make it true and was rejected twice over: against `RRF_K = 60`
/// it clears the spread by 3%, one change to [`DEFAULT_LIMIT`] from being false
/// again, and it leaves the reported score as flat as it found it; against a
/// corrected `RRF_K` it drops the band's reach to two ranks, which is a
/// re-ranker that cannot re-rank. So the band is unchanged and `RRF_K` moved.
/// Its reach at `RRF_K = 12` is rank 6 — a fully salient hit can climb over a
/// completely faded one up to six places above it, and no further, which is
/// inside the page and not across it.
pub const SALIENCE_BAND_FLOOR: f64 = 0.7;
pub const SALIENCE_BAND_SPAN: f64 = 0.3;

/// What a disputed edge keeps of its salience weight.
///
/// The schema asks for exactly this — a disputed edge is "still returned by
/// search, demoted below the normal salience band and flagged in output" — and
/// halving puts it at 0.35–0.50, clear of the 0.7 floor an undisputed edge can
/// never fall below. So a contested fact loses to an uncontested one of equal
/// relevance whatever their ages, and still beats an uncontested edge that is
/// half as relevant.
///
/// Demoted rather than dropped, for the reason the schema gives at length:
/// hiding it makes the *dispute* invisible rather than the fact harmless. An
/// edge search will not return cannot be rated, so the quorum that flagged it
/// could never be revised, and a disputed fact would be a permanent one.
const DISPUTED_WEIGHT: f64 = 0.5;

/// The fraction of the best fused score a hit has to reach to be returned, by
/// requested precision.
///
/// Relative rather than absolute: an RRF score is a function of how deep the
/// lists were read, not of how good the answer is, so an absolute floor would
/// be a number calibrated against one corpus and meaningless on the next. What
/// carries across corpora is the shape of the drop-off below the best hit.
///
/// The scale these sit on is worth stating, because RRF's range is narrow. An
/// edge both lists return first scores `2/13`; an edge only one list has heard
/// of scores at most `1/13`, exactly half.
///
/// **A fraction of the best score is a bound on rank, and the rank it solves to
/// moves with [`RRF_K`].** Writing `k` for `RRF_K`, a leader that both lists
/// rank first scores `2/(k + 1)`, so a hit only one list has survives to rank
/// `(k + 1)/(2f) − k` and a corroborated hit to twice that distance,
/// `(k + 1)/f − k`. Every one of those depths shrank when `k` went from 60 to
/// 12, and leaving the fractions alone would have silently retuned all three
/// settings. They are re-solved here to name the same things they named
/// before:
///
/// - `Relaxed` at 0.03 keeps everything either list produced — single-list hits
///   to rank 204, past [`CANDIDATE_DEPTH`], exactly as 0.10 did at `k = 60`.
/// - `Standard` at 0.15 keeps a single-list hit to rank 31 and a corroborated
///   one to rank 74. The depth has to sit strictly between [`DEFAULT_LIMIT`]
///   and [`MAX_LIMIT`]: below the page it docks a page nobody asked it to,
///   at the cap it cannot bite at all. 31 is near the middle of that range and
///   within four ranks of the 27 this setting meant at `k = 60`.
/// - `Strict` at 0.60 is unchanged and still means "the two lists must agree":
///   the property is `f > 0.5`, which is what excludes a hit scoring half the
///   leader. It now also bounds *how deep* they may agree, to rank 9 of each
///   list, because `f > 0.5` forces that depth below `k + 2` — which is the
///   same rank at which agreement stops outscoring a list-leader at all. The
///   tightest setting keeping only agreements that beat a leader is the
///   coherent reading, so the number stays.
///
/// 0.35 could not stay. At `k = 12` it puts `Standard` — the default — at rank
/// 6, and no `k` satisfying the spread requirement in [`RRF_K`] can rescue it:
/// keeping 0.35 non-binding across a page of corroborated hits needs `k > 30`,
/// and the band requires `k < 20`. The fractions had to move with the constant.
///
/// **All of which presumes two lists.** With one, every candidate scores
/// `1/(RRF_K + rank)` — a sequence identical for every query and every corpus —
/// and a fraction of its first term is arithmetic on `RRF_K`, not a reading of
/// the results: `Relaxed` keeps rank 200, `Standard` rank 74, `Strict` rank 9.
/// The first two are past anything a caller can be served, so the setting is
/// inert; the third is now *inside* [`DEFAULT_LIMIT`], so it would dock the
/// default page itself rather than only a tail past 41. Inert or arbitrary,
/// neither is a meaning, so [`lists_can_disagree`] declines to apply the cutoff
/// at all when there is nothing to fuse — a guard that was protecting a caller
/// who asked for more than 41 and now protects every caller who asks for
/// nothing. See its doc for why the obvious alternative — a floor on raw
/// similarity — is not available.
pub const CUTOFF_FRACTION_RELAXED: f64 = 0.03;
pub const CUTOFF_FRACTION_STANDARD: f64 = 0.15;
pub const CUTOFF_FRACTION_STRICT: f64 = 0.60;

/// Half-life of the exponential salience decay, in days.
///
/// Applied by the SQL `effective_salience`, never recomputed here: a decay
/// curve implemented twice is a decay curve that will disagree with itself.
///
/// Visible to the rest of the store for the same reason. A rating rebases an
/// edge's salience to the present before it moves it, so the writer and this
/// reader have to age a row by the same curve — two half-lives would make a
/// rating's effect depend on how long ago search last looked at the edge.
pub const SALIENCE_HALF_LIFE_DAYS: f32 = 27.1;

/// How much of a proof excerpt a hit carries unless the caller asks for all of
/// it.
///
/// Counted in characters rather than bytes, because the excerpt is quoted
/// verbatim from text this server did not write and is as likely to be Japanese
/// as English — a byte budget would give a CJK proof a third of the words an
/// English one gets, and slicing to a byte offset would panic outright.
///
/// Two hundred is roughly two sentences: enough for a model to tell whether
/// this is the fact it wanted, and short enough that ten of them do not cost a
/// noticeable fraction of its context.
const PROOF_SNIPPET_CHARS: usize = 200;

/// Appended to a snippet that had more after it. Belt and braces with
/// [`SearchHit::proof_truncated`]: the flag is for the caller's code, the
/// ellipsis is for the model that reads the text and would otherwise take a
/// sentence cut in half as the whole of what was written.
const TRUNCATION_MARKER: char = '…';

/// The shortest query fragment the lexical half will match a node name on.
///
/// Below this, fragments like `to` or `de` appear in half a tenant's names and
/// the lexical list stops being a signal.
const LEXICAL_TERM_CHARS: usize = 3;

/// How a query is matched against the graph.
///
/// Optional at the call site and defaulting to [`SearchMethod::Hybrid`],
/// because an agent is not well placed to know which half of the index will
/// answer its question — and getting it wrong silently loses recall.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchMethod {
    /// Cosine similarity against `edge.proof_vector` only.
    Semantic,
    /// Node name and alias matching only, expanded to the edges touching those
    /// nodes.
    Lexical,
    /// Both lists, fused by rank.
    #[default]
    Hybrid,
}

/// How aggressively the tail of the result list is cut.
///
/// Read only when [`SearchMethod::Hybrid`] actually produced two rankings. It
/// asks how far the two halves of the index have to corroborate a hit, and one
/// half on its own corroborates nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Precision {
    /// Keep almost everything the candidate lists produced.
    Relaxed,
    #[default]
    Standard,
    /// Keep only what the two lists agree on.
    Strict,
}

impl Precision {
    /// The fraction of the top fused score a hit has to reach.
    fn cutoff_fraction(self) -> f64 {
        match self {
            Self::Relaxed => CUTOFF_FRACTION_RELAXED,
            Self::Standard => CUTOFF_FRACTION_STANDARD,
            Self::Strict => CUTOFF_FRACTION_STRICT,
        }
    }
}

/// Everything about a search that is not the query itself.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchOptions {
    pub search_method: Option<SearchMethod>,
    pub precision: Option<Precision>,
    pub limit: Option<usize>,
    /// Return `proof_excerpt` in full rather than a snippet.
    pub include_full_proof: bool,
}

/// One edge, as search reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// The handle the rating loop takes. Every hit carries one: an agent that
    /// cannot name what it read cannot tell the server the answer was useful.
    pub edge_id: Uuid,
    pub source_node_id: String,
    pub target_node_id: String,
    pub relationship_type: String,
    /// Salience as of this search, after decay.
    pub effective_salience: f64,
    /// What this hit was ranked on: its fused relevance, re-weighted by
    /// salience and by dispute.
    ///
    /// Comparable only against the other hits in the same response. An RRF
    /// score is a function of how deep the candidate lists were read, so it is
    /// a position dressed as a number and means nothing across two searches.
    pub score: f64,
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    /// Negative ratings from distinct principals reached quorum. Reported
    /// rather than hidden: the agent can judge it from the proof and the
    /// writer.
    pub disputed: bool,
    pub writer_client: Option<String>,
    pub writer_principal: Option<String>,
    /// The proof text, or as much of it as this search returns.
    pub proof: String,
    /// Whether `proof` is a prefix of the stored excerpt rather than all of it.
    pub proof_truncated: bool,
}

impl Store {
    /// Ranks this tenant's active edges against `query`.
    ///
    /// Four steps: retrieve two candidate lists independently, fuse them by
    /// rank, re-weight by salience within a bounded band, then cut the tail
    /// relative to the best hit and take the caller's limit.
    ///
    /// Two things this never does, both of them one-way doors. It never filters
    /// on salience — an edge search will not return cannot be rated, and an
    /// edge that cannot be rated cannot recover, so age demotes and nothing
    /// else. And it never reads outside `tenant`, which is taken from a
    /// validated credential rather than a tool argument: a query string is
    /// untrusted text, and this is the call that would hand another customer's
    /// memory to whoever wrote it.
    pub async fn search(
        &self,
        tenant: &str,
        query: &str,
        options: &SearchOptions,
        embedder: &dyn Embedder,
    ) -> Result<Vec<SearchHit>, StoreError> {
        // Refused before anything is embedded or read. Neither half degrades
        // gracefully on an empty string — the vector half ranks the whole
        // tenant by distance to nothing, and `strpos(anything, '')` is 1, so
        // the lexical half matches every node — and both produce results that
        // are indistinguishable from answers. Nothing is the only honest reply
        // to a question that was not asked.
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let method = options.search_method.unwrap_or_default();

        // A list the caller excluded is not merely ignored, it is not
        // retrieved: skipping the semantic half skips a forward pass through
        // the embedding model, which is the most expensive thing a search does.
        let semantic = if method == SearchMethod::Lexical {
            Vec::new()
        } else {
            semantic_candidates(self.pool(), tenant, query, embedder).await?
        };

        let lexical = if method == SearchMethod::Semantic {
            Vec::new()
        } else {
            lexical_candidates(self.pool(), tenant, query).await?
        };

        let lists = [semantic, lexical];
        let mut fused = reciprocal_rank_fusion(&lists);

        // Before the rows are read, which is the point. Salience arrives with
        // the hydration below, so at the moment the tail is cut it is not
        // merely unused, it is unavailable: nothing can be dropped here for
        // being old. Salience reorders what survives and decides nothing about
        // what survives.
        //
        // And only when there were two rankings to fuse. The cutoff reads the
        // spread fusion produced; with one list there is no spread, only
        // `1/(RRF_K + rank)`, and the caller's precision would be doing
        // arithmetic on a constant rather than judging the results.
        if lists_can_disagree(&lists) {
            retain_above_cutoff(
                &mut fused,
                options.precision.unwrap_or_default().cutoff_fraction(),
            );
        }

        let ids: Vec<Uuid> = fused.iter().map(|candidate| candidate.edge_id).collect();
        let rows = hydrate(self.pool(), tenant, &ids).await?;

        // A candidate with no row was retired between the candidate query and
        // this one — another agent's write superseding it mid-search — and is
        // dropped rather than reported: it is no longer current, and the whole
        // point of the status filter is that a stale fact reads exactly like a
        // live one.
        //
        // Linear lookups over two lists bounded by `CANDIDATE_DEPTH`, the same
        // trade the edge writer's type cache makes: a few hundred comparisons
        // are cheaper than the map that would avoid them.
        let mut hits: Vec<SearchHit> = fused
            .iter()
            .filter_map(|candidate| {
                let row = rows.iter().find(|row| row.edge_id == candidate.edge_id)?;
                let score = candidate.relevance
                    * salience_weight(row.effective_salience)
                    * if row.disputed { DISPUTED_WEIGHT } else { 1.0 };
                Some(row.clone().into_hit(score, options.include_full_proof))
            })
            .collect();

        // `total_cmp` rather than `partial_cmp().unwrap()`: a NaN score would
        // panic the second and is a total order under the first. Ties fall back
        // to the edge id so that one ranking is one answer, whichever replica
        // produced it.
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.edge_id.cmp(&right.edge_id))
        });

        hits.truncate(effective_limit(options.limit));

        Ok(hits)
    }
}

/// Edges whose proof text sits closest to the query in embedding space.
///
/// The query goes through `embed_query`, never `embed_documents`. The
/// production model expects an instruction prefix on the query side and none on
/// the document side; using the document side for a search string does not
/// fail, it quietly degrades every ranking this function produces.
///
/// Models are compared explicitly, the way the node resolver and type coercion
/// compare them. A cosine between two different embedding spaces is a
/// meaningless number that still sorts, so a row from another model is not a
/// worse candidate — it is not a candidate.
///
/// The rank is `RANK() OVER` the distance and not the row number, for the
/// reason [`lexical_candidates`] gives at length. The flaw is far smaller here:
/// a cosine is a property of the *edge*, so this list discriminates on
/// something real and ties only when two edges embed to the same point — which
/// in practice means two facts quoting the same sentence as their proof, one
/// paragraph evidencing two relationships. That is uncommon rather than
/// impossible, and when it happens `edge_id` is again the only thing separating
/// them. Costing nothing to close, it is closed: the same rule holds in both
/// halves, and "a position is not a rank" is easier to keep true than "a
/// position is not a rank except here".
///
/// The window sits *outside* the `LIMIT` rather than beside it, which is not
/// cosmetic. A window function is evaluated before `LIMIT`, so writing
/// `rank() OVER (…)` alongside the vector `ORDER BY` would make Postgres sort
/// every active edge in the tenant before taking 200 — throwing away the HNSW
/// index scan the store's session settings exist to guarantee. Wrapping the
/// unchanged query in a subselect leaves that plan alone and numbers what it
/// returned.
async fn semantic_candidates(
    pool: &sqlx::PgPool,
    tenant: &str,
    query: &str,
    embedder: &dyn Embedder,
) -> Result<Vec<RankedEdge>, StoreError> {
    let vector = embedder.embed_query(query).await?;

    // Vectors travel as `real[]` and are cast with `::real[]::vector`, the cast
    // pgvector installs: the `pgvector` crate's encoders are built against a
    // different sqlx version and do not apply here.
    let rows = sqlx::query(
        "SELECT edge_id, rank() OVER (ORDER BY distance) AS rank \
         FROM (SELECT edge_id, proof_vector <=> $2::real[]::vector AS distance \
               FROM edge \
               WHERE tenant_id = $1 \
                 AND status = 'active' \
                 AND embedding_model = $3 \
               ORDER BY proof_vector <=> $2::real[]::vector, edge_id \
               LIMIT $4) AS candidate \
         ORDER BY distance, edge_id",
    )
    .bind(tenant)
    .bind(&vector)
    .bind(embedder.model_id())
    .bind(CANDIDATE_DEPTH)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(ranked_edge).collect())
}

/// Edges touching a node whose name or alias the query names.
///
/// Matching is by `strpos` on lowercased text, never `LIKE`. The query is
/// caller text, and `LIKE` would read a `%` in it as a pattern — a search for
/// `100%` would match every node in the tenant.
///
/// The ordering is the lexical list's whole contribution to the fusion, so it
/// is written out rather than left to the planner: an exact name first, then
/// the count of query terms the name contains, then the shortest name — a term
/// found in a two-word name is a stronger signal than the same term found in a
/// sentence — and finally the edge id, so the list is a function of the data
/// rather than of the plan.
///
/// **The rank is not the row number, and the difference is the whole point of
/// this function.** All three meaningful keys are properties of the matched
/// *node*: whether its name matched exactly, how many query terms it carries,
/// how long it is. Every edge touching one matched node therefore ties on all
/// three — the ordinary case for a query naming one person — and the row order
/// falls through to `edge.edge_id`, a random UUID. Numbering the rows 1, 2, 3
/// hands that UUID to [`reciprocal_rank_fusion`] as a ranking, which pays
/// `1/(k + rank)` for it and reports the result as *corroborated* by two lists
/// when only one of them ranked anything.
///
/// So the rank is `RANK() OVER` the meaningful keys and **not** over
/// `edge.edge_id`: competition ranking, where tied rows share a rank and the
/// next distinct value skips (1, 1, 3). Tied edges are paid identically and
/// the semantic list decides the order among them, which is the honest
/// outcome — a list that cannot discriminate says so instead of inventing an
/// order.
///
/// `edge.edge_id` stays in the `ORDER BY`, because the two jobs are separate
/// and both are needed. It decides which rows the `LIMIT` keeps and what order
/// they are emitted in, so one dataset is one list on every replica; it just no
/// longer reaches the rank *value* that fusion scores.
///
/// The window runs over the whole matched set and the `LIMIT` is taken after
/// it, so a rank means the same thing whatever the depth. Competition rank
/// never exceeds the row number it replaces — a row can only be tied with or
/// behind the rows above it — so ranks here still run within `1..=$4`.
async fn lexical_candidates(
    pool: &sqlx::PgPool,
    tenant: &str,
    query: &str,
) -> Result<Vec<RankedEdge>, StoreError> {
    let needle = query.trim().to_lowercase();

    // `strpos(anything, '') = 1`, so a blank needle would match every node in
    // the tenant and hand back an arbitrary 200 edges as though they were
    // answers.
    if needle.is_empty() {
        return Ok(Vec::new());
    }

    let terms = lexical_terms(&needle);

    let rows = sqlx::query(
        "WITH matched AS ( \
             SELECT node_id, \
                    (lower(display_name) = $2) AS exact, \
                    (SELECT count(*) FROM unnest($3::text[]) AS term(t) \
                     WHERE strpos(lower(display_name), t) > 0) AS terms, \
                    length(display_name) AS name_length \
             FROM node \
             WHERE tenant_id = $1 \
               AND merged_into IS NULL \
               AND (strpos(lower(display_name), $2) > 0 \
                    OR EXISTS (SELECT 1 FROM unnest($3::text[]) AS term(t) \
                               WHERE strpos(lower(display_name), t) > 0) \
                    OR EXISTS (SELECT 1 FROM unnest(aliases) AS alias(a) \
                               WHERE strpos(lower(a), $2) > 0 \
                                  OR EXISTS (SELECT 1 FROM unnest($3::text[]) AS term(t) \
                                             WHERE strpos(lower(a), t) > 0)))), \
             scored AS ( \
             SELECT edge.edge_id AS edge_id, \
                    bool_or(matched.exact) AS exact, \
                    max(matched.terms) AS terms, \
                    min(matched.name_length) AS name_length \
             FROM edge \
             JOIN matched \
               ON matched.node_id IN (edge.source_node_id, edge.target_node_id) \
             WHERE edge.tenant_id = $1 \
               AND edge.status = 'active' \
             GROUP BY edge.edge_id) \
         SELECT edge_id, \
                rank() OVER (ORDER BY exact DESC, terms DESC, name_length) AS rank \
         FROM scored \
         ORDER BY exact DESC, terms DESC, name_length, edge_id \
         LIMIT $4",
    )
    .bind(tenant)
    .bind(&needle)
    .bind(&terms)
    .bind(CANDIDATE_DEPTH)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(ranked_edge).collect())
}

/// The fragments of a lowercased query worth matching a node name on.
///
/// Split on anything that is not alphanumeric, so `who does alice-w report to`
/// yields the words rather than the punctuation between them.
/// [`char::is_alphanumeric`] is Unicode-aware for the reason the node id
/// canonicaliser gives: an ASCII rule would discard every query written in a
/// non-Latin script.
fn lexical_terms(needle: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();

    for term in needle.split(|character: char| !character.is_alphanumeric()) {
        if term.chars().count() < LEXICAL_TERM_CHARS || terms.iter().any(|seen| seen == term) {
            continue;
        }
        terms.push(term.to_string());
    }

    terms
}

/// How many hits this search may return.
fn effective_limit(requested: Option<usize>) -> usize {
    requested.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT)
}

/// Whether the fusion had more than one ranking to fuse, and therefore whether
/// a relative cutoff reads anything.
///
/// Keyed on the lists that came back rather than on [`SearchMethod`], because a
/// hybrid search whose lexical half matched no node is a single-list search
/// wearing the other name, and it degenerates in exactly the same way.
///
/// **Why there is no single-list precision to fall back on.** The only per-hit
/// signal a lone list carries that rank does not is the raw score — a cosine,
/// for the semantic half — and that regime was measured against the production
/// model rather than assumed: real answers to a query span 0.5791–0.7808 while
/// same-topic non-answers reach 0.7010, so the two populations overlap and a
/// floor tight enough to exclude the non-answers loses three real answers in
/// ten. There is no number there to cut on. Nor is depth a substitute:
/// "precision" that shrank the result count would be `limit` under a second
/// name, and a second knob controlling the first is a pair a caller can set
/// against each other.
///
/// So the honest single-list behaviour is none, said out loud in the tool
/// description, rather than a second meaning invented for the same word.
fn lists_can_disagree(lists: &[Vec<RankedEdge>]) -> bool {
    lists.iter().filter(|list| !list.is_empty()).count() > 1
}

/// Drops everything scoring below `fraction` of the best candidate.
///
/// `candidates` must already be ordered best-first, which fusion leaves them
/// in. There is deliberately no salience anywhere in this signature: the type
/// it takes cannot express one.
fn retain_above_cutoff(candidates: &mut Vec<Candidate>, fraction: f64) {
    let Some(best) = candidates.first().map(|candidate| candidate.relevance) else {
        return;
    };

    let floor = best * fraction;
    candidates.retain(|candidate| candidate.relevance >= floor);
}

/// Maps effective salience onto the multiplier a fused score is re-ranked by.
///
/// Clamped before it is mapped. The column is constrained to `[0, 1]` and the
/// decay only shrinks it, so a value outside that range means something else
/// has already gone wrong — and the one outcome this function must not have is
/// letting it through to widen the band it is supposed to bound.
fn salience_weight(effective_salience: f64) -> f64 {
    SALIENCE_BAND_FLOOR + SALIENCE_BAND_SPAN * effective_salience.clamp(0.0, 1.0)
}

/// One edge id and the fused relevance it earned.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    edge_id: Uuid,
    relevance: f64,
}

/// One edge as a candidate list ranked it.
///
/// **The rank is carried rather than read off the position, because a position
/// cannot express a tie.** A list handed to [`reciprocal_rank_fusion`] as a
/// bare `Vec<Uuid>` is a list that has been made to commit to an order it may
/// not have: the lexical half ranks on properties of the matched *node*, so
/// every edge touching one matched node ties on every key it reads, and
/// numbering the rows turns the `edge_id` they fell back on into ranks 1, 2, 3.
/// Fusion then pays `1/(k + rank)` for a UUID sort and reports the result as a
/// second list's agreement. Agreement with a coin flip is not evidence.
///
/// So both candidate queries state the rank, and both state it as SQL's
/// `RANK()`: **competition ranking**, where tied rows share a rank and the next
/// distinct value skips it (1, 1, 3, 4, 4, 6). Equal rows are paid equally, and
/// the list that *can* discriminate decides the order between them.
///
/// Two things follow that are worth writing down.
///
/// * **Rank never exceeds the row number it replaces.** A row's competition
///   rank is one plus the number of rows strictly ahead of it, which is at most
///   the number of rows above it, so `rank(i) ≤ i`. Skipping makes ranks
///   *sparse*, never *large*: the deepest rank in a list of `n` is still `n`.
///   Everything [`RRF_K`] is solved from therefore survives; the arithmetic
///   is in `competition_ranking_never_ranks_a_row_deeper_than_its_position`.
/// * **Determinism is unaffected.** The rank value is what fusion scores; the
///   order rows are emitted in, and which ones the depth limit keeps, are still
///   settled by `edge_id`. One dataset is one answer on every replica, and it
///   is no longer one answer *because of* the ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RankedEdge {
    edge_id: Uuid,
    rank: usize,
}

/// Reads the `(edge_id, rank)` pair both candidate queries select.
///
/// `RANK()` is `bigint`; a rank is never negative and never deeper than
/// [`CANDIDATE_DEPTH`], so the narrowing cannot lose anything a candidate query
/// could produce.
fn ranked_edge(row: sqlx::postgres::PgRow) -> RankedEdge {
    let rank: i64 = row.get("rank");

    RankedEdge {
        edge_id: row.get("edge_id"),
        rank: rank.max(1) as usize,
    }
}

/// Fuses candidate lists by rank: `score = Σ 1/(k + rank)`, ranks 1-based.
///
/// Nothing about the *values* the lists were ordered by reaches this function,
/// which is the property being bought. A cosine of 0.83 and "third lexical hit"
/// cannot be added together in any way a reader could defend; their ranks can.
/// An edge missing from a list simply contributes nothing for it, so an edge
/// both lists agree on outranks one either list ranks first.
///
/// **The rank is the one the list stated, never the position the row arrived
/// in.** Counting positions cannot tell a ranking from an ordering, so a list
/// that tied every row — the lexical half, for any query naming one person —
/// would be read as ranking them 1, 2, 3 and paid three different amounts for
/// a UUID sort. Edges a list tied are paid identically here and leave this
/// function still tied, which lets the other list decide between them. See
/// [`RankedEdge`].
///
/// Ties in the *fused* score are broken on the edge id so that the same data
/// ranks the same way on every replica and every replay.
fn reciprocal_rank_fusion(lists: &[Vec<RankedEdge>]) -> Vec<Candidate> {
    let mut fused: Vec<Candidate> = Vec::new();

    for list in lists {
        for ranked in list {
            let contribution = 1.0 / (RRF_K + ranked.rank as f64);

            match fused
                .iter_mut()
                .find(|candidate| candidate.edge_id == ranked.edge_id)
            {
                Some(candidate) => candidate.relevance += contribution,
                None => fused.push(Candidate {
                    edge_id: ranked.edge_id,
                    relevance: contribution,
                }),
            }
        }
    }

    fused.sort_by(|left, right| {
        right
            .relevance
            .total_cmp(&left.relevance)
            .then_with(|| left.edge_id.cmp(&right.edge_id))
    });

    fused
}

/// Everything a hit reports, for one edge.
#[derive(Debug, Clone, PartialEq)]
struct EdgeRow {
    edge_id: Uuid,
    source_node_id: String,
    target_node_id: String,
    relationship_type: String,
    effective_salience: f64,
    valid_from: DateTime<Utc>,
    valid_until: Option<DateTime<Utc>>,
    disputed: bool,
    writer_client: Option<String>,
    writer_principal: Option<String>,
    proof_excerpt: String,
}

impl EdgeRow {
    fn into_hit(self, score: f64, include_full_proof: bool) -> SearchHit {
        let (proof, proof_truncated) = if include_full_proof {
            (self.proof_excerpt, false)
        } else {
            snippet(&self.proof_excerpt)
        };

        SearchHit {
            edge_id: self.edge_id,
            source_node_id: self.source_node_id,
            target_node_id: self.target_node_id,
            relationship_type: self.relationship_type,
            effective_salience: self.effective_salience,
            score,
            valid_from: self.valid_from,
            valid_until: self.valid_until,
            disputed: self.disputed,
            writer_client: self.writer_client,
            writer_principal: self.writer_principal,
            proof,
            proof_truncated,
        }
    }
}

/// Cuts a proof excerpt down to [`PROOF_SNIPPET_CHARS`], reporting whether it
/// had to.
///
/// The cut is taken on a character boundary found by walking the string, not on
/// a byte offset. `&excerpt[..200]` reads as the same thing and panics the
/// moment a proof contains an accent, an emoji or a word of Japanese — and
/// proof text is quoted verbatim from sources this server has no say over.
///
/// Visible to the rest of the store because [`explore`](super::explore) returns
/// proof text too. A second snippet length would mean two answers to "how much
/// of a proof does this memory show you", differing by tool, with nothing in
/// either answer to say which one a caller was reading.
pub(crate) fn snippet(excerpt: &str) -> (String, bool) {
    match excerpt.char_indices().nth(PROOF_SNIPPET_CHARS) {
        Some((boundary, _)) => {
            let mut text = String::with_capacity(boundary + TRUNCATION_MARKER.len_utf8());
            text.push_str(&excerpt[..boundary]);
            text.push(TRUNCATION_MARKER);
            (text, true)
        }
        None => (excerpt.to_string(), false),
    }
}

/// Reads the columns a hit reports for every surviving candidate.
///
/// Decay is applied by the SQL function the migration ships, evaluated at one
/// instant for the whole result set: computing it per row against `now()` would
/// let two edges be measured against two different clocks in one ranking.
///
/// `status = 'active'` is asserted again here even though both candidate
/// queries already filter on it. It is the one condition in this module that a
/// caller could not detect the loss of — a superseded fact reads exactly like a
/// current one — so it is checked on the path that actually produces output.
async fn hydrate(
    pool: &sqlx::PgPool,
    tenant: &str,
    edge_ids: &[Uuid],
) -> Result<Vec<EdgeRow>, StoreError> {
    if edge_ids.is_empty() {
        return Ok(Vec::new());
    }

    let rows = sqlx::query(
        "SELECT edge_id, source_node_id, target_node_id, relationship_type, \
                effective_salience(salience_base, salience_updated_at, $3, now()) \
                    ::double precision AS effective_salience, \
                valid_from, valid_until, (disputed_at IS NOT NULL) AS disputed, \
                writer_client, writer_principal, proof_excerpt \
         FROM edge \
         WHERE tenant_id = $1 AND edge_id = ANY($2::uuid[]) AND status = 'active'",
    )
    .bind(tenant)
    .bind(edge_ids)
    .bind(SALIENCE_HALF_LIFE_DAYS)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|row| EdgeRow {
            edge_id: row.get("edge_id"),
            source_node_id: row.get("source_node_id"),
            target_node_id: row.get("target_node_id"),
            relationship_type: row.get("relationship_type"),
            effective_salience: row.get("effective_salience"),
            valid_from: row.get("valid_from"),
            valid_until: row.get("valid_until"),
            disputed: row.get("disputed"),
            writer_client: row.get("writer_client"),
            writer_principal: row.get("writer_principal"),
            proof_excerpt: row.get("proof_excerpt"),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(byte: u8) -> Uuid {
        Uuid::from_bytes([byte; 16])
    }

    /// A candidate list with no ties in it: the `n`th id holds rank `n`.
    ///
    /// Named for the claim it makes rather than for what it does. Deriving a
    /// rank from a position is the thing [`reciprocal_rank_fusion`] stopped
    /// doing; it is sound only where a fixture has *said* its rows are
    /// distinct, which is what calling this is.
    fn distinctly_ranked(edge_ids: &[Uuid]) -> Vec<RankedEdge> {
        edge_ids
            .iter()
            .enumerate()
            .map(|(index, edge_id)| RankedEdge {
                edge_id: *edge_id,
                rank: index + 1,
            })
            .collect()
    }

    fn relevance_of(fused: &[Candidate], edge_id: Uuid) -> f64 {
        fused
            .iter()
            .find(|candidate| candidate.edge_id == edge_id)
            .unwrap_or_else(|| panic!("edge {edge_id} is missing from {fused:#?}"))
            .relevance
    }

    /// What fusion is *for*. Neither list can promote a candidate the way two
    /// lists agreeing on it can, so an edge both halves return third beats an
    /// edge either half returns first — and that has to hold without either
    /// list's own scores being comparable to the other's.
    #[test]
    fn agreement_between_the_lists_outranks_being_first_in_one() {
        let agreed = edge(1);
        let semantic_best = edge(2);
        let lexical_best = edge(3);

        let fused = reciprocal_rank_fusion(&[
            distinctly_ranked(&[semantic_best, edge(4), agreed]),
            distinctly_ranked(&[lexical_best, edge(5), agreed]),
        ]);

        assert_eq!(fused[0].edge_id, agreed, "got {fused:#?}");
    }

    /// Ranks are 1-based. Off by one and the head of every list is scored as
    /// though it were the second hit — which changes nothing about the order
    /// within one list and quietly changes every fused comparison between two.
    #[test]
    fn the_first_rank_scores_one_over_k_plus_one() {
        let fused = reciprocal_rank_fusion(&[distinctly_ranked(&[edge(1)])]);

        assert!(
            (fused[0].relevance - 1.0 / (RRF_K + 1.0)).abs() < f64::EPSILON,
            "rank 1 scored {}, expected {}",
            fused[0].relevance,
            1.0 / (RRF_K + 1.0)
        );
    }

    /// Two candidates the fusion cannot separate must not be separated by the
    /// order the lists happened to arrive in: this server is stateless behind a
    /// load balancer, and a ranking that depends on a hash iteration order
    /// answers the same question differently on every replica.
    #[test]
    fn an_exact_tie_is_broken_on_the_edge_id() {
        let low = edge(1);
        let high = edge(9);

        let one_way =
            reciprocal_rank_fusion(&[distinctly_ranked(&[high]), distinctly_ranked(&[low])]);
        let other_way =
            reciprocal_rank_fusion(&[distinctly_ranked(&[low]), distinctly_ranked(&[high])]);

        assert_eq!(one_way[0].edge_id, low);
        assert_eq!(other_way[0].edge_id, low);
    }

    /// **A list that cannot separate its rows must contribute the same to
    /// every one of them — the test that fails against positional ranking.**
    ///
    /// The lexical half orders on properties of the matched *node*, so when
    /// several edges hang off one node — a query naming one person — every
    /// candidate ties on every key and the list has exactly one honest thing to
    /// say: these are equally good. Reading rank off a *position* cannot say
    /// it. It turns whatever order the rows arrived in — `edge_id`, a random
    /// UUID — into ranks 1, 2, 3 and pays `1/(k + rank)` for each, so an edge
    /// comes back scored as corroborated by two lists when only one of them
    /// ranked anything.
    ///
    /// So the assertion is on the *contribution*, not merely on the order:
    /// subtract each edge's known semantic term and what is left is what the
    /// tied list paid, which must be the same number three times. An
    /// implementation that goes back to counting positions pays `1/13`, `1/14`
    /// and `1/15` here and fails on the arithmetic rather than on a happy
    /// accident of which id sorts first.
    #[test]
    fn a_list_that_cannot_rank_its_rows_contributes_the_same_to_each_of_them() {
        let (best, middle, worst) = (edge(9), edge(5), edge(1));

        // The semantic half can tell the three apart, and ranks them against
        // their id order so that a positional reading of the tie is visible.
        let semantic = distinctly_ranked(&[best, middle, worst]);

        // The lexical half tied on every key it reads. The rows arrive in the
        // order the database emitted them — `edge_id` — and that order must not
        // be readable as a ranking.
        let lexical = vec![
            RankedEdge {
                edge_id: worst,
                rank: 1,
            },
            RankedEdge {
                edge_id: middle,
                rank: 1,
            },
            RankedEdge {
                edge_id: best,
                rank: 1,
            },
        ];

        let fused = reciprocal_rank_fusion(&[semantic, lexical]);

        let tied_share: Vec<f64> = [best, middle, worst]
            .iter()
            .zip(1..)
            .map(|(edge_id, semantic_rank)| {
                relevance_of(&fused, *edge_id) - 1.0 / (RRF_K + f64::from(semantic_rank))
            })
            .collect();

        let equal_share = 1.0 / (RRF_K + 1.0);
        assert!(
            tied_share
                .iter()
                .all(|share| (share - equal_share).abs() < 1e-12),
            "three edges the lexical half tied at rank 1 must each be paid \
             {equal_share}; they were paid {tied_share:?}, which is a position \
             being read as a rank"
        );

        assert_eq!(
            fused
                .iter()
                .map(|candidate| candidate.edge_id)
                .collect::<Vec<_>>(),
            vec![best, middle, worst],
            "with the tie declared, the list that can rank decides the order \
             among them: {fused:#?}"
        );
    }

    /// The band, stated as the two numbers a reader would check.
    #[test]
    fn the_salience_band_spans_exactly_zero_point_seven_to_one() {
        assert!((salience_weight(0.0) - 0.7).abs() < f64::EPSILON);
        assert!((salience_weight(1.0) - 1.0).abs() < f64::EPSILON);
    }

    /// Out-of-range input cannot widen the band. The column is constrained to
    /// `[0, 1]` and decay only shrinks it, so a value outside that range means
    /// something upstream is already wrong — and the one thing this must not do
    /// is turn that into a score that outranks everything else.
    #[test]
    fn salience_outside_its_range_cannot_widen_the_band() {
        assert!((salience_weight(4.0) - 1.0).abs() < f64::EPSILON);
        assert!((salience_weight(-1.0) - 0.7).abs() < f64::EPSILON);
    }

    /// The guarantee the band buys, as arithmetic rather than as a story: past
    /// a relevance ratio of `1 / 0.7`, no combination of saliences can invert
    /// the order. Multiplying by raw salience has no such ratio — a salience of
    /// 0.001 will lose to anything.
    ///
    /// **This is only half the property, and on its own it is the half that
    /// cannot fail.** It is an implication — *if* relevance differs by more
    /// than the band is wide, salience cannot flip it — and implications are
    /// true whatever [`RRF_K`] is. It passed unchanged while `RRF_K = 60` made
    /// its premise unreachable inside a result page. The other half, the one
    /// that asks whether fusion ever produces a gap that wide, is
    /// [`the_fusion_outspreads_the_band_across_the_default_page`].
    ///
    /// The baseline is taken from the fusion rather than written as a literal,
    /// so this reads the constant it is reasoning about instead of a number
    /// left behind by an older one.
    #[test]
    fn relevance_beyond_the_bands_width_can_never_be_overturned() {
        let junk = 1.0 / (RRF_K + 1.0);
        let relevant = junk * (1.0 / SALIENCE_BAND_FLOOR + 0.001);

        assert!(
            relevant * salience_weight(0.0) > junk * salience_weight(1.0),
            "an all-but-dead edge that is half again as relevant must still win"
        );
    }

    /// **What relevance is actually worth, measured against what salience can
    /// spend — the test whose absence let [`RRF_K`] stay at 60.**
    ///
    /// The band promises that salience breaks ties and never overturns
    /// relevance. That promise is not a property of the band alone: it holds
    /// only while the fusion spreads relevance *wider* across a page than the
    /// band is wide. Nothing tied the two constants together, and they
    /// disagreed. At `RRF_K = 60` rank 1 to rank 10 spanned `70/61 = 1.148x`
    /// against a band `1/0.7 = 1.429x` wide, so a fresh edge at the bottom of
    /// the default page outscored a faded one at the top: the exact inversion
    /// the band was introduced to prevent, inside the page every caller gets.
    ///
    /// The window is [`DEFAULT_LIMIT`] rather than [`MAX_LIMIT`] because the
    /// spread `(k + n)/(k + 1)` grows with `n` — the shorter page is the harder
    /// constraint, and satisfying it satisfies every longer one.
    ///
    /// The fixture is the two lists agreeing on the order, which is the
    /// flattest shape fusion can produce over a window: every score is
    /// `2/(RRF_K + rank)`, the factor of two cancels out of the ratio, and it
    /// is what the live server was observed doing (`2/61, 2/62, 2/63`). Lists
    /// that *disagree* can leave two hits nearly tied, and separating those is
    /// the band's job rather than a violation of it — as is separating hits a
    /// list tied outright, which is the flattest case of the same thing and the
    /// reason [`RankedEdge`] exists.
    ///
    /// Fails if [`RRF_K`] rises, if the band widens, or if [`DEFAULT_LIMIT`]
    /// drops below the band's reach. Every value is read from the constants, so
    /// none of them can be moved without this being re-run.
    #[test]
    fn the_fusion_outspreads_the_band_across_the_default_page() {
        let page: Vec<Uuid> = (0..DEFAULT_LIMIT).map(|index| edge(index as u8)).collect();
        let page = distinctly_ranked(&page);
        let fused = reciprocal_rank_fusion(&[page.clone(), page]);

        let top = fused[0].relevance;
        let bottom = fused[DEFAULT_LIMIT - 1].relevance;
        let spread = top / bottom;
        let band = salience_weight(1.0) / salience_weight(0.0);

        assert!(
            spread > band,
            "relevance spreads only {spread:.4}x from rank 1 to rank \
             {DEFAULT_LIMIT} while salience can move a score {band:.4}x, so the \
             band can reorder the whole default page rather than break ties in it"
        );

        // The same claim as the comparison it is a proxy for: the bottom of the
        // page at full salience against the top of it at none.
        assert!(
            bottom * salience_weight(1.0) < top * salience_weight(0.0),
            "a fresh hit at rank {DEFAULT_LIMIT} scores {} and a faded hit at \
             rank 1 scores {} — salience is overturning relevance, not \
             breaking a tie",
            bottom * salience_weight(1.0),
            top * salience_weight(0.0)
        );
    }

    /// **The other end of the same interval: [`RRF_K`] cannot be lowered
    /// without limit.**
    ///
    /// Fusion is worth doing because two lists agreeing beats one list being
    /// certain. An edge at rank `r` in both lists scores `2/(k + r)` against
    /// `1/(k + 1)` for an edge only one list has, so agreement wins exactly
    /// while `r < k + 2`. Shrink `k` and that depth shrinks with it, until
    /// corroboration stops outranking confidence and the fusion is doing
    /// nothing a single sorted list would not.
    ///
    /// The depth this pins is the page a caller reads: agreement *anywhere* on
    /// the default page must beat being first in one list, which needs
    /// `RRF_K > DEFAULT_LIMIT - 2`. Together with
    /// [`the_fusion_outspreads_the_band_across_the_default_page`] that brackets
    /// the constant from both sides.
    #[test]
    fn agreement_anywhere_on_the_default_page_beats_being_first_in_one_list() {
        let agreed = edge(0);
        let single_list_best = edge(1);

        // `agreed` sits at the very bottom of the default page in both lists.
        // `single_list_best` heads one list and the other has never heard of it.
        let mut semantic: Vec<Uuid> = vec![single_list_best];
        semantic.extend((2..DEFAULT_LIMIT as u8).map(edge));
        semantic.push(agreed);

        let mut lexical: Vec<Uuid> = (100..99 + DEFAULT_LIMIT as u8).map(edge).collect();
        lexical.push(agreed);

        assert_eq!(
            (semantic.len(), lexical.len()),
            (DEFAULT_LIMIT, DEFAULT_LIMIT),
            "the fixture only tests anything if the agreed edge is last on the page"
        );

        let fused =
            reciprocal_rank_fusion(&[distinctly_ranked(&semantic), distinctly_ranked(&lexical)]);

        assert_eq!(
            fused[0].edge_id, agreed,
            "an edge both lists rank {DEFAULT_LIMIT} loses to an edge one list \
             ranks first, so RRF_K = {RRF_K} is too small for fusion to reward \
             agreement across the page it serves: {fused:#?}"
        );
    }

    /// **And the band still has to do something.**
    ///
    /// Constraint one bounds [`RRF_K`] from above by requiring the page to
    /// spread wider than the band. Pushed too far the band becomes decoration:
    /// if a fully salient hit cannot climb over even the hit immediately above
    /// it, ratings have stopped ranking anything. The band's *reach* — the
    /// deepest rank a salience-1.0 hit can hold and still outrank a salience-0
    /// hit at rank 1 — must therefore sit strictly between 1 and
    /// [`DEFAULT_LIMIT`]: above 1 so ratings matter, below the page so they
    /// cannot overturn it.
    #[test]
    fn the_band_reaches_into_the_page_without_reaching_across_it() {
        let faded_first = 1.0 / (RRF_K + 1.0) * salience_weight(0.0);
        let reach = (1..=DEFAULT_LIMIT)
            .filter(|rank| 1.0 / (RRF_K + *rank as f64) * salience_weight(1.0) > faded_first)
            .count();

        assert!(
            reach > 1,
            "salience cannot lift a hit past even the one above it, so the band \
             is decoration"
        );
        assert!(
            reach < DEFAULT_LIMIT,
            "salience reaches rank {reach} of a {DEFAULT_LIMIT}-hit page, which \
             is the whole page"
        );
    }

    /// **[`RRF_K`] was solved on the assumption that a list's ranks run
    /// `1..=n`. Ties make them skip. This is the arithmetic that says the
    /// assumption survives.**
    ///
    /// 1, 1, 3, 4, 4, 6 reaches 6 in six rows and reads as though a seventh
    /// pattern might reach 7. None can. Competition rank is *defined* as one
    /// plus the number of rows strictly ahead — which is what SQL's `RANK()`
    /// computes and what this recomputes from the definition rather than from
    /// the recurrence — and at most every row above a row is strictly ahead of
    /// it. So no row outranks its own position, skipping makes ranks sparse
    /// rather than deep, and the deepest rank on a page of [`DEFAULT_LIMIT`] is
    /// [`DEFAULT_LIMIT`].
    ///
    /// Checked over every tie pattern a page can hold, together with the bound
    /// those patterns could have broken:
    /// [`agreement_anywhere_on_the_default_page_beats_being_first_in_one_list`]
    /// needs the worst rank on the page to stay under `RRF_K + 2`, and a rank
    /// that never exceeds its position cannot get there first.
    #[test]
    fn competition_ranking_never_ranks_a_row_deeper_than_its_position() {
        /// One plus the number of rows strictly ahead, for keys given
        /// best-first — the definition of `RANK() OVER (ORDER BY …)`.
        fn competition_ranks(keys: &[u32]) -> Vec<usize> {
            keys.iter()
                .map(|key| 1 + keys.iter().filter(|other| *other < key).count())
                .collect()
        }

        // A page's tie pattern is one bit per gap: set means the next row is
        // strictly worse, clear means it ties with the one above.
        for pattern in 0..(1u32 << (DEFAULT_LIMIT - 1)) {
            let mut keys = vec![0u32];
            for gap in 0..DEFAULT_LIMIT - 1 {
                let previous = keys[gap];
                keys.push(previous + ((pattern >> gap) & 1));
            }

            let ranks = competition_ranks(&keys);

            for (index, rank) in ranks.iter().enumerate() {
                assert!(
                    *rank <= index + 1,
                    "row {} of {keys:?} ranked {rank}, deeper than its own \
                     position — RRF_K is solved assuming it cannot",
                    index + 1
                );
                assert!(
                    2.0 / (RRF_K + *rank as f64) > 1.0 / (RRF_K + 1.0),
                    "two lists agreeing at rank {rank}, which a page of \
                     {keys:?} can produce, no longer beats one list's leader"
                );
            }

            assert!(
                ranks.iter().copied().max().expect("a page has rows") <= DEFAULT_LIMIT,
                "a page of {keys:?} ranked something past {DEFAULT_LIMIT}"
            );
        }
    }

    /// Proof text is quoted verbatim from sources this server has no say over,
    /// so it is exactly as likely to be Japanese as English. `&excerpt[..200]`
    /// reads as the same thing this does and panics on the first one.
    #[test]
    fn a_snippet_cuts_on_a_character_boundary_not_a_byte_offset() {
        let excerpt = "東京の倉庫".repeat(60);
        assert!(
            !excerpt.is_char_boundary(PROOF_SNIPPET_CHARS),
            "this fixture only tests anything if byte 200 is mid-character"
        );

        let (text, truncated) = snippet(&excerpt);

        assert!(truncated);
        assert_eq!(text.chars().count(), PROOF_SNIPPET_CHARS + 1);
        assert!(text.ends_with(TRUNCATION_MARKER));
        assert!(excerpt.starts_with(text.trim_end_matches(TRUNCATION_MARKER)));
    }

    /// An excerpt that already fits comes back untouched, marker and all:
    /// appending an ellipsis to a complete sentence tells the model there is
    /// more to know when there is not.
    #[test]
    fn an_excerpt_that_fits_is_left_exactly_as_it_was_written() {
        let exact = "a".repeat(PROOF_SNIPPET_CHARS);

        assert_eq!(snippet(&exact), (exact.clone(), false));
        assert_eq!(snippet("short"), ("short".to_string(), false));
    }

    /// The cap is a cap, not a suggestion, and the default is what a caller
    /// that says nothing gets.
    #[test]
    fn the_limit_is_defaulted_and_capped() {
        assert_eq!(effective_limit(None), DEFAULT_LIMIT);
        assert_eq!(effective_limit(Some(500)), MAX_LIMIT);
        assert_eq!(effective_limit(Some(3)), 3);
    }

    /// A candidate exactly at the floor is kept. The floor is a fraction of a
    /// float, so making it exclusive would drop hits on rounding alone.
    #[test]
    fn the_cutoff_keeps_a_candidate_sitting_exactly_on_the_floor() {
        let mut candidates = vec![
            Candidate {
                edge_id: edge(1),
                relevance: 1.0,
            },
            Candidate {
                edge_id: edge(2),
                relevance: 0.5,
            },
            Candidate {
                edge_id: edge(3),
                relevance: 0.499,
            },
        ];

        retain_above_cutoff(&mut candidates, 0.5);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.edge_id)
                .collect::<Vec<_>>(),
            vec![edge(1), edge(2)]
        );
    }

    /// **Why the cutoff is not applied to a single list.**
    ///
    /// Fed one ranking, the cutoff stops being a judgement about the results
    /// and becomes arithmetic on [`RRF_K`]: every candidate scores
    /// `1/(RRF_K + rank)` whatever the query asked or the corpus holds, so
    /// `fraction * best` solves to a fixed rank. The three settings mean "keep
    /// 200", "keep 74" and "keep 9" — the first two past any answer a caller is
    /// served, the third *inside* [`DEFAULT_LIMIT`] and therefore able to dock
    /// the default page itself.
    ///
    /// That third number is why the guard in [`Store::search`] matters more
    /// than it did. At `RRF_K = 60` these were 200, 114 and 41, and only a
    /// caller asking for more than 41 could notice; at 12 a single-list search
    /// at `Strict` would come back one hit short of a page a caller did not
    /// even have to ask for.
    ///
    /// Pinned as numbers because they are the whole argument for the branch in
    /// [`Store::search`], and because they move if `RRF_K` does — which is the
    /// point: a precision setting whose depth is a function of the fusion
    /// constant is not reading the results.
    #[test]
    fn the_cutoff_on_a_single_list_solves_to_a_fixed_rank_whatever_the_corpus() {
        let single_list =
            distinctly_ranked(&(0..CANDIDATE_DEPTH as u8).map(edge).collect::<Vec<Uuid>>());

        for (fraction, expected) in [
            (CUTOFF_FRACTION_RELAXED, CANDIDATE_DEPTH as usize),
            (CUTOFF_FRACTION_STANDARD, 74),
            (CUTOFF_FRACTION_STRICT, 9),
        ] {
            let mut candidates = reciprocal_rank_fusion(std::slice::from_ref(&single_list));
            retain_above_cutoff(&mut candidates, fraction);

            assert_eq!(
                candidates.len(),
                expected,
                "a fraction of {fraction} kept {} of one list, not {expected}",
                candidates.len()
            );
        }
    }

    /// So the cutoff is offered a list count, not a search method: a hybrid
    /// search whose lexical half named nothing has one ranking and the
    /// degeneracy above, whatever the caller called it.
    #[test]
    fn only_two_rankings_can_disagree() {
        assert!(lists_can_disagree(&[
            distinctly_ranked(&[edge(1)]),
            distinctly_ranked(&[edge(2)])
        ]));

        assert!(
            !lists_can_disagree(&[distinctly_ranked(&[edge(1), edge(2)]), Vec::new()]),
            "a semantic-only search, and a hybrid one whose lexical half matched \
             no node, are the same shape"
        );
        assert!(!lists_can_disagree(&[distinctly_ranked(&[edge(1)])]));
        assert!(!lists_can_disagree(&[Vec::new(), Vec::new()]));
    }

    /// Nothing found is an ordinary answer, not a division by an empty maximum.
    #[test]
    fn the_cutoff_on_an_empty_candidate_list_is_not_an_error() {
        let mut candidates: Vec<Candidate> = Vec::new();

        retain_above_cutoff(&mut candidates, 0.5);

        assert!(candidates.is_empty());
    }

    /// Fragments shorter than the floor appear in half a tenant's names and
    /// would make the lexical list match everything; repeats would count one
    /// word twice in the ordering.
    #[test]
    fn query_terms_drop_short_fragments_and_repeats() {
        assert_eq!(
            lexical_terms("who does alice-whitfield report to, and does bob"),
            vec!["who", "does", "alice", "whitfield", "report", "and", "bob"]
        );
    }

    /// The same Unicode rule the node id canonicaliser uses, for the same
    /// reason: an ASCII-only split discards every query written in another
    /// script instead of matching it. The length floor is in characters, so a
    /// short word in a multi-byte script is short rather than long.
    #[test]
    fn query_terms_survive_a_non_latin_script() {
        assert_eq!(
            lexical_terms("東京の倉庫 タワー 在庫"),
            vec!["東京の倉庫", "タワー"]
        );
    }
}
