//! Entity resolution and the node upsert path.

use sha2::{Digest, Sha256};

use super::{
    MAX_DISPLAY_NAME_CHARS, Provenance, Store, reject_overlong_text, reject_oversized_batch,
};
use crate::embed::Embedder;
use crate::error::StoreError;

/// One entity as an agent described it. There is deliberately no `node_id`
/// field: the server mints ids, and a model permitted to supply one will supply
/// a plausible id that does not exist, which then fails every edge write that
/// references it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeInput {
    pub display_name: String,
    pub aliases: Vec<String>,
}

impl NodeInput {
    pub fn new(display_name: impl Into<String>) -> Self {
        Self {
            display_name: display_name.into(),
            aliases: Vec::new(),
        }
    }

    pub fn with_aliases<A, S>(display_name: impl Into<String>, aliases: A) -> Self
    where
        A: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            display_name: display_name.into(),
            aliases: aliases.into_iter().map(Into::into).collect(),
        }
    }
}

/// A node the resolver considered but did not commit to.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeCandidate {
    pub node_id: String,
    pub display_name: String,
    pub similarity: f64,
}

/// A node offered back to a caller that named one this tenant does not have.
///
/// Deliberately not a [`NodeCandidate`]: there is no cosine here to report. A
/// caller that names a node names an id the server minted, so a miss is a
/// spelling problem rather than a semantic one, and inventing a similarity for
/// it would put a number in front of an agent that means nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSuggestion {
    pub node_id: String,
    pub display_name: String,
}

/// What the caller decided about an input whose resolution came out ambiguous.
///
/// The second half of a disambiguation: [`Store::resolve_nodes`] handed the
/// caller a set of near misses, and this is the answer coming back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Choice {
    /// No usable answer. The caller declined, cancelled, or was never asked, so
    /// a node is minted and the near misses are reported — which is exactly
    /// what an upsert that consulted nobody does.
    CreateNew,
    /// The caller says this name belongs to an entity that already exists.
    ///
    /// Nothing here checks that the id was one of the candidates offered. That
    /// check belongs one layer up, against the sealed record of what was
    /// actually asked: re-deriving the candidate list here would be checking
    /// the answer against a second, later reading of the graph rather than
    /// against the question.
    AttachTo(String),
}

/// What resolution did with one input, for a caller that may be consulted.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeResolution {
    /// A node was written, with the outcome an ordinary upsert would report.
    Written(NodeUpsertOutcome),
    /// **Nothing was written.** Resolution was ambiguous and the caller asked
    /// to be consulted before anything is minted; these are the near misses to
    /// consult it about.
    Deferred { candidates: Vec<NodeCandidate> },
}

/// How one input is handled if it turns out to be ambiguous.
///
/// The internal form of [`Choice`], plus the deferral only the first round of a
/// disambiguation uses. Private because `Defer` is not a decision a caller
/// makes per item — it is what a whole first round does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy<'a> {
    /// Mint a node and report the near misses.
    Create,
    /// Write nothing and hand the near misses back.
    Defer,
    /// Attach to the node the caller picked.
    AttachTo(&'a str),
}

/// Which tier of the resolution ladder committed to a node.
///
/// Reported by [`Store::look_up_node`] and by nothing else. The upsert path has
/// no use for it — a caller that asked for a node to exist does not care which
/// tier found it — but a caller that asked to *read* around a name does: tier 1
/// and tier 2 are facts about the graph, and tier 3 is a model's opinion. A
/// traversal that presented the two identically would let a guess about which
/// entity was meant arrive looking like a lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchedBy {
    /// Tier 1: the id this text canonicalises to is already taken, so this is
    /// the same entity arriving again — or an id the server minted, quoted
    /// back.
    NodeId,
    /// Tier 2: some node answers to this exact text, either as its display name
    /// under different capitalisation or as a recorded alias.
    NameOrAlias,
    /// Tier 3: nothing in the graph carries this text, and the nearest name in
    /// embedding space cleared the confident bar. The only tier that reports a
    /// similarity, because it is the only one that read one.
    Similarity,
}

/// What a read-only walk of the ladder concluded about one name.
///
/// The same three tiers [`Store::upsert_nodes`] runs, with nothing written: no
/// node minted, no surface form recorded, no tombstone followed into an
/// `UPDATE`. See [`Store::look_up_node`].
#[derive(Debug, Clone, PartialEq)]
pub enum NodeLookup {
    /// The ladder committed to one node.
    Matched {
        node_id: String,
        matched_by: MatchedBy,
        /// The cosine, when [`MatchedBy::Similarity`] found it and never
        /// otherwise.
        similarity: Option<f64>,
    },
    /// Nothing was certain. `candidates` holds the near misses the ladder would
    /// have reported beside a freshly minted node, and is empty when this
    /// tenant has nothing resembling the name at all.
    Unresolved { candidates: Vec<NodeCandidate> },
}

/// What resolution did with one input.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeUpsertOutcome {
    /// Resolved to a node that already existed.
    Matched { node_id: String },
    /// Nothing resolved, so a node was minted.
    Created { node_id: String },
    /// Something resolved, but not confidently enough to commit to. A node was
    /// still minted — `node_id` is that new node — and the near misses come
    /// back as `candidates`.
    Ambiguous {
        node_id: String,
        candidates: Vec<NodeCandidate>,
    },
}

impl Store {
    /// Resolves each input to an existing node or mints a new one, in input
    /// order.
    pub async fn upsert_nodes(
        &self,
        tenant: &str,
        inputs: &[NodeInput],
        embedder: &dyn Embedder,
        provenance: &Provenance,
    ) -> Result<Vec<NodeUpsertOutcome>, StoreError> {
        self.upsert_nodes_choosing(tenant, inputs, &[], embedder, provenance)
            .await
    }

    /// As [`Store::upsert_nodes`], with the caller's answers to a previous
    /// round of [`Store::resolve_nodes`] applied.
    ///
    /// `choices` runs parallel to `inputs`, and a missing one is
    /// [`Choice::CreateNew`] — which is what a declined, cancelled or
    /// unanswered question means, and what an upsert that asked nobody does
    /// anyway. An empty list is therefore exactly [`Store::upsert_nodes`].
    ///
    /// The names are embedded again here rather than carried over from the
    /// round that asked the question. The alternative is shipping 768 floats
    /// per item out through the client and back on every round — tens of
    /// kilobytes of base64 — and trusting a vector the client could have
    /// edited on the way.
    pub async fn upsert_nodes_choosing(
        &self,
        tenant: &str,
        inputs: &[NodeInput],
        choices: &[Choice],
        embedder: &dyn Embedder,
        provenance: &Provenance,
    ) -> Result<Vec<NodeUpsertOutcome>, StoreError> {
        let policies: Vec<Policy<'_>> = (0..inputs.len())
            .map(|index| match choices.get(index) {
                Some(Choice::AttachTo(node_id)) => Policy::AttachTo(node_id),
                Some(Choice::CreateNew) | None => Policy::Create,
            })
            .collect();

        Ok(self
            .resolve_each(tenant, inputs, &policies, embedder, provenance)
            .await?
            .into_iter()
            .map(|resolution| match resolution {
                NodeResolution::Written(outcome) => outcome,
                // Unreachable by construction: only `Policy::Defer` defers, and
                // no `Choice` maps onto it.
                NodeResolution::Deferred { .. } => {
                    unreachable!("an answered input is never deferred again")
                }
            })
            .collect())
    }

    /// Resolves each input, writing what it is certain about and leaving what
    /// it is not for the caller to settle.
    ///
    /// The first round of a disambiguation. An ambiguous input comes back as
    /// [`NodeResolution::Deferred`] with **no row written for it**: minting a
    /// node and merging it away once the answer arrives would leave a tombstone
    /// behind for every question ever asked, and the duplicate is the whole
    /// cost the question exists to avoid. Everything unambiguous in the same
    /// batch is still written, so one contested name does not hold up the
    /// nineteen settled ones beside it.
    pub async fn resolve_nodes(
        &self,
        tenant: &str,
        inputs: &[NodeInput],
        embedder: &dyn Embedder,
        provenance: &Provenance,
    ) -> Result<Vec<NodeResolution>, StoreError> {
        let policies = vec![Policy::Defer; inputs.len()];

        self.resolve_each(tenant, inputs, &policies, embedder, provenance)
            .await
    }

    /// Runs the resolution ladder over one name and **writes nothing**.
    ///
    /// The read-only half of [`Store::upsert_nodes`], for callers that want to
    /// know which entity a caller meant without asserting that it exists. It
    /// reaches the same [`resolve`] every write path reaches, so the thresholds
    /// ([`CONFIDENT_SIMILARITY`], [`CONFIDENT_SIMILARITY_SHORT`],
    /// [`AMBIGUOUS_SIMILARITY`]), the [`CANDIDATE_LIMIT`] and the
    /// [`names_contradict`] veto are stated once, in one place, and cannot
    /// drift between the tool that writes and the tool that reads.
    ///
    /// What it deliberately does not do is everything [`Store::resolve_each`]
    /// wraps around that call: no row is inserted, no surface form is appended
    /// to a matched node's aliases, and no transaction is opened. A read that
    /// recorded the name it was asked about would teach the graph a new alias
    /// every time an agent guessed at one.
    ///
    /// `name` is embedded even when tier 1 answers, exactly as the upsert path
    /// embeds it: the ladder takes the vector as an argument, and reordering
    /// that to save a forward pass would be a second ladder.
    ///
    /// [`Store::resolve_each`]: Store::resolve_each
    pub async fn look_up_node(
        &self,
        tenant: &str,
        name: &str,
        embedder: &dyn Embedder,
    ) -> Result<NodeLookup, StoreError> {
        // The same two argument checks the write path makes, in the same order
        // and with the same errors: a name is a name whether it is being
        // recorded or looked up.
        reject_overlong_text("node", 0, name, MAX_DISPLAY_NAME_CHARS)?;

        if name.trim().is_empty() {
            return Err(StoreError::BlankDisplayName { index: 0 });
        }

        let input = NodeInput::new(name);
        let node_id = canonical_node_id(name);
        let forms = surface_forms(&input);

        // Stored text, so the document side of the embedder — the ladder
        // compares this against `node.name_vector`, which was written the same
        // way. A query-side vector would be measured against documents.
        let vector = embedder
            .embed_documents(std::slice::from_ref(&input.display_name))
            .await?
            .pop()
            .ok_or_else(|| {
                crate::error::EmbedError::Inference(format!("no vector returned for `{name}`"))
            })?;

        // A pooled connection rather than a transaction: nothing here writes,
        // and the ladder's three tiers are each a single statement whose answer
        // is not made truer by being read inside one.
        let mut connection = self.pool.acquire().await?;

        let resolution = resolve(
            &mut connection,
            tenant,
            &input,
            &node_id,
            &forms,
            &vector,
            embedder.model_id(),
        )
        .await?;

        Ok(match resolution {
            Resolution::Existing {
                node_id,
                matched_by,
                similarity,
            } => NodeLookup::Matched {
                node_id,
                matched_by,
                similarity,
            },
            Resolution::Fresh(candidates) => NodeLookup::Unresolved { candidates },
        })
    }

    /// The one resolution pass every entry point above runs, in input order.
    async fn resolve_each(
        &self,
        tenant: &str,
        inputs: &[NodeInput],
        policies: &[Policy<'_>],
        embedder: &dyn Embedder,
        provenance: &Provenance,
    ) -> Result<Vec<NodeResolution>, StoreError> {
        // Checked for the whole batch before anything is embedded or written,
        // so a bad item costs the caller a rejection rather than leaving half
        // its batch committed and half not.
        //
        // Length first, because it is the cheapest question and the one whose
        // answer decides whether the rest is worth asking: the names below go
        // into one forward pass, so an unbounded `inputs` is an unbounded
        // allocation before any per-item check has run.
        reject_oversized_batch("nodes", inputs.len())?;

        // One entity's alias list is an array argument too, and an unbounded
        // one is the same hole in a different shape: `surface_forms` compares
        // every alias against every alias it has already kept, so a single node
        // carrying a hundred thousand of them is quadratic work no batch cap on
        // `nodes` would have refused.
        for (index, input) in inputs.iter().enumerate() {
            reject_oversized_batch("aliases", input.aliases.len())?;
            reject_overlong_text(
                "display_name",
                index,
                &input.display_name,
                MAX_DISPLAY_NAME_CHARS,
            )?;

            for alias in &input.aliases {
                reject_overlong_text("alias", index, alias, MAX_DISPLAY_NAME_CHARS)?;
            }
        }

        if let Some(index) = inputs
            .iter()
            .position(|input| input.display_name.trim().is_empty())
        {
            return Err(StoreError::BlankDisplayName { index });
        }

        let names: Vec<String> = inputs
            .iter()
            .map(|input| input.display_name.clone())
            .collect();

        // One forward pass for the whole batch. Node names are stored text, not
        // search strings, so they go through the document side of the embedder.
        //
        // Vectors travel to Postgres as `real[]` and are cast with
        // `::real[]::vector`, a cast pgvector installs itself. The `pgvector`
        // crate would normally supply an encoder, but it is built against sqlx
        // 0.8 while this crate is on 0.9, so its `Type` impls are for a
        // different crate version and do not apply. The cast costs one array
        // copy per row and keeps full `f32` precision.
        let vectors = embedder.embed_documents(&names).await?;

        let mut transaction = self.pool.begin().await?;
        let mut outcomes = Vec::with_capacity(inputs.len());

        for ((input, vector), policy) in inputs.iter().zip(vectors).zip(policies) {
            let node_id = canonical_node_id(&input.display_name);
            let forms = surface_forms(input);

            let (existing, candidates) = match resolve(
                &mut transaction,
                tenant,
                input,
                &node_id,
                &forms,
                &vector,
                embedder.model_id(),
            )
            .await?
            {
                // Which tier found it is not something an upsert has any use
                // for: the caller asked for the node to exist, and it does.
                Resolution::Existing { node_id, .. } => (Some(node_id), Vec::new()),
                Resolution::Fresh(candidates) => (None, candidates),
            };

            // Where the caller's answer sends this name, if it gave one. Asked
            // only when the ladder itself was uncertain: an id the server
            // minted or a name the graph already carries is better evidence
            // than a choice made a round ago from a list of near misses. A
            // choice naming a node that has since been removed resolves to
            // nothing and mints, rather than failing the item.
            let chosen = match policy {
                Policy::AttachTo(node_id) if existing.is_none() => {
                    follow_merges(&mut transaction, tenant, node_id).await?
                }
                _ => None,
            };

            let outcome = if let Some(matched) = existing.or(chosen) {
                record_surface_forms(&mut transaction, tenant, &matched, &forms).await?;
                NodeResolution::Written(NodeUpsertOutcome::Matched { node_id: matched })
            } else if matches!(policy, Policy::Defer) && !candidates.is_empty() {
                // The caller is about to be asked which entity this is, so
                // nothing is written for it. The answer arrives in another
                // call, quite possibly served by another instance, and this
                // one keeps no note that it asked.
                NodeResolution::Deferred { candidates }
            } else {
                // `ON CONFLICT DO NOTHING` rather than a bare insert: the
                // resolver read an empty table a moment ago, but another
                // agent may have written this exact id since, and under
                // read-committed neither transaction can see the other's
                // uncommitted row. Postgres blocks the second insert until
                // the first commits and then reports no row inserted, which
                // is the signal that this is a match after all.
                let inserted = sqlx::query(
                    "INSERT INTO node \
                     (tenant_id, node_id, display_name, aliases, name_vector, \
                      embedding_model, writer_client, writer_principal) \
                     VALUES ($1, $2, $3, $4, $5::real[]::vector, $6, $7, $8) \
                     ON CONFLICT (tenant_id, node_id) DO NOTHING",
                )
                .bind(tenant)
                .bind(&node_id)
                .bind(&input.display_name)
                // `forms` minus the display name, by the same rule the
                // append path uses: a node should not carry an alias that
                // merely recases the name it is already displayed under.
                .bind(&forms[1..])
                .bind(&vector)
                .bind(embedder.model_id())
                .bind(provenance.client.as_deref())
                .bind(provenance.principal.as_deref())
                .execute(&mut *transaction)
                .await?
                .rows_affected()
                    > 0;

                if !inserted {
                    let matched = follow_merges(&mut transaction, tenant, &node_id)
                        .await?
                        .unwrap_or(node_id);
                    record_surface_forms(&mut transaction, tenant, &matched, &forms).await?;
                    NodeResolution::Written(NodeUpsertOutcome::Matched { node_id: matched })
                } else if candidates.is_empty() {
                    NodeResolution::Written(NodeUpsertOutcome::Created { node_id })
                } else {
                    NodeResolution::Written(NodeUpsertOutcome::Ambiguous {
                        node_id,
                        candidates,
                    })
                }
            };

            outcomes.push(outcome);
        }

        transaction.commit().await?;

        Ok(outcomes)
    }
}

/// A display name of at most this many characters is treated as short.
///
/// Not a claim about the embedding space. Measured over 422 labelled pairs,
/// nothing about the cosines changes at this boundary: negatives average 0.5955
/// in the 1–6 character band, 0.6028 in 7–12 and 0.5934 in 13–20, and the worst
/// negative in the whole corpus is a *long* pair. Short names are not crowded
/// together, and an earlier version of this comment said they were.
///
/// What the count actually stands in for is whether [`names_contradict`] can
/// defend the merge at all. That guard needs a word on each side the other side
/// lacks, so a name of a single token can never be vetoed by it, whatever it is
/// compared against — there is nothing to disagree with. Twelve characters is
/// about the longest single word an entity gets called (`Kubernetes`,
/// `Postgres`, `Phoenix`), so a character count is a usable proxy for a token
/// count.
///
/// The proxy is lossy, and lossy in the safe direction: `Jane Doe` is two
/// tokens in eight characters and gets a stricter bar than it needs. That costs
/// a duplicate node. Being lossy the other way would cost a fusion.
pub const SHORT_NAME_CHARS: usize = 12;

/// Cosine at or above which a vector neighbour is the same entity.
///
/// Deliberately high. Sentence embedders of the `bge-base` class compress their
/// similarity range: unrelated short English strings routinely sit at 0.6–0.8,
/// so a threshold that sounds strict in the abstract merges half the graph.
pub const CONFIDENT_SIMILARITY: f64 = 0.92;

/// The same bar for a short name, stricter for one reason: a short name has one
/// defence where a long name has two.
///
/// [`names_contradict`] is the second defence and it is nearly inert here. It
/// vetoes 10 of 22 long negatives and 2 of 30 short ones, because a one-token
/// name has no second token to disagree with. What that leaves is not a small
/// difference between the regimes. At 0.92 a long name sits 0.2019 above the
/// highest long negative the guard cannot veto; the highest short negative it
/// cannot veto is `Phoenix` against `Phoenix, AZ` at 0.8977, and no value in
/// `[0, 1]` buys a short name that much room. The margins cannot be equalised,
/// so this number is not derived by trying to.
///
/// It is derived from where the guard's blind spot is instead. The blind spot
/// is one token plus a qualifier: the guard reads the extra word as elaboration
/// and allows it. That band holds `Phoenix` / `Phoenix, AZ` at 0.8977 and
/// `Barcelona` / `Barcelona FC` at 0.8534 on the false side, and `AWS` /
/// `Amazon AWS` at 0.9294 and `Acme` / `ACME Inc` at 0.9259 on the true side.
/// The two populations are 0.0282 apart, they are the same lexical shape, and
/// nothing else the resolver can see distinguishes them. So the bar is put
/// above the whole band rather than through the middle of it.
///
/// 0.93 is what that comes to. It clears the worst short negative in the corpus
/// (`v1.2.0` / `v1.3.0`, 0.9115) by 0.0185 and the worst one the guard cannot
/// veto by 0.0323, and it admits 2 of the 14 true short matches: `GPT-4` /
/// `GPT4` and `Postgres` / `PostgreSQL`. Both of those are pairs whose names
/// either agree completely or share no word at all, which is the case where the
/// guard's silence means agreement rather than blindness.
///
/// It replaces 0.97, which admitted 0 of 14 and left the vector tier doing
/// nothing whatsoever for short names.
pub const CONFIDENT_SIMILARITY_SHORT: f64 = 0.93;

/// Below the confident bar but above this, a neighbour is worth reporting as a
/// candidate even though it is not worth attaching to.
///
/// One floor for every length, unlike the confident bar. The guard-coverage
/// asymmetry that splits the bar above is a fact about *attaching*:
/// [`names_contradict`] is a veto at the merge site and is never asked about
/// the candidate list, so it gives no reason for the floor to move. Neither
/// does the model — unrelated short pairs measure 0.3631–0.5071 against
/// 0.3205–0.4589 for long ones, which is the same floor. A short candidate
/// floor of 0.90 used to sit here on the theory that "below this, a short name
/// is near everything"; that is measurably false, and it made 10 of 14 true
/// short matches into duplicates with no candidate list to say so. The strict
/// confident bar is only defensible because the failure it buys is visible.
pub const AMBIGUOUS_SIMILARITY: f64 = 0.82;

/// How many vector neighbours to pull back. Enough to show an agent that a name
/// is contested, few enough that the answer stays readable.
pub(crate) const CANDIDATE_LIMIT: i64 = 5;

/// Marks an id that had to be derived from a digest because the display name
/// canonicalised to nothing. Chosen to be obviously machine-made, so an operator
/// reading the table can tell these apart from names an agent actually chose.
pub const UNNAMEABLE_ID_PREFIX: &str = "unnamed-";

/// How many hex characters of the digest go into an unnameable id. Six bytes is
/// enough that a collision inside one tenant is not a practical concern, and
/// short enough that an agent can quote the id back without transcription
/// errors.
const UNNAMEABLE_ID_HEX_CHARS: usize = 12;

/// Mints the server-side handle for a display name.
///
/// Lowercase, every run of non-alphanumeric characters becomes a single hyphen,
/// and the ends are trimmed. `"Project Phoenix"` becomes `project-phoenix`.
///
/// Alphanumeric is [`char::is_alphanumeric`], which is Unicode-aware: an
/// ASCII-only rule would flatten every entity named in a non-Latin script onto
/// the same empty id.
///
/// A name with no alphanumeric characters at all — `"!!!"` — would otherwise
/// produce an empty `node_id`: a primary key value that is unquotable, and that
/// every other unnameable name in the tenant would collide with, silently
/// merging unrelated entities. Such names instead get
/// `unnamed-<12 hex chars of SHA-256(display_name)>`, which is derived rather
/// than random, so upserting the same unnameable name twice still resolves to
/// the same node.
pub fn canonical_node_id(display_name: &str) -> String {
    let mut id = String::with_capacity(display_name.len());
    let mut hyphen_pending = false;

    for character in display_name.chars() {
        if character.is_alphanumeric() {
            if hyphen_pending && !id.is_empty() {
                id.push('-');
            }
            hyphen_pending = false;
            id.extend(character.to_lowercase());
        } else {
            hyphen_pending = true;
        }
    }

    if id.is_empty() {
        let digest = Sha256::digest(display_name.as_bytes());
        let hex: String = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
            .chars()
            .take(UNNAMEABLE_ID_HEX_CHARS)
            .collect();
        return format!("{UNNAMEABLE_ID_PREFIX}{hex}");
    }

    id
}

/// The similarity bars a given name has to clear.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Thresholds {
    confident: f64,
    ambiguous: f64,
}

impl Thresholds {
    /// Short names get the stricter confident bar and the same candidate floor.
    /// Length is counted in characters of the trimmed name, not bytes: a name
    /// in a non-Latin script is not long merely for being multi-byte.
    ///
    /// Only the confident bar moves with length, because only the confident bar
    /// has a second defence to be missing. See [`SHORT_NAME_CHARS`] for what
    /// the count is really standing in for.
    fn for_name(display_name: &str) -> Self {
        let short = display_name.trim().chars().count() <= SHORT_NAME_CHARS;

        Self {
            confident: if short {
                CONFIDENT_SIMILARITY_SHORT
            } else {
                CONFIDENT_SIMILARITY
            },
            ambiguous: AMBIGUOUS_SIMILARITY,
        }
    }
}

/// Whether two names disagree about *which* entity they name.
///
/// A cosine cannot answer that question. Measured over 422 labelled pairs,
/// `Acme Corporation` / `Acme Holdings` — two companies — sits at 0.9451 while
/// `billing-service-v2` / `the v2 billing service` — one service — sits at
/// 0.9295, so the populations interleave and no bar separates them.
///
/// What separates them is what the extra words *do*. One name may elaborate on
/// another: an article, an honorific, a legal suffix. Both names carrying a
/// word the other lacks is a different thing — a substitution, and a
/// substitution is how sibling entities are named.
///
/// This is a veto, not a score. It says only that two names disagree; it never
/// says they agree, and it is asked only about neighbours a cosine has already
/// called confident. A pair with little in the way of shared words —
/// `GPT-4` / `GPT4`, or a paraphrase that shares nothing — has no common name
/// to disagree about, so nothing is vetoed and the vector decides alone, as it
/// did before this guard existed.
pub fn names_contradict(left: &str, right: &str) -> bool {
    let left_tokens = name_tokens(left);
    let right_tokens = name_tokens(right);
    let (left_matched, right_matched) = pair_tokens(&left_tokens, &right_tokens);

    // Nothing to disagree about. Two names that barely overlap are two names,
    // not a contradiction, and the tier being guarded exists for exactly that
    // case — the paraphrase no agent has written down.
    if !comparable(&left_matched) || !comparable(&right_matched) {
        return false;
    }

    // A version or an ordinal decides an entity by itself, so it is held to
    // equality rather than to the containment rule the words are.
    if numbered(&left_tokens) != numbered(&right_tokens) {
        return true;
    }

    // Extra words on one side elaborate; extra words on both sides substitute.
    unmatched(&left_matched) > 0 && unmatched(&right_matched) > 0
}

/// The words of a name: lowercased, split on everything that is not
/// alphanumeric, so `Dr. Katherine Chen`, `billing-service-v2` and
/// `the v2 billing service` all decompose the same way whatever punctuation
/// their writer used.
///
/// [`char::is_alphanumeric`] rather than an ASCII rule, for the reason
/// [`canonical_node_id`] gives: a name in a non-Latin script must not tokenise
/// to nothing.
fn name_tokens(name: &str) -> Vec<String> {
    name.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Marks, for each token on each side, whether the other side carries it.
///
/// Greedy and order-free: `Project Phoenix` and `Phoenix Project` are the same
/// two words, and which one a writer put first is not evidence about anything.
fn pair_tokens(left: &[String], right: &[String]) -> (Vec<bool>, Vec<bool>) {
    let mut left_matched = vec![false; left.len()];
    let mut right_matched = vec![false; right.len()];

    for (index, token) in left.iter().enumerate() {
        let found = right
            .iter()
            .enumerate()
            .position(|(other, candidate)| !right_matched[other] && same_word(candidate, token));
        if let Some(other) = found {
            left_matched[index] = true;
            right_matched[other] = true;
        }
    }

    // Second, because an initialism consumes several tokens at once and must
    // not eat words the first pass could have paired off one for one.
    match_initialisms(left, &mut left_matched, right, &mut right_matched);
    match_initialisms(right, &mut right_matched, left, &mut left_matched);

    (left_matched, right_matched)
}

/// The lengths a token may have and still be read as an initialism. Two
/// letters at least, or every stray word is an abbreviation of the two beside
/// it; five at most, because past that an acronym is a word in its own right
/// and the graph should not be guessing.
const INITIALISM_CHARS: std::ops::RangeInclusive<usize> = 2..=5;

/// Pairs off any token on one side that spells out the initials of a run of
/// tokens on the other — `US` against `United States`, `SF` against
/// `San Francisco`.
///
/// Only unmatched tokens on either side take part, so an initialism can never
/// steal a word that already has a partner, and the run has to be consecutive:
/// initials taken out of order are a coincidence, not an abbreviation.
fn match_initialisms(
    tokens: &[String],
    matched: &mut [bool],
    other: &[String],
    other_matched: &mut [bool],
) {
    for index in 0..tokens.len() {
        if matched[index] {
            continue;
        }

        let letters: Vec<char> = tokens[index].chars().collect();
        if !INITIALISM_CHARS.contains(&letters.len())
            || !letters.iter().all(|letter| letter.is_alphabetic())
        {
            continue;
        }

        let Some(start) = spelled_out_at(&letters, other, other_matched) else {
            continue;
        };

        matched[index] = true;
        for position in other_matched.iter_mut().skip(start).take(letters.len()) {
            *position = true;
        }
    }
}

/// Where in `tokens` an unmatched run begins whose first letters are exactly
/// `letters`, if anywhere.
fn spelled_out_at(letters: &[char], tokens: &[String], matched: &[bool]) -> Option<usize> {
    if tokens.len() < letters.len() {
        return None;
    }

    (0..=tokens.len() - letters.len()).find(|start| {
        letters.iter().enumerate().all(|(offset, letter)| {
            !matched[start + offset] && tokens[start + offset].starts_with(*letter)
        })
    })
}

/// How many tokens on one side the other side does not carry.
fn unmatched(matched: &[bool]) -> usize {
    matched.iter().filter(|found| !**found).count()
}

/// The shortest a word may be before something appended to it still counts as
/// the same word. `Review` and `Reviews` are one word; `Data` and `Database`,
/// `Corp` and `Corporation` are not, because four characters is a prefix half
/// a vocabulary shares.
const SAME_WORD_STEM_CHARS: usize = 5;

/// How much may be appended before the longer form is a different word. A
/// plural adds one and `PostgreSQL` adds two to `Postgres`; `BillingService`
/// adds seven to `Billing` and is a compound, not an inflection.
const SAME_WORD_SUFFIX_CHARS: usize = 3;

/// Whether two tokens are the same word written twice.
///
/// Suffixes only, and short ones. This is not a spelling-distance measure: a
/// distance that admitted `v1` against `v2` or `Success` against `Support`
/// would readmit exactly the substitutions the guard exists to catch, and both
/// of those are one character apart.
fn same_word(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }

    let (shorter, longer) = if left.chars().count() <= right.chars().count() {
        (left, right)
    } else {
        (right, left)
    };
    let stem = shorter.chars().count();

    stem >= SAME_WORD_STEM_CHARS
        && longer.chars().count() - stem <= SAME_WORD_SUFFIX_CHARS
        && longer.starts_with(shorter)
}

/// The tokens of a name that carry a digit, sorted and deduplicated so two
/// names can be compared on them alone.
///
/// A version, an ordinal or a year decides an entity by itself, and it is the
/// one token an embedder has least to work with: `v1` and `v2` are two
/// characters, one of them shared, hung off a name whose every other word is
/// identical. So these are held to equality rather than to the containment
/// rule the words are — `billing-service-v2` is not `Billing Service`, even
/// though the only difference is on one side.
fn numbered(tokens: &[String]) -> Vec<&String> {
    let mut numbered: Vec<&String> = tokens
        .iter()
        .filter(|token| token.chars().any(|character| character.is_numeric()))
        .collect();
    numbered.sort_unstable();
    numbered.dedup();
    numbered
}

/// Whether one side of a pair holds enough of the other for a difference
/// between them to read as a disagreement rather than as two different names.
///
/// Stated without a fraction on purpose: the names must agree on at least as
/// many words as they differ on. `Acme Corporation` against `Acme Holdings` is
/// one word shared and one word apart, which is a substitution. `Phoenix Data
/// Platform` against `Ingest Platform` shares only the head noun, and calling
/// that a contradiction would veto every paraphrase the vector tier exists to
/// catch — the tier would collapse into a slower copy of the exact-match tier
/// above it.
fn comparable(matched: &[bool]) -> bool {
    matched.len() - unmatched(matched) >= unmatched(matched)
}

/// What the ladder concluded, before anything has been written.
enum Resolution {
    /// Attach to this existing node.
    ///
    /// Carries which tier found it, because one caller does need to know:
    /// [`Store::look_up_node`] hands the answer to a model, and "this is the id
    /// you quoted" and "this is the nearest name I could find" are not the same
    /// claim. The write path ignores both extra fields.
    Existing {
        node_id: String,
        matched_by: MatchedBy,
        similarity: Option<f64>,
    },
    /// Mint a node. A non-empty candidate list means the decision was close
    /// enough that the caller should be told what it nearly matched.
    Fresh(Vec<NodeCandidate>),
}

/// Runs the resolution ladder for one input. Reads only.
///
/// The three tiers are ordered by how much they can be trusted: an id the
/// server itself minted, then a name some agent explicitly recorded, then a
/// number produced by a model. Each tier only runs when the one above it found
/// nothing.
async fn resolve(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    input: &NodeInput,
    node_id: &str,
    forms: &[String],
    vector: &[f32],
    model_id: &str,
) -> Result<Resolution, StoreError> {
    // Tier 1: the id the server would mint for this name is already taken, so
    // this is the same entity arriving again.
    if let Some(matched) = follow_merges(connection, tenant, node_id).await? {
        return Ok(Resolution::Existing {
            node_id: matched,
            matched_by: MatchedBy::NodeId,
            similarity: None,
        });
    }

    // Tier 2: some node already answers to one of these names, either as its
    // display name under different capitalisation or as a recorded alias.
    let exact: Vec<(String,)> = sqlx::query_as(
        "SELECT node_id FROM node \
         WHERE tenant_id = $1 \
           AND EXISTS ( \
                 SELECT 1 FROM unnest($2::text[]) AS form(f) \
                 WHERE lower(f) = lower(node.display_name) \
                    OR EXISTS ( \
                         SELECT 1 FROM unnest(node.aliases) AS existing(a) \
                         WHERE lower(a) = lower(f)) ) \
         ORDER BY node_id",
    )
    .bind(tenant)
    .bind(forms)
    .fetch_all(&mut *connection)
    .await?;

    // Two rows can be one entity if one has since been merged into the other,
    // so the tombstones are followed before the answers are counted.
    let mut targets: Vec<String> = Vec::new();
    for (candidate,) in &exact {
        let target = follow_merges(&mut *connection, tenant, candidate)
            .await?
            .unwrap_or_else(|| candidate.clone());
        if !targets.contains(&target) {
            targets.push(target);
        }
    }

    match targets.len() {
        0 => {}
        1 => {
            return Ok(Resolution::Existing {
                node_id: targets.remove(0),
                matched_by: MatchedBy::NameOrAlias,
                similarity: None,
            });
        }
        // An exact match is only decisive when it is unique. Two nodes
        // answering to one name is a collision, and taking either would be an
        // over-merge dressed up as certainty.
        _ => {
            return Ok(Resolution::Fresh(
                describe_candidates(connection, tenant, &targets, vector).await?,
            ));
        }
    }

    // Tier 3: nothing in the graph carries this name, so the only remaining
    // evidence is how close it sits to a name that is there.
    //
    // Merged nodes are excluded rather than followed: the partial index this
    // ordering rides on is `WHERE merged_into IS NULL`, and a tombstone's
    // vector is a stale copy of a name its successor already carries. Models
    // are compared explicitly — a cosine between two different embedding spaces
    // is a number with no meaning, and the schema keeps `embedding_model`
    // precisely so that is detectable rather than silent.
    let neighbours: Vec<(String, String, f64)> = sqlx::query_as(
        "SELECT node_id, display_name, 1 - (name_vector <=> $2::real[]::vector) \
         FROM node \
         WHERE tenant_id = $1 \
           AND merged_into IS NULL \
           AND name_vector IS NOT NULL \
           AND embedding_model = $3 \
         ORDER BY name_vector <=> $2::real[]::vector \
         LIMIT $4",
    )
    .bind(tenant)
    .bind(vector)
    .bind(model_id)
    .bind(CANDIDATE_LIMIT)
    .fetch_all(&mut *connection)
    .await?;

    let candidates: Vec<NodeCandidate> = neighbours
        .into_iter()
        .map(|(node_id, display_name, similarity)| NodeCandidate {
            node_id,
            display_name,
            similarity,
        })
        .collect();

    let thresholds = Thresholds::for_name(&input.display_name);
    let confident: Vec<&NodeCandidate> = candidates
        .iter()
        .filter(|candidate| candidate.similarity >= thresholds.confident)
        .collect();

    // Exactly one confident neighbour is the only shape that justifies
    // attaching. Two of them means the name is genuinely shared, which is
    // ambiguity however high the scores are.
    //
    // And the cosine does not get the last word even then. Measured on the
    // real model, the same-entity and distinct-entity populations interleave
    // across this band — 0.9451 for two companies, 0.9295 for one service
    // written two ways — so a confident neighbour whose name *contradicts* the
    // incoming one is dropped back into the ambiguous band. The names are
    // compared, not the ids: `display_name` is the text the vector was
    // computed from, so this checks exactly the claim the cosine made, and it
    // reads a column tier 3 already selected rather than asking the database
    // anything more.
    if confident.len() == 1 && !names_contradict(&input.display_name, &confident[0].display_name) {
        return Ok(Resolution::Existing {
            node_id: confident[0].node_id.clone(),
            matched_by: MatchedBy::Similarity,
            similarity: Some(confident[0].similarity),
        });
    }

    // Everything still in contention. Reported, never attached to: the caller —
    // or, one layer up, the agent — is better placed to say whether these are
    // the same entity, and creating a duplicate meanwhile is the failure a
    // later merge can undo.
    Ok(Resolution::Fresh(
        candidates
            .into_iter()
            .filter(|candidate| candidate.similarity >= thresholds.ambiguous)
            .collect(),
    ))
}

/// Fills in display names and true cosines for nodes the exact tiers found, so
/// a contested lexical match reports the same shape as a contested vector one
/// rather than a placeholder score.
async fn describe_candidates(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_ids: &[String],
    vector: &[f32],
) -> Result<Vec<NodeCandidate>, StoreError> {
    let rows: Vec<(String, String, Option<f64>)> = sqlx::query_as(
        "SELECT node_id, display_name, 1 - (name_vector <=> $3::real[]::vector) \
         FROM node \
         WHERE tenant_id = $1 AND node_id = ANY($2::text[]) \
         ORDER BY name_vector <=> $3::real[]::vector",
    )
    .bind(tenant)
    .bind(node_ids)
    .bind(vector)
    .fetch_all(connection)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(node_id, display_name, similarity)| NodeCandidate {
            node_id,
            display_name,
            // A node with no `name_vector` — one written before an embedder was
            // configured — is reported at zero rather than omitted. It is still
            // a contender on lexical grounds and dropping it would hide a
            // collision.
            similarity: similarity.unwrap_or(0.0),
        })
        .collect())
}

/// How many near misses to offer a caller that named a node this tenant does
/// not have. Enough to cover a plausible mistake, few enough that an agent can
/// read the list and pick.
const SUGGESTION_LIMIT: i64 = 5;

/// The shortest fragment of a mistyped id worth matching on. Below this,
/// fragments like `to` or `de` appear in half a tenant's ids and the
/// suggestions stop being suggestions.
const SUGGESTION_FRAGMENT_CHARS: usize = 3;

/// Ids in this tenant that a caller naming `named` may have meant.
///
/// Lexical rather than vector-based, because the thing being matched is an id
/// the server itself minted and the caller is quoting back. A miss is therefore
/// a spelling or a transcription problem — `alicia-whitfield` for
/// `alice-whitfield`, or the display name where the handle belongs — and the
/// nearest name in embedding space answers a question nobody asked.
///
/// `named` is run through [`canonical_node_id`] first, so a caller that sent
/// `"Alice Whitfield"` is told the id it should have sent. Matching is by
/// substring containment in either direction plus shared fragments, all through
/// `strpos` rather than `LIKE`: the caller's text is untrusted, and `LIKE` would
/// read a `%` in it as a pattern.
pub(crate) async fn suggest_nodes(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    named: &str,
) -> Result<Vec<NodeSuggestion>, StoreError> {
    let probe = canonical_node_id(named);
    let fragments: Vec<String> = probe
        .split('-')
        .filter(|fragment| fragment.chars().count() >= SUGGESTION_FRAGMENT_CHARS)
        .map(str::to_string)
        .collect();

    // Merged nodes are left out. A tombstone id still resolves — that is what
    // `follow_merges` is for — so a caller only reaches this path with an id
    // that answers to nothing, and pointing it at a dead node would send it
    // back with a second id that also cannot carry an edge.
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT node_id, display_name FROM node \
         WHERE tenant_id = $1 \
           AND merged_into IS NULL \
           AND (strpos(node_id, $2) > 0 \
                OR strpos($2, node_id) > 0 \
                OR EXISTS ( \
                      SELECT 1 FROM unnest($3::text[]) AS fragment(f) \
                      WHERE strpos(node_id, f) > 0)) \
         ORDER BY (node_id = $2) DESC, \
                  (SELECT count(*) FROM unnest($3::text[]) AS fragment(f) \
                   WHERE strpos(node_id, f) > 0) DESC, \
                  abs(length(node_id) - length($2)), \
                  node_id \
         LIMIT $4",
    )
    .bind(tenant)
    .bind(&probe)
    .bind(&fragments)
    .bind(SUGGESTION_LIMIT)
    .fetch_all(connection)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(node_id, display_name)| NodeSuggestion {
            node_id,
            display_name,
        })
        .collect())
}

/// Walks `merged_into` from `node_id` to the node that is still live, returning
/// `None` if `node_id` does not exist in this tenant at all.
///
/// A merge leaves the old row in place as a tombstone, because an agent may be
/// holding the old id in a context this server cannot reach or correct. The
/// chain can be longer than one hop — a successor can itself be merged later —
/// so this follows it to the end.
pub(crate) async fn follow_merges(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_id: &str,
) -> Result<Option<String>, StoreError> {
    Ok(merge_chain(connection, tenant, node_id)
        .await?
        .map(|chain| {
            chain
                .into_iter()
                .next_back()
                .expect("a chain always holds at least the node it started from")
        }))
}

/// Every node from `node_id` to the live one at the end of its `merged_into`
/// chain, in order, or `None` if `node_id` does not exist in this tenant.
///
/// [`follow_merges`] wants only the last element and is written in terms of
/// this, so there is one walk and one cycle guard rather than two. The merge
/// path wants the whole chain: to know whether writing `loser -> winner` would
/// close a loop, it has to ask whether the loser is somewhere on the winner's
/// chain, and "somewhere" is a question the endpoint alone cannot answer.
pub(crate) async fn merge_chain(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_id: &str,
) -> Result<Option<Vec<String>>, StoreError> {
    let mut current = node_id.to_string();
    let mut seen = vec![current.clone()];

    loop {
        let row: Option<(Option<String>,)> =
            sqlx::query_as("SELECT merged_into FROM node WHERE tenant_id = $1 AND node_id = $2")
                .bind(tenant)
                .bind(&current)
                .fetch_optional(&mut *connection)
                .await?;

        let Some((successor,)) = row else {
            // Only reachable on the first iteration: `merged_into` is a foreign
            // key, so every later hop is guaranteed to exist.
            return Ok(None);
        };

        let Some(successor) = successor else {
            return Ok(Some(seen));
        };

        // The schema forbids a node merging into itself, but nothing in it
        // forbids A into B into A, which two separately reasonable merges can
        // produce. Left unguarded that is an unbounded loop holding an open
        // transaction: it does not fail the request, it takes the process with
        // it.
        //
        // `merge_nodes` refuses to write the second of those two merges, so a
        // graph this server built alone cannot reach here. An operator's UPDATE
        // or an import can, and this stays the last line.
        if seen.contains(&successor) {
            return Err(StoreError::MergeCycle {
                tenant: tenant.to_string(),
                node_id: successor,
            });
        }

        seen.push(successor.clone());
        current = successor;
    }
}

/// Appends whichever of `forms` the node does not already answer to.
///
/// The filtering happens inside the `UPDATE` rather than in a read-then-write,
/// so two writers recording different new names for the same node cannot lose
/// one another's append. `WITH ORDINALITY` keeps the caller's order; `array_agg`
/// over a bare `unnest` would be free to permute it.
pub(super) async fn record_surface_forms(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    node_id: &str,
    forms: &[String],
) -> Result<(), StoreError> {
    sqlx::query(
        "UPDATE node SET aliases = node.aliases || COALESCE(( \
             SELECT array_agg(f ORDER BY ord) \
             FROM unnest($3::text[]) WITH ORDINALITY AS form(f, ord) \
             WHERE lower(f) <> lower(node.display_name) \
               AND NOT EXISTS ( \
                     SELECT 1 FROM unnest(node.aliases) AS existing(a) \
                     WHERE lower(a) = lower(f)) \
         ), '{}'::text[]) \
         WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(tenant)
    .bind(node_id)
    .bind(forms)
    .execute(connection)
    .await?;

    Ok(())
}

/// Every name an input claims for its entity: the display name first, then the
/// aliases, with case-insensitive repeats dropped so a caller passing
/// `["Phoenix", "phoenix"]` does not record both.
///
/// Order is preserved rather than sorted, because the display name has to stay
/// in front — it is the form the node is named after.
fn surface_forms(input: &NodeInput) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    let mut forms: Vec<String> = Vec::new();

    for form in std::iter::once(&input.display_name).chain(&input.aliases) {
        let folded = form.to_lowercase();
        if form.trim().is_empty() || seen.contains(&folded) {
            continue;
        }
        seen.push(folded);
        forms.push(form.clone());
    }

    forms
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The id is the handle an agent threads back into every later write, so it
    /// has to be predictable from the name the agent already said.
    #[test]
    fn a_display_name_canonicalises_to_a_kebab_case_id() {
        assert_eq!(canonical_node_id("Project Phoenix"), "project-phoenix");
        assert_eq!(canonical_node_id("Ada Lovelace"), "ada-lovelace");
    }

    /// Punctuation runs collapse rather than each becoming a hyphen, and the
    /// ends are trimmed: `"  Acme, Inc. "` and `"Acme Inc"` are the same entity
    /// as far as a caller is concerned.
    #[test]
    fn punctuation_runs_collapse_and_the_ends_are_trimmed() {
        assert_eq!(canonical_node_id("  Acme,  Inc. "), "acme-inc");
        assert_eq!(canonical_node_id("--Phoenix--"), "phoenix");
        assert_eq!(canonical_node_id("A / B / C"), "a-b-c");
    }

    /// Case is the most common way the same entity arrives twice.
    #[test]
    fn canonicalisation_is_case_insensitive() {
        assert_eq!(
            canonical_node_id("PROJECT PHOENIX"),
            canonical_node_id("project phoenix")
        );
    }

    /// `is_alphanumeric` is Unicode-aware on purpose: stripping non-ASCII would
    /// turn every entity named in a non-Latin script into the same empty id.
    #[test]
    fn non_latin_scripts_survive_canonicalisation() {
        assert_eq!(canonical_node_id("東京 タワー"), "東京-タワー");
    }

    /// The two pairs a calibration run of `BAAI/bge-base-en-v1.5` over 422
    /// labelled pairs caught the vector tier fusing. Both clear
    /// [`CONFIDENT_SIMILARITY`] on the real model, both are distinct entities,
    /// and no threshold separates them from the true matches around them — the
    /// same-entity pair `billing-service-v2` / `the v2 billing service` scores
    /// 0.9295, *between* the two. The names are the only evidence left.
    #[test]
    fn a_substituted_token_is_a_contradiction_no_cosine_can_override() {
        assert!(
            names_contradict("Acme Corporation", "Acme Holdings"),
            "0.9451 measured: sibling legal entities, one token apart"
        );
        assert!(
            names_contradict("billing-service-v2", "billing-service-v1"),
            "0.9250 measured: distinct deployed versions"
        );
    }

    /// The true matches that sit in and around the same band, and the whole
    /// reason a threshold cannot do this job: 0.9295 and 0.9286 are *below* the
    /// 0.9451 over-merge above, and 0.9454 is barely above it. What is
    /// different about them is not the number, it is that one name only ever
    /// adds to the other — a word order, an article, an honorific, a hyphen.
    #[test]
    fn word_order_articles_honorifics_and_punctuation_are_not_contradictions() {
        for (left, right, cosine, difference) in [
            (
                "billing-service-v2",
                "the v2 billing service",
                0.9295,
                "slug against prose",
            ),
            ("Project Phoenix", "Phoenix Project", 0.9454, "word order"),
            (
                "Acme Corporation",
                "The Acme Corporation",
                0.9875,
                "article",
            ),
            ("Dr. Katherine Chen", "Katherine Chen", 0.9286, "honorific"),
            ("GPT-4", "GPT4", 0.9409, "punctuation"),
        ] {
            assert!(
                !names_contradict(left, right),
                "{left} / {right} is one entity written twice ({difference}), \
                 measured at {cosine}"
            );
        }
    }

    /// A version or an ordinal is the one word that decides an entity on its
    /// own, and it is exactly the word an embedder is worst at: `v1` and `v2`
    /// are two characters, one of them shared, in a name whose every other
    /// token is identical.
    ///
    /// Both of these are *one-sided* — the rule above reads extra words on one
    /// side as an elaboration and would merge a numbered deployment into the
    /// unnumbered service it was cut from.
    #[test]
    fn a_version_or_ordinal_token_is_decisive_even_on_one_side_alone() {
        assert!(
            names_contradict("billing-service-v2", "Billing Service"),
            "a versioned deployment is not the service it was cut from"
        );
        assert!(
            names_contradict("Q3 Planning", "Planning"),
            "the quarter is the entity, not a decoration on it"
        );
    }

    /// The same word with a suffix on it is not a second word. Both of these
    /// clear the confident bar comfortably on the real model — 0.9607 and
    /// 0.9827 — and a guard that read `Reviews` as contradicting `Review`
    /// would fragment the graph on an English plural.
    ///
    /// The rule has to stay narrow enough to keep telling `Success` and
    /// `Support` apart, which is the same shape as the over-merges it exists
    /// to catch.
    #[test]
    fn a_plural_or_a_spelled_out_product_name_is_not_a_contradiction() {
        assert!(
            !names_contradict("Quarterly Business Review", "Quarterly Business Reviews"),
            "0.9607 measured: an English plural"
        );
        assert!(
            !names_contradict("Postgres Migration Runbook", "PostgreSQL Migration Runbook"),
            "0.9827 measured: one product, two spellings"
        );
        assert!(
            names_contradict("Customer Success Platform", "Customer Support Platform"),
            "0.8423 measured: two platforms sharing a prefix is still a substitution"
        );
    }

    /// An abbreviation is the words it stands for. Without this the guard
    /// costs the single highest-scoring true match in the whole measured
    /// corpus — 0.9938 — for the crime of writing two words as two letters.
    #[test]
    fn an_initialism_does_not_contradict_the_words_it_stands_for() {
        assert!(
            !names_contradict(
                "United States Department of Energy",
                "US Department of Energy"
            ),
            "0.9938 measured: US is United States"
        );
        assert!(
            !names_contradict("San Francisco Bay Area", "the SF Bay Area"),
            "0.9367 measured: SF is San Francisco"
        );
        assert!(
            names_contradict("Acme Corporation", "Acme Holdings"),
            "an initialism is a run of first letters, not a licence to \
             leave a word unaccounted for"
        );
    }

    /// Two names that share little are not disagreeing, they are just two
    /// names, and the third tier exists precisely for the paraphrase no agent
    /// has ever written down. A guard that vetoed those would not be a guard on
    /// the vector tier, it would be its removal — everything left would already
    /// have matched exactly one tier higher.
    ///
    /// So the veto needs the two names to be substantially the same name
    /// before a difference between them counts as a contradiction.
    #[test]
    fn names_with_little_in_common_are_left_to_the_vector() {
        assert!(!names_contradict(
            "Distributed Tracing Pipeline",
            "Observability Stack"
        ));
        assert!(!names_contradict(
            "Phoenix Data Platform",
            "Ingest Platform"
        ));
    }

    /// The guard's blind spot, pinned rather than hidden.
    ///
    /// An abbreviation that is not a run of initials is indistinguishable from
    /// a substitution: `K8s` is a numeronym, so the words around it match and
    /// the one word that differs looks exactly like `Corporation` against
    /// `Holdings`. The pair measures 0.8190 — far below any bar in this file —
    /// so the guard is never asked about it today. It would start costing a
    /// true match the moment somebody lowered [`CONFIDENT_SIMILARITY`], and
    /// this is where they should find that out.
    #[test]
    fn an_abbreviation_that_is_not_an_initialism_reads_as_a_substitution() {
        assert!(names_contradict(
            "K8s Control Plane",
            "Kubernetes Control Plane"
        ));
    }

    /// An empty `node_id` would be a primary key value that no agent can quote
    /// back and that collides with every other unnameable name in the tenant.
    /// A digest of the original text keeps it unique, non-empty and — because it
    /// is derived, not random — idempotent across calls.
    #[test]
    fn a_name_with_no_alphanumerics_gets_a_non_empty_derived_id() {
        let id = canonical_node_id("!!!");

        assert!(!id.is_empty());
        assert!(
            id.starts_with(UNNAMEABLE_ID_PREFIX),
            "{id} should be visibly synthetic"
        );
        assert_eq!(id, canonical_node_id("!!!"), "the fallback must be stable");
        assert_ne!(
            id,
            canonical_node_id("???"),
            "different unnameable names must not collide"
        );
    }
}
