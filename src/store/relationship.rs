//! Relationship type coercion.
//!
//! The type vocabulary is an open set: an agent may write a verb the server has
//! never seen, and the server registers it. Coercion runs first so that the set
//! does not fragment into `USES`, `uses`, `utilizes` and `Utilises`, all of them
//! true, none of them retrievable together.
//!
//! The set is open *per tenant*, and that is a security boundary rather than a
//! nicety. What an agent coins is a phrase out of a conversation this server
//! does not trust, so it is registered in the caller's own scope; the curated
//! set — [`GLOBAL_SCOPE`], the rows the migration seeded — is readable by every
//! tenant and writable by none. Each coercion therefore sees exactly two halves
//! of the table, `tenant_id IN ('', caller)`, and prefers the curated half when
//! a name is in both. Nothing else is visible: a vocabulary one tenant could
//! read across is a vocabulary that leaks the other's content, and the leak
//! would arrive at the second tenant looking like its own canonical type.
//!
//! Three tiers, stopping at the first hit, ordered by how much each can be
//! trusted: a spelling rule, then a synonym some operator wrote down, then a
//! number produced by a model.
//!
//! Only the third tier can be wrong in a way that matters, and it can be wrong
//! in a way that nothing downstream detects. Inverse relations — `OWNS`/
//! `OWNED_BY`, `REPORTS_TO`/`MANAGES` — are as close in embedding space as any
//! two synonyms while meaning the opposite thing, so folding one onto the other
//! reverses the direction of every fact written with it and leaves a
//! `proof_excerpt` that still reads correctly. That is why tier 3, and only tier
//! 3, is overruled by a direction guard.
//!
//! The guard has two halves and both are consulted: `relationship_inverse_pair`,
//! a curated table, and [`are_mechanical_inverses`], a rule about English
//! spelling. Neither subsumes the other. The table holds pairs made of unrelated
//! words — `REPORTS_TO`/`MANAGES`, `LOCATED_IN`/`CONTAINS` — that no rule will
//! ever derive; the rule covers the pairs nobody has curated yet, which in an
//! open vocabulary is the majority of the ones that will actually arrive.
//!
//! **The guard cannot be retired in favour of a better threshold, and it is
//! worth being explicit about why, because the shape of the guard invites the
//! assumption that it is belt-and-braces.** Calibration against the real
//! `bge-base-en-v1.5` measured both populations: the highest inverse pair,
//! `AUTHORED_BY`/`AUTHORED` at 0.8692, outscores 10 of the 14 true synonyms in
//! the same corpus. The two populations are not adjacent, they are interleaved,
//! so no value of [`CONFIDENT_SIMILARITY`] separates them. A bar high enough to
//! exclude every inverse excludes most real synonyms with it, which is the
//! fragmentation this module exists to prevent. Direction is not a matter of
//! degree and no amount of cosine will make it one.

use super::Store;
use crate::embed::Embedder;
use crate::error::{EmbedError, StoreError};

/// The half of the vocabulary every tenant shares.
///
/// Empty on purpose: no deployment names a tenant that, so the sentinel cannot
/// collide with one. It is also refused as a *caller's* tenant, a few lines into
/// [`coerce_relationship_type`], because a credential carrying an empty claim
/// would otherwise arrive here indistinguishable from permission to write the
/// curated set. Readable by every tenant, writable by none of them.
pub const GLOBAL_SCOPE: &str = "";

/// One relationship type: as the vocabulary knows it, where in the vocabulary
/// that is, and as the caller wrote it.
///
/// `raw` is carried verbatim, never normalised, and is persisted beside every
/// edge. Coercion is a judgement call made by a normaliser, a lookup table and a
/// model; without the caller's own words next to the result, a coercion that
/// went wrong is neither detectable nor reversible.
///
/// `scope` is [`GLOBAL_SCOPE`] for a curated type and the owning tenant for a
/// private one. It is not decoration: `edge` references the vocabulary by
/// `(scope, name)`, because a name alone stopped identifying a row the moment
/// two tenants could both hold one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoercedType {
    pub canonical: String,
    pub raw: String,
    pub scope: String,
}

impl CoercedType {
    fn new(scope: String, canonical: String, raw: &str) -> Self {
        Self {
            canonical,
            raw: raw.to_string(),
            scope,
        }
    }
}

/// The cardinality every auto-registered type gets.
///
/// `many` is the conservative half of the enum, not merely the common one. Task
/// 6 supersedes on cardinality: a `one` type retires the incumbent edge when a
/// newer fact arrives, a `many` type never does. Nobody has described what an
/// auto-registered type means, so guessing `one` would silently retire facts on
/// the strength of a name the server has never seen before.
const AUTO_REGISTERED_CARDINALITY: &str = "many";

/// Cosine at or above which two relationship names are the same relation.
///
/// Provisional, and knowingly so: Task 12 exists to replace it with a measured
/// value, because the numbers in the original spec were guesses and models of
/// the `bge-base` class compress their similarity range badly enough that a
/// figure which sounds strict in the abstract merges half the vocabulary.
///
/// Sits above the 0.92 the node resolver uses for long names and below the 0.97
/// it uses for short ones. Relationship names are short strings, which is the
/// regime where unrelated embeddings crowd together, so the long-name bar would
/// be too generous. The short-name bar is not adopted wholesale because the two
/// failures are not equally bad: an over-merged *node* silently fuses two
/// entities and cannot be told apart afterwards, whereas an over-merged *type*
/// lands in a small table an operator can read at a glance, and every edge keeps
/// the caller's own wording in `raw_relationship_type`. The exception is the
/// inverse pair, where a wrong coercion is not untidy but false — which is why
/// that case has an explicit guard rather than a higher number.
pub const CONFIDENT_SIMILARITY: f64 = 0.95;

impl Store {
    /// Resolves `raw` onto the vocabulary `tenant` can see — the curated set
    /// plus that tenant's own — registering it privately if nothing there fits.
    pub async fn coerce_relationship_type(
        &self,
        tenant: &str,
        raw: &str,
        embedder: &dyn Embedder,
    ) -> Result<CoercedType, StoreError> {
        let mut transaction = self.pool.begin().await?;
        let coerced = coerce_relationship_type(&mut transaction, tenant, raw, embedder).await?;
        transaction.commit().await?;

        Ok(coerced)
    }
}

/// Resolves `raw` onto the vocabulary `tenant` can see, registering it in that
/// tenant's own scope if nothing there fits.
///
/// Takes a connection rather than a pool so an edge write can run this inside
/// the transaction that will reference the type, the same way the node resolver
/// runs inside the transaction that writes the node. The returned
/// [`CoercedType::scope`] is the other half of that reference: `edge` points at
/// `(scope, canonical)`, not at a name.
pub async fn coerce_relationship_type(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    raw: &str,
    embedder: &dyn Embedder,
) -> Result<CoercedType, StoreError> {
    // Refused before anything is read or written. `GLOBAL_SCOPE` is the empty
    // string, so a caller whose tenant is empty is not an unusual tenant, it is
    // the curated vocabulary — and would auto-register into the half of the
    // table every other tenant reads. A credential without a tenant claim is
    // already rejected by `Authenticator`; a claim that is present and empty is
    // one IdP mapping away, and this is where that becomes a privilege.
    if tenant == GLOBAL_SCOPE {
        return Err(StoreError::BlankTenant);
    }

    let normalised = normalise(raw);

    // Refused before anything is read or embedded. A name that folds to nothing
    // is a caller bug rather than a new relation, and the empty string is a
    // primary key value that `edge.relationship_type` would then reference.
    if normalised.is_empty() {
        return Err(StoreError::BlankRelationshipType {
            raw: raw.to_string(),
        });
    }

    // Tier 1: the vocabulary already holds the form this name folds onto.
    if let Some((scope, canonical)) = registered_type(&mut *connection, tenant, &normalised).await?
    {
        return Ok(CoercedType::new(scope, canonical, raw));
    }

    // Tier 2: an operator has already written down that this word means an
    // existing type. Both tiers are exact lookups, so neither needs the
    // direction guard tier 3 carries: an alias row is a human decision about
    // this exact pair of names, not an inference from a distance.
    if let Some(canonical) = aliased_type(&mut *connection, &normalised).await? {
        return Ok(CoercedType::new(GLOBAL_SCOPE.to_string(), canonical, raw));
    }

    // Still the exact tiers, with one more spelling to try. It is offered last,
    // after both tables have missed on the name as written, because stripping a
    // plural is a guess about English while a row in either table is a fact
    // about this vocabulary — and half the seeded vocabulary is itself plural.
    if let Some(singular) = singular_candidate(&normalised) {
        if let Some((scope, canonical)) =
            registered_type(&mut *connection, tenant, &singular).await?
        {
            return Ok(CoercedType::new(scope, canonical, raw));
        }

        if let Some(canonical) = aliased_type(&mut *connection, &singular).await? {
            return Ok(CoercedType::new(GLOBAL_SCOPE.to_string(), canonical, raw));
        }
    }

    // Tier 3: nothing in the vocabulary is spelled this way or recorded as
    // meaning this, so the only remaining evidence is how close it sits to a
    // name that is there.
    backfill_missing_name_vectors(&mut *connection, tenant, embedder).await?;

    // `pop` rather than indexing: the trait promises one vector per input, but
    // an embedder that broke that promise would panic a request handler here
    // instead of failing it.
    let vector = embedder
        .embed_documents(std::slice::from_ref(&normalised))
        .await?
        .pop()
        .ok_or_else(|| {
            EmbedError::Inference(format!(
                "no vector returned for relationship type `{normalised}`"
            ))
        })?;

    // The direction guard, in two halves that both have to pass. The spelling
    // rule is asked first because it costs nothing and covers the pairs nobody
    // has curated yet; the table is asked second because it covers the pairs no
    // spelling rule can reach. Neither subsumes the other, so neither replaces
    // the other.
    //
    // A close neighbour that is an inverse is refused outright rather than
    // skipped over in favour of the next-nearest: an inverse sitting closest is
    // evidence that this corner of the space is direction-ambiguous, and the
    // second choice was made in the same space by the same model. Registering
    // the name as its own type is the only outcome here that cannot reverse a
    // fact.
    if let Some((scope, candidate, similarity)) =
        nearest_type(&mut *connection, tenant, &vector, embedder.model_id()).await?
        && similarity >= CONFIDENT_SIMILARITY
        && !are_mechanical_inverses(&normalised, &candidate)
        && !are_registered_inverses(&mut *connection, &normalised, &candidate).await?
    {
        return Ok(CoercedType::new(scope, candidate, raw));
    }

    // Written into the caller's own scope, never the curated one. A coined name
    // is untrusted text out of one tenant's conversation, and the curated set is
    // the half every tenant reads.
    //
    // `ON CONFLICT DO NOTHING` rather than a bare insert. A tenant runs many
    // agents, and two of them coining the same verb at the same moment both read
    // a vocabulary without it, because under read-committed neither can see the
    // other's uncommitted row. Postgres blocks the second insert until the first
    // commits and then reports no row inserted, which is the signal that this is
    // a match after all rather than a failure the caller could have avoided.
    sqlx::query(
        "INSERT INTO relationship_type \
         (tenant_id, name, cardinality, name_vector, embedding_model, auto_registered) \
         VALUES ($1, $2, $3::relationship_cardinality, $4::real[]::vector, $5, TRUE) \
         ON CONFLICT (tenant_id, name) DO NOTHING",
    )
    .bind(tenant)
    .bind(&normalised)
    .bind(AUTO_REGISTERED_CARDINALITY)
    // Already computed for the tier-3 comparison a few lines up. Storing it
    // makes the new type a candidate for the next coercion immediately, instead
    // of a row the backfill has to come back for.
    .bind(&vector)
    // And the model beside it, without which that candidacy is void: tier 3
    // will not compare a vector it cannot attribute.
    .bind(embedder.model_id())
    .execute(connection)
    .await?;

    Ok(CoercedType::new(tenant.to_string(), normalised, raw))
}

/// Resolves `raw` onto a type the vocabulary already holds, **writing
/// nothing**, or `None` when nothing there is spelled that way.
///
/// For a caller that is *filtering* on a type rather than asserting a fact:
/// `explore_entity` takes a `relationship_type` and has to reach `DEPENDS_ON`
/// from `"depends on"` the same way [`coerce_relationship_type`] does. It runs
/// the same [`normalise`], then the same exact tiers in the same order —
/// [`registered_type`], [`aliased_type`], and both again against
/// [`singular_candidate`] — so a filter and a write agree about what a caller's
/// wording means, and neither restates the other's rules.
///
/// **Tier 3 is deliberately not here, and the reason is not only that it
/// writes.** It does write — it backfills missing name vectors and registers
/// the name when nothing fits — and a read that enlarged the shared vocabulary
/// every time an agent mistyped a filter would leave junk types behind that
/// later *writes* would then coerce onto. But the semantics are the stronger
/// argument. On the write path a tier-3 coercion is recorded with the caller's
/// own words beside it in `raw_relationship_type`, so a wrong guess is visible
/// and reversible. On a filter there is nowhere to record anything: a cosine
/// that folded `PROXIES_TO` onto `PUBLISHES_TO` would quietly answer a question
/// the caller did not ask, and the caller would read the result as the answer
/// to the one it did.
///
/// A name that folds to nothing is refused rather than treated as "no filter",
/// for the reason the write path refuses it: silently returning the unfiltered
/// neighbourhood to a caller that asked for one type is the one outcome it
/// cannot detect.
pub(crate) async fn registered_type_for_lookup(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    raw: &str,
) -> Result<Option<String>, StoreError> {
    let normalised = normalise(raw);

    if normalised.is_empty() {
        return Err(StoreError::BlankRelationshipType {
            raw: raw.to_string(),
        });
    }

    for candidate in std::iter::once(normalised.clone()).chain(singular_candidate(&normalised)) {
        if let Some((_, canonical)) = registered_type(&mut *connection, tenant, &candidate).await? {
            return Ok(Some(canonical));
        }

        if let Some(canonical) = aliased_type(&mut *connection, &candidate).await? {
            return Ok(Some(canonical));
        }
    }

    Ok(None)
}

/// Returns the scope and name of `name` if the vocabulary this tenant can see
/// already holds it.
///
/// The visible vocabulary is the curated set plus this tenant's own, and no
/// third thing.
///
/// A name held in both is the curated one. The curated set is what an operator
/// maintains and what every tenant already shares, so a private row that
/// shadowed it would leave one tenant meaning something of its own by a word the
/// rest of the deployment has agreed on — with nothing to see, because the two
/// are spelled the same. The ordering is written out rather than left to the
/// planner: which of two matching rows comes back first is not something a query
/// without `ORDER BY` promises, and this one is a rule, not a preference.
async fn registered_type(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    name: &str,
) -> Result<Option<(String, String)>, StoreError> {
    let row: Option<(String, String)> = sqlx::query_as(
        "SELECT tenant_id, name FROM relationship_type \
         WHERE name = $2 AND tenant_id IN ('', $1) \
         ORDER BY (tenant_id = '') DESC \
         LIMIT 1",
    )
    .bind(tenant)
    .bind(name)
    .fetch_optional(connection)
    .await?;

    Ok(row)
}

/// Whether the vocabulary records these two names as opposite directions of one
/// relation.
///
/// Checked in both column orders because the table records an unordered pair:
/// `('REPORTS_TO', 'MANAGES')` is one row, not two, and a guard that only looked
/// one way would let half of every pair through.
///
/// One of the two ways tier 3 can be overruled — [`are_mechanical_inverses`] is
/// the other — and it exists because the failure it prevents is invisible.
/// `MANAGES` and `REPORTS_TO` are as close in embedding space as any two
/// synonyms, so a similarity match between them looks exactly like a good one;
/// accepting it would flip the direction of every fact written with the incoming
/// type while leaving a `proof_excerpt` that still reads correctly. Nothing
/// downstream — not search, not an audit, not a rating — can tell a reversed
/// edge from a true one.
///
/// The table is what the spelling rule cannot be: `REPORTS_TO`/`MANAGES`,
/// `LOCATED_IN`/`CONTAINS` and `DEPENDS_ON`/`REQUIRED_BY` are inverse pairs
/// built from unrelated words, and the only thing that knows they are pairs is
/// a person who wrote them down. Curation is also the escape hatch for what the
/// rule gets wrong in the other direction — an irregular participle like
/// `PAYS`/`PAID_BY` is a row, not a patch.
async fn are_registered_inverses(
    connection: &mut sqlx::PgConnection,
    incoming: &str,
    candidate: &str,
) -> Result<bool, StoreError> {
    let (inverse,): (bool,) = sqlx::query_as(
        "SELECT EXISTS ( \
             SELECT 1 FROM relationship_inverse_pair \
             WHERE (type_a = $1 AND type_b = $2) \
                OR (type_a = $2 AND type_b = $1))",
    )
    .bind(incoming)
    .bind(candidate)
    .fetch_one(connection)
    .await?;

    Ok(inverse)
}

/// The suffix English uses to name the passive half of a relation: the agent
/// the action was performed *by*.
const PASSIVE_AGENT_SUFFIX: &str = "_BY";

/// The two ends of the possessive shape. `A HAS_X B` and `B X_OF A` state one
/// fact from opposite ends, so the two names are inverses of each other.
const POSSESSIVE_PREFIX: &str = "HAS_";
const POSSESSIVE_SUFFIX: &str = "_OF";

/// Verb endings stripped to reach a stem, beyond the trailing `S` that
/// [`plural_stem`] already knows the rules for. `ES` is the third person of a
/// stem ending in a sibilant (`WATCHES`); `ED` and `D` are the two spellings of
/// the past participle (`FUNDED`, `MANAGED`); `ING` is the present participle,
/// which an agent writes as readily as the finite verb (`MENTORING` beside
/// `MENTORS`).
///
/// All four are conjugations of one verb rather than four kinds of evidence, so
/// this list widens what counts as the same stem — it does not add a shape a
/// pair can have. Each is tried against the name as written, never against the
/// output of another: one ending is a conjugation, two stacked is a guess.
const INFLECTED_ENDINGS: [&str; 4] = ["ING", "ES", "ED", "D"];

/// The shortest remainder an ending may be stripped down to.
///
/// Below three letters what is left is a fragment rather than a word, and a
/// match between two fragments is a coincidence: `USED` minus `ED` is `US`,
/// which would happily equal the `US` of some unrelated name. The floor applies
/// only to stripping — a name is always compared as written as well, so a
/// genuinely two-letter stem is still matched against its own spelling.
const MIN_STEM_CHARS: usize = 3;

/// Whether two relationship names are opposite directions of one relation on
/// the evidence of their spelling alone.
///
/// This is the half of the direction guard that needs no table.
/// `relationship_inverse_pair` is curated, and curation only ever covers the
/// pairs somebody thought of: calibration against the real model found
/// `MENTORS`/`MENTORED_BY` at 0.8450, `MANAGES`/`MANAGED_BY` at 0.7375,
/// `FUNDS`/`FUNDED_BY` at 0.7243 and `SUPERSEDES`/`SUPERSEDED_BY` at 0.6159,
/// none of them in the table. The four share one shape, and a shape is a rule
/// rather than four more rows that the fifth coinage would miss — which matters
/// because the vocabulary is an open set, so the fifth coinage is not
/// hypothetical.
///
/// Two morphologies, both of which have to survive the objection that
/// `AUTHORED_BY` is itself a seeded canonical type:
///
/// * **`X` against `X_BY`.** The passive agent. Read on the suffix alone this
///   would refuse every coercion onto `AUTHORED_BY`, including the paraphrases
///   tier 3 exists to fold in, so the suffix is not the evidence: the *stem*
///   under it has to match the other name. `AUTHORED_BY`/`AUTHORED` is a pair;
///   `AUTHORED_BY`/`USES` is not, and neither is `AUTHORED_BY`/`WRITTEN_BY`.
///   The stems are compared through one inflection on each side, because the
///   pairs that actually occur are inflected differently on the two halves —
///   `MENTORS` beside `MENTORED_BY`, not `MENTORED` beside `MENTORED_BY`.
/// * **`HAS_X` against `X_OF`.** `A HAS_PART B` and `B PART_OF A` are the same
///   fact. Both ends of the shape are required, because `_OF` on its own is the
///   commonest ending in this vocabulary — `MEMBER_OF`, `PART_OF`,
///   `COMPONENT_OF` — and says nothing about direction by itself.
///
/// Deliberately no more than that. `REPORTS_TO`/`MANAGES` and
/// `LOCATED_IN`/`CONTAINS` are inverse pairs made of unrelated words, and no
/// rule about English endings will ever reach them, so the curated table is not
/// replaced by this and both are consulted. The known gaps in what is here are
/// irregular participles (`PAYS`/`PAID_BY`) and copula-prefixed forms
/// (`IS_PART_OF` against `HAS_PART`); the second is really a tier-1
/// normalisation the server does not do, and doing it there would fix this for
/// free rather than teaching this rule more English than the normaliser knows.
///
/// The rule leans towards saying yes, because the two failures are not
/// comparable. A false positive refuses a coercion that should have happened:
/// the name registers as its own type, an operator sees two words for one
/// relation in a small table and writes one alias row. A false negative folds a
/// relation onto its own opposite and reverses the direction of every fact
/// written with it, leaving a `proof_excerpt` that still reads correctly. The
/// first is untidy and recoverable; the second is silent and is not.
pub fn are_mechanical_inverses(left: &str, right: &str) -> bool {
    is_passive_agent_of(left, right)
        || is_passive_agent_of(right, left)
        || is_possessive_inverse_of(left, right)
        || is_possessive_inverse_of(right, left)
}

/// Whether `passive` is the passive-agent form of the verb in `active`.
///
/// Asked in one direction only; [`are_mechanical_inverses`] asks it both ways.
fn is_passive_agent_of(passive: &str, active: &str) -> bool {
    let Some(participle) = passive.strip_suffix(PASSIVE_AGENT_SUFFIX) else {
        return false;
    };

    share_a_stem(participle, active)
}

/// Whether `having` and `belonging` are the two ends of `HAS_X` / `X_OF`.
fn is_possessive_inverse_of(having: &str, belonging: &str) -> bool {
    let Some(had) = having.strip_prefix(POSSESSIVE_PREFIX) else {
        return false;
    };
    let Some(belonged) = belonging.strip_suffix(POSSESSIVE_SUFFIX) else {
        return false;
    };

    share_a_stem(had, belonged)
}

/// Whether two words are the same word under one inflection on either side.
///
/// Empty on either side is not a match. Nothing reaches tier 3 with an empty
/// name — the caller refuses it long before — but a stripped suffix can leave
/// one, and two empty stems comparing equal would make every such name the
/// inverse of every other.
fn share_a_stem(left: &str, right: &str) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }

    let right_stems = stems(right);
    stems(left).iter().any(|stem| right_stems.contains(stem))
}

/// Every form of `name` this rule is willing to treat as its stem: the name as
/// written, plus the result of undoing one inflection.
///
/// The trailing `S` goes through [`plural_stem`] rather than being stripped
/// here, so the one rule this file holds about English `S` — `ACCESS` and
/// `STATUS` are not plurals — serves the exact tiers and the direction guard
/// alike. It stands in for the third person singular as well as the plural:
/// `MENTORS` is the former and takes the same `S` off.
fn stems(name: &str) -> Vec<&str> {
    let mut stems = Vec::with_capacity(1 + INFLECTED_ENDINGS.len());
    stems.push(name);

    if let Some(stem) = plural_stem(name) {
        stems.push(stem);
    }

    for ending in INFLECTED_ENDINGS {
        if let Some(stem) = name.strip_suffix(ending)
            && stem.chars().count() >= MIN_STEM_CHARS
        {
            stems.push(stem);
        }
    }

    stems
}

/// The nearest vectored type to `vector`, with its scope and cosine.
///
/// A type whose `name_vector` is NULL simply cannot match. The seeded rows ship
/// that way — the migration cannot embed anything, because the model is a
/// runtime choice — which is why [`backfill_missing_name_vectors`] runs first.
///
/// Models are compared explicitly, the way the node resolver compares them. A
/// cosine between vectors from two different embedding spaces is a number with
/// no meaning, and it is a number that still sorts, so an unfiltered nearest
/// neighbour would answer confidently and wrongly with nothing to show for it.
/// A row from another model is therefore not a worse candidate, it is not a
/// candidate.
async fn nearest_type(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    vector: &[f32],
    model_id: &str,
) -> Result<Option<(String, String, f64)>, StoreError> {
    let row: Option<(String, String, f64)> = sqlx::query_as(
        "SELECT tenant_id, name, 1 - (name_vector <=> $2::real[]::vector) \
         FROM relationship_type \
         WHERE tenant_id IN ('', $1) \
           AND name_vector IS NOT NULL \
           AND embedding_model = $3 \
         ORDER BY name_vector <=> $2::real[]::vector \
         LIMIT 1",
    )
    .bind(tenant)
    .bind(vector)
    .bind(model_id)
    .fetch_optional(connection)
    .await?;

    Ok(row)
}

/// Embeds the name of every type that has no vector yet.
///
/// Done here rather than in the migration because a migration has no embedder:
/// which model produces these vectors is a runtime choice, and a vector written
/// by SQL would be a vector from no model at all. Done lazily rather than at
/// startup because coercion is the only thing that reads the column, and a
/// server that never coerces should not pay for it.
///
/// Only NULL rows are touched, so this is idempotent and costs one `SELECT` on
/// every call after the first. The curated set is filled in as readily as the
/// caller's own vocabulary: those are the rows the migration seeded, and every
/// tenant needs them vectored to reach them at tier 3.
///
/// Every filled row records the model that filled it, so tier 3 can refuse to
/// compare across spaces rather than silently doing it.
///
/// Note for whoever swaps in a real model: this backfill is not the migration
/// path. It only touches rows with no vector at all, so rows carrying the old
/// model's vectors stay as they are — visible to the old model and invisible to
/// the new one, which is the safe direction but not a working deployment.
/// Changing models means re-embedding every row, which is a scripted operator
/// action, not something a request should do on its way past.
async fn backfill_missing_name_vectors(
    connection: &mut sqlx::PgConnection,
    tenant: &str,
    embedder: &dyn Embedder,
) -> Result<(), StoreError> {
    // Ordered so that two writers backfilling at the same moment take row locks
    // in the same sequence and cannot deadlock against each other.
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT tenant_id, name FROM relationship_type \
         WHERE tenant_id IN ('', $1) AND name_vector IS NULL \
         ORDER BY tenant_id, name",
    )
    .bind(tenant)
    .fetch_all(&mut *connection)
    .await?;

    if rows.is_empty() {
        return Ok(());
    }

    let names: Vec<String> = rows.iter().map(|(_, name)| name.clone()).collect();
    let vectors = embedder.embed_documents(&names).await?;

    for ((scope, name), vector) in rows.iter().zip(vectors) {
        // `AND name_vector IS NULL` so a writer that lost the race overwrites
        // nothing: the row it was about to fill has already been filled, and
        // rewriting it would churn a value another transaction may have read.
        sqlx::query(
            "UPDATE relationship_type \
             SET name_vector = $3::real[]::vector, embedding_model = $4 \
             WHERE tenant_id = $1 AND name = $2 AND name_vector IS NULL",
        )
        .bind(scope)
        .bind(name)
        .bind(&vector)
        .bind(embedder.model_id())
        .execute(&mut *connection)
        .await?;
    }

    Ok(())
}

/// Returns the canonical type `alias` is a recorded synonym of.
///
/// The one lookup here with no tenant in it, because there is no tenant-scoped
/// alias to find. Aliases are curated — nothing an agent can reach writes one —
/// and the table's foreign key names `relationship_type (tenant_id, name)` with
/// a generated `''` on its own side, so a row pointing anywhere but the curated
/// vocabulary cannot be inserted. Every answer this returns is therefore
/// [`GLOBAL_SCOPE`], and it is the schema that guarantees it rather than this
/// query.
async fn aliased_type(
    connection: &mut sqlx::PgConnection,
    alias: &str,
) -> Result<Option<String>, StoreError> {
    let row: Option<(String,)> =
        sqlx::query_as("SELECT canonical FROM relationship_type_alias WHERE alias = $1")
            .bind(alias)
            .fetch_optional(connection)
            .await?;

    Ok(row.map(|(canonical,)| canonical))
}

/// The separator every run of spacing punctuation collapses to.
const SEPARATOR: char = '_';

/// Terminal letter pairs that only look like a plural. Stripping the `S` from
/// `ACCESS`, `STATUS`, `BASIS` or `ALIAS` leaves a stem that is not a word, and
/// a stem that is not a word is a lookup that can only match by accident.
const FALSE_PLURAL_ENDINGS: [&str; 4] = ["SS", "US", "IS", "AS"];

/// How long the final token has to be before its trailing `S` is read as a
/// plural. `OWNS` and `USES` clear it; `HAS`, `IS` and initialisms do not.
const MIN_PLURAL_TOKEN_CHARS: usize = 4;

/// Folds one written relationship type onto the single form this server would
/// register it under: upper case, `_` between tokens, nothing on the ends.
///
/// Only whitespace, `-` and `_` are treated as token separators. Broadening
/// that to "every non-alphanumeric character" would quietly rewrite names that
/// carry punctuation deliberately, and relationship types are verbs, not
/// arbitrary text.
///
/// Returns an empty string when the input contains nothing but separators. That
/// is a caller error rather than a type, and the caller refuses it — registering
/// it would put an unquotable empty primary key in the vocabulary that every
/// other blank input would then collide with.
fn normalise(raw: &str) -> String {
    let mut normalised = String::with_capacity(raw.len());
    let mut separator_pending = false;

    for character in raw.chars() {
        if character.is_whitespace() || character == '-' || character == '_' {
            separator_pending = true;
            continue;
        }

        if separator_pending && !normalised.is_empty() {
            normalised.push(SEPARATOR);
        }
        separator_pending = false;
        normalised.extend(character.to_uppercase());
    }

    rejoin_dangling_letter(normalised)
}

/// Reattaches a one-character final token to the token before it, so `use s`
/// reaches `USES` rather than registering `USE_S`.
///
/// A single letter is never a meaningful token in a relationship type, so a
/// trailing one is a mistyped suffix — almost always the plural `S`. The rule is
/// deliberately confined to the last token and to exactly one character:
/// `A_TEAM` and `PART_OF` are two real tokens each and must survive intact.
fn rejoin_dangling_letter(normalised: String) -> String {
    let Some((head, tail)) = normalised.rsplit_once(SEPARATOR) else {
        return normalised;
    };

    if head.is_empty() || tail.chars().count() != 1 {
        return normalised;
    }

    format!("{head}{tail}")
}

/// The singular form of `name`, when its trailing `S` is plausibly a plural.
///
/// This is a *fallback* candidate for the exact-lookup tiers, offered only after
/// the name as written has missed. The order matters: half the seeded vocabulary
/// — `USES`, `OWNS`, `REPORTS_TO`, `PREFERS` — is itself plural, so a
/// normalisation that stripped the `S` up front would make those types
/// unreachable and register `USE` beside `USES`.
///
/// Because the candidate is only ever accepted when it names a type that already
/// exists, the heuristics in [`plural_stem`] are a cheap first filter rather
/// than the safeguard. They exist so the obvious English false plurals never
/// reach a lookup at all.
fn singular_candidate(name: &str) -> Option<String> {
    plural_stem(name).map(str::to_string)
}

/// `name` with a trailing plural `S` taken off, when the `S` is plausibly one.
///
/// Split out of [`singular_candidate`] because the direction guard's morphology
/// rule asks the same question of a stem it is about to compare, and one rule
/// about English `S` in two places is one rule that will diverge. The `S` of
/// the third person singular is the same character under the same constraints —
/// `MENTORS` is a verb, not a plural noun, and loses it identically.
fn plural_stem(name: &str) -> Option<&str> {
    let stem = name.strip_suffix('S')?;

    let last_token = name.rsplit(SEPARATOR).next().unwrap_or(name);
    if last_token.chars().count() < MIN_PLURAL_TOKEN_CHARS {
        return None;
    }

    if FALSE_PLURAL_ENDINGS
        .iter()
        .any(|ending| name.ends_with(ending))
    {
        return None;
    }

    Some(stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of tier 1: the vocabulary must not fragment into `USES`,
    /// `uses`, `Uses` and `use-s` because four agents typed the same verb four
    /// ways.
    #[test]
    fn case_and_separator_variants_collapse_to_one_upper_snake_form() {
        for raw in ["uses", "Uses", "USES", "  uses  "] {
            assert_eq!(normalise(raw), "USES", "{raw} should normalise to USES");
        }

        for raw in [
            "reports to",
            "reports-to",
            "reports_to",
            "Reports   To",
            "reports--to",
            "reports _ to",
        ] {
            assert_eq!(
                normalise(raw),
                "REPORTS_TO",
                "{raw} should normalise to REPORTS_TO"
            );
        }
    }

    /// Leading and trailing separators carry no meaning and would otherwise
    /// register `_OWNS` as a type distinct from `OWNS`.
    #[test]
    fn leading_and_trailing_separators_are_trimmed() {
        assert_eq!(normalise("_owns_"), "OWNS");
        assert_eq!(normalise("--depends-on--"), "DEPENDS_ON");
    }

    /// `use s` is a slip, not a two-token type: a one-letter trailing segment is
    /// never a meaningful part of a relationship name, so it rejoins the word it
    /// was split off from rather than registering `USE_S`.
    #[test]
    fn a_dangling_one_letter_trailing_segment_rejoins_the_word_before_it() {
        assert_eq!(normalise("use s"), "USES");
        assert_eq!(normalise("owns"), "OWNS");
        // Only the trailing segment, and only when it is a single character:
        // `A_TEAM` is two real tokens.
        assert_eq!(normalise("a team"), "A_TEAM");
    }

    /// A name with nothing but separators in it has no type behind it. It is
    /// reported as empty rather than being registered as an unquotable
    /// primary key that every other blank input would then collide with.
    #[test]
    fn a_name_with_no_usable_characters_normalises_to_nothing() {
        assert!(normalise("").is_empty());
        assert!(normalise("   ").is_empty());
        assert!(normalise("___").is_empty());
        assert!(normalise(" - _ - ").is_empty());
    }

    /// The singular candidate is a *fallback* form, offered to the exact-lookup
    /// tiers only after the name as written has missed. `USES` is itself a
    /// registered type, so it must never be rewritten to `USE` before that
    /// lookup happens — the ordering is what keeps the seeded vocabulary
    /// reachable.
    #[test]
    fn a_trailing_plural_s_yields_a_singular_candidate() {
        assert_eq!(singular_candidate("MENTORS").as_deref(), Some("MENTOR"));
        assert_eq!(singular_candidate("USES").as_deref(), Some("USE"));
        assert_eq!(
            singular_candidate("HAS_MEMBERS").as_deref(),
            Some("HAS_MEMBER")
        );
    }

    /// English words ending in `SS`, `US`, `IS` or `AS` are not plurals, and a
    /// name whose final token is short is more likely an abbreviation than a
    /// pluralised verb. Stripping either produces a stem that is not a word.
    #[test]
    fn an_s_that_is_not_a_plural_is_left_alone() {
        for name in ["ACCESS", "STATUS", "BASIS", "ALIAS", "HAS", "IS", "OWNS_AS"] {
            assert_eq!(
                singular_candidate(name),
                None,
                "{name} does not end in a plural S"
            );
        }
    }

    /// Nothing to strip is not an error, it is the common case.
    #[test]
    fn a_name_that_does_not_end_in_s_has_no_singular_candidate() {
        assert_eq!(singular_candidate("PART_OF"), None);
        assert_eq!(singular_candidate("DEPENDS_ON"), None);
        assert_eq!(singular_candidate(""), None);
    }

    /// Asserts the rule both ways round. The table it stands beside records an
    /// unordered pair and is queried in both column orders; a morphology rule
    /// that only worked one way would let half of every pair through for
    /// exactly the same reason.
    fn assert_mechanical(left: &str, right: &str) {
        assert!(
            are_mechanical_inverses(left, right),
            "{left} / {right} is a mechanical inverse pair"
        );
        assert!(
            are_mechanical_inverses(right, left),
            "{right} / {left} is the same pair, read the other way"
        );
    }

    fn assert_not_mechanical(left: &str, right: &str) {
        assert!(
            !are_mechanical_inverses(left, right),
            "{left} / {right} is not a mechanical inverse pair"
        );
        assert!(
            !are_mechanical_inverses(right, left),
            "{right} / {left} is not one either"
        );
    }

    /// **The hole this rule exists to close.** Calibration against the real
    /// model found these four inverse pairs scoring high and absent from
    /// `relationship_inverse_pair` — the highest, `MENTORS`/`MENTORED_BY` at
    /// 0.8450, one threshold adjustment away from being coerced onto its own
    /// opposite.
    ///
    /// Every one of them is `X` against `X_ED_BY`: the active verb on one side
    /// and its passive agent on the other. That is a rule, so it is written as
    /// one rather than as four more table rows that the fifth coinage would
    /// miss.
    #[test]
    fn the_unseeded_inverse_pairs_the_calibration_found_are_detected_mechanically() {
        assert_mechanical("MENTORS", "MENTORED_BY");
        assert_mechanical("MANAGES", "MANAGED_BY");
        assert_mechanical("FUNDS", "FUNDED_BY");
        assert_mechanical("SUPERSEDES", "SUPERSEDED_BY");
    }

    /// **The trap.** `AUTHORED_BY` is itself a seeded canonical type, so a rule
    /// that read "ends in `_BY`" as "is somebody's inverse" would refuse every
    /// coercion onto it — including the paraphrases the whole tier exists to
    /// fold in. The suffix is not the evidence; the stem underneath it, matched
    /// against the other name, is.
    #[test]
    fn a_by_suffix_alone_does_not_make_two_types_inverse() {
        for other in ["USES", "OWNS", "RELATED_TO", "LOCATED_IN"] {
            assert_not_mechanical("AUTHORED_BY", other);
        }

        // The ones that cost something if this is got wrong: real synonyms of
        // `AUTHORED_BY`, two of them already tier-2 aliases of it.
        for synonym in ["WRITTEN_BY", "CREATED_BY", "PENNED_BY", "DRAFTED_BY"] {
            assert_not_mechanical("AUTHORED_BY", synonym);
        }
    }

    /// The passive-agent morphology, in its plainest form: the stem is spelled
    /// identically on both sides, and one of them carries `_BY`. This is the
    /// pair the migration already seeds, derived rather than looked up.
    #[test]
    fn the_passive_agent_morphology_pairs_a_participle_with_its_agent() {
        assert_mechanical("AUTHORED", "AUTHORED_BY");
        assert_mechanical("HOSTED", "HOSTED_BY");
        // And the inflected form of the same rule, which is what the seeded
        // vocabulary actually contains.
        assert_mechanical("OWNS", "OWNED_BY");
        assert_mechanical("USES", "USED_BY");

        // The present participle is the same verb, and an agent writing
        // `MENTORING` has said what `MENTORS` says.
        assert_mechanical("MENTORING", "MENTORED_BY");
        assert_mechanical("FUNDING", "FUNDED_BY");

        // Two agents of different verbs are not each other's inverse — they are
        // two unrelated relations that happen to share a suffix.
        assert_not_mechanical("AUTHORED_BY", "HOSTED_BY");
        assert_not_mechanical("MENTORING", "FUNDED_BY");
        assert_not_mechanical("REVIEWED", "AUTHORED_BY");
        // A stem that merely starts the same is not the same stem.
        assert_not_mechanical("OWES", "OWNED_BY");
    }

    /// The possessive morphology: `A HAS_X B` says the same thing as
    /// `B X_OF A`, so the two names describe one relation read from opposite
    /// ends. It is productive — `HAS_OWNER`/`OWNER_OF`, `HAS_PARENT`/
    /// `PARENT_OF` — which is what makes it worth a rule.
    ///
    /// Both halves of the shape are required. `_OF` on its own is the commonest
    /// ending in the vocabulary (`MEMBER_OF`, `PART_OF`, `COMPONENT_OF`) and
    /// says nothing about direction by itself.
    #[test]
    fn the_possessive_morphology_pairs_has_x_with_x_of() {
        assert_mechanical("HAS_PART", "PART_OF");
        assert_mechanical("HAS_MEMBER", "MEMBER_OF");
        assert_mechanical("HAS_OWNER", "OWNER_OF");
        // Number is an inflection like any other: `HAS_PARTS` is `HAS_PART`.
        assert_mechanical("HAS_PARTS", "PART_OF");

        // Two `_OF` names are two relations, not a pair.
        assert_not_mechanical("PART_OF", "MEMBER_OF");
        assert_not_mechanical("MEMBER_OF", "COMPONENT_OF");
        // Two `HAS_` names likewise.
        assert_not_mechanical("HAS_PART", "HAS_MEMBER");
        // And the shape is only a pair when the stems agree.
        assert_not_mechanical("HAS_PART", "MEMBER_OF");
        assert_not_mechanical("HAS_OWNER", "PART_OF");
        // `_OF` is not the passive agent: `IS_OWNER_OF` is a synonym of `OWNS`,
        // and refusing it would refuse a coercion the tier exists to make.
        assert_not_mechanical("OWNS", "IS_OWNER_OF");
    }

    /// **Why the curated table is not replaced.** These four are inverse pairs
    /// that share no morphology at all: the two names are different words, and
    /// no rule about English suffixes can connect them. They are why
    /// `relationship_inverse_pair` stays, and why this rule is written to run
    /// beside it rather than in place of it.
    #[test]
    fn the_suppletive_inverse_pairs_are_invisible_to_any_morphology_rule() {
        assert_not_mechanical("REPORTS_TO", "MANAGES");
        assert_not_mechanical("LOCATED_IN", "CONTAINS");
        assert_not_mechanical("DEPENDS_ON", "REQUIRED_BY");
        assert_not_mechanical("PART_OF", "HAS_WHOLE");
    }

    /// The cost of a false positive is a coercion refused, which is recoverable
    /// — the type registers separately and an operator can add an alias. It is
    /// still a cost, and this is the population that would pay it: every
    /// `Same`-labelled relationship pair the calibration corpus measures.
    ///
    /// None of them may be refused, and the reason none of them is is
    /// structural rather than lucky: the rule never looks at two names unless
    /// one of them carries `_BY`, or one carries `HAS_` and the other `_OF`.
    #[test]
    fn no_true_synonym_in_the_calibration_corpus_is_refused() {
        for (left, right) in [
            ("USES", "UTILIZES"),
            ("USES", "LEVERAGES"),
            ("USES", "EMPLOYS"),
            ("DEPENDS_ON", "RELIES_ON"),
            ("DEPENDS_ON", "REQUIRES"),
            ("OWNS", "IS_OWNER_OF"),
            ("OWNS", "POSSESSES"),
            ("AUTHORED_BY", "WRITTEN_BY"),
            ("AUTHORED_BY", "PENNED_BY"),
            ("LOCATED_IN", "SITUATED_IN"),
            ("MEMBER_OF", "BELONGS_TO"),
            ("PREFERS", "FAVOURS"),
            ("REPORTS_TO", "REPORTING_TO"),
            ("MANAGES", "OVERSEES"),
        ] {
            assert_not_mechanical(left, right);
        }
    }

    /// A name is not its own inverse, and a fragment of a word is not a stem.
    /// Neither case can arise from a real coercion — tier 1 answers before tier
    /// 3 is asked about an identical name — but a rule that said yes to either
    /// would be saying yes for a reason that has nothing to do with direction.
    #[test]
    fn degenerate_names_are_not_inverses() {
        assert_not_mechanical("AUTHORED_BY", "AUTHORED_BY");
        assert_not_mechanical("OWNS", "OWNS");
        assert_not_mechanical("BY", "BY");
        assert_not_mechanical("HAS_PART", "HAS_PART");
        assert_not_mechanical("", "");
        assert_not_mechanical("_BY", "");
    }
}
