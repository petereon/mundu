//! Integration tests for relationship type coercion, against a real Postgres.
//!
//! `relationship_type` is scoped in two halves. `tenant_id = ''` is the global
//! curated set — the seeded vocabulary and the aliases that reach it — and every
//! other value is one tenant's private vocabulary. So tenancy isolates these
//! tests the way it isolates the node suite, but only for what a test registers:
//! a test that seeds *curated* rows is writing to a table every other test can
//! see, and the suite runs in parallel.
//!
//! Both mechanisms are therefore in use. Every test coerces under a tenant named
//! after itself, and every test also names the types it owns after itself. Two
//! rules follow from the second:
//!
//! * a test may create, read and delete types carrying its own suffix, in any
//!   scope;
//! * a test must never delete or rewrite a seeded row, because a concurrently
//!   running test — or the next task — is reading it.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use mundu::embed::{DeterministicEmbedder, Embedder};
use mundu::error::StoreError;
use mundu::store::Store;
use mundu::store::relationship::{GLOBAL_SCOPE, coerce_relationship_type};

/// One [`Store`] per test, for the same reason the node suite builds one per
/// test: a sqlx pool is bound to the reactor that created it, and every
/// `#[tokio::test]` gets its own.
async fn store() -> Store {
    common::store().await
}

fn embedder() -> DeterministicEmbedder {
    DeterministicEmbedder::new()
}

/// Names a tenant after the test that owns it and empties its private
/// vocabulary, the way the node suite does with its nodes.
///
/// A tenant id is never a tool argument in this server — it is derived from a
/// validated credential — so a test standing in for a caller has to supply one
/// explicitly, and two tests must not share.
///
/// The delete is confined to that tenant's own scope, so it can never reach the
/// curated set a concurrently running test is reading.
async fn fresh_tenant(store: &Store, name: &str) -> String {
    let tenant = format!("test-{name}");

    sqlx::query("DELETE FROM relationship_type WHERE tenant_id = $1")
        .bind(&tenant)
        .execute(store.pool())
        .await
        .expect("clearing the tenant's vocabulary should succeed");

    tenant
}

/// Claims a per-test corner of the vocabulary — every scope of it — and clears
/// whatever a previous run left in it, so "this call registered the type" means
/// the same thing on the first run and the hundredth.
///
/// Matching on the suffix rather than on an exact list is deliberate. A test
/// that fails midway can leave a *badly* normalised row behind — the run that
/// first drove out `case_variants_all_canonicalise_to_the_seeded_type` deposited
/// a literal `uses` row — and an exact-name delete would not find it, so the
/// next run would inherit it. Every suffix below is a word that appears in no
/// seeded name, which is what keeps the pattern from reaching the curated
/// vocabulary.
async fn own_suffix(store: &Store, suffix: &str) {
    assert!(
        suffix.len() >= 6 && suffix.chars().all(|c| c.is_ascii_uppercase()),
        "{suffix} is too short or too generic to be a safe per-test marker"
    );

    let pattern = format!("%{suffix}%");

    sqlx::query(
        "DELETE FROM relationship_inverse_pair \
         WHERE upper(type_a) LIKE $1 OR upper(type_b) LIKE $1",
    )
    .bind(&pattern)
    .execute(store.pool())
    .await
    .expect("clearing the test's inverse pairs should succeed");

    // Cascades to relationship_type_alias.
    sqlx::query("DELETE FROM relationship_type WHERE upper(name) LIKE $1")
        .bind(&pattern)
        .execute(store.pool())
        .await
        .expect("clearing the test's types should succeed");
}

/// Reads back the registration flags of one type in one scope.
async fn registration_of(store: &Store, scope: &str, name: &str) -> Option<(String, bool)> {
    sqlx::query_as(
        "SELECT cardinality::text, auto_registered FROM relationship_type \
         WHERE tenant_id = $1 AND name = $2",
    )
    .bind(scope)
    .bind(name)
    .fetch_optional(store.pool())
    .await
    .expect("reading the type should succeed")
}

/// Every scope holding a type of this name, in tenant order.
///
/// The shape of the answer is the whole point: one row means one vocabulary,
/// which is what the global table used to force on every tenant at once.
async fn scopes_of(store: &Store, name: &str) -> Vec<String> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT tenant_id FROM relationship_type WHERE name = $1 ORDER BY tenant_id",
    )
    .bind(name)
    .fetch_all(store.pool())
    .await
    .expect("reading the scopes should succeed");

    rows.into_iter().map(|(tenant_id,)| tenant_id).collect()
}

/// The vocabulary is an open set on purpose: an agent that can only choose from
/// a fixed list will pick the closest wrong type rather than say something new.
/// Anything that survives all three coercion tiers is registered as it was
/// written.
///
/// Cardinality is `many`, and that is not a default worth glossing over. Task 6
/// supersedes on cardinality: a `one` type retires the incumbent edge, a `many`
/// type never does. Guessing `one` for a type nobody has described would
/// silently retire facts.
#[tokio::test]
async fn a_novel_type_auto_registers_as_many() {
    let store = &store().await;
    own_suffix(store, "AUTOREG").await;
    let tenant = fresh_tenant(store, "autoreg").await;
    let name = "SPONSORS_AUTOREG";

    let coerced = store
        .coerce_relationship_type(&tenant, name, &embedder())
        .await
        .expect("registering a novel type should succeed");

    assert_eq!(coerced.canonical, name);
    assert_eq!(coerced.raw, name);

    let (cardinality, auto_registered) = registration_of(store, &tenant, name)
        .await
        .expect("the novel type should have been registered");
    assert_eq!(
        cardinality, "many",
        "an undescribed type must not be able to supersede an existing fact"
    );
    assert!(
        auto_registered,
        "an operator has to be able to tell the curated vocabulary from what agents invented"
    );

    // The name was embedded on the way here to ask tier 3 about it, so leaving
    // the column NULL would throw that away and leave the new type invisible to
    // the very tier that is supposed to fold the next paraphrase of it in.
    let (has_vector, model) = vector_and_model(store, &tenant, name).await;
    assert!(
        has_vector,
        "a registered type must be a candidate for the next coercion, not a blind spot"
    );
    // And a vector nothing identifies the model of is the same blind spot with
    // an extra step: tier 3 will not compare it, because it cannot know whether
    // the comparison would mean anything.
    assert_eq!(
        model.as_deref(),
        Some(embedder().model_id()),
        "the model that produced name_vector must be recorded beside it"
    );
}

/// Whether one type has a vector, and which model is recorded as having
/// produced it.
async fn vector_and_model(store: &Store, scope: &str, name: &str) -> (bool, Option<String>) {
    sqlx::query_as(
        "SELECT name_vector IS NOT NULL, embedding_model FROM relationship_type \
         WHERE tenant_id = $1 AND name = $2",
    )
    .bind(scope)
    .bind(name)
    .fetch_one(store.pool())
    .await
    .expect("reading the type should succeed")
}

/// **The property this scoping exists for.** The vocabulary used to be one
/// global table, so a verb coined by one tenant's agent — a verb derived from
/// untrusted text in that tenant's conversation — became part of every other
/// tenant's vocabulary. The moment anything can list the types, that is a
/// cross-tenant read of one tenant's content.
///
/// Two tenants coin the same name, one after the other. Each must end up with
/// its own private row: the second must not find the first's, and the curated
/// global set must not have grown at all.
#[tokio::test]
async fn a_type_one_tenant_coins_is_invisible_to_another_tenant() {
    let store = &store().await;
    own_suffix(store, "TENANTISO").await;
    let acme = fresh_tenant(store, "tenantiso-acme").await;
    let globex = fresh_tenant(store, "tenantiso-globex").await;
    let name = "SPONSORS_TENANTISO";

    let first = store
        .coerce_relationship_type(&acme, name, &embedder())
        .await
        .expect("the first tenant should register the type");
    assert_eq!(first.canonical, name);

    let second = store
        .coerce_relationship_type(&globex, name, &embedder())
        .await
        .expect("the second tenant should register its own type");
    assert_eq!(second.canonical, name);

    assert_eq!(
        scopes_of(store, name).await,
        vec![acme.clone(), globex.clone()],
        "each tenant must hold its own row; one shared row is the leak"
    );

    // Said again from the other direction, because it is the whole point: an
    // agent's coinage must never reach the curated set, which is the one scope
    // every tenant can read.
    assert!(
        !scopes_of(store, name)
            .await
            .contains(&GLOBAL_SCOPE.to_string()),
        "an agent-coined type must never land in the global vocabulary"
    );

    // Each row is genuinely its owner's: auto-registered under the tenant that
    // wrote it, not a second tenant attaching to the first's row.
    assert!(registration_of(store, &acme, name).await.is_some());
    assert!(registration_of(store, &globex, name).await.is_some());
}

/// The resolved type is only half an answer without the scope it was resolved
/// in. Task 6 writes both into `edge`, whose foreign key is composite: a
/// coercion that reported the wrong scope would point the edge at a row that
/// does not exist, or — worse, and only where the name happens to exist twice —
/// at a different tenant's type of the same name.
#[tokio::test]
async fn the_scope_of_the_resolved_type_comes_back_with_it() {
    let store = &store().await;
    own_suffix(store, "SCOPEOUT").await;
    let tenant = fresh_tenant(store, "scopeout").await;

    let curated = store
        .coerce_relationship_type(&tenant, "uses", &embedder())
        .await
        .expect("coercing a curated type should succeed");
    assert_eq!(curated.canonical, "USES");
    assert_eq!(
        curated.scope, GLOBAL_SCOPE,
        "USES is curated, and an edge pointing at it has to say so"
    );

    let registered = store
        .coerce_relationship_type(&tenant, "sponsors scopeout", &embedder())
        .await
        .expect("registering a novel type should succeed");
    assert_eq!(registered.canonical, "SPONSORS_SCOPEOUT");
    assert_eq!(
        registered.scope, tenant,
        "a type this tenant coined lives in this tenant's scope"
    );

    // The second arrival takes the other path — tier 1, not registration — and
    // has to describe the same row the same way.
    let again = store
        .coerce_relationship_type(&tenant, "sponsors scopeout", &embedder())
        .await
        .expect("re-coercing should succeed");
    assert_eq!(again, registered);
}

/// Auto-registration writes to the caller's scope and nowhere else.
///
/// `''` is the curated set: the one scope shared by every tenant, and the one
/// no credential can name. A registration that landed there would publish one
/// tenant's coinage to all of them, which is the same leak as the shared table
/// with an extra step.
#[tokio::test]
async fn auto_registration_files_the_type_under_the_calling_tenant() {
    let store = &store().await;
    own_suffix(store, "CALLERSCOPE").await;
    let tenant = fresh_tenant(store, "callerscope").await;

    store
        .coerce_relationship_type(&tenant, "sponsors callerscope", &embedder())
        .await
        .expect("registering a novel type should succeed");

    assert_eq!(
        scopes_of(store, "SPONSORS_CALLERSCOPE").await,
        vec![tenant],
        "the new type belongs to its caller, not to everyone"
    );
}

/// Counts the rows carrying one type name, in every scope. That total is the
/// interesting one: a second write of the same type by the same tenant must
/// find the first rather than attempt its own row, and a write by a *different*
/// tenant must not touch it at all.
async fn type_count(store: &Store, name: &str) -> i64 {
    let (count,): (i64,) = sqlx::query_as("SELECT count(*) FROM relationship_type WHERE name = $1")
        .bind(name)
        .fetch_one(store.pool())
        .await
        .expect("counting the type should succeed");
    count
}

/// Agents are stateless between calls, so the same verb arrives again and again.
/// The second arrival has to resolve to the type the first one registered.
#[tokio::test]
async fn coercing_the_same_novel_type_twice_registers_it_once() {
    let store = &store().await;
    own_suffix(store, "TWICEREG").await;
    let tenant = fresh_tenant(store, "twicereg").await;
    let name = "SPONSORS_TWICEREG";

    let first = store
        .coerce_relationship_type(&tenant, name, &embedder())
        .await
        .expect("the first coercion should succeed");
    let second = store
        .coerce_relationship_type(&tenant, name, &embedder())
        .await
        .expect("the second coercion should succeed");

    assert_eq!(first, second);
    assert_eq!(first.canonical, name);
    assert_eq!(type_count(store, name).await, 1);
}

/// The other half of the scoping. Isolation that also walled off the curated
/// vocabulary would leave every tenant to reinvent `USES` privately, which is
/// the fragmentation coercion exists to prevent — and it would do it silently,
/// because each tenant's own answers would look perfectly consistent.
///
/// The seeded set therefore answers for everyone, and answers *as itself*: a
/// private copy per tenant would satisfy `canonical == "USES"` while being
/// exactly the failure this rules out.
#[tokio::test]
async fn a_curated_type_resolves_for_any_tenant() {
    let store = &store().await;
    let acme = fresh_tenant(store, "curated-acme").await;
    let globex = fresh_tenant(store, "curated-globex").await;

    for tenant in [&acme, &globex] {
        let coerced = store
            .coerce_relationship_type(tenant, "uses", &embedder())
            .await
            .expect("coercing a curated type should succeed");

        assert_eq!(
            coerced.canonical, "USES",
            "{tenant} should have reached the seeded USES"
        );
    }

    let scopes = scopes_of(store, "USES").await;
    assert!(
        scopes.contains(&GLOBAL_SCOPE.to_string()),
        "the seeded USES must still be curated, not moved"
    );
    for tenant in [&acme, &globex] {
        assert!(
            !scopes.contains(tenant),
            "{tenant} forked its own USES instead of reading the curated one"
        );
    }
}

/// A name can exist in both halves, and when it does the curated half wins.
///
/// The state is ordinary rather than exotic: auto-registration only ever writes
/// privately, so a tenant coins `ESCALATES`, an operator later decides the word
/// is worth curating, and now there are two rows. Every tenant must converge on
/// the operator's, because the alternative is one tenant quietly keeping its own
/// meaning of a word the rest of the deployment has agreed on — a divergence
/// that is invisible, since both sides spell it identically.
#[tokio::test]
async fn a_name_in_both_scopes_resolves_to_the_curated_one() {
    let store = &store().await;
    own_suffix(store, "BOTHSCOPES").await;
    let tenant = fresh_tenant(store, "bothscopes").await;

    seed_type(store, GLOBAL_SCOPE, "ESCALATES_BOTHSCOPES").await;
    seed_type(store, &tenant, "ESCALATES_BOTHSCOPES").await;

    let coerced = store
        .coerce_relationship_type(&tenant, "escalates bothscopes", &embedder())
        .await
        .expect("coercing should succeed");

    assert_eq!(coerced.canonical, "ESCALATES_BOTHSCOPES");
    assert_eq!(
        coerced.scope, GLOBAL_SCOPE,
        "the curated vocabulary is the authoritative one; a private row must not shadow it"
    );
}

/// Tier 1. Four agents write the same verb four ways in one afternoon; without
/// this the vocabulary grows a row for each of them and no two of their facts
/// are ever retrieved together.
///
/// `USES` is seeded, so this also pins that the curated vocabulary is reachable
/// from what an agent actually types.
#[tokio::test]
async fn case_variants_all_canonicalise_to_the_seeded_type() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "case-variants").await;

    for raw in ["uses", "Uses", "USES", "  uses "] {
        let coerced = store
            .coerce_relationship_type(&tenant, raw, &embedder())
            .await
            .expect("coercing a case variant should succeed");

        assert_eq!(
            coerced.canonical, "USES",
            "{raw} should have reached the seeded USES"
        );
        assert_eq!(
            coerced.raw, raw,
            "the caller's own words must survive coercion unchanged"
        );
    }

    // Nothing new may have been registered on the way: the whole point of tier
    // 1 is that these four are one type, not four.
    assert_eq!(type_count(store, "uses").await, 0);
    assert_eq!(type_count(store, "Uses").await, 0);
    assert!(
        !scopes_of(store, "USES").await.contains(&tenant),
        "the four must have reached the curated USES, not a private copy of it"
    );
}

/// `raw` is what makes a coercion auditable. It is persisted on every edge, so
/// an operator who thinks a type was resolved wrongly can see precisely what the
/// agent wrote — including the case and spacing that the normaliser threw away —
/// and undo it. Folding it to the canonical form on the way out would erase the
/// only evidence.
#[tokio::test]
async fn the_raw_string_is_returned_exactly_as_the_caller_wrote_it() {
    let store = &store().await;
    own_suffix(store, "RAWFORM").await;
    let tenant = fresh_tenant(store, "rawform").await;
    let name = "MENTORS_RAWFORM";

    // A registration and a coercion take different paths out of the function,
    // and `raw` has to survive both.
    for raw in ["  Mentors-RawForm  ", "mentors rawform", "MENTORS_RAWFORM"] {
        let coerced = store
            .coerce_relationship_type(&tenant, raw, &embedder())
            .await
            .expect("coercing should succeed");

        assert_eq!(coerced.canonical, name);
        assert_eq!(coerced.raw, raw);
    }
}

/// Tier 2. `utilizes` is a synonym of `USES`, not a spelling of it, so no amount
/// of lexical folding reaches it — and leaving it to the model would make a
/// resolution that an operator already wrote down by hand depend on a similarity
/// score. The alias table is the cheapest and most certain evidence available.
#[tokio::test]
async fn a_seeded_alias_resolves_to_its_canonical_type() {
    let store = &store().await;
    // Removes the stray `UTILIZES` *type* row that a failing run of this test
    // deposits — which would then make the next run pass at tier 1 and never
    // reach the alias table at all. No seeded type is named this; the seeded
    // `UTILIZES` lives in relationship_type_alias, which this does not touch.
    own_suffix(store, "UTILIZES").await;
    let tenant = fresh_tenant(store, "seeded-alias").await;

    for raw in ["utilizes", "Utilizes", "UTILIZES"] {
        let coerced = store
            .coerce_relationship_type(&tenant, raw, &embedder())
            .await
            .expect("coercing a known alias should succeed");

        assert_eq!(coerced.canonical, "USES", "{raw} is a seeded alias of USES");
        assert_eq!(coerced.raw, raw);
    }

    assert_eq!(
        type_count(store, "UTILIZES").await,
        0,
        "an alias must resolve, not fork the vocabulary"
    );
}

async fn embed(text: &str) -> Vec<f32> {
    embedder()
        .embed_documents(&[text.to_string()])
        .await
        .expect("the deterministic embedder cannot fail")
        .remove(0)
}

/// Builds a unit vector whose cosine with `base` is exactly `cosine`.
///
/// `DeterministicEmbedder` is a hash: two texts either embed identically or land
/// near-orthogonal, with nothing in between. Tier 3 lives in between, so a
/// near-neighbour has to be constructed rather than found. `other` is projected
/// off `base` first, so the result is `cosine * base + sin * u` for a genuine
/// orthonormal `u` and the angle is exact rather than approximate.
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

/// Registers a type in `scope` sitting at a chosen cosine from `neighbour_of`,
/// vectored by the embedder the tests coerce with.
///
/// The vector is not the embedding of the type's own name, which is the only way
/// to put a controlled distance in front of tier 3.
async fn seed_type_near(store: &Store, scope: &str, name: &str, neighbour_of: &str, cosine: f64) {
    seed_type_near_from(
        store,
        scope,
        name,
        neighbour_of,
        cosine,
        embedder().model_id(),
    )
    .await;
}

/// The same row, recorded as the output of some other model.
///
/// The vector is still produced here, because the point is not to obtain a
/// genuinely foreign vector — it is that a *number* cannot tell you which space
/// it came from. Only the recorded model can.
async fn seed_type_near_from_another_model(
    store: &Store,
    scope: &str,
    name: &str,
    neighbour_of: &str,
    cosine: f64,
) {
    seed_type_near_from(
        store,
        scope,
        name,
        neighbour_of,
        cosine,
        "some-other-embedder-256",
    )
    .await;
}

async fn seed_type_near_from(
    store: &Store,
    scope: &str,
    name: &str,
    neighbour_of: &str,
    cosine: f64,
    model: &str,
) {
    let vector = at_cosine(
        &embed(neighbour_of).await,
        &embed("unrelated anchor").await,
        cosine,
    );

    sqlx::query(
        "INSERT INTO relationship_type \
         (tenant_id, name, cardinality, description, name_vector, embedding_model) \
         VALUES ($1, $2, 'many', 'seeded by a test', $3::real[]::vector, $4)",
    )
    .bind(scope)
    .bind(name)
    .bind(&vector)
    .bind(model)
    .execute(store.pool())
    .await
    .expect("seeding a type should succeed");
}

/// Registers a type in `scope` with no vector — the shape every seeded row has
/// until something backfills it.
async fn seed_type(store: &Store, scope: &str, name: &str) {
    sqlx::query(
        "INSERT INTO relationship_type (tenant_id, name, cardinality, description) \
         VALUES ($1, $2, 'many', 'seeded by a test')",
    )
    .bind(scope)
    .bind(name)
    .execute(store.pool())
    .await
    .expect("seeding a type should succeed");
}

/// Records a curated synonym. There is no scope argument because there is no
/// tenant-scoped alias: the table's foreign key reaches the global vocabulary
/// and nothing else, so `canonical` must already be a curated type.
async fn seed_alias(store: &Store, alias: &str, canonical: &str) {
    sqlx::query("INSERT INTO relationship_type_alias (alias, canonical) VALUES ($1, $2)")
        .bind(alias)
        .bind(canonical)
        .execute(store.pool())
        .await
        .expect("seeding an alias should succeed");
}

async fn record_inverse_pair(store: &Store, type_a: &str, type_b: &str) {
    sqlx::query("INSERT INTO relationship_inverse_pair (type_a, type_b) VALUES ($1, $2)")
        .bind(type_a)
        .bind(type_b)
        .execute(store.pool())
        .await
        .expect("recording an inverse pair should succeed");
}

/// Asserts that neither exact tier can resolve `name`, so whatever happens next
/// happened at tier 3.
async fn assert_unresolvable_by_exact_tiers(store: &Store, name: &str) {
    assert_eq!(
        type_count(store, name).await,
        0,
        "{name} must not already be a type, or tier 1 answers and tier 3 never runs"
    );

    let (aliased,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM relationship_type_alias WHERE alias = $1")
            .bind(name)
            .fetch_one(store.pool())
            .await
            .expect("counting aliases should succeed");
    assert_eq!(
        aliased, 0,
        "{name} must not be a recorded alias, or tier 2 answers and tier 3 never runs"
    );
}

/// **The direction guard.** Inverse relations are the one place where a coercion
/// is not merely untidy but false: fold `OWNED_BY` onto `OWNS` and every fact
/// written with it points the wrong way, while its `proof_excerpt` still reads
/// correctly and survives any audit that does not re-read the source.
///
/// The test is built so it cannot pass for the wrong reason. Both halves present
/// tier 3 with the *same* evidence — a curated neighbour at cosine 0.99, a name
/// no exact tier can resolve — and differ only in whether the pair is registered
/// as inverse:
///
/// * the control **must** be coerced, which fails if tier 3 is missing, broken,
///   or thresholded out of reach;
/// * the guarded pair **must not** be, which fails if the guard is missing.
///
/// A guard that refused everything would pass the second half and fail the
/// first. Tier 3 being absent entirely would pass the second half and fail the
/// first. Only a working tier 3 with a working guard passes both.
#[tokio::test]
async fn a_registered_inverse_is_refused_at_tier_three_and_registered_instead() {
    let store = &store().await;
    own_suffix(store, "DIRGUARD").await;
    let tenant = fresh_tenant(store, "dirguard").await;

    // The guarded pair. `OWNS_DIRGUARD` is the incumbent; `OWNED_BY_DIRGUARD` is
    // what the agent wrote, and it is its inverse.
    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "OWNS_DIRGUARD",
        "OWNED_BY_DIRGUARD",
        0.99,
    )
    .await;
    // Deliberately the opposite column order from (incoming, candidate): the
    // table records an unordered pair, and a guard that only checked one
    // orientation would let half of every pair through.
    record_inverse_pair(store, "OWNS_DIRGUARD", "OWNED_BY_DIRGUARD").await;

    // The control. Identical shape, no inverse pair.
    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "EATS_DIRGUARD",
        "DEVOURS_DIRGUARD",
        0.99,
    )
    .await;

    assert_unresolvable_by_exact_tiers(store, "OWNED_BY_DIRGUARD").await;
    assert_unresolvable_by_exact_tiers(store, "DEVOURS_DIRGUARD").await;

    let control = store
        .coerce_relationship_type(&tenant, "devours dirguard", &embedder())
        .await
        .expect("coercing the control should succeed");
    assert_eq!(
        control.canonical, "EATS_DIRGUARD",
        "tier 3 did not fire at cosine 0.99, so this test proves nothing about the guard"
    );

    let guarded = store
        .coerce_relationship_type(&tenant, "owned by dirguard", &embedder())
        .await
        .expect("coercing the guarded name should succeed");
    assert_eq!(
        guarded.canonical, "OWNED_BY_DIRGUARD",
        "an inverse of the nearest neighbour must be registered, never coerced onto it"
    );
    assert_eq!(guarded.raw, "owned by dirguard");

    let (cardinality, auto_registered) = registration_of(store, &tenant, "OWNED_BY_DIRGUARD")
        .await
        .expect("the refused type should have been registered in its own right");
    assert_eq!(cardinality, "many");
    assert!(auto_registered);
}

/// Asserts that the curated table says nothing about this pair, so a refusal
/// can only have come from the morphology rule.
///
/// Without it these tests would pass just as well against the table alone, and
/// the thing under test would be untested.
async fn assert_not_a_recorded_inverse(store: &Store, left: &str, right: &str) {
    let (recorded,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM relationship_inverse_pair \
         WHERE (type_a = $1 AND type_b = $2) OR (type_a = $2 AND type_b = $1)",
    )
    .bind(left)
    .bind(right)
    .fetch_one(store.pool())
    .await
    .expect("counting inverse pairs should succeed");

    assert_eq!(
        recorded, 0,
        "{left} / {right} is in relationship_inverse_pair, so a refusal proves nothing \
         about the morphology rule"
    );
}

/// **The direction guard's other half.** `relationship_inverse_pair` is
/// curated, and curation covers the pairs somebody thought of. Calibration
/// against the real model found four inverse pairs missing from it —
/// `MENTORS`/`MENTORED_BY` the highest at 0.8450 — all of them the same shape:
/// an active verb against its passive agent. A shape is a rule, and a rule
/// covers the coinage nobody has made yet, which is the case an open vocabulary
/// guarantees will arrive.
///
/// Built like the table-driven guard test above so it cannot pass for the wrong
/// reason. Three arrivals, identical evidence — a curated neighbour at cosine
/// 0.99, a name no exact tier can resolve, no row in the table for any of them
/// — differing only in their spelling:
///
/// * the control **must** be coerced, which fails if tier 3 is missing, broken
///   or thresholded out of reach;
/// * the passive-agent pair and the possessive pair **must not** be, which
///   fails if the morphology rule is missing.
///
/// A rule that refused everything would pass two of the three and fail the
/// control.
#[tokio::test]
async fn a_mechanical_inverse_is_refused_at_tier_three_and_registered_instead() {
    let store = &store().await;
    own_suffix(store, "MORPHGUARD").await;
    let tenant = fresh_tenant(store, "morphguard").await;

    // The control. No morphology connects these two names.
    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "MORPHGUARD_EATS",
        "MORPHGUARD_DEVOURS",
        0.99,
    )
    .await;

    // `X` against `X_ED_BY` — the shape of all four pairs the calibration found
    // unguarded. The marker leads rather than trails, because the suffix under
    // test has to be the last thing in the name.
    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "MORPHGUARD_MENTORS",
        "MORPHGUARD_MENTORED_BY",
        0.99,
    )
    .await;

    // `HAS_X` against `X_OF`. `A HAS_PATRON B` and `B PATRON_OF A` are one fact
    // stated from opposite ends.
    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "HAS_MORPHGUARD_PATRON",
        "MORPHGUARD_PATRON_OF",
        0.99,
    )
    .await;

    for name in [
        "MORPHGUARD_DEVOURS",
        "MORPHGUARD_MENTORED_BY",
        "MORPHGUARD_PATRON_OF",
    ] {
        assert_unresolvable_by_exact_tiers(store, name).await;
    }
    assert_not_a_recorded_inverse(store, "MORPHGUARD_MENTORS", "MORPHGUARD_MENTORED_BY").await;
    assert_not_a_recorded_inverse(store, "HAS_MORPHGUARD_PATRON", "MORPHGUARD_PATRON_OF").await;

    let control = store
        .coerce_relationship_type(&tenant, "morphguard devours", &embedder())
        .await
        .expect("coercing the control should succeed");
    assert_eq!(
        control.canonical, "MORPHGUARD_EATS",
        "tier 3 did not fire at cosine 0.99, so this test proves nothing about the rule"
    );

    let passive = store
        .coerce_relationship_type(&tenant, "morphguard mentored by", &embedder())
        .await
        .expect("coercing the passive-agent name should succeed");
    assert_eq!(
        passive.canonical, "MORPHGUARD_MENTORED_BY",
        "the passive agent of the nearest neighbour's verb must be registered, never \
         coerced onto it"
    );
    assert_eq!(
        passive.scope, tenant,
        "what the guard refused was registered privately"
    );

    let possessive = store
        .coerce_relationship_type(&tenant, "morphguard patron of", &embedder())
        .await
        .expect("coercing the possessive name should succeed");
    assert_eq!(
        possessive.canonical, "MORPHGUARD_PATRON_OF",
        "`X_OF` states the nearest neighbour's `HAS_X` backwards, so it is a new type"
    );

    // Refused at tier 3 is not refused outright: the caller said something, and
    // it is recorded as its own relation with the conservative cardinality.
    let (cardinality, auto_registered) = registration_of(store, &tenant, "MORPHGUARD_MENTORED_BY")
        .await
        .expect("the refused type should have been registered in its own right");
    assert_eq!(cardinality, "many");
    assert!(auto_registered);
}

/// **The trap.** `AUTHORED_BY` is itself a seeded canonical type, so "ends in
/// `_BY`" cannot be read as "is somebody's inverse". A rule that pattern-matched
/// the suffix would refuse every coercion onto it — including the paraphrases
/// tier 3 exists to fold in — and the refusals would look exactly like the
/// correct ones.
///
/// Two arrivals with the *same suffix* and the same evidence, differing only in
/// whether the stem under the suffix is the neighbour's verb:
///
/// * `PENNED_BY` against an incumbent `SCRIBBLES` is two ways of saying one
///   thing, and **must** be coerced;
/// * `MINTED_BY` against an incumbent `MINTS` is one thing said backwards, and
///   **must not** be.
#[tokio::test]
async fn a_by_suffix_alone_does_not_refuse_a_coercion() {
    let store = &store().await;
    own_suffix(store, "SUFFIXONLY").await;
    let tenant = fresh_tenant(store, "suffixonly").await;

    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "SUFFIXONLY_SCRIBBLES",
        "SUFFIXONLY_PENNED_BY",
        0.99,
    )
    .await;
    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "SUFFIXONLY_MINTS",
        "SUFFIXONLY_MINTED_BY",
        0.99,
    )
    .await;

    for name in ["SUFFIXONLY_PENNED_BY", "SUFFIXONLY_MINTED_BY"] {
        assert_unresolvable_by_exact_tiers(store, name).await;
    }
    assert_not_a_recorded_inverse(store, "SUFFIXONLY_SCRIBBLES", "SUFFIXONLY_PENNED_BY").await;
    assert_not_a_recorded_inverse(store, "SUFFIXONLY_MINTS", "SUFFIXONLY_MINTED_BY").await;

    let unrelated_stem = store
        .coerce_relationship_type(&tenant, "suffixonly penned by", &embedder())
        .await
        .expect("coercing should succeed");
    assert_eq!(
        unrelated_stem.canonical, "SUFFIXONLY_SCRIBBLES",
        "a `_BY` name whose stem is not the neighbour's verb is a synonym, not an inverse, \
         and refusing it would cost a coercion the tier exists to make"
    );

    let shared_stem = store
        .coerce_relationship_type(&tenant, "suffixonly minted by", &embedder())
        .await
        .expect("coercing should succeed");
    assert_eq!(
        shared_stem.canonical, "SUFFIXONLY_MINTED_BY",
        "same suffix, and this time it is the neighbour's own verb: that is a reversal"
    );
    assert_eq!(shared_stem.scope, tenant);
}

/// The same boundary, at the tier that cannot be reasoned about by eye.
///
/// Tier 3 answers with whatever sits nearest in embedding space, so a vocabulary
/// it can see across is a vocabulary it can *quote* across: another tenant's
/// coined name would come back as this caller's canonical type and be written
/// onto this caller's edges.
///
/// Built the way the direction-guard test is, so it cannot pass for the wrong
/// reason. Both halves present the same evidence — a neighbour at cosine 0.99
/// that no exact tier can reach — and differ only in which tenant is asking:
///
/// * inside the owning tenant the coercion **must** happen, which fails if tier
///   3 is broken or scoped out of existence;
/// * from the other tenant it **must not**.
#[tokio::test]
async fn tier_three_does_not_reach_across_tenants() {
    let store = &store().await;
    own_suffix(store, "VECTORISO").await;
    let acme = fresh_tenant(store, "vectoriso-acme").await;
    let globex = fresh_tenant(store, "vectoriso-globex").await;

    // Acme's private vocabulary, sitting one hundredth away from a name Globex
    // is about to write.
    seed_type_near(
        store,
        &acme,
        "PATRONISES_VECTORISO",
        "BANKROLLS_VECTORISO",
        0.99,
    )
    .await;

    assert_unresolvable_by_exact_tiers(store, "BANKROLLS_VECTORISO").await;

    let inside = store
        .coerce_relationship_type(&acme, "bankrolls vectoriso", &embedder())
        .await
        .expect("coercing inside the owning tenant should succeed");
    assert_eq!(
        inside.canonical, "PATRONISES_VECTORISO",
        "tier 3 did not fire at cosine 0.99 inside the owning tenant, \
         so this test proves nothing about the boundary"
    );
    assert_eq!(
        inside.scope, acme,
        "a tier-3 hit on a private type is reported as private"
    );

    let across = store
        .coerce_relationship_type(&globex, "bankrolls vectoriso", &embedder())
        .await
        .expect("coercing from another tenant should succeed");
    assert_eq!(
        across.canonical, "BANKROLLS_VECTORISO",
        "another tenant's private type must not be quotable through a similarity score"
    );
    assert_eq!(
        scopes_of(store, "BANKROLLS_VECTORISO").await,
        vec![globex],
        "the name Globex wrote belongs to Globex alone"
    );
}

/// Still tier 1: a pluralised verb is a spelling of the type, not a different
/// one, and the exact tiers get one more form to try before the model is asked
/// anything.
///
/// The ordering is the substance here. Half the seeded vocabulary is itself
/// plural — `USES`, `OWNS`, `PREFERS` — so the singular is offered strictly
/// *after* the name as written has missed, and after the alias table has missed
/// too. A depluralisation is a guess about English; an exact row is a fact about
/// this vocabulary, and a guess must never outrank one.
#[tokio::test]
async fn a_plural_of_a_registered_type_resolves_to_the_singular() {
    let store = &store().await;
    own_suffix(store, "PLURALTIER").await;
    let tenant = fresh_tenant(store, "pluraltier").await;

    // The per-test marker leads rather than trails, because the `S` under test
    // has to be the last character of the whole name — which is the only place
    // an English plural ever is.
    //
    // Curated rather than tenant-private, because the alias below can only
    // reach a curated type, and the four rows should sit in one scope.
    seed_type(store, GLOBAL_SCOPE, "PLURALTIER_MENTOR").await;
    // Both forms registered, so the exact match has something to beat.
    seed_type(store, GLOBAL_SCOPE, "PLURALTIER_PATROLS").await;
    seed_type(store, GLOBAL_SCOPE, "PLURALTIER_PATROL").await;
    seed_type(store, GLOBAL_SCOPE, "PLURALTIER_GUIDES").await;
    seed_alias(store, "PLURALTIER_SHEPHERD", "PLURALTIER_GUIDES").await;

    let coerced = store
        .coerce_relationship_type(&tenant, "pluraltier mentors", &embedder())
        .await
        .expect("coercing a plural should succeed");
    assert_eq!(coerced.canonical, "PLURALTIER_MENTOR");
    assert_eq!(
        type_count(store, "PLURALTIER_MENTORS").await,
        0,
        "a plural must not fork the type it is a plural of"
    );

    let exact = store
        .coerce_relationship_type(&tenant, "pluraltier patrols", &embedder())
        .await
        .expect("coercing should succeed");
    assert_eq!(
        exact.canonical, "PLURALTIER_PATROLS",
        "the name as written is registered, so it must win over its own singular"
    );

    let via_alias = store
        .coerce_relationship_type(&tenant, "pluraltier shepherds", &embedder())
        .await
        .expect("coercing should succeed");
    assert_eq!(
        via_alias.canonical, "PLURALTIER_GUIDES",
        "the singular form is offered to the alias table too"
    );
}

/// The tier-3 threshold has to sit somewhere, and a threshold nothing tests is a
/// threshold that can be deleted without a failure. Two arrivals with identical
/// evidence apart from their distance: the near one is folded in, the far one is
/// registered in its own right.
///
/// Erring towards registration is the cheap direction. A vocabulary carrying two
/// words for one relation is untidy and an operator fixes it with a single alias
/// row; a relation folded onto a type that does not mean the same thing is a
/// claim the graph now asserts about facts nobody re-reads.
#[tokio::test]
async fn a_neighbour_below_the_similarity_threshold_is_registered_rather_than_coerced() {
    let store = &store().await;
    own_suffix(store, "NEARMISS").await;
    let tenant = fresh_tenant(store, "nearmiss").await;

    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "NEARMISS_CLOSE_HOST",
        "NEARMISS_CLOSE_GUEST",
        0.99,
    )
    .await;
    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "NEARMISS_FAR_HOST",
        "NEARMISS_FAR_GUEST",
        0.90,
    )
    .await;

    assert_unresolvable_by_exact_tiers(store, "NEARMISS_CLOSE_GUEST").await;
    assert_unresolvable_by_exact_tiers(store, "NEARMISS_FAR_GUEST").await;

    let close = store
        .coerce_relationship_type(&tenant, "nearmiss close guest", &embedder())
        .await
        .expect("coercing should succeed");
    assert_eq!(
        close.canonical, "NEARMISS_CLOSE_HOST",
        "0.99 is inside the confident band"
    );
    assert_eq!(
        close.scope, GLOBAL_SCOPE,
        "the host it folded onto is curated"
    );

    let far = store
        .coerce_relationship_type(&tenant, "nearmiss far guest", &embedder())
        .await
        .expect("coercing should succeed");
    assert_eq!(
        far.canonical, "NEARMISS_FAR_GUEST",
        "0.90 is a resemblance, not an identity, and must not be folded in"
    );
    assert_eq!(
        far.scope, tenant,
        "what tier 3 declined to fold in was registered privately"
    );
    assert!(
        registration_of(store, &tenant, "NEARMISS_FAR_GUEST")
            .await
            .is_some_and(|(_, auto_registered)| auto_registered)
    );
}

/// Tier 3 compares vectors, and a cosine between vectors from two different
/// models is a meaningless number that still sorts — so the failure has no
/// symptom. `node` has filtered on `embedding_model` since it was written; this
/// is the same rule for the vocabulary, which until now had nowhere to record
/// the model at all.
///
/// Control-and-guard again, because a tier 3 that matched nothing would pass the
/// half that matters and prove nothing. Two neighbours at the same cosine,
/// differing only in which model is recorded against them.
#[tokio::test]
async fn tier_three_ignores_a_vector_from_another_embedding_model() {
    let store = &store().await;
    own_suffix(store, "MODELMIX").await;
    let tenant = fresh_tenant(store, "modelmix").await;

    seed_type_near(
        store,
        GLOBAL_SCOPE,
        "MODELMIX_SAME_HOST",
        "MODELMIX_SAME_GUEST",
        0.99,
    )
    .await;
    seed_type_near_from_another_model(
        store,
        GLOBAL_SCOPE,
        "MODELMIX_OTHER_HOST",
        "MODELMIX_OTHER_GUEST",
        0.99,
    )
    .await;

    assert_unresolvable_by_exact_tiers(store, "MODELMIX_SAME_GUEST").await;
    assert_unresolvable_by_exact_tiers(store, "MODELMIX_OTHER_GUEST").await;

    let same = store
        .coerce_relationship_type(&tenant, "modelmix same guest", &embedder())
        .await
        .expect("coercing the control should succeed");
    assert_eq!(
        same.canonical, "MODELMIX_SAME_HOST",
        "tier 3 did not fire at cosine 0.99 for its own model, \
         so this test proves nothing about the filter"
    );

    let other = store
        .coerce_relationship_type(&tenant, "modelmix other guest", &embedder())
        .await
        .expect("coercing across models should succeed");
    assert_eq!(
        other.canonical, "MODELMIX_OTHER_GUEST",
        "0.99 between two models' vectors is not a similarity, it is a coincidence"
    );
    assert_eq!(other.scope, tenant, "so the name was registered instead");
}

/// The seeded vocabulary arrives with no vectors, because a migration has no
/// embedder to make them with, and the server fills them in on first use. The
/// model has to be recorded in the same breath: a vector whose provenance is
/// unknown is one tier 3 can never use again, so a backfill that skipped the
/// column would leave the curated set permanently unreachable by similarity —
/// silently, since every other tier still works.
#[tokio::test]
async fn the_backfill_records_the_model_that_produced_each_vector() {
    let store = &store().await;
    own_suffix(store, "BACKFILL").await;
    let tenant = fresh_tenant(store, "backfill").await;

    // Exactly the shape the migration leaves behind.
    seed_type(store, GLOBAL_SCOPE, "ARBITRATES_BACKFILL").await;

    let (has_vector, _) = vector_and_model(store, GLOBAL_SCOPE, "ARBITRATES_BACKFILL").await;
    assert!(!has_vector, "the row under test starts out unvectored");

    // Any coercion that reaches tier 3 runs the backfill first.
    store
        .coerce_relationship_type(&tenant, "instigates backfill", &embedder())
        .await
        .expect("coercing should succeed");

    let (has_vector, model) = vector_and_model(store, GLOBAL_SCOPE, "ARBITRATES_BACKFILL").await;
    assert!(has_vector, "the backfill exists to fill exactly this row");
    assert_eq!(
        model.as_deref(),
        Some(embedder().model_id()),
        "a vector with no recorded model is a vector nothing may be compared to"
    );
}

/// A name with nothing in it is a caller bug, not a new relation. Registered, it
/// would put an empty string in a primary key that `edge.relationship_type`
/// references — a type no agent can quote back, no operator can grep for, and
/// that every other blank arrival would then silently share.
#[tokio::test]
async fn a_name_with_no_usable_characters_is_refused() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "blank-name").await;

    for raw in ["", "   ", " - __ - "] {
        let error = store
            .coerce_relationship_type(&tenant, raw, &embedder())
            .await
            .expect_err("a name that normalises to nothing names nothing");

        assert!(
            matches!(error, StoreError::BlankRelationshipType { .. }),
            "expected BlankRelationshipType for {raw:?}, got {error:?}"
        );
    }

    assert_eq!(
        type_count(store, "").await,
        0,
        "the empty string must never reach the vocabulary"
    );
}

/// The scope sentinel is a string, and the tenant is a string, so the two can
/// collide. `Authenticator` rejects a credential with no tenant *claim*, but a
/// claim present and empty — one misconfigured IdP mapping away — validates and
/// arrives here as `""`. That caller would be writing into the curated set: the
/// half of the vocabulary every other tenant reads.
///
/// Refused before anything is read or written, so the hole is closed where the
/// meaning of `''` is known rather than wherever a tenant string was last
/// handled. Only the empty string is refused: a tenant of `" "` is merely an odd
/// name for a scope of its own, and quietly trimming callers' tenant ids would
/// mean silently moving one caller's memory into another's.
#[tokio::test]
async fn a_caller_with_no_tenant_may_not_write_the_curated_vocabulary() {
    let store = &store().await;
    own_suffix(store, "NOTENANT").await;

    let error = store
        .coerce_relationship_type(GLOBAL_SCOPE, "sponsors notenant", &embedder())
        .await
        .expect_err("the curated scope is not a tenant anyone may call as");

    assert!(
        matches!(error, StoreError::BlankTenant),
        "expected BlankTenant, got {error:?}"
    );
    assert_eq!(
        type_count(store, "SPONSORS_NOTENANT").await,
        0,
        "a caller with no tenant must not leave anything behind"
    );
}

/// Scoping the vocabulary narrows this race to one tenant, it does not remove
/// it: a tenant runs many agents, they are stateless between calls, and two of
/// them inventing the same verb at the same instant is ordinary traffic rather
/// than an edge case — both will have read a vocabulary without it before either
/// writes.
///
/// Surfacing that as a duplicate-key error would fail a write the caller did
/// nothing wrong in making, and would fail it at the point where the edge behind
/// it is about to be stored.
/// The window is held open deliberately rather than raced for. Two coercions
/// launched with `join!` almost always serialise by luck — the loser's read
/// happens after the winner's commit and there is no conflict to survive — so
/// the interesting interleaving has to be constructed: one writer registers the
/// type and *stays uncommitted* while the second reads the vocabulary without
/// it.
///
/// If the pause is ever too short the second writer simply sees the committed
/// row, and the test degrades into a weaker assertion rather than a flaky
/// failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_writer_that_loses_the_registration_race_still_resolves_the_type() {
    let store = store().await;
    own_suffix(&store, "RACEREG").await;
    let tenant = fresh_tenant(&store, "racereg").await;
    let name = "RACEREG_ESCALATES";

    // The winner: registered, not yet committed, so nothing else can see it.
    let mut pending = store
        .pool()
        .begin()
        .await
        .expect("beginning a transaction should succeed");
    let winner = coerce_relationship_type(&mut pending, &tenant, name, &embedder())
        .await
        .expect("the first writer should register the type");
    assert_eq!(winner.canonical, name);

    // The loser: reads an empty vocabulary, then collides on a primary key it
    // had every reason to believe was free. Same tenant as the winner, because
    // that is the only pair that can collide at all now.
    let loser = tokio::spawn({
        let store = store.clone();
        let tenant = tenant.clone();
        async move {
            store
                .coerce_relationship_type(&tenant, name, &embedder())
                .await
        }
    });

    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    pending
        .commit()
        .await
        .expect("committing the winner should succeed");

    let loser = loser
        .await
        .expect("the losing writer should not panic")
        .expect("losing a race the caller could not have avoided is not an error");

    assert_eq!(
        loser.canonical, name,
        "the loser must resolve to the type that won, not fail"
    );
    assert_eq!(type_count(&store, name).await, 1);
}
