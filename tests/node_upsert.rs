//! Integration tests for the node upsert path, against a real Postgres.
//!
//! Every test owns a `tenant_id` derived from its own name. The suite runs in
//! parallel against one shared database, so isolation cannot come from
//! truncating shared tables — that would delete a concurrently running test's
//! rows. Tenancy is the isolation mechanism the schema already has.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use mundu::embed::{DeterministicEmbedder, Embedder};
use mundu::error::StoreError;
use mundu::store::node::{Choice, NodeInput, NodeResolution, NodeUpsertOutcome};
use mundu::store::{Provenance, Store};

/// One [`Store`] per test, deliberately not a shared `static`.
///
/// Each `#[tokio::test]` builds and tears down its own runtime, and a sqlx pool
/// is bound to the reactor that created it. A pool cached in a `static` and
/// first built inside test A fails inside test B the moment A's runtime shuts
/// down, with `A Tokio 1.x context was found, but it is being shutdown` — a
/// failure that reads as a database problem and is not one.
///
/// `migrate` is idempotent and takes a Postgres advisory lock, so running it
/// from every test is safe even though the suite runs in parallel.
async fn store() -> Store {
    common::store().await
}

/// Names a tenant after the test that owns it, and clears anything a previous
/// run of that same test left behind, so assertions like "this created a node"
/// mean the same thing on the first run and the hundredth.
async fn fresh_tenant(store: &Store, name: &str) -> String {
    let tenant = format!("test-{name}");

    sqlx::query("DELETE FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .execute(store.pool())
        .await
        .expect("clearing the tenant should succeed");

    tenant
}

fn embedder() -> DeterministicEmbedder {
    DeterministicEmbedder::new()
}

fn anonymous() -> Provenance {
    Provenance::default()
}

/// The id is minted by the server from the display name the agent supplied.
/// Agents never propose ids: a model allowed to invent one will invent a
/// plausible id that does not exist, and every later edge write against it
/// fails.
#[tokio::test]
async fn a_new_display_name_creates_a_node_with_a_kebab_case_id() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "creates-a-node").await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Project Phoenix")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting a new name should succeed");

    assert_eq!(outcomes.len(), 1);
    let NodeUpsertOutcome::Created { node_id } = &outcomes[0] else {
        panic!("expected Created, got {:?}", outcomes[0]);
    };
    assert_eq!(node_id, "project-phoenix");

    let (display_name, model): (String, Option<String>) = sqlx::query_as(
        "SELECT display_name, embedding_model FROM node WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(&tenant)
    .bind(node_id)
    .fetch_one(store.pool())
    .await
    .expect("the created row should be readable");

    assert_eq!(display_name, "Project Phoenix");
    assert_eq!(
        model.as_deref(),
        Some(embedder().model_id()),
        "the model that produced name_vector must be recorded beside it"
    );
}

/// Agents are stateless between calls, so the same entity arrives again and
/// again. If a repeat minted a second node, the graph would fork under an agent
/// that did nothing wrong, and half its facts would hang off each fork.
#[tokio::test]
async fn upserting_the_same_display_name_twice_is_idempotent() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "idempotent").await;

    let first = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Project Phoenix")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("the first upsert should succeed");
    let second = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Project Phoenix")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("the second upsert should succeed");

    assert_eq!(
        first[0],
        NodeUpsertOutcome::Created {
            node_id: "project-phoenix".to_string()
        }
    );
    assert_eq!(
        second[0],
        NodeUpsertOutcome::Matched {
            node_id: "project-phoenix".to_string()
        },
        "the second upsert must recognise the node it already has"
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting the tenant's nodes should succeed");
    assert_eq!(rows, 1, "the repeat must not have duplicated the node");
}

/// Agents name the same entity differently from call to call. An alias the
/// graph already knows is the cheapest possible evidence that two names are one
/// entity, and ignoring it forks the graph on a synonym.
#[tokio::test]
async fn a_known_alias_resolves_to_the_existing_node() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "known-alias").await;

    store
        .upsert_nodes(
            &tenant,
            &[NodeInput::with_aliases(
                "Project Phoenix",
                ["The Phoenix Programme"],
            )],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("seeding the node should succeed");

    // Deliberately a different case from the stored alias: an agent quoting a
    // name back rarely reproduces its capitalisation.
    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("the phoenix programme")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting a known alias should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched {
            node_id: "project-phoenix".to_string()
        },
        "a known alias must resolve, not fork"
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting the tenant's nodes should succeed");
    assert_eq!(rows, 1, "no second node should have been minted");
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
/// `DeterministicEmbedder` is a hash: two texts either embed identically or
/// land near-orthogonal, with nothing in between. The interesting thresholds
/// live in between, so the near misses have to be constructed. `other` is
/// projected off `base` first, so the result is `cosine * base + sin * u` for a
/// genuine orthonormal `u` and the angle is exact rather than approximate.
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

/// Writes a node whose `name_vector` is not the embedding of its own display
/// name — the shape a rename leaves behind, and the only way to put a
/// controlled similarity in front of the resolver.
async fn seed_node(store: &Store, tenant: &str, display_name: &str, vector: &[f32]) -> String {
    let node_id = display_name
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");

    sqlx::query(
        "INSERT INTO node (tenant_id, node_id, display_name, name_vector, embedding_model) \
         VALUES ($1, $2, $3, $4::real[]::vector, $5)",
    )
    .bind(tenant)
    .bind(&node_id)
    .bind(display_name)
    .bind(vector)
    .bind(embedder().model_id())
    .execute(store.pool())
    .await
    .expect("seeding a node should succeed");

    node_id
}

/// Reads the alias array off a node.
async fn aliases_of(store: &Store, tenant: &str, node_id: &str) -> Vec<String> {
    let (aliases,): (Vec<String>,) =
        sqlx::query_as("SELECT aliases FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(tenant)
            .bind(node_id)
            .fetch_one(store.pool())
            .await
            .expect("the node should be readable");
    aliases
}

/// Resolution that throws away the name it resolved *from* learns nothing. The
/// next agent to use that name pays the same similarity search, and if the name
/// ever drifts out of the confident band it forks instead. Recording the
/// surface form turns one successful match into a permanent exact match.
#[tokio::test]
async fn a_new_surface_form_of_a_matched_node_is_appended_to_its_aliases() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "surface-form").await;

    store
        .upsert_nodes(
            &tenant,
            &[NodeInput::with_aliases(
                "Project Phoenix",
                ["PROJECT PHOENIX"],
            )],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("seeding the node should succeed");

    assert!(
        aliases_of(store, &tenant, "project-phoenix")
            .await
            .is_empty(),
        "an alias that only recases the display name carries no information"
    );

    let again =
        NodeInput::with_aliases("Project Phoenix", ["Phoenix Programme", "PROJECT PHOENIX"]);

    let outcomes = store
        .upsert_nodes(
            &tenant,
            std::slice::from_ref(&again),
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting a known node under a new name should succeed");
    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched {
            node_id: "project-phoenix".to_string()
        }
    );

    assert_eq!(
        aliases_of(store, &tenant, "project-phoenix").await,
        vec!["Phoenix Programme".to_string()],
        "the unseen name should be recorded, and a bare case variant of the \
         display name should not be"
    );

    store
        .upsert_nodes(&tenant, &[again], &embedder(), &anonymous())
        .await
        .expect("the repeat should succeed");

    assert_eq!(
        aliases_of(store, &tenant, "project-phoenix").await,
        vec!["Phoenix Programme".to_string()],
        "a surface form the node already carries must not be appended twice"
    );
}

/// The third tier of the ladder. Exact matching only catches names the graph
/// has literally seen; a paraphrase that no agent has ever written down before
/// has nothing but the vector to go on.
#[tokio::test]
async fn a_confident_vector_match_resolves_to_that_node() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "vector-match").await;

    let incoming = "Distributed Tracing Pipeline";
    let near = at_cosine(
        &embed(incoming).await,
        &embed("unrelated anchor").await,
        0.95,
    );
    let seeded = seed_node(store, &tenant, "Observability Stack", &near).await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new(incoming)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched {
            node_id: seeded.clone()
        },
        "a long name at cosine 0.95 is a confident match"
    );
    assert_eq!(
        aliases_of(store, &tenant, &seeded).await,
        vec![incoming.to_string()],
        "the name that resolved should become an exact match next time"
    );
}

/// The asymmetry that shapes the whole resolver. Fragmentation — two nodes for
/// one entity — is undone by a later merge. Over-merge is not: once `Phoenix`
/// the city has been fused into `project-phoenix` the boundary is gone, and
/// only a human reading each proof can guess which entity a fact meant. So the
/// uncertain case takes the recoverable error and hands the near misses back.
#[tokio::test]
async fn an_uncertain_vector_match_creates_a_new_node_and_reports_the_candidates() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "ambiguous-band").await;

    let incoming = "Distributed Tracing Pipeline";
    let near = at_cosine(
        &embed(incoming).await,
        &embed("unrelated anchor").await,
        0.87,
    );
    let seeded = seed_node(store, &tenant, "Observability Stack", &near).await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new(incoming)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    let NodeUpsertOutcome::Ambiguous {
        node_id,
        candidates,
    } = &outcomes[0]
    else {
        panic!("expected Ambiguous, got {:?}", outcomes[0]);
    };

    assert_eq!(
        node_id, "distributed-tracing-pipeline",
        "the uncertain case still has to give the caller a usable handle"
    );
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].node_id, seeded);
    assert_eq!(candidates[0].display_name, "Observability Stack");
    assert!(
        (candidates[0].similarity - 0.87).abs() < 1e-4,
        "expected a similarity near 0.87, got {}",
        candidates[0].similarity
    );

    assert!(
        aliases_of(store, &tenant, &seeded).await.is_empty(),
        "the near miss must not have been quietly attached to"
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting the tenant's nodes should succeed");
    assert_eq!(rows, 2, "the new node should exist alongside the candidate");
}

/// A cosine cannot tell a paraphrase from a sibling. Measured over 422
/// labelled pairs, `Acme Corporation` against `Acme Holdings` — two companies —
/// scores 0.9451, while `billing-service-v2` against `the v2 billing service` —
/// one service — scores 0.9295. The populations interleave, so no bar
/// separates them and the vector tier needs a second opinion.
///
/// The names supply it: both of these carry a word the other contradicts. The
/// outcome is the recoverable one — a duplicate node and a candidate list —
/// rather than the irreversible one.
#[tokio::test]
async fn a_confident_vector_match_whose_names_contradict_is_not_merged() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "contradicting-names").await;

    let incoming = "Acme Corporation";
    let near = at_cosine(
        &embed(incoming).await,
        &embed("unrelated anchor").await,
        0.95,
    );
    let seeded = seed_node(store, &tenant, "Acme Holdings", &near).await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new(incoming)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    let NodeUpsertOutcome::Ambiguous {
        node_id,
        candidates,
    } = &outcomes[0]
    else {
        panic!(
            "0.95 clears the confident bar, but `Corporation` against `Holdings` \
             is a substitution and must not merge; got {:?}",
            outcomes[0]
        );
    };

    assert_eq!(
        node_id, "acme-corporation",
        "the caller still needs a handle"
    );
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].node_id, seeded);
    assert_eq!(candidates[0].display_name, "Acme Holdings");
    assert!(
        (candidates[0].similarity - 0.95).abs() < 1e-4,
        "the near miss is reported with the cosine it actually scored, got {}",
        candidates[0].similarity
    );

    assert!(
        aliases_of(store, &tenant, &seeded).await.is_empty(),
        "the vetoed neighbour must be left exactly as it was"
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting the tenant's nodes should succeed");
    assert_eq!(rows, 2, "both entities should still exist separately");
}

/// The other half of the guard, and the reason it is a veto rather than a
/// requirement: a confident vector whose names do not contradict resolves
/// exactly as it did before. Word order is not a disagreement.
#[tokio::test]
async fn a_confident_vector_match_whose_names_agree_still_merges() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "agreeing-names").await;

    let incoming = "Project Phoenix";
    let near = at_cosine(
        &embed(incoming).await,
        &embed("unrelated anchor").await,
        0.95,
    );
    let seeded = seed_node(store, &tenant, "Phoenix Project", &near).await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new(incoming)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched {
            node_id: seeded.clone()
        },
        "the same two words in the other order name the same entity"
    );
    assert_eq!(
        aliases_of(store, &tenant, &seeded).await,
        vec![incoming.to_string()],
        "the name that resolved should become an exact match next time"
    );
}

/// A version token is the one word that decides an entity on its own, and the
/// one an embedder has least to work with: `v1` and `v2` are two characters,
/// one of them shared, on a name that is otherwise identical. Measured at
/// 0.9250, which clears the confident bar.
#[tokio::test]
async fn a_version_token_blocks_a_confident_vector_match() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "version-token").await;

    let incoming = "billing-service-v2";
    let near = at_cosine(
        &embed(incoming).await,
        &embed("unrelated anchor").await,
        0.95,
    );
    let seeded = seed_node(store, &tenant, "billing-service-v1", &near).await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new(incoming)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    let NodeUpsertOutcome::Ambiguous {
        node_id,
        candidates,
    } = &outcomes[0]
    else {
        panic!(
            "two deployed versions are two entities however close the prose \
             around them; got {:?}",
            outcomes[0]
        );
    };

    assert_eq!(node_id, "billing-service-v2");
    assert_eq!(candidates[0].node_id, seeded);
}

/// A name that two nodes both answer to confidently is not a confident match,
/// it is a collision. Picking the higher score would be picking at random.
#[tokio::test]
async fn two_confident_neighbours_are_ambiguous_rather_than_a_best_match() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "two-confident").await;

    let incoming = "Distributed Tracing Pipeline";
    let anchor = embed("unrelated anchor").await;
    let base = embed(incoming).await;

    let first = seed_node(
        store,
        &tenant,
        "Tracing Stack",
        &at_cosine(&base, &anchor, 0.99),
    )
    .await;
    let second = seed_node(
        store,
        &tenant,
        "Span Collector",
        &at_cosine(&base, &anchor, 0.96),
    )
    .await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new(incoming)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    let NodeUpsertOutcome::Ambiguous { candidates, .. } = &outcomes[0] else {
        panic!("expected Ambiguous, got {:?}", outcomes[0]);
    };

    let reported: Vec<&str> = candidates.iter().map(|c| c.node_id.as_str()).collect();
    assert_eq!(
        reported,
        vec![first.as_str(), second.as_str()],
        "both contenders should be reported, closest first"
    );
}

/// The same cosine means different things at different name lengths — though
/// not for the reason this comment used to give. Unrelated short names do not
/// crowd together: measured, they sit at 0.36–0.51, no higher than long ones.
/// What differs is that `names_contradict` cannot defend a short name. A
/// one-token name has no second token to disagree with, so the cosine is the
/// only thing standing between `Phoenix` the city and `project-phoenix`, and it
/// is held to a higher number because it is holding the line by itself.
#[tokio::test]
async fn a_short_name_needs_a_stricter_similarity_than_a_long_one() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "short-name").await;
    let anchor = embed("unrelated anchor").await;

    // Identical evidence in both halves: cosine 0.925 against one seeded node,
    // which is over the long bar of 0.92 and under the short one of 0.93.
    let long_name = "Phoenix Data Platform";
    let long_seed = seed_node(
        store,
        &tenant,
        "Ingest Platform",
        &at_cosine(&embed(long_name).await, &anchor, 0.925),
    )
    .await;

    let short_name = "Phoenix";
    let short_seed = seed_node(
        store,
        &tenant,
        "Firebird",
        &at_cosine(&embed(short_name).await, &anchor, 0.925),
    )
    .await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new(long_name), NodeInput::new(short_name)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched { node_id: long_seed },
        "0.925 is confident for a 21-character name"
    );

    let NodeUpsertOutcome::Ambiguous {
        node_id,
        candidates,
    } = &outcomes[1]
    else {
        panic!(
            "expected the same 0.925 to be merely ambiguous for a short name, got {:?}",
            outcomes[1]
        );
    };
    assert_eq!(node_id, "phoenix");
    assert_eq!(candidates[0].node_id, short_seed);
}

/// The short-name confident bar, pinned from both sides.
///
/// Where it sits is not arbitrary. Measured on the real model, the highest
/// short-name negative `names_contradict` cannot veto is `Phoenix` against
/// `Phoenix, AZ` at 0.8977 — one token plus a qualifier, a shape the guard
/// reads as elaboration and allows through. The true matches of that same
/// shape, `AWS` / `Amazon AWS` at 0.9294 and `Acme` / `ACME Inc` at 0.9259, sit
/// barely 0.03 above it and are lexically indistinguishable from it. So the bar
/// is placed above that whole band rather than inside it, and a short name has
/// to clear 0.93 on the strength of the cosine alone.
#[tokio::test]
async fn a_short_name_merges_above_the_confident_bar_and_not_below_it() {
    let store = &store().await;
    let anchor = embed("unrelated anchor").await;
    let short_name = "Phoenix";

    // Above. Two tenants rather than two seeds in one, so neither probe can see
    // the other's neighbour and change `confident.len() == 1` underneath it.
    let above = fresh_tenant(store, "short-bar-above").await;
    let cleared = seed_node(
        store,
        &above,
        "Firebird",
        &at_cosine(&embed(short_name).await, &anchor, 0.94),
    )
    .await;

    let outcomes = store
        .upsert_nodes(
            &above,
            &[NodeInput::new(short_name)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched { node_id: cleared },
        "0.94 clears the short-name confident bar, so the vector tier must fire — \
         a bar no short name can reach is a tier that does nothing"
    );

    // Below.
    let below = fresh_tenant(store, "short-bar-below").await;
    let missed = seed_node(
        store,
        &below,
        "Firebird",
        &at_cosine(&embed(short_name).await, &anchor, 0.92),
    )
    .await;

    let outcomes = store
        .upsert_nodes(
            &below,
            &[NodeInput::new(short_name)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    let NodeUpsertOutcome::Ambiguous {
        node_id,
        candidates,
    } = &outcomes[0]
    else {
        panic!(
            "0.92 is inside the band the guard is blind to and must not merge, got {:?}",
            outcomes[0]
        );
    };
    assert_eq!(node_id, "phoenix");
    assert_eq!(candidates[0].node_id, missed);
}

/// The candidate floor does not move with name length, and the confident bar
/// does. That asymmetry is the point.
///
/// `names_contradict` is a veto on *attaching*; it is never consulted about the
/// candidate list, so the guard-coverage gap that justifies a stricter short
/// confident bar says nothing at all about where near misses get reported. Nor
/// does the model: unrelated short pairs measure 0.3631–0.5071 against
/// 0.3205–0.4589 for long ones, which is the same floor. A short name therefore
/// reports from the same 0.82 a long one does — which is what keeps the
/// duplicates the strict bar creates visible instead of silent.
#[tokio::test]
async fn a_short_name_reports_near_misses_from_the_same_floor_as_a_long_one() {
    let store = &store().await;
    let anchor = embed("unrelated anchor").await;
    let short_name = "Mercury";

    let reported = fresh_tenant(store, "short-floor-above").await;
    let seeded = seed_node(
        store,
        &reported,
        "Quicksilver",
        &at_cosine(&embed(short_name).await, &anchor, 0.85),
    )
    .await;

    let outcomes = store
        .upsert_nodes(
            &reported,
            &[NodeInput::new(short_name)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    let NodeUpsertOutcome::Ambiguous {
        node_id,
        candidates,
    } = &outcomes[0]
    else {
        panic!(
            "0.85 is over the candidate floor, so the near miss must be handed back \
             rather than silently duplicated; got {:?}",
            outcomes[0]
        );
    };
    assert_eq!(node_id, "mercury");
    assert_eq!(candidates[0].node_id, seeded);

    let quiet = fresh_tenant(store, "short-floor-below").await;
    seed_node(
        store,
        &quiet,
        "Quicksilver",
        &at_cosine(&embed(short_name).await, &anchor, 0.80),
    )
    .await;

    let outcomes = store
        .upsert_nodes(
            &quiet,
            &[NodeInput::new(short_name)],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Created {
            node_id: "mercury".to_string()
        },
        "0.80 is under the floor: reporting it would make every short name contested"
    );
}

/// `tenant_id` is the whole isolation boundary, and it comes from a validated
/// credential rather than a tool argument. Two customers naming a project the
/// same thing is the ordinary case, not an edge case, and neither may see the
/// other's node — including as a resolution candidate, which would leak a name
/// without ever returning a row.
#[tokio::test]
async fn two_tenants_hold_the_same_node_id_without_seeing_each_other() {
    let store = &store().await;
    let acme = fresh_tenant(store, "isolation-acme").await;
    let globex = fresh_tenant(store, "isolation-globex").await;

    for tenant in [&acme, &globex] {
        let outcome = store
            .upsert_nodes(
                tenant,
                &[NodeInput::new("Project Phoenix")],
                &embedder(),
                &anonymous(),
            )
            .await
            .expect("upserting should succeed");

        assert_eq!(
            outcome[0],
            NodeUpsertOutcome::Created {
                node_id: "project-phoenix".to_string()
            },
            "{tenant} must mint its own node, not find the other tenant's"
        );
    }

    // The alias one tenant records must not resolve for the other.
    store
        .upsert_nodes(
            &acme,
            &[NodeInput::with_aliases(
                "Project Phoenix",
                ["Acme Internal Codename"],
            )],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("recording an alias should succeed");

    let outcome = store
        .upsert_nodes(
            &globex,
            &[NodeInput::new("Acme Internal Codename")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");
    assert_eq!(
        outcome[0],
        NodeUpsertOutcome::Created {
            node_id: "acme-internal-codename".to_string()
        },
        "one tenant's alias must be invisible to another"
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&acme)
        .fetch_one(store.pool())
        .await
        .expect("counting should succeed");
    assert_eq!(
        rows, 1,
        "the other tenant's write must not have landed here"
    );
}

async fn merge_into(store: &Store, tenant: &str, from: &str, into: &str) {
    sqlx::query("UPDATE node SET merged_into = $3 WHERE tenant_id = $1 AND node_id = $2")
        .bind(tenant)
        .bind(from)
        .bind(into)
        .execute(store.pool())
        .await
        .expect("recording a merge should succeed");
}

/// Entity resolution never deletes. Agents hold `node_id`s across stateless
/// calls, so a deleted id would dangle in a context window that nothing can
/// correct. A merged node stays in the table as a tombstone and keeps
/// resolving — to its successor, and through a chain of them, because the
/// successor may itself have been merged later.
#[tokio::test]
async fn a_merged_node_resolves_to_its_successor() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "merged-into").await;

    for name in ["Legacy Billing", "Billing Service", "Payments Platform"] {
        store
            .upsert_nodes(&tenant, &[NodeInput::new(name)], &embedder(), &anonymous())
            .await
            .expect("seeding should succeed");
    }
    merge_into(store, &tenant, "legacy-billing", "billing-service").await;
    merge_into(store, &tenant, "billing-service", "payments-platform").await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Legacy Billing")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting a merged name should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched {
            node_id: "payments-platform".to_string()
        },
        "resolution must follow the chain to the end, not stop at the first hop"
    );

    assert_eq!(
        aliases_of(store, &tenant, "payments-platform").await,
        vec!["Legacy Billing".to_string()],
        "the retired name belongs on the successor, which is what answers to it now"
    );
    assert!(
        aliases_of(store, &tenant, "legacy-billing")
            .await
            .is_empty(),
        "nothing should be written back onto the tombstone"
    );
}

/// The schema forbids a node merging into itself but says nothing about A into
/// B into A, which two independent merges can produce without either being
/// wrong on its own. Following that chain is an unbounded loop inside an open
/// transaction: it does not fail the request, it takes the process with it.
#[tokio::test]
async fn a_merge_cycle_is_refused_rather_than_followed_forever() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "merge-cycle").await;

    for name in ["Alpha Service", "Beta Service"] {
        store
            .upsert_nodes(&tenant, &[NodeInput::new(name)], &embedder(), &anonymous())
            .await
            .expect("seeding should succeed");
    }
    merge_into(store, &tenant, "alpha-service", "beta-service").await;
    merge_into(store, &tenant, "beta-service", "alpha-service").await;

    let error = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Alpha Service")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect_err("a cyclic merge chain has no successor to resolve to");

    assert!(
        matches!(error, StoreError::MergeCycle { .. }),
        "expected MergeCycle, got {error:?}"
    );
}

/// Puts an alias on a node behind the resolver's back. Upserting it the normal
/// way would resolve to whichever node already carries it, which is the very
/// situation being set up.
async fn force_alias(store: &Store, tenant: &str, node_id: &str, alias: &str) {
    sqlx::query(
        "UPDATE node SET aliases = array_append(aliases, $3) \
         WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(tenant)
    .bind(node_id)
    .bind(alias)
    .execute(store.pool())
    .await
    .expect("forcing an alias should succeed");
}

/// An exact match is only decisive when it is unique. Two nodes carrying the
/// same alias is the lexical form of the same collision the vector tier guards
/// against, and picking the alphabetically first one would be picking at random
/// — an over-merge with no signal that anything was chosen.
#[tokio::test]
async fn a_surface_form_that_two_nodes_answer_to_is_ambiguous() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "contested-alias").await;

    for name in ["Phoenix Arizona", "Project Phoenix"] {
        store
            .upsert_nodes(&tenant, &[NodeInput::new(name)], &embedder(), &anonymous())
            .await
            .expect("seeding should succeed");
    }
    force_alias(store, &tenant, "phoenix-arizona", "Phoenix").await;
    force_alias(store, &tenant, "project-phoenix", "Phoenix").await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Phoenix")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting should succeed");

    let NodeUpsertOutcome::Ambiguous {
        node_id,
        candidates,
    } = &outcomes[0]
    else {
        panic!("expected Ambiguous, got {:?}", outcomes[0]);
    };
    assert_eq!(node_id, "phoenix");

    let mut reported: Vec<&str> = candidates.iter().map(|c| c.node_id.as_str()).collect();
    reported.sort_unstable();
    assert_eq!(reported, vec!["phoenix-arizona", "project-phoenix"]);

    for contender in ["phoenix-arizona", "project-phoenix"] {
        assert_eq!(
            aliases_of(store, &tenant, contender).await,
            vec!["Phoenix".to_string()],
            "{contender} must be left exactly as it was"
        );
    }
}

/// Writes are batched, and the caller zips the outcomes back against its own
/// input list by position. A reordered or deduplicated result files one
/// entity's id under another's name, which then binds every edge to the wrong
/// node — with a proof excerpt that still reads correctly.
#[tokio::test]
async fn a_batch_returns_one_outcome_per_input_in_input_order() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "batch-order").await;

    store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Billing Service")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("seeding should succeed");

    // Deliberately unsorted, with a known node in the middle and the same
    // entity named twice — sorting or deduplicating internally is the plausible
    // optimisation, and both would be caught here.
    let inputs = [
        NodeInput::new("Zephyr Cluster"),
        NodeInput::new("Billing Service"),
        NodeInput::new("Apex Gateway"),
        NodeInput::new("zephyr cluster"),
    ];

    let outcomes = store
        .upsert_nodes(&tenant, &inputs, &embedder(), &anonymous())
        .await
        .expect("upserting a batch should succeed");

    assert_eq!(
        outcomes,
        vec![
            NodeUpsertOutcome::Created {
                node_id: "zephyr-cluster".to_string()
            },
            NodeUpsertOutcome::Matched {
                node_id: "billing-service".to_string()
            },
            NodeUpsertOutcome::Created {
                node_id: "apex-gateway".to_string()
            },
            // A repeat inside one batch has to see the row its predecessor just
            // wrote, or a single call forks the graph on its own.
            NodeUpsertOutcome::Matched {
                node_id: "zephyr-cluster".to_string()
            },
        ]
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting should succeed");
    assert_eq!(rows, 3);
}

/// `"!!!"` has no alphanumerics to build an id from, and an empty `node_id`
/// would be a primary key value no agent can quote back — one that every other
/// unnameable name in the tenant would then collide with, silently fusing
/// unrelated entities. The digest fallback is derived rather than random, so
/// the name stays idempotent like any other.
#[tokio::test]
async fn a_name_with_no_alphanumerics_still_gets_a_usable_idempotent_id() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "unnameable").await;

    let outcomes = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("!!!"), NodeInput::new("???")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("an unnameable name is unusual, not invalid");

    let (
        NodeUpsertOutcome::Created { node_id: bangs },
        NodeUpsertOutcome::Created { node_id: queries },
    ) = (&outcomes[0], &outcomes[1])
    else {
        panic!("expected two Created outcomes, got {outcomes:?}");
    };

    assert!(bangs.starts_with("unnamed-"), "{bangs} should be derived");
    assert_ne!(bangs, queries, "two unnameable names must not collide");

    let repeat = store
        .upsert_nodes(&tenant, &[NodeInput::new("!!!")], &embedder(), &anonymous())
        .await
        .expect("upserting should succeed");
    assert_eq!(
        repeat[0],
        NodeUpsertOutcome::Matched {
            node_id: bangs.clone()
        }
    );

    let (display_name,): (String,) =
        sqlx::query_as("SELECT display_name FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(&tenant)
            .bind(bangs)
            .fetch_one(store.pool())
            .await
            .expect("the row should be readable");
    assert_eq!(
        display_name, "!!!",
        "the id may be derived, but the name the agent used is kept verbatim"
    );
}

/// A blank name is not an unnameable entity, it is a caller bug: there is
/// nothing to mint an id from and nothing for a later agent to search for. It
/// is refused before any row is written, so a batch never half-lands.
#[tokio::test]
async fn a_blank_display_name_is_refused_and_writes_nothing() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "blank-name").await;

    let error = store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Real Entity"), NodeInput::new("   ")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect_err("a blank display name has no entity behind it");

    assert!(
        matches!(error, StoreError::BlankDisplayName { index: 1 }),
        "expected BlankDisplayName {{ index: 1 }}, got {error:?}"
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting should succeed");
    assert_eq!(
        rows, 0,
        "the valid sibling must not have landed either; the batch is rejected whole"
    );
}

/// Several agents share one graph and nothing serialises them. Two of them
/// naming the same entity at the same moment both read an empty table, so the
/// loser's `INSERT` collides on a primary key it had every reason to think was
/// free. Surfacing that as an error would make a correct write fail for reasons
/// the caller cannot act on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_writers_of_one_name_produce_one_node() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "concurrent").await;

    let write = || async {
        store
            .upsert_nodes(
                &tenant,
                &[NodeInput::new("Shared Entity")],
                &embedder(),
                &anonymous(),
            )
            .await
            .expect("neither writer should fail")
    };

    let (left, right) = tokio::join!(write(), write());

    let created = [&left[0], &right[0]]
        .into_iter()
        .filter(|outcome| matches!(outcome, NodeUpsertOutcome::Created { .. }))
        .count();
    assert_eq!(
        created, 1,
        "exactly one of the two should have minted the node, got {left:?} and {right:?}"
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting should succeed");
    assert_eq!(rows, 1);
}

/// Several agents write into one graph. Without the writer on the row there is
/// no way to tell which of them introduced an entity, which is the first
/// question asked of anything that looks wrong.
#[tokio::test]
async fn provenance_is_persisted_on_a_created_node() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "provenance").await;

    let provenance = Provenance {
        client: Some("claude-desktop/1.4".to_string()),
        principal: Some("user:alice".to_string()),
    };

    store
        .upsert_nodes(
            &tenant,
            &[NodeInput::new("Billing Service")],
            &embedder(),
            &provenance,
        )
        .await
        .expect("upserting a new name should succeed");

    let (client, principal): (Option<String>, Option<String>) = sqlx::query_as(
        "SELECT writer_client, writer_principal FROM node WHERE tenant_id = $1 AND node_id = $2",
    )
    .bind(&tenant)
    .bind("billing-service")
    .fetch_one(store.pool())
    .await
    .expect("the created row should be readable");

    assert_eq!(client, provenance.client);
    assert_eq!(principal, provenance.principal);
}

// ---------------------------------------------------------------------------
// Resolution and commitment, separated
//
// A disambiguation round trip needs to *decide* in one call and *write* in
// another: the first round hands the caller its near misses and the second one
// arrives with an answer. These are the two halves that makes that possible.
// ---------------------------------------------------------------------------

/// The first half. An ambiguous input comes back deferred with **no row of its
/// own written**, so a caller that is about to be asked which entity it meant
/// is not asked about a node this call has already minted — and a client that
/// never answers has not left a duplicate behind.
///
/// Everything unambiguous in the same batch is still written. One contested
/// name must not hold up the nineteen settled ones beside it.
#[tokio::test]
async fn resolve_nodes_defers_an_ambiguous_input_and_writes_nothing_for_it() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "resolve-defers").await;

    let contested = "Distributed Tracing Pipeline";
    let near = at_cosine(
        &embed(contested).await,
        &embed("unrelated anchor").await,
        0.87,
    );
    let seeded = seed_node(store, &tenant, "Observability Stack", &near).await;

    let resolutions = store
        .resolve_nodes(
            &tenant,
            &[NodeInput::new(contested), NodeInput::new("Payroll Service")],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("resolving should succeed");

    let NodeResolution::Deferred { candidates } = &resolutions[0] else {
        panic!(
            "expected the contested name to be deferred, got {:?}",
            resolutions[0]
        );
    };
    assert_eq!(candidates[0].node_id, seeded);

    assert_eq!(
        resolutions[1],
        NodeResolution::Written(NodeUpsertOutcome::Created {
            node_id: "payroll-service".to_string()
        }),
        "an unambiguous name in the same batch is still written"
    );

    let (rows,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(&tenant)
            .bind("distributed-tracing-pipeline")
            .fetch_one(store.pool())
            .await
            .expect("counting should succeed");
    assert_eq!(rows, 0, "a deferred input must leave nothing behind");
}

/// The second half. The caller has said which entity it meant, so the name is
/// attached to that node rather than minting a rival — and the surface form is
/// recorded there, which is what makes the next call resolve exactly instead of
/// asking again.
#[tokio::test]
async fn upsert_nodes_choosing_attaches_the_name_to_the_node_the_caller_picked() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "choosing-attaches").await;

    let contested = "Distributed Tracing Pipeline";
    let near = at_cosine(
        &embed(contested).await,
        &embed("unrelated anchor").await,
        0.87,
    );
    let seeded = seed_node(store, &tenant, "Observability Stack", &near).await;

    let outcomes = store
        .upsert_nodes_choosing(
            &tenant,
            &[NodeInput::new(contested)],
            &[Choice::AttachTo(seeded.clone())],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting with a choice should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Matched {
            node_id: seeded.clone()
        },
        "the caller's answer decides where the name lands"
    );
    assert_eq!(
        aliases_of(store, &tenant, &seeded).await,
        vec![contested.to_string()],
        "the chosen node has to learn the name it was chosen for"
    );

    let (rows,): (i64,) = sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting the tenant's nodes should succeed");
    assert_eq!(rows, 1, "answering must not also mint a duplicate");
}

/// An answer naming nothing is not an answer. The candidate ids offered in
/// round one are read back out of a sealed state a round later, and in between
/// a node can be deleted — so a choice that resolves to no live node falls back
/// to the behaviour of never having asked, rather than failing the item.
#[tokio::test]
async fn a_choice_naming_a_node_this_tenant_does_not_have_mints_one_instead() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "choosing-unknown").await;

    let outcomes = store
        .upsert_nodes_choosing(
            &tenant,
            &[NodeInput::new("Distributed Tracing Pipeline")],
            &[Choice::AttachTo("a-node-that-was-deleted".to_string())],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting with a stale choice should succeed");

    assert_eq!(
        outcomes[0],
        NodeUpsertOutcome::Created {
            node_id: "distributed-tracing-pipeline".to_string()
        },
        "a choice that names nothing leaves the caller with a usable handle"
    );
}

/// `CreateNew` is what a declined, cancelled or unanswered question means, and
/// it has to be indistinguishable from an upsert that never asked anybody —
/// that is the fallback the whole feature rests on.
#[tokio::test]
async fn choosing_to_create_reports_exactly_what_an_unasked_upsert_would() {
    let store = &store().await;
    let tenant = fresh_tenant(store, "choosing-create-new").await;

    let contested = "Distributed Tracing Pipeline";
    let near = at_cosine(
        &embed(contested).await,
        &embed("unrelated anchor").await,
        0.87,
    );
    let seeded = seed_node(store, &tenant, "Observability Stack", &near).await;

    let outcomes = store
        .upsert_nodes_choosing(
            &tenant,
            &[NodeInput::new(contested)],
            &[Choice::CreateNew],
            &embedder(),
            &anonymous(),
        )
        .await
        .expect("upserting with a declined choice should succeed");

    let NodeUpsertOutcome::Ambiguous {
        node_id,
        candidates,
    } = &outcomes[0]
    else {
        panic!("expected Ambiguous, got {:?}", outcomes[0]);
    };
    assert_eq!(node_id, "distributed-tracing-pipeline");
    assert_eq!(candidates[0].node_id, seeded);
}
