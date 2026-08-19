//! The disambiguation round trip, against a real Postgres.
//!
//! These drive `upsert_node` one MRTR round at a time: the call that asks, and
//! the retry that answers. There is no live MCP session here — `rmcp` will not
//! let one be built outside the crate — so the rounds are driven through the
//! same method the `#[tool]` adapter calls, with the two things the adapter
//! pulls out of the request context passed in explicitly: who the caller is,
//! and whether it said it can answer a question.
//!
//! **Reaching `Ambiguous` on purpose.** `DeterministicEmbedder` is a hash, so
//! two similar names are not similar vectors and nothing in a natural batch
//! ever lands in the uncertain band. `tests/node_upsert.rs` solves this by
//! writing `name_vector` directly to construct an exact cosine, and this suite
//! follows it — a disambiguation suite that never produced an ambiguity would
//! be testing the fallback path twelve times over.
//!
//! Every test owns a `tenant_id` derived from its own name: the suite runs in
//! parallel against one shared database, so isolation is tenancy rather than
//! truncation.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use std::sync::Arc;

use rmcp::model::{CallToolResponse, ElicitRequestParams, InputRequest, InputResponses};
use serde_json::{Value, json};

use mundu::auth::Principal;
use mundu::embed::{DeterministicEmbedder, Embedder};
use mundu::mcp::MunduTools;
use mundu::mcp::disambiguation::{Elicitation, Retry};
use mundu::mcp::tools::{NodeReport, NodeReports, NodeSpec, UpsertNodeInput};
use mundu::store::Store;

/// The signing key these tests share. In a deployment it comes from
/// configuration and has to be the same on every instance that could serve a
/// retry; here it only has to be the same between two calls.
const KEY: &[u8] = b"a-request-state-key-of-32-bytes!!";

async fn store() -> Store {
    common::store().await
}

fn embedder() -> DeterministicEmbedder {
    DeterministicEmbedder::new()
}

/// The tool surface with a signing key, which is what a deployment that can ask
/// questions looks like.
async fn tools() -> MunduTools {
    MunduTools::new(store().await, Arc::new(embedder())).sealing_state_with(KEY)
}

/// The same surface with no key configured — a deployment that never asks.
async fn tools_without_a_key() -> MunduTools {
    MunduTools::new(store().await, Arc::new(embedder()))
}

async fn fresh_tenant(tools: &MunduTools, name: &str) -> String {
    let tenant = format!("test-mrtr-{name}");

    sqlx::query("DELETE FROM node WHERE tenant_id = $1")
        .bind(&tenant)
        .execute(tools.operations().store().pool())
        .await
        .expect("clearing the tenant should succeed");

    tenant
}

fn principal(tenant: &str, subject: &str) -> Principal {
    Principal {
        tenant: tenant.to_string(),
        subject: Some(subject.to_string()),
    }
}

fn asking_about(display_names: &[&str]) -> UpsertNodeInput {
    UpsertNodeInput {
        nodes: display_names
            .iter()
            .map(|display_name| NodeSpec {
                display_name: display_name.to_string(),
                aliases: Vec::new(),
            })
            .collect(),
    }
}

async fn embed(text: &str) -> Vec<f32> {
    embedder()
        .embed_documents(&[text.to_string()])
        .await
        .expect("the deterministic embedder cannot fail")
        .remove(0)
}

/// Builds a unit vector whose cosine with `base` is exactly `cosine`, by the
/// method `tests/node_upsert.rs` documents: `other` is projected off `base`, so
/// the result is `cos * base + sin * u` for a genuine orthonormal `u`.
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

/// Puts `display_name` in the uncertain band against a node named
/// `near_display_name`, and hands back that node's id.
async fn contested(
    store: &Store,
    tenant: &str,
    display_name: &str,
    near_display_name: &str,
) -> String {
    let near = at_cosine(
        &embed(display_name).await,
        &embed("unrelated anchor").await,
        0.87,
    );

    seed_node(store, tenant, near_display_name, &near).await
}

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

async fn node_exists(store: &Store, tenant: &str, node_id: &str) -> bool {
    let (rows,): (i64,) =
        sqlx::query_as("SELECT count(*) FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(tenant)
            .bind(node_id)
            .fetch_one(store.pool())
            .await
            .expect("counting should succeed");
    rows > 0
}

/// The questions and the sealed state of a round that asked, or a failure
/// naming what came back instead.
fn asked(response: CallToolResponse) -> (Vec<(String, InputRequest)>, String) {
    let CallToolResponse::InputRequired(result) = response else {
        panic!("expected the call to ask something, got {response:#?}");
    };

    let requests = result
        .input_requests
        .expect("a round that asks has to say what it is asking")
        .into_iter()
        .collect();
    let state = result
        .request_state
        .expect("a round that asks has to carry the progress the retry needs");

    (requests, state)
}

/// The reports of a round that finished, or a failure naming what came back.
///
/// Deserialised as [`NodeReports`] rather than a bare `Vec`: the tool publishes
/// an object-typed `outputSchema` so that clients older than `2026-07-28` can
/// read its tool list at all, and the value has to match the schema. Going
/// through the same type the server sends is what keeps this test honest about
/// the shape a client actually receives.
fn completed(response: CallToolResponse) -> Vec<NodeReport> {
    let CallToolResponse::Complete(result) = response else {
        panic!("expected a finished call, got {response:#?}");
    };

    let reports: NodeReports = serde_json::from_value(
        result
            .structured_content
            .expect("every upsert_node answer is structured content"),
    )
    .expect("the answer should be an object carrying the node reports");

    reports.nodes
}

/// The message an elicitation carries, or a failure naming what it was instead.
fn elicited_message(request: &InputRequest) -> String {
    let InputRequest::Elicitation(elicitation) = request else {
        panic!("a disambiguation asks a person, not a model: {request:#?}");
    };
    let ElicitRequestParams::FormElicitationParams { message, .. } = &elicitation.params else {
        panic!("the answer is a choice from a list, not a visit to a URL");
    };

    message.clone()
}

/// An accepted answer naming `node_id`.
fn answer(node_id: &str) -> Value {
    json!({"action": "accept", "content": {"node_id": node_id}})
}

fn responses(answers: &[(&str, Value)]) -> InputResponses {
    answers
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

/// **One round for the whole batch.** Two contested names in a batch of three
/// produce two questions in a single `input_required`, not two round trips —
/// which is the only thing that makes asking affordable alongside batched
/// writes.
///
/// The settled name in the same batch is written on the way past: one contested
/// name must not hold up the ones beside it. The contested ones are *not*
/// written, because the whole point of asking is to avoid minting the duplicate
/// the question is about.
#[tokio::test]
async fn two_ambiguous_names_are_asked_about_together_in_one_round() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "one-round").await;
    let caller = principal(&tenant, "agent-a");
    let store = tools.operations().store();

    let tracing_stack = contested(
        store,
        &tenant,
        "Distributed Tracing Pipeline",
        "Observability Stack",
    )
    .await;
    let ledger = contested(store, &tenant, "Billing Service", "Ledger Service").await;

    let response = tools
        .upsert_node_round(
            &caller,
            Some("test-harness/1.0"),
            Elicitation::Supported,
            asking_about(&[
                "Distributed Tracing Pipeline",
                "Payroll Service",
                "Billing Service",
            ]),
            Retry::default(),
        )
        .await
        .expect("asking should succeed");

    let (requests, state) = asked(response);

    let keys: Vec<&str> = requests.iter().map(|(key, _)| key.as_str()).collect();
    assert_eq!(
        keys.len(),
        2,
        "one question per contested name, in one round: {keys:?}"
    );
    assert!(
        elicited_message(&requests[0].1).contains("Distributed Tracing Pipeline")
            && elicited_message(&requests[0].1).contains(&tracing_stack),
        "the first question should be about the first contested name and offer its candidate"
    );
    assert!(
        elicited_message(&requests[1].1).contains("Billing Service")
            && elicited_message(&requests[1].1).contains(&ledger),
        "the second question should be about the second contested name"
    );
    assert!(
        state.starts_with("rs1."),
        "the progress should travel sealed, got {state}"
    );

    assert!(
        node_exists(store, &tenant, "payroll-service").await,
        "the unambiguous name in the batch is still written"
    );
    for pending in ["distributed-tracing-pipeline", "billing-service"] {
        assert!(
            !node_exists(store, &tenant, pending).await,
            "`{pending}` is what the question is about; minting it would answer it"
        );
    }
}

/// The answer, applied. The caller says which entity it meant, and the write
/// lands on that node — including the surface form, which is what makes the
/// next call resolve exactly instead of asking again.
#[tokio::test]
async fn a_valid_retry_applies_the_choice_and_completes_the_write() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "retry-applies").await;
    let caller = principal(&tenant, "agent-a");
    let store = tools.operations().store();

    let seeded = contested(
        store,
        &tenant,
        "Distributed Tracing Pipeline",
        "Observability Stack",
    )
    .await;
    let input = asking_about(&["Payroll Service", "Distributed Tracing Pipeline"]);

    let (requests, state) = asked(
        tools
            .upsert_node_round(
                &caller,
                Some("test-harness/1.0"),
                Elicitation::Supported,
                input.clone(),
                Retry::default(),
            )
            .await
            .expect("asking should succeed"),
    );
    let key = requests[0].0.clone();

    let reports = completed(
        tools
            .upsert_node_round(
                &caller,
                Some("test-harness/1.0"),
                Elicitation::Supported,
                input,
                Retry {
                    request_state: Some(state),
                    responses: Some(responses(&[(&key, answer(&seeded))])),
                },
            )
            .await
            .expect("answering should succeed"),
    );

    assert_eq!(reports.len(), 2, "every item in the batch is reported on");
    assert_eq!(
        reports[0].node_id, "payroll-service",
        "the item settled in the first round keeps the id it was given"
    );
    assert!(
        !reports[0].matched,
        "a name the first round created must not be reported as matched by the second \
         for having been created by the first"
    );
    assert_eq!(
        reports[1].node_id, seeded,
        "the contested name should land on the node the caller picked"
    );
    assert!(
        reports[1].matched,
        "attaching to an existing entity is a match, not a creation"
    );
    assert!(
        reports[1].candidates.is_none(),
        "there is nothing left to disambiguate once it has been disambiguated"
    );

    assert_eq!(
        aliases_of(store, &tenant, &seeded).await,
        vec!["Distributed Tracing Pipeline".to_string()],
        "the chosen node has to learn the name it was chosen for"
    );
    assert!(
        !node_exists(store, &tenant, "distributed-tracing-pipeline").await,
        "answering must not also mint the duplicate the question existed to avoid"
    );
}

/// The client echoes `requestState` back verbatim, so an edited one is
/// attacker-controlled input naming whichever node the editor chose. It has to
/// be refused rather than believed.
#[tokio::test]
async fn a_tampered_request_state_is_refused() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "tampered").await;
    let caller = principal(&tenant, "agent-a");
    let store = tools.operations().store();

    let seeded = contested(
        store,
        &tenant,
        "Distributed Tracing Pipeline",
        "Observability Stack",
    )
    .await;
    let input = asking_about(&["Distributed Tracing Pipeline"]);

    let (requests, state) = asked(
        tools
            .upsert_node_round(
                &caller,
                None,
                Elicitation::Supported,
                input.clone(),
                Retry::default(),
            )
            .await
            .expect("asking should succeed"),
    );

    let mut sections: Vec<String> = state.split('.').map(str::to_string).collect();
    sections[1] = sections[1].replacen('A', "B", 1).replacen('C', "D", 1);
    let tampered = sections.join(".");
    assert_ne!(tampered, state, "the edit has to change something");

    let error = tools
        .upsert_node_round(
            &caller,
            None,
            Elicitation::Supported,
            input,
            Retry {
                request_state: Some(tampered),
                responses: Some(responses(&[(&requests[0].0, answer(&seeded))])),
            },
        )
        .await
        .expect_err("a tampered request state must not be honoured");

    assert_eq!(error.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    assert!(
        !node_exists(store, &tenant, "distributed-tracing-pipeline").await
            && aliases_of(store, &tenant, &seeded).await.is_empty(),
        "a refused retry must not have written anything"
    );
}

/// A sealed state is bound to the call it was minted for and to the caller that
/// asked. Neither another batch nor another tenant can replay it: the answers
/// are applied by index, and the same index in another batch is another name.
#[tokio::test]
async fn a_state_sealed_for_another_call_or_another_caller_is_refused() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "not-replayable").await;
    let caller = principal(&tenant, "agent-a");
    let store = tools.operations().store();

    let seeded = contested(
        store,
        &tenant,
        "Distributed Tracing Pipeline",
        "Observability Stack",
    )
    .await;
    let input = asking_about(&["Distributed Tracing Pipeline"]);

    let (requests, state) = asked(
        tools
            .upsert_node_round(
                &caller,
                None,
                Elicitation::Supported,
                input.clone(),
                Retry::default(),
            )
            .await
            .expect("asking should succeed"),
    );
    let answers = responses(&[(&requests[0].0, answer(&seeded))]);

    let elsewhere = tools
        .upsert_node_round(
            &caller,
            None,
            Elicitation::Supported,
            asking_about(&["Payroll Service"]),
            Retry {
                request_state: Some(state.clone()),
                responses: Some(answers.clone()),
            },
        )
        .await
        .expect_err("a state sealed for one call must not open against another");
    assert_eq!(elsewhere.code, rmcp::model::ErrorCode::INVALID_PARAMS);

    let stranger = tools
        .upsert_node_round(
            &principal("test-mrtr-not-replayable-rival", "agent-a"),
            None,
            Elicitation::Supported,
            input,
            Retry {
                request_state: Some(state),
                responses: Some(answers),
            },
        )
        .await
        .expect_err("another tenant must not be able to open this caller's state");
    assert_eq!(stranger.code, rmcp::model::ErrorCode::INVALID_PARAMS);
}

/// A client that declines, cancels, or simply does not answer is never blocked:
/// the call finishes exactly as it would have if nobody had been asked — a new
/// node, and the near misses reported beside it.
#[tokio::test]
async fn a_declined_or_unanswered_question_falls_back_to_create_and_report() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "declined").await;
    let caller = principal(&tenant, "agent-a");
    let store = tools.operations().store();

    let seeded = contested(
        store,
        &tenant,
        "Distributed Tracing Pipeline",
        "Observability Stack",
    )
    .await;
    let input = asking_about(&["Distributed Tracing Pipeline"]);

    for (name, given) in [
        ("declined", Some(json!({"action": "decline"}))),
        ("cancelled", Some(json!({"action": "cancel"}))),
        ("unanswered", None),
    ] {
        sqlx::query("DELETE FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(&tenant)
            .bind("distributed-tracing-pipeline")
            .execute(store.pool())
            .await
            .expect("clearing the previous round should succeed");

        let (requests, state) = asked(
            tools
                .upsert_node_round(
                    &caller,
                    None,
                    Elicitation::Supported,
                    input.clone(),
                    Retry::default(),
                )
                .await
                .expect("asking should succeed"),
        );

        let reports = completed(
            tools
                .upsert_node_round(
                    &caller,
                    None,
                    Elicitation::Supported,
                    input.clone(),
                    Retry {
                        request_state: Some(state),
                        responses: given
                            .clone()
                            .map(|value| responses(&[(&requests[0].0, value)])),
                    },
                )
                .await
                .expect("a declined question still finishes the call"),
        );

        assert_eq!(reports[0].node_id, "distributed-tracing-pipeline", "{name}");
        assert!(!reports[0].matched, "{name}: nothing was resolved");
        let candidates = reports[0]
            .candidates
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: the near misses should still be reported"));
        assert_eq!(candidates[0].node_id, seeded, "{name}");
    }
}

/// The no-regression case, and the common one. A batch with nothing contested
/// in it finishes in a single round with no MRTR anywhere near it.
#[tokio::test]
async fn an_unambiguous_batch_still_completes_in_one_round() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "no-questions").await;
    let caller = principal(&tenant, "agent-a");

    let reports = completed(
        tools
            .upsert_node_round(
                &caller,
                Some("test-harness/1.0"),
                Elicitation::Supported,
                asking_about(&["Payroll Service", "Ada Lovelace"]),
                Retry::default(),
            )
            .await
            .expect("an unambiguous batch should just work"),
    );

    let ids: Vec<&str> = reports.iter().map(|r| r.node_id.as_str()).collect();
    assert_eq!(ids, vec!["payroll-service", "ada-lovelace"]);
    assert!(
        reports.iter().all(|report| !report.matched),
        "both names are new to this tenant"
    );
}

/// A client that never said it could answer an elicitation is not asked one.
/// Under `2026-07-28` the capability arrives on every request, so this is
/// knowable before the question is composed — and a question posted to a client
/// that cannot answer it is a call that hangs where it could have completed.
#[tokio::test]
async fn a_client_that_cannot_answer_is_never_asked() {
    let tools = tools().await;
    let tenant = fresh_tenant(&tools, "cannot-answer").await;
    let caller = principal(&tenant, "agent-a");
    let store = tools.operations().store();

    let seeded = contested(
        store,
        &tenant,
        "Distributed Tracing Pipeline",
        "Observability Stack",
    )
    .await;

    let reports = completed(
        tools
            .upsert_node_round(
                &caller,
                None,
                Elicitation::Unsupported,
                asking_about(&["Distributed Tracing Pipeline"]),
                Retry::default(),
            )
            .await
            .expect("a client that cannot be asked still gets an answer"),
    );

    assert_eq!(reports[0].node_id, "distributed-tracing-pipeline");
    assert_eq!(
        reports[0]
            .candidates
            .as_ref()
            .expect("the near misses should be reported instead")[0]
            .node_id,
        seeded
    );
}

/// With no signing key configured there is nowhere safe to put the progress, so
/// the server does not ask. It behaves exactly as it did before MRTR existed —
/// which is the point: the alternative, a key minted per process, would work on
/// one instance and fail the first time a retry landed on another.
#[tokio::test]
async fn a_deployment_with_no_signing_key_never_asks() {
    let tools = tools_without_a_key().await;
    let tenant = fresh_tenant(&tools, "no-key").await;
    let caller = principal(&tenant, "agent-a");
    let store = tools.operations().store();

    let seeded = contested(
        store,
        &tenant,
        "Distributed Tracing Pipeline",
        "Observability Stack",
    )
    .await;

    let reports = completed(
        tools
            .upsert_node_round(
                &caller,
                None,
                Elicitation::Supported,
                asking_about(&["Distributed Tracing Pipeline"]),
                Retry::default(),
            )
            .await
            .expect("a server that cannot seal state should still answer"),
    );

    assert_eq!(reports[0].node_id, "distributed-tracing-pipeline");
    assert_eq!(
        reports[0]
            .candidates
            .as_ref()
            .expect("the near misses should be reported instead")[0]
            .node_id,
        seeded
    );
}
