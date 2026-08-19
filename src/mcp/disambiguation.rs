//! Asking the caller which entity it meant, mid-call.
//!
//! When a name sits close to something this tenant already holds but not close
//! enough to commit to, the store hands back near misses instead of a decision.
//! The plain answer — mint a node, report the near misses, let the agent sort
//! it out on its next call — is still here as the fallback. This module adds
//! the better one: answer the `tools/call` with an MRTR (SEP-2322)
//! `input_required`, ask the caller which entity it meant, and finish the write
//! when the answer comes back.
//!
//! Four constraints shape everything below, and none of them is negotiable.
//!
//! **One round for the whole batch.** [`InputRequests`] is a keyed map, so a
//! batch of twenty names with five ambiguities asks five questions in one round
//! trip rather than five round trips. That is the only reason asking is
//! affordable alongside batched writes at all, and it is why the questions are
//! collected before any of them is asked.
//!
//! **Nothing is remembered between the rounds.** The retry is a fresh
//! `tools/call` that the load balancer sends wherever it likes, so anything
//! this process held is gone by the time the answer arrives. Keeping progress
//! server-side behind an opaque handle would work perfectly on one instance and
//! fail the moment a second one served a retry — which is instance affinity
//! smuggled back in, and `2026-07-28` exists to remove it. Progress therefore
//! travels with the client, in `requestState`.
//!
//! **`requestState` is untrusted.** The client echoes it back verbatim, so it
//! is attacker-controlled input: an edited one names whichever node the editor
//! chose, and the next round would bind the caller's name to it — and with it
//! every edge later written against that name. It is sealed with HMAC-SHA256
//! ([`RequestStateCodec`]) and bound to the caller and to a digest of the
//! arguments, so a state that was tampered with, minted elsewhere, taken from
//! another tenant or lifted from another call does not open. The answers are
//! checked too: an id that was not among the candidates this server offered is
//! not attached to, however well sealed the state it arrived with.
//!
//! **Decisions, never vectors.** The sealed record holds ids and indexes.
//! Sealing embeddings instead would put 768 floats per item — tens of kilobytes
//! of base64 — through the client on every round, and a vector the client can
//! edit is a vector this server would be trusting. Round two re-embeds the
//! names it still has to write, which costs one forward pass over the contested
//! items and nothing else.

use std::time::Duration;

use rmcp::ErrorData;
use rmcp::handler::server::tool::IntoCallToolResult;
use rmcp::handler::server::wrapper::Json;
use rmcp::model::{
    CallToolResponse, ElicitRequestParams, ElicitResult, ElicitationAction, ElicitationSchema,
    EnumSchema, InputRequest, InputRequests, InputRequiredResult, InputResponses,
    RequestStateCodec, SealOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::auth::Principal;
use crate::mcp::tools::{NodeReport, NodeReports, ToolError, Tools, UpsertNodeInput};
use crate::store::node::{Choice, NodeCandidate, NodeResolution, NodeUpsertOutcome};

/// How long a caller has to answer before the state stops opening.
///
/// Long enough for a person to read a prompt and pick from a list, short enough
/// that a state lifted out of a log tomorrow is already dead. Nothing else
/// expires it: there is no server-side record to expire.
const ANSWER_WINDOW: Duration = Duration::from_secs(15 * 60);

/// How many hex characters of the argument digest go into the sealed state.
///
/// 128 bits. The digest is inside the seal, so nothing can grind it without the
/// key; all it has to survive is accidental collision between two calls by the
/// same caller.
const REQUEST_DIGEST_HEX_CHARS: usize = 32;

/// Names this server, this tool and this key's purpose in the HMAC binding, so
/// a state sealed for some later tool cannot be opened as an `upsert_node` one.
const BINDING_PURPOSE: &str = "mundu/upsert_node/disambiguation/v1";

/// The single field a disambiguation answer carries.
pub(crate) const ANSWER_PROPERTY: &str = "node_id";

/// The answer meaning "none of these".
///
/// Deliberately unspellable as a `node_id`: [`canonical_node_id`] maps every
/// run of non-alphanumeric characters onto a hyphen, so no display name
/// canonicalises to anything containing an underscore. A sentinel that a real
/// entity could be named after would let a caller pick "create new" by naming a
/// node, or worse, the other way round.
///
/// [`canonical_node_id`]: crate::store::node::canonical_node_id
pub(crate) const CREATE_NEW: &str = "__create_new__";

/// Prefixes the key each question is filed under in [`InputRequests`].
const QUESTION_KEY_PREFIX: &str = "node-";

/// Why an echoed `requestState` was not honoured.
#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error(
        "the requestState echoed back with this call could not be verified; it was not sealed \
         by this memory, or not for this caller. Send the call again without it"
    )]
    Unverified,
    #[error(
        "the requestState echoed back with this call was sealed for a different call; \
         disambiguation answers cannot be moved between calls. Send the call again without it"
    )]
    ForeignRequest,
    #[error(
        "the requestState echoed back with this call has expired; send the call again \
         without it"
    )]
    Expired,
    #[error(
        "this memory could not seal the state for a disambiguation round; the call was not \
         completed"
    )]
    Unsealable,
}

/// Seals and opens the progress a disambiguation carries between its rounds.
///
/// One codec per signing key, reused for the life of the key. The key has to be
/// the same on every instance that could serve the retry, which is why it comes
/// from configuration rather than from the process.
#[derive(Clone)]
pub struct Disambiguator {
    codec: RequestStateCodec,
}

/// Hand-written so the key cannot reach a log through a `{:?}` on something
/// that happens to contain one.
impl std::fmt::Debug for Disambiguator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Disambiguator").finish()
    }
}

impl Disambiguator {
    pub fn new(key: &[u8]) -> Self {
        Self {
            codec: RequestStateCodec::new(key.to_vec()),
        }
    }

    /// Seals what this round settled and what it still has to ask about.
    pub(crate) fn seal(
        &self,
        caller: &Principal,
        input: &UpsertNodeInput,
        progress: &Progress,
    ) -> Result<String, StateError> {
        let state = SealedState {
            request: digest_of(input),
            progress: progress.clone(),
        };

        self.codec
            .seal_json_with(
                &state,
                &SealOptions::new()
                    .associated_data(&binding(caller))
                    .ttl(ANSWER_WINDOW),
            )
            .map_err(|error| {
                // Only reachable if this module's own types stop serializing,
                // which no input can cause.
                tracing::error!(%error, "could not seal a disambiguation state");
                StateError::Unsealable
            })
    }

    /// Opens a state a client echoed back, against the call it arrived on.
    ///
    /// Two bindings, both fail-closed. The caller is mixed into the HMAC, so
    /// another tenant's copy of the string does not verify at all. The digest
    /// of the arguments is inside the sealed body and compared here, so a state
    /// this caller legitimately holds from *another* `upsert_node` call cannot
    /// be replayed against this one — which matters because the answers are
    /// applied by index, and the same index in another batch is another name.
    pub(crate) fn open(
        &self,
        caller: &Principal,
        input: &UpsertNodeInput,
        sealed: &str,
    ) -> Result<Progress, StateError> {
        let state: SealedState = self
            .codec
            .open_json_with(sealed, &binding(caller))
            .map_err(|error| match error {
                rmcp::model::RequestStateError::Expired => StateError::Expired,
                error => {
                    tracing::debug!(%error, "refused a request state");
                    StateError::Unverified
                }
            })?;

        // A plain comparison, not a constant-time one: the digest is derived
        // from arguments the caller sent and can recompute at will, so there is
        // no secret here to leak through timing. The comparison that does need
        // to be constant-time is of the HMAC tag, and `open_json_with` has
        // already done it.
        if state.request != digest_of(input) {
            return Err(StateError::ForeignRequest);
        }

        Ok(state.progress)
    }
}

/// What a disambiguation carries between its rounds.
///
/// The field names are one character because every one of them is base64'd into
/// a string that travels to the client and back on each round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Progress {
    /// The inputs this round wrote, so the next one neither writes nor embeds
    /// them again.
    #[serde(rename = "s")]
    pub(crate) settled: Vec<Settled>,
    /// The inputs this round asked about, and what it offered for each.
    #[serde(rename = "a")]
    pub(crate) asked: Vec<Asked>,
}

/// One input a round resolved, by its position in the caller's batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Settled {
    #[serde(rename = "i")]
    pub(crate) index: usize,
    #[serde(rename = "n")]
    pub(crate) node_id: String,
    #[serde(rename = "m")]
    pub(crate) matched: bool,
}

/// One input a round could not settle, and the ids it offered for it.
///
/// The candidate ids are here so the next round can check the answer against
/// what was actually offered. Without them a client could name any node in the
/// tenant and have the caller's name bound to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Asked {
    #[serde(rename = "i")]
    pub(crate) index: usize,
    #[serde(rename = "c")]
    pub(crate) candidates: Vec<String>,
}

/// The sealed body: the progress, plus the call it belongs to.
#[derive(Debug, Serialize, Deserialize)]
struct SealedState {
    #[serde(rename = "r")]
    request: String,
    #[serde(rename = "p")]
    progress: Progress,
}

/// What the state is bound to besides the key: this purpose and this caller.
///
/// Built through `serde_json` rather than by concatenation, so a tenant that
/// contains the separator cannot be made to look like another tenant plus a
/// subject.
fn binding(caller: &Principal) -> Vec<u8> {
    serde_json::to_vec(&(
        BINDING_PURPOSE,
        &caller.tenant,
        caller.subject.as_deref().unwrap_or_default(),
    ))
    .unwrap_or_else(|_| BINDING_PURPOSE.as_bytes().to_vec())
}

/// Identifies the arguments a state was sealed for.
///
/// Over the re-serialized typed arguments rather than the raw JSON the client
/// sent, so the same call formatted differently — key order, whitespace, an
/// absent `aliases` against an empty one — still digests the same. What it must
/// not do is collide across *different* calls, and it does not: the names, the
/// aliases and their order are all in it.
fn digest_of(input: &UpsertNodeInput) -> String {
    let encoded = serde_json::to_vec(input).unwrap_or_default();

    Sha256::digest(&encoded)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
        .chars()
        .take(REQUEST_DIGEST_HEX_CHARS)
        .collect()
}

/// Where the question about the input at `index` is filed in the keyed map.
///
/// By position, because that is what the answer has to be put back against, and
/// because two items in one batch can carry the same display name.
pub(crate) fn question_key(index: usize) -> String {
    format!("{QUESTION_KEY_PREFIX}{index}")
}

/// The question one contested name turns into.
///
/// An elicitation rather than a sampling request: the useful answer here is a
/// person's, and asking a model to guess which of two entities was meant only
/// moves the coin flip from this server into the client.
///
/// The enum values are `node_id`s because an id is what the answer has to name
/// — a display name is not a handle, and two nodes may share one. The display
/// names and cosines go in the message instead, where a person reads them,
/// which also keeps the schema to the plain `enum` form every elicitation
/// client understands rather than the titled `oneOf` only newer ones do.
pub(crate) fn question(display_name: &str, candidates: &[NodeCandidate]) -> InputRequest {
    let offered: Vec<String> = candidates
        .iter()
        .map(|candidate| candidate.node_id.clone())
        .chain(std::iter::once(CREATE_NEW.to_string()))
        .collect();

    let listed: Vec<String> = candidates
        .iter()
        .map(|candidate| {
            format!(
                "{} ({}, {:.2} similar)",
                candidate.display_name, candidate.node_id, candidate.similarity
            )
        })
        .collect();

    let message = format!(
        "`{display_name}` is close to {} this memory already holds: {}. \
         Answer with the node_id of the one it is, or `{CREATE_NEW}` to record it as a \
         new entity.",
        if candidates.len() == 1 {
            "something"
        } else {
            "several entities"
        },
        listed.join("; ")
    );

    let schema = ElicitationSchema::builder()
        .required_enum_schema(
            ANSWER_PROPERTY,
            EnumSchema::builder(offered)
                .description(format!("Which entity `{display_name}` names"))
                .build(),
        )
        .build()
        // The one way this fails is a required property that was never added,
        // and the line above adds it.
        .unwrap_or_else(|_| ElicitationSchema::new(Default::default()));

    InputRequest::Elicitation(rmcp::model::ElicitRequest::new(
        ElicitRequestParams::FormElicitationParams {
            meta: None,
            message,
            requested_schema: schema,
        },
    ))
}

/// What one client answer means for the item it answered.
///
/// Everything that is not an accepted, well-formed choice of an id this server
/// actually offered comes out as [`Choice::CreateNew`]: a decline, a cancel, an
/// answer to a question nobody asked, an id from somewhere else, or no answer
/// at all. That is the same outcome as never having asked, which is what makes
/// a client that will not or cannot answer a client that is never blocked.
pub(crate) fn choice_of(answer: Option<&Value>, offered: &[String]) -> Choice {
    let Some(answer) = answer else {
        return Choice::CreateNew;
    };

    // The client's half of MRTR is typed as opaque JSON, so this is where it
    // becomes a result again — and where a malformed one is a decline rather
    // than a failed call.
    let Ok(result) = serde_json::from_value::<ElicitResult>(answer.clone()) else {
        tracing::debug!("ignoring a disambiguation answer that is not an elicitation result");
        return Choice::CreateNew;
    };

    if result.action != ElicitationAction::Accept {
        return Choice::CreateNew;
    }

    let chosen = result
        .content
        .as_ref()
        .and_then(|content| content.get(ANSWER_PROPERTY))
        .and_then(Value::as_str)
        .unwrap_or(CREATE_NEW);

    // Checked against what was offered, which is read out of the sealed state
    // rather than recomputed: an id the client supplies is an id the client
    // chose, and attaching to it unchecked would let any node in the tenant be
    // named as the answer to a question about any other.
    if offered.iter().any(|candidate| candidate == chosen) {
        Choice::AttachTo(chosen.to_string())
    } else {
        if chosen != CREATE_NEW {
            tracing::debug!("ignoring a disambiguation answer naming a node that was not offered");
        }
        Choice::CreateNew
    }
}

// ---------------------------------------------------------------------------
// The rounds
// ---------------------------------------------------------------------------

/// Whether the client on the other end of this call said it can answer a
/// question.
///
/// Read from the capabilities `2026-07-28` puts in every request's `_meta`,
/// which is what makes it knowable *before* the question is composed. A client
/// that never advertised elicitation is handed the plain answer instead: `rmcp`
/// would deliver the `input_required` anyway, and a question nobody can answer
/// is a call that stalls where it could have finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Elicitation {
    Supported,
    Unsupported,
}

/// What a retry brought back with it.
///
/// Both halves are optional and both are the client's: answers with no state
/// answer nothing this server can verify, and a state with no answers is a
/// caller that was asked and said nothing. Neither is an error — the first is
/// treated as a fresh call, the second as a decline.
#[derive(Debug, Default, Clone)]
pub struct Retry {
    pub request_state: Option<String>,
    pub responses: Option<InputResponses>,
}

/// Why one round of `upsert_node` could not be completed as asked.
#[derive(Debug, thiserror::Error)]
pub enum RoundError {
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    State(#[from] StateError),
    /// The reports could not be encoded as structured content. Reachable only
    /// if a `NodeReport` stops serializing, which no input can cause.
    #[error("{0}")]
    Unreportable(ErrorData),
}

/// One round of `upsert_node`: the call that asks, or the retry that answers.
///
/// Which one this is depends entirely on what the client sent. There is no
/// server-side record of a round having happened, and deliberately so — the
/// retry is a fresh `tools/call` that a load balancer may hand to any instance.
pub(crate) async fn upsert_node_round(
    tools: &Tools,
    disambiguator: Option<&Disambiguator>,
    caller: &Principal,
    client: Option<&str>,
    elicitation: Elicitation,
    input: UpsertNodeInput,
    retry: Retry,
) -> Result<CallToolResponse, RoundError> {
    if let Some(sealed) = retry.request_state.as_deref() {
        // Fail closed. A state this server could not have minted is a state it
        // cannot verify, and the one thing it must never do with an unverified
        // one is act on the ids inside it.
        let disambiguator = disambiguator.ok_or(StateError::Unverified)?;
        let progress = disambiguator.open(caller, &input, sealed)?;

        return answer(tools, caller, client, &input, progress, retry.responses).await;
    }

    // Nowhere to put the progress, or nobody to ask: finish the call the way
    // this server did before it could ask anything. Every other path — a
    // decline, a cancel, an answer nobody gave — lands on the same behaviour,
    // which is what makes an uncooperative client a slower client rather than a
    // blocked one.
    let Some(disambiguator) = disambiguator.filter(|_| elicitation == Elicitation::Supported)
    else {
        return reported(tools.upsert_node(caller, client, input).await?);
    };

    ask(tools, disambiguator, caller, client, input).await
}

/// The first round: resolve the batch, write what is certain, and ask about the
/// rest — all of it, in one `input_required`.
async fn ask(
    tools: &Tools,
    disambiguator: &Disambiguator,
    caller: &Principal,
    client: Option<&str>,
    input: UpsertNodeInput,
) -> Result<CallToolResponse, RoundError> {
    let resolutions = tools.resolve_node(caller, client, &input).await?;

    let mut progress = Progress {
        settled: Vec::new(),
        asked: Vec::new(),
    };
    let mut questions = InputRequests::new();
    let mut reports = Vec::with_capacity(input.nodes.len());

    for (index, (spec, resolution)) in input.nodes.iter().zip(resolutions).enumerate() {
        match resolution {
            NodeResolution::Written(outcome) => {
                progress.settled.push(Settled {
                    index,
                    node_id: written_id(&outcome).to_string(),
                    matched: matches!(outcome, NodeUpsertOutcome::Matched { .. }),
                });
                reports.push(NodeReport::of(&spec.display_name, outcome));
            }
            NodeResolution::Deferred { candidates } => {
                questions.insert(
                    question_key(index),
                    question(&spec.display_name, &candidates),
                );
                progress.asked.push(Asked {
                    index,
                    candidates: candidates
                        .iter()
                        .map(|candidate| candidate.node_id.clone())
                        .collect(),
                });
            }
        }
    }

    // The common case, and the one that must stay cheap: nothing was contested,
    // so the batch is finished and nobody is asked anything.
    if progress.asked.is_empty() {
        return reported(reports);
    }

    let state = disambiguator.seal(caller, &input, &progress)?;

    tracing::debug!(
        questions = progress.asked.len(),
        settled = progress.settled.len(),
        "asking the caller which entities it meant"
    );

    Ok(CallToolResponse::InputRequired(InputRequiredResult::new(
        Some(questions),
        Some(state),
    )))
}

/// The second round: apply whatever came back and finish the write.
///
/// Only the contested names are written here. The rest were written by the
/// round that asked, and their outcomes travelled in the sealed state — so they
/// are neither re-embedded nor written twice, and a name that was `Created` in
/// round one is not reported as `Matched` in round two for having been created
/// in round one.
async fn answer(
    tools: &Tools,
    caller: &Principal,
    client: Option<&str>,
    input: &UpsertNodeInput,
    progress: Progress,
    responses: Option<InputResponses>,
) -> Result<CallToolResponse, RoundError> {
    let mut contested = UpsertNodeInput { nodes: Vec::new() };
    let mut choices = Vec::with_capacity(progress.asked.len());

    for asked in &progress.asked {
        let spec = input
            .nodes
            .get(asked.index)
            .ok_or(StateError::ForeignRequest)?;

        contested.nodes.push(spec.clone());
        choices.push(choice_of(
            responses
                .as_ref()
                .and_then(|responses| responses.get(&question_key(asked.index))),
            &asked.candidates,
        ));
    }

    let written = tools
        .upsert_node_choosing(caller, client, &contested, &choices)
        .await?;

    // Rebuilt in the caller's order, because the caller matches answers to
    // questions by position and a batch of twenty is unreadable otherwise.
    let mut reports: Vec<Option<NodeReport>> = vec![None; input.nodes.len()];

    for settled in &progress.settled {
        let spec = input
            .nodes
            .get(settled.index)
            .ok_or(StateError::ForeignRequest)?;

        reports[settled.index] = Some(NodeReport {
            node_id: settled.node_id.clone(),
            display_name: spec.display_name.clone(),
            matched: settled.matched,
            candidates: None,
        });
    }

    for (asked, report) in progress.asked.iter().zip(written) {
        reports[asked.index] = Some(report);
    }

    // A state whose indexes do not cover the batch cannot have been sealed for
    // it, whatever the digest says. Refused rather than reported with a hole.
    let reports: Vec<NodeReport> = reports
        .into_iter()
        .collect::<Option<Vec<NodeReport>>>()
        .ok_or(StateError::ForeignRequest)?;

    reported(reports)
}

/// The id a written outcome minted or matched.
fn written_id(outcome: &NodeUpsertOutcome) -> &str {
    match outcome {
        NodeUpsertOutcome::Matched { node_id }
        | NodeUpsertOutcome::Created { node_id }
        // Unreachable from a deferring round, which defers exactly the
        // ambiguous ones — but an id is an id, and the candidates that would be
        // dropped here are the ones the caller is being asked about anyway.
        | NodeUpsertOutcome::Ambiguous { node_id, .. } => node_id,
    }
}

/// The finished answer, as structured content.
///
/// Wrapped in [`NodeReports`] rather than sent as a bare array, because
/// that is the shape the tool publishes an `outputSchema` for. A client
/// that validates the result against the schema would reject a bare array,
/// which is the same incompatibility as an array-typed schema, moved one
/// step downstream.
fn reported(nodes: Vec<NodeReport>) -> Result<CallToolResponse, RoundError> {
    Json(NodeReports { nodes })
        .into_call_tool_result()
        .map_err(RoundError::Unreportable)
}

#[cfg(test)]
mod tests {
    use rmcp::model::{ElicitRequestParams, InputRequest};
    use serde_json::json;

    use super::*;
    use crate::auth::Principal;
    use crate::mcp::tools::NodeSpec;
    use crate::store::node::NodeCandidate;

    const KEY: &[u8] = b"a-request-state-key-of-32-bytes!!";

    fn caller() -> Principal {
        Principal {
            tenant: "acme".to_string(),
            subject: Some("agent-a".to_string()),
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

    fn progress() -> Progress {
        Progress {
            settled: vec![Settled {
                index: 0,
                node_id: "billing-service".to_string(),
                matched: true,
            }],
            asked: vec![Asked {
                index: 1,
                candidates: vec!["observability-stack".to_string()],
            }],
        }
    }

    fn candidate(node_id: &str, display_name: &str, similarity: f64) -> NodeCandidate {
        NodeCandidate {
            node_id: node_id.to_string(),
            display_name: display_name.to_string(),
            similarity,
        }
    }

    /// The round trip the whole feature rests on: what one round sealed is what
    /// the next one reads back, so no progress has to be held on the instance
    /// that asked the question.
    #[test]
    fn a_sealed_state_opens_against_the_call_it_was_sealed_for() {
        let disambiguator = Disambiguator::new(KEY);
        let input = asking_about(&["Billing Service", "Distributed Tracing Pipeline"]);

        let sealed = disambiguator
            .seal(&caller(), &input, &progress())
            .expect("sealing should succeed");
        let opened = disambiguator
            .open(&caller(), &input, &sealed)
            .expect("the state this call sealed should open against this call");

        assert_eq!(opened, progress());
    }

    /// The client echoes `requestState` back verbatim, which makes it
    /// attacker-controlled input. An edited one names whatever node the editor
    /// chose, and the next round would bind the caller's name — and every edge
    /// later written against that name — to it.
    #[test]
    fn a_tampered_request_state_is_refused() {
        let disambiguator = Disambiguator::new(KEY);
        let input = asking_about(&["Billing Service", "Distributed Tracing Pipeline"]);
        let sealed = disambiguator
            .seal(&caller(), &input, &progress())
            .expect("sealing should succeed");

        // One character of the sealed body changed, with the tag left as it
        // was: the cheapest edit there is, and the one every other builds on.
        let mut sections: Vec<String> = sealed.split('.').map(str::to_string).collect();
        let body = &sections[1];
        let midpoint = body.len() / 2;
        sections[1] = body
            .char_indices()
            .map(|(at, character)| match (at == midpoint, character) {
                (true, 'A') => 'B',
                (true, _) => 'A',
                (false, character) => character,
            })
            .collect();
        let tampered = sections.join(".");
        assert_ne!(tampered, sealed, "the edit has to change something");

        assert!(
            matches!(
                disambiguator.open(&caller(), &input, &tampered),
                Err(StateError::Unverified)
            ),
            "an edited request state must not open"
        );
    }

    /// Same key, same caller, a different call. Without the digest check, the
    /// state from a batch that asked about `Distributed Tracing Pipeline` would
    /// be replayable against a batch that asked about something else, and the
    /// answer would be applied to whichever name happened to sit at that index.
    #[test]
    fn a_state_sealed_for_another_call_is_not_replayable_against_this_one() {
        let disambiguator = Disambiguator::new(KEY);

        let sealed = disambiguator
            .seal(
                &caller(),
                &asking_about(&["Billing Service", "Distributed Tracing Pipeline"]),
                &progress(),
            )
            .expect("sealing should succeed");

        let elsewhere = asking_about(&["Billing Service", "Payroll Service"]);
        assert!(
            matches!(
                disambiguator.open(&caller(), &elsewhere, &sealed),
                Err(StateError::ForeignRequest)
            ),
            "a state sealed for one call must not open against another"
        );
    }

    /// The state is bound to the credential as well as to the arguments: one
    /// tenant's sealed state, replayed by another, would apply an answer about
    /// nodes the second tenant cannot even see.
    #[test]
    fn a_state_sealed_for_another_caller_is_not_replayable() {
        let disambiguator = Disambiguator::new(KEY);
        let input = asking_about(&["Billing Service", "Distributed Tracing Pipeline"]);
        let sealed = disambiguator
            .seal(&caller(), &input, &progress())
            .expect("sealing should succeed");

        for other in [
            Principal {
                tenant: "rival".to_string(),
                subject: Some("agent-a".to_string()),
            },
            Principal {
                tenant: "acme".to_string(),
                subject: Some("agent-b".to_string()),
            },
        ] {
            assert!(
                matches!(
                    disambiguator.open(&other, &input, &sealed),
                    Err(StateError::Unverified)
                ),
                "{other:?} must not be able to open another caller's state"
            );
        }
    }

    /// A state sealed under one key must not open under another, which is what
    /// makes the key — rather than the format — the thing every instance that
    /// could serve the retry has to share.
    #[test]
    fn a_state_sealed_under_another_key_is_refused() {
        let input = asking_about(&["Billing Service", "Distributed Tracing Pipeline"]);
        let sealed = Disambiguator::new(KEY)
            .seal(&caller(), &input, &progress())
            .expect("sealing should succeed");

        assert!(matches!(
            Disambiguator::new(b"a-different-key-of-32-bytes-long!").open(
                &caller(),
                &input,
                &sealed
            ),
            Err(StateError::Unverified)
        ));
    }

    /// The question a contested name turns into: the name that was sent, the
    /// nodes it might be, and an option for none of them. The candidate ids are
    /// the enum values because they are what the answer has to name — a display
    /// name is not a handle.
    #[test]
    fn a_contested_name_becomes_one_elicitation_offering_its_candidates() {
        let request = question(
            "Distributed Tracing Pipeline",
            &[
                candidate("observability-stack", "Observability Stack", 0.87),
                candidate("tracing-stack", "Tracing Stack", 0.84),
            ],
        );

        let InputRequest::Elicitation(elicitation) = request else {
            panic!("a disambiguation asks a person, not a model");
        };
        let ElicitRequestParams::FormElicitationParams {
            message,
            requested_schema,
            ..
        } = elicitation.params
        else {
            panic!("the answer is a choice from a list, not a visit to a URL");
        };

        assert!(
            message.contains("Distributed Tracing Pipeline"),
            "the question has to name what was asked about: {message}"
        );

        let schema = serde_json::to_value(&requested_schema).expect("a schema serializes");
        let offered = schema["properties"][ANSWER_PROPERTY]["enum"]
            .as_array()
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| panic!("the answer should be a choice from a list: {schema}"));

        assert_eq!(
            offered,
            vec!["observability-stack", "tracing-stack", CREATE_NEW],
            "every candidate, plus the option of none of them"
        );
        assert_eq!(
            schema["required"],
            json!([ANSWER_PROPERTY]),
            "an answer that names nothing is not an answer: {schema}"
        );
    }

    /// One key per contested item, derived from its position in the batch, so
    /// twenty names with five ambiguities ask five questions in **one** round
    /// and each answer can be put back where it came from.
    #[test]
    fn every_contested_item_gets_its_own_key_in_one_round() {
        let keys: Vec<String> = [0usize, 3, 17].into_iter().map(question_key).collect();

        let mut unique = keys.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), keys.len(), "one key per item, not per round");
    }

    /// The answer is checked against what was actually offered. Without that,
    /// a client could name any node in the tenant — including one it learned
    /// about somewhere else entirely — and have the caller's name bound to it.
    #[test]
    fn an_answer_naming_a_node_that_was_never_offered_is_not_honoured() {
        let offered = vec!["observability-stack".to_string()];

        assert_eq!(
            choice_of(
                Some(
                    &json!({"action": "accept", "content": {ANSWER_PROPERTY: "observability-stack"}})
                ),
                &offered
            ),
            Choice::AttachTo("observability-stack".to_string()),
            "an answer naming something that was offered is the whole point"
        );
        assert_eq!(
            choice_of(
                Some(
                    &json!({"action": "accept", "content": {ANSWER_PROPERTY: "someone-elses-node"}})
                ),
                &offered
            ),
            Choice::CreateNew,
            "a node that was never on the list must not be attached to"
        );
    }

    /// Declining, cancelling, answering with nonsense, or not answering at all
    /// all mean the same thing: nobody said which entity this was, so mint one
    /// and report the near misses — exactly what an upsert that asked nobody
    /// does.
    #[test]
    fn a_declined_cancelled_or_missing_answer_falls_back_to_creating() {
        let offered = vec!["observability-stack".to_string()];

        for answer in [
            Some(json!({"action": "decline"})),
            Some(json!({"action": "cancel"})),
            Some(json!({"action": "accept"})),
            Some(json!({"action": "accept", "content": {}})),
            Some(json!({"action": "accept", "content": {ANSWER_PROPERTY: CREATE_NEW}})),
            Some(json!("not even an elicitation result")),
            None,
        ] {
            assert_eq!(
                choice_of(answer.as_ref(), &offered),
                Choice::CreateNew,
                "{answer:?} is not an instruction to attach to anything"
            );
        }
    }
}
