//! End-to-end tests over the real HTTP surface.
//!
//! These start an actual listener on an ephemeral port and speak to it with
//! `reqwest`. Nothing here reaches into the tool layer directly: a request that
//! is answered was answered by the router, the authenticating middleware, the
//! `StreamableHttpService` and a tool, in that order. That whole chain is what
//! is under test, because every one of its links is new and the interesting
//! failures are between them.
//!
//! **Hand-built JSON-RPC, not an MCP client.** `rmcp`'s client transports are
//! not compiled into this crate — the feature set enables only the server side
//! — and under `2026-07-28` there is no handshake to drive anyway: every
//! request carries its own protocol version and client identity in `_meta`, so
//! a request built here is byte-for-byte what a conforming client would send.
//! The two tests that matter for reachability assert on the *tool's own
//! output* — the five published tool names, and a `node_id` that only
//! `upsert_node` can mint — so a router that merely matched the path would fail
//! them.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use mundu::auth::Authenticator;
use mundu::config::ProtocolCompat;
use mundu::embed::{DeterministicEmbedder, Embedder};
use mundu::server::AppState;
use mundu::store::Store;

/// The only protocol revision this server speaks by default.
const PROTOCOL_VERSION: &str = "2026-07-28";

/// The revision a client that is not this server's contemporary asks for.
///
/// `rmcp`'s own `ProtocolVersion::LATEST`, which is exactly what makes it the
/// one that matters: every client built on the same SDK this server uses sends
/// this and nothing else, the Devin CLI included.
const LEGACY_PROTOCOL_VERSION: &str = "2025-11-25";

/// What the legacy client in these tests calls itself in its handshake.
///
/// Deliberately nothing `rmcp` could produce on its own. If a legacy write ever
/// comes back attributed to `rmcp/<something>` instead of this, the provenance
/// is fabricated rather than remembered, and the assertion says so.
const LEGACY_CLIENT_NAME: &str = "legacy-probe";
const LEGACY_CLIENT_VERSION: &str = "9.9.9";

/// Tokens are 32 characters because `parse_static_tokens` refuses anything
/// shorter, on the grounds that a static token is the whole credential.
const ACME_TOKEN: &str = "acme-0123456789abcdef0123456789";
const RIVAL_TOKEN: &str = "rival-0123456789abcdef0123456789";

/// The key every server in this suite seals a disambiguation's `requestState`
/// with. A deployment reads it from configuration, because the instance that
/// answers a question is not necessarily the one that asked it.
const REQUEST_STATE_KEY: &[u8] = b"a-request-state-key-of-32-bytes!!";

/// A running server, and the base URL to reach it on.
///
/// The port is whatever the OS handed out: tests run in parallel, and a
/// hard-coded port is a suite that fails on a machine where something else is
/// already listening.
struct TestServer {
    url: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl TestServer {
    /// Binds port 0, reads back the assigned port, and serves until dropped.
    ///
    /// Modern-only, because that is the default and most of this suite is about
    /// what the default does.
    async fn start(auth: Authenticator) -> Self {
        Self::start_with(auth, None, ProtocolCompat::ModernOnly).await
    }

    /// As [`TestServer::start`], but with the legacy escape hatch open.
    async fn start_dual_era(auth: Authenticator) -> Self {
        Self::start_with(auth, None, ProtocolCompat::DualEra).await
    }

    /// As [`TestServer::start`], but answering to the named hostnames.
    async fn start_with_hosts(auth: Authenticator, hosts: &[&str]) -> Self {
        Self::start_with(
            auth,
            Some(hosts.iter().map(|h| h.to_string()).collect()),
            ProtocolCompat::ModernOnly,
        )
        .await
    }

    async fn start_with(
        auth: Authenticator,
        allowed_hosts: Option<Vec<String>>,
        compat: ProtocolCompat,
    ) -> Self {
        let store = store().await;
        let mut state = AppState::new(
            store,
            Arc::new(DeterministicEmbedder::new()),
            Arc::new(auth),
        )
        .with_request_state_key(REQUEST_STATE_KEY)
        .with_protocol_compat(compat);
        if let Some(hosts) = allowed_hosts {
            state = state.with_allowed_hosts(hosts);
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binding an ephemeral port should succeed");
        let port = listener
            .local_addr()
            .expect("a bound listener has an address")
            .port();

        let (shutdown, shutdown_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            axum::serve(listener, mundu::server::router(state))
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("the test server should serve");
        });

        Self {
            url: format!("http://127.0.0.1:{port}/mcp"),
            shutdown: Some(shutdown),
        }
    }

    /// Posts a JSON-RPC body, optionally bearing a token.
    ///
    /// The `Mcp-*` headers are derived from the body rather than passed in,
    /// because that is what SEP-2243 asks a client to do: they duplicate the
    /// method and the called tool name into headers so a proxy can route and
    /// audit without parsing the body. The server checks them against the body
    /// and rejects a mismatch, so building them by hand here would only test
    /// the test.
    async fn post(&self, token: Option<&str>, body: Value) -> reqwest::Response {
        self.post_as(token, PROTOCOL_VERSION, body).await
    }

    /// As [`TestServer::post`], but naming the protocol version explicitly.
    ///
    /// The header has to travel with the `_meta` value, not independently: the
    /// server checks the two against each other and rejects a mismatch on its
    /// own, which would mask whatever the caller was actually trying to test.
    async fn post_as(
        &self,
        token: Option<&str>,
        protocol_version: &str,
        body: Value,
    ) -> reqwest::Response {
        self.post_with(token, Some(protocol_version), body).await
    }

    /// A request that names no protocol version at all — the shape an
    /// unmodified pre-2026 client sends.
    async fn post_without_protocol_version(
        &self,
        token: Option<&str>,
        body: Value,
    ) -> reqwest::Response {
        self.post_with(token, None, body).await
    }

    /// A request presenting a `Host` other than the one it connected to, which
    /// is what arrives when a load balancer forwards for a public hostname.
    async fn post_with_host(&self, host: &str, token: &str, body: Value) -> reqwest::Response {
        reqwest::Client::new()
            .post(&self.url)
            .header("Host", host)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .header("MCP-Protocol-Version", PROTOCOL_VERSION)
            .header(
                "Mcp-Method",
                body["method"].as_str().expect("a request names a method"),
            )
            .header("Authorization", format!("Bearer {token}"))
            .json(&body)
            .send()
            .await
            .expect("the server should answer")
    }

    async fn post_with(
        &self,
        token: Option<&str>,
        protocol_version: Option<&str>,
        body: Value,
    ) -> reqwest::Response {
        let mut request = reqwest::Client::new()
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .header(
                "Mcp-Method",
                body["method"].as_str().expect("a request names a method"),
            );

        if let Some(protocol_version) = protocol_version {
            request = request.header("MCP-Protocol-Version", protocol_version);
        }
        if let Some(name) = body["params"].get("name").and_then(Value::as_str) {
            request = request.header("Mcp-Name", name);
        }
        if let Some(token) = token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }

        request
            .json(&body)
            .send()
            .await
            .expect("the server should answer")
    }

    /// Posts and unwraps the JSON-RPC `result`, failing loudly on an error
    /// object — an assertion that reads `hits.len() == 1` against an error
    /// response is an assertion that says nothing useful when it fails.
    async fn call(&self, token: &str, body: Value) -> Value {
        let response = self.post(Some(token), body).await;
        let status = response.status();
        let payload: Value = response
            .json()
            .await
            .expect("the server should answer with JSON");

        assert!(
            status.is_success(),
            "expected a successful call, got {status}: {payload}"
        );
        assert!(
            payload.get("error").is_none(),
            "expected a result, got a JSON-RPC error: {payload}"
        );

        payload["result"].clone()
    }
}

impl TestServer {
    /// Posts a legacy-shaped request: no `_meta` in the body, no SEP-2243
    /// `Mcp-*` headers, and the session id the handshake handed back if there
    /// is one.
    ///
    /// This is not a stripped-down modern request — it is the whole of what a
    /// `2025-11-25` client sends. Everything the modern path relies on to say
    /// who is calling is absent because no such client has anywhere to put it.
    async fn post_legacy(
        &self,
        token: &str,
        session: Option<&str>,
        protocol_version: Option<&str>,
        body: Value,
    ) -> reqwest::Response {
        let mut request = reqwest::Client::new()
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {token}"));

        // Absent on `initialize`, which is where a client learns the version,
        // and present on everything after it — exactly the 2025-06-18 rule.
        if let Some(version) = protocol_version {
            request = request.header("MCP-Protocol-Version", version);
        }
        if let Some(session) = session {
            request = request.header("Mcp-Session-Id", session);
        }

        request
            .json(&body)
            .send()
            .await
            .expect("the server should answer")
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

/// A pre-2026 client, mid-conversation.
///
/// Holds the session id, because that is the whole difference between the two
/// eras: a legacy client says who it is once, in the handshake, and everything
/// afterwards is identified by a header rather than by anything in the request.
struct LegacyClient<'a> {
    server: &'a TestServer,
    token: &'a str,
    session: String,
}

impl<'a> LegacyClient<'a> {
    /// The handshake a legacy client actually performs: `initialize` naming the
    /// version in the body and in no header, then the `notifications/initialized`
    /// that follows it.
    ///
    /// Returns the negotiated `initialize` result and the session id beside it,
    /// or the payload on its own when the server handed back no session — which
    /// is what a modern-only server does, and is the thing the refusal test
    /// asserts on.
    async fn connect(server: &'a TestServer, token: &'a str) -> Result<(Self, Value), Value> {
        let response = server
            .post_legacy(token, None, None, legacy_initialize_body())
            .await;
        let session = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let payload = json_rpc_body(response).await;

        let Some(session) = session else {
            return Err(payload);
        };

        let client = Self {
            server,
            token,
            session,
        };
        let accepted = client
            .server
            .post_legacy(
                token,
                Some(&client.session),
                Some(LEGACY_PROTOCOL_VERSION),
                json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
            )
            .await;
        assert!(
            accepted.status().is_success(),
            "a legacy session should accept the notification that opens it, got {}",
            accepted.status()
        );

        Ok((client, payload["result"].clone()))
    }

    /// One request inside the session, returning the whole JSON-RPC envelope so
    /// a caller can assert on an error as easily as on a result.
    async fn request(&self, body: Value) -> Value {
        let response = self
            .server
            .post_legacy(
                self.token,
                Some(&self.session),
                Some(LEGACY_PROTOCOL_VERSION),
                body,
            )
            .await;
        json_rpc_body(response).await
    }

    /// As [`LegacyClient::request`], but insisting on a result.
    async fn call(&self, body: Value) -> Value {
        let payload = self.request(body).await;
        assert!(
            payload.get("error").is_none(),
            "expected a result from a legacy call, got a JSON-RPC error: {payload}"
        );
        payload["result"].clone()
    }
}

/// The JSON-RPC message in a response, whether it arrived as JSON or as the
/// `text/event-stream` a legacy session answers with.
///
/// Read chunk by chunk rather than with `.text()`, because a session's SSE
/// stream is not obliged to close the moment the response is on it — waiting
/// for the end of the body would wait out the keep-alive for nothing.
async fn json_rpc_body(mut response: reqwest::Response) -> Value {
    let is_event_stream = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));

    if !is_event_stream {
        return response
            .json()
            .await
            .expect("a JSON response should carry a JSON-RPC message");
    }

    let mut buffer = String::new();
    loop {
        let chunk = tokio::time::timeout(Duration::from_secs(10), response.chunk())
            .await
            .unwrap_or_else(|_| panic!("the event stream produced no message: {buffer}"))
            .expect("the event stream should be readable");

        let Some(chunk) = chunk else {
            panic!("the event stream ended without a JSON-RPC message: {buffer}");
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        if let Some(message) = first_json_rpc_frame(&buffer) {
            return message;
        }
    }
}

/// The first `data:` line in an SSE buffer that is a whole JSON-RPC message.
///
/// The whole buffer is re-read each time because a chunk boundary can land in
/// the middle of one, and a half-parsed frame is indistinguishable from a
/// priming event that carries no data at all.
fn first_json_rpc_frame(buffer: &str) -> Option<Value> {
    buffer
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
        .find(|value| value.get("jsonrpc").is_some())
}

async fn store() -> Store {
    common::store().await
}

/// How many nodes a tenant holds, read straight out of Postgres.
///
/// Asserting on the database rather than on the HTTP response is the point: a
/// refused request could still have run a tool and written a row before failing
/// on the way out, and only the store can rule that out.
async fn nodes_in(store: &Store, tenant: &str) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM node WHERE tenant_id = $1")
        .bind(tenant)
        .fetch_one(store.pool())
        .await
        .expect("counting a tenant's nodes should succeed")
}

/// Two tokens for two tenants, so a test can prove that what one sees the other
/// does not. `suffix` keeps every test in its own pair of tenants: the suite
/// runs in parallel against one shared database, and tenancy is the isolation
/// the schema already has.
async fn two_tenant_authenticator(suffix: &str) -> (Authenticator, String, String) {
    let acme = format!("test-http-acme-{suffix}");
    let rival = format!("test-http-rival-{suffix}");

    // Cleared, not merely named: the counts these tests assert on start from
    // zero, and rows a previous run left behind would make them pass or fail
    // for reasons that have nothing to do with the code under test.
    let store = store().await;
    for tenant in [&acme, &rival] {
        for statement in [
            "DELETE FROM edge_rating WHERE tenant_id = $1",
            "DELETE FROM edge WHERE tenant_id = $1",
            "DELETE FROM node WHERE tenant_id = $1",
        ] {
            sqlx::query(statement)
                .bind(tenant)
                .execute(store.pool())
                .await
                .expect("clearing the tenant should succeed");
        }
    }

    let tokens = std::collections::HashMap::from([
        (mundu::auth::hash_token(ACME_TOKEN), acme.clone()),
        (mundu::auth::hash_token(RIVAL_TOKEN), rival.clone()),
    ]);

    (Authenticator::StaticToken { tokens }, acme, rival)
}

/// The `_meta` block SEP-2575 requires on every `2026-07-28` request. There is
/// no session to remember any of it, so it rides on each request.
fn request_meta() -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": { "name": "mundu-http-test", "version": "0.0.0" },
    })
}

fn tools_list_body() -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": { "_meta": request_meta() },
    })
}

fn tools_call_body(name: &str, arguments: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "_meta": request_meta(),
            "name": name,
            "arguments": arguments,
        },
    })
}

/// The same call from a client that says it can answer a question.
///
/// The capability rides in the per-request `_meta` like everything else: there
/// is no handshake under `2026-07-28` to have declared it in, which is exactly
/// what makes it available on the call that has to decide whether to ask.
fn tools_call_body_from_an_answering_client(name: &str, arguments: Value) -> Value {
    let mut body = tools_call_body(name, arguments);
    body["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"] =
        json!({ "elicitation": {} });
    body
}

/// The handshake body of a pre-2026 client.
///
/// Everything it will ever say about itself is in here: the version, the
/// capabilities, and the name that has to end up in `writer_client` if the
/// provenance is real rather than invented.
fn legacy_initialize_body() -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": LEGACY_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": LEGACY_CLIENT_NAME, "version": LEGACY_CLIENT_VERSION },
        },
    })
}

fn legacy_tools_list_body() -> Value {
    json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {} })
}

fn legacy_tools_call_body(name: &str, arguments: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": { "name": name, "arguments": arguments },
    })
}

/// The `writer_client` recorded against a tenant's one node, read straight out
/// of Postgres.
///
/// The column, not the tool's echo of it: what a client is *told* about its own
/// write is not evidence about what was stored, and the audit trail is the
/// stored value.
async fn writer_client_of_the_only_node(store: &Store, tenant: &str) -> Option<String> {
    sqlx::query_scalar("SELECT writer_client FROM node WHERE tenant_id = $1")
        .bind(tenant)
        .fetch_one(store.pool())
        .await
        .expect("the node the legacy client wrote should be readable")
}

/// Builds a unit vector whose cosine with `base` is exactly `cosine`, by the
/// method `tests/node_upsert.rs` documents. `DeterministicEmbedder` is a hash,
/// so nothing written through the tools ever lands in the uncertain band; the
/// only way to put a contested name in front of the resolver is to construct
/// the neighbour's vector directly.
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

/// Seeds a node that sits in the uncertain band against `contested`, and hands
/// back its id.
async fn seed_neighbour_of(
    store: &Store,
    tenant: &str,
    contested: &str,
    display_name: &str,
) -> String {
    let embedder = DeterministicEmbedder::new();
    let mut vectors = embedder
        .embed_documents(&[contested.to_string(), "unrelated anchor".to_string()])
        .await
        .expect("the deterministic embedder cannot fail");
    let anchor = vectors.remove(1);
    let base = vectors.remove(0);

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
    .bind(at_cosine(&base, &anchor, 0.87))
    .bind(embedder.model_id())
    .execute(store.pool())
    .await
    .expect("seeding a node should succeed");

    node_id
}

/// The names a client actually gets back, in whatever order the router lists
/// them.
fn listed_tool_names(result: &Value) -> Vec<String> {
    let mut names: Vec<String> = result["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("tools/list should return an array of tools, got {result}"))
        .iter()
        .map(|tool| {
            tool["name"]
                .as_str()
                .expect("a tool has a name")
                .to_string()
        })
        .collect();
    names.sort();
    names
}

/// The structured payload a tool returned, which is where every one of this
/// server's tools puts its answer.
fn structured(result: &Value) -> &Value {
    assert_ne!(
        result["isError"],
        json!(true),
        "the tool reported a failure: {result}"
    );
    result
        .get("structuredContent")
        .unwrap_or_else(|| panic!("a tool result should carry structured content, got {result}"))
}

/// The array inside a tool's structured content.
///
/// Every tool wraps its results in an object carrying exactly one array field,
/// because an object-typed `outputSchema` is what lets a client older than
/// `2026-07-28` read the tool list at all — a bare array makes the whole list
/// unreadable to it. Indexing the payload directly is what this replaces: a
/// `Value` yields `null` for a bad index rather than failing, so the old shape
/// silently produced null ids and an empty search instead of an error.
///
/// Asserting there is exactly one field keeps the helper honest if a tool ever
/// grows a second one.
fn items(result: &Value) -> &Value {
    let payload = structured(result);
    let fields = payload
        .as_object()
        .unwrap_or_else(|| panic!("a tool payload should be an object, got {payload}"));

    assert_eq!(
        fields.len(),
        1,
        "a tool payload wraps exactly one array, got {payload}"
    );

    fields
        .values()
        .next()
        .expect("the payload has exactly one field")
}

/// The whole tool surface has to be reachable through the transport, not just
/// present in the binary. This is the test that would catch a router mounted at
/// the wrong path, a handler wired to an empty tool router, or a middleware
/// that swallows the request.
#[tokio::test]
async fn tools_list_returns_all_five_tools() {
    let (auth, _, _) = two_tenant_authenticator("list").await;
    let server = TestServer::start(auth).await;

    let result = server.call(ACME_TOKEN, tools_list_body()).await;

    assert_eq!(
        listed_tool_names(&result),
        vec![
            "create_provenance_relationship",
            "explore_entity",
            "rate_memory_utility",
            "search_knowledge_graph",
            "upsert_node",
        ],
    );
}

/// The end-to-end case: a real socket, a real credential, a real tool, a real
/// row in Postgres.
///
/// The assertion is on the `node_id`, not on the shape of the envelope.
/// `upsert_node` is the only thing that mints one, so a `node_id` in the
/// response is proof the request reached the tool and the tool reached the
/// store — which no amount of correct routing could fake.
#[tokio::test]
async fn a_tools_call_with_a_valid_token_reaches_the_tool_and_returns_its_result() {
    let (auth, _, _) = two_tenant_authenticator("call").await;
    let server = TestServer::start(auth).await;

    let result = server
        .call(
            ACME_TOKEN,
            tools_call_body(
                "upsert_node",
                json!({ "nodes": [{ "display_name": "Ada Lovelace" }] }),
            ),
        )
        .await;

    let reports = items(&result);
    assert_eq!(
        reports[0]["display_name"], "Ada Lovelace",
        "the tool should echo the name it was sent, got {reports}"
    );
    assert!(
        reports[0]["node_id"]
            .as_str()
            .is_some_and(|id| !id.is_empty()),
        "only upsert_node mints a node_id; its absence means no tool ran: {reports}"
    );
}

/// An unauthenticated write must be refused *before* a tool sees it, not by a
/// tool that finds no tenant to file the write under. The distinction is
/// visible from outside: a 401 at the transport, rather than a 200 carrying a
/// JSON-RPC error.
///
/// The `WWW-Authenticate` header is asserted because a bare 401 leaves a client
/// unable to tell "wrong credential" from "wrong URL".
#[tokio::test]
async fn a_request_with_no_authorization_header_is_refused_before_any_tool_runs() {
    let (auth, acme, _) = two_tenant_authenticator("anonymous").await;
    let server = TestServer::start(auth).await;
    let store = store().await;

    let response = server
        .post(
            None,
            tools_call_body(
                "upsert_node",
                json!({ "nodes": [{ "display_name": "Anonymous Write" }] }),
            ),
        )
        .await;

    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    assert!(
        response
            .headers()
            .get("WWW-Authenticate")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|challenge| challenge.starts_with("Bearer")),
        "a 401 should say how to authenticate, got {:?}",
        response.headers().get("WWW-Authenticate")
    );

    assert_eq!(
        nodes_in(&store, &acme).await,
        0,
        "an unauthenticated request must not have reached the store"
    );
}

/// A token that is well-formed but not one this server knows is refused the
/// same way, and — the part that matters — refused at the transport rather
/// than resolved to some default tenant.
#[tokio::test]
async fn a_request_with_an_invalid_token_is_refused() {
    let (auth, acme, _) = two_tenant_authenticator("forged").await;
    let server = TestServer::start(auth).await;
    let store = store().await;

    let response = server
        .post(
            Some("forged-0123456789abcdef0123456789"),
            tools_call_body(
                "upsert_node",
                json!({ "nodes": [{ "display_name": "Forged Write" }] }),
            ),
        )
        .await;

    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    assert_eq!(
        nodes_in(&store, &acme).await,
        0,
        "a forged token must not have reached the store"
    );
}

/// The tenant boundary, asserted through the transport rather than under it.
///
/// `tests/mcp_tools.rs` already proves two `Principal`s see disjoint data. What
/// it cannot prove is that the *credential* is what produces the `Principal` —
/// that is decided in the middleware, and only a request over a real socket
/// exercises it. A token swap here has to change which tenant is written to and
/// which is searched.
#[tokio::test]
async fn two_tokens_for_two_tenants_see_disjoint_data() {
    let (auth, acme, rival) = two_tenant_authenticator("isolation").await;
    let server = TestServer::start(auth).await;
    let store = store().await;

    // Each tenant records the same fact about differently-named entities, so
    // any leak shows up as the other tenant's entity name.
    for (token, subject) in [(ACME_TOKEN, "Acme Turbine"), (RIVAL_TOKEN, "Rival Turbine")] {
        let nodes = server
            .call(
                token,
                tools_call_body(
                    "upsert_node",
                    json!({ "nodes": [{ "display_name": subject }, { "display_name": "Steel" }] }),
                ),
            )
            .await;
        let nodes = items(&nodes);

        server
            .call(
                token,
                tools_call_body(
                    "create_provenance_relationship",
                    json!({ "relationships": [{
                        "source_node_id": nodes[0]["node_id"],
                        "relationship_type": "USES",
                        "target_node_id": nodes[1]["node_id"],
                        "proof_excerpt": format!("{subject} is machined from steel."),
                    }] }),
                ),
            )
            .await;
    }

    // Both tenants hold a fact matching this query; each must see only its own.
    let body = tools_call_body(
        "search_knowledge_graph",
        json!({ "query": "turbine steel" }),
    );
    let acme_hits = items(&server.call(ACME_TOKEN, body.clone()).await).clone();
    let rival_hits = items(&server.call(RIVAL_TOKEN, body).await).clone();

    for (hits, own, other) in [
        (&acme_hits, "Acme Turbine", "Rival Turbine"),
        (&rival_hits, "Rival Turbine", "Acme Turbine"),
    ] {
        let text = hits.to_string();
        assert!(
            text.contains(own),
            "{own}'s own fact should come back to it, got {hits}"
        );
        assert!(
            !text.contains(other),
            "{own} must not see {other}'s facts, got {hits}"
        );
    }

    // And the writes landed under the tenant the token names, not merely the
    // reads being filtered on the way out.
    assert_eq!(nodes_in(&store, &acme).await, 2);
    assert_eq!(nodes_in(&store, &rival).await, 2);
}

/// Older clients are turned away on purpose.
///
/// Everything this server records about *who* wrote a fact comes from the
/// per-request `_meta` that only `2026-07-28` carries. A `2025-06-18` client
/// would be served statelessly all the same, and every edge it wrote would have
/// no writer attribution at all — a silent, permanent hole in the provenance
/// the whole design rests on. Better a client that cannot connect than a graph
/// nobody can audit.
#[tokio::test]
async fn a_client_on_an_older_protocol_version_is_rejected() {
    let (auth, acme, _) = two_tenant_authenticator("legacy").await;
    let server = TestServer::start(auth).await;
    let store = store().await;

    // A coherent older client: the version it names in `_meta` is the version
    // it names in the header. An inconsistent pair is rejected as a mismatch
    // long before anything consults the supported set, which would make this
    // test pass without proving anything about it.
    const LEGACY_VERSION: &str = "2025-06-18";
    let mut body = tools_call_body(
        "upsert_node",
        json!({ "nodes": [{ "display_name": "Legacy Write" }] }),
    );
    body["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] = json!(LEGACY_VERSION);

    let response = server.post_as(Some(ACME_TOKEN), LEGACY_VERSION, body).await;
    let status = response.status();
    let payload: Value = response.json().await.expect("the server should answer");

    assert!(
        payload.get("error").is_some(),
        "a request naming an unsupported protocol version should be an error, \
         got {status}: {payload}"
    );
    assert_eq!(
        nodes_in(&store, &acme).await,
        0,
        "a rejected protocol version must not have reached the store"
    );
}

/// A request that declares nothing about itself — no `MCP-Protocol-Version`
/// header, no `_meta` — is refused rather than assumed to be a legacy client.
///
/// This is the case `with_stateless_protocol_metadata_required(true)` exists
/// for. Without it, `rmcp` reads an absent header as `2025-03-26` and serves
/// the request anyway; with `legacy_session_mode` off it would be served
/// statelessly, so the write would land with no client attribution at all.
/// Declaring nothing must not be the way to opt out of provenance.
#[tokio::test]
async fn a_request_declaring_no_protocol_version_at_all_is_rejected() {
    let (auth, acme, _) = two_tenant_authenticator("bare").await;
    let server = TestServer::start(auth).await;
    let store = store().await;

    let mut body = tools_call_body(
        "upsert_node",
        json!({ "nodes": [{ "display_name": "Bare Write" }] }),
    );
    body["params"]
        .as_object_mut()
        .expect("params is an object")
        .remove("_meta");

    let response = server
        .post_without_protocol_version(Some(ACME_TOKEN), body)
        .await;
    let status = response.status();
    let payload: Value = response.json().await.expect("the server should answer");

    assert!(
        payload.get("error").is_some(),
        "a request carrying no protocol metadata should be an error, got {status}: {payload}"
    );
    assert_eq!(
        nodes_in(&store, &acme).await,
        0,
        "a request with no provenance metadata must not have written anything"
    );
}

/// The other half of the same rule: a request that *names* `2026-07-28` in its
/// header but leaves the matching `_meta` out is refused too, so a client
/// cannot claim the modern protocol and then omit the identity that protocol
/// carries.
#[tokio::test]
async fn a_modern_request_with_no_meta_is_rejected() {
    let (auth, acme, _) = two_tenant_authenticator("nometa").await;
    let server = TestServer::start(auth).await;
    let store = store().await;

    let mut body = tools_call_body(
        "upsert_node",
        json!({ "nodes": [{ "display_name": "Metaless Write" }] }),
    );
    body["params"]
        .as_object_mut()
        .expect("params is an object")
        .remove("_meta");

    let response = server.post(Some(ACME_TOKEN), body).await;
    let status = response.status();
    let payload: Value = response.json().await.expect("the server should answer");

    assert!(
        payload.get("error").is_some(),
        "a modern request with no _meta should be an error, got {status}: {payload}"
    );
    assert_eq!(nodes_in(&store, &acme).await, 0);
}

/// **The escape hatch, end to end.** A `2025-11-25` client handshakes, lists
/// the tools and writes a node, sending no `_meta` anywhere because no such
/// client has anywhere to put it.
///
/// This is the case the whole setting exists for. Against the default it is the
/// failure the change was reported for: `initialize` answers `200` advertising
/// a version the client did not ask for, and every call after it dies on a
/// header the client does not know to send.
#[tokio::test]
async fn a_legacy_client_is_served_end_to_end_under_dual_era() {
    let (auth, acme, _) = two_tenant_authenticator("dual-era").await;
    let server = TestServer::start_dual_era(auth).await;
    let store = store().await;

    let (client, handshake) = LegacyClient::connect(&server, ACME_TOKEN)
        .await
        .unwrap_or_else(|payload| panic!("a legacy handshake should open a session: {payload}"));

    assert_eq!(
        handshake["protocolVersion"], LEGACY_PROTOCOL_VERSION,
        "the server has to agree to the version the client asked for, not talk over it: \
         {handshake}"
    );

    let listed = client.call(legacy_tools_list_body()).await;
    assert_eq!(
        listed_tool_names(&listed),
        vec![
            "create_provenance_relationship",
            "explore_entity",
            "rate_memory_utility",
            "search_knowledge_graph",
            "upsert_node",
        ],
    );

    let written = client
        .call(legacy_tools_call_body(
            "upsert_node",
            json!({ "nodes": [{ "display_name": "Legacy Ada" }] }),
        ))
        .await;
    let reports = items(&written);
    assert_eq!(reports[0]["display_name"], "Legacy Ada");

    assert_eq!(
        nodes_in(&store, &acme).await,
        1,
        "the legacy call has to have reached the store, not merely been answered"
    );
}

/// **What a legacy write is actually attributed to.**
///
/// The reason `legacy_session_mode` is on under `dual-era` and not off. A
/// legacy client states its identity once, in the handshake; the session is the
/// only thing that remembers it, and `RequestContext::client_info()` reads it
/// back from there. Sessionless, `rmcp` synthesises a peer from
/// `Implementation::default()` and the same call reports every legacy writer as
/// `rmcp/<the version of rmcp this binary was built against>` — a fabricated
/// name, not a missing one, and indistinguishable from a genuine `rmcp` client.
///
/// So this asserts the positive *and* the negative: the handshake's name is
/// stored, and the fabricated one is not.
#[tokio::test]
async fn a_legacy_write_is_attributed_to_the_client_that_handshook() {
    let (auth, acme, _) = two_tenant_authenticator("dual-era-provenance").await;
    let server = TestServer::start_dual_era(auth).await;
    let store = store().await;

    let (client, _) = LegacyClient::connect(&server, ACME_TOKEN)
        .await
        .unwrap_or_else(|payload| panic!("a legacy handshake should open a session: {payload}"));

    client
        .call(legacy_tools_call_body(
            "upsert_node",
            json!({ "nodes": [{ "display_name": "Provenance Probe" }] }),
        ))
        .await;

    let writer = writer_client_of_the_only_node(&store, &acme).await;

    assert_eq!(
        writer.as_deref(),
        Some(format!("{LEGACY_CLIENT_NAME}/{LEGACY_CLIENT_VERSION}").as_str()),
        "a legacy write must be stamped with the client that handshook; anything else is \
         either a hole in the audit trail or, worse, somebody else's name in it"
    );
    assert!(
        !writer.as_deref().is_some_and(|w| w.starts_with("rmcp/")),
        "`rmcp/...` here would mean the identity was synthesised by the SDK rather than \
         remembered from the handshake, got {writer:?}"
    );
}

/// The escape hatch has to be genuinely opt-in: under the unchanged default the
/// same client gets exactly the treatment it gets today.
///
/// Which is worse than a refusal, and that is the point of asserting it
/// precisely. `initialize` succeeds with `200`, hands back no session, and
/// advertises `2026-07-28` — a version the client did not ask for and cannot
/// speak — and the first real call after it fails on the per-request metadata
/// it has no way to send.
#[tokio::test]
async fn a_legacy_client_is_still_turned_away_under_the_default_modern_only() {
    let (auth, acme, _) = two_tenant_authenticator("modern-only-legacy").await;
    let server = TestServer::start(auth).await;
    let store = store().await;

    let handshake = LegacyClient::connect(&server, ACME_TOKEN)
        .await
        .err()
        .expect("a modern-only server must not open a session for a legacy client");

    assert_eq!(
        handshake["result"]["protocolVersion"], PROTOCOL_VERSION,
        "today's shape: the handshake is answered with the modern version the client never \
         asked for, got {handshake}"
    );

    // And the call the client would make next, exactly as it would make it.
    let refused = json_rpc_body(
        server
            .post_legacy(
                ACME_TOKEN,
                None,
                None,
                legacy_tools_call_body(
                    "upsert_node",
                    json!({ "nodes": [{ "display_name": "Legacy Write" }] }),
                ),
            )
            .await,
    )
    .await;

    assert!(
        refused.get("error").is_some(),
        "a legacy call must still be refused under the default, got {refused}"
    );
    assert_eq!(
        nodes_in(&store, &acme).await,
        0,
        "a refused legacy client must not have written anything"
    );
}

/// Widening the server must not narrow it. A `2026-07-28` client sees the same
/// stateless, `_meta`-carrying, JSON-answering server it saw before — `rmcp`
/// routes a modern request statelessly whatever `legacy_session_mode` says, and
/// this is the test that would notice if it stopped.
#[tokio::test]
async fn a_modern_client_is_unchanged_by_dual_era() {
    let (auth, acme, _) = two_tenant_authenticator("dual-era-modern").await;
    let server = TestServer::start_dual_era(auth).await;
    let store = store().await;

    let listed = server.call(ACME_TOKEN, tools_list_body()).await;
    assert_eq!(listed_tool_names(&listed).len(), 5);

    let written = server
        .call(
            ACME_TOKEN,
            tools_call_body(
                "upsert_node",
                json!({ "nodes": [{ "display_name": "Modern Ada" }] }),
            ),
        )
        .await;
    assert_eq!(items(&written)[0]["display_name"], "Modern Ada");

    assert_eq!(
        writer_client_of_the_only_node(&store, &acme)
            .await
            .as_deref(),
        Some("mundu-http-test/0.0.0"),
        "a modern client's provenance still comes from its own per-request _meta"
    );
}

/// `rmcp` answers only to hostnames it has been told about, which defaults to
/// loopback — a DNS-rebinding guard for servers running on a laptop. Behind the
/// round-robin load balancer this server is designed for, the `Host` is the
/// deployment's public name, and without a way to name it every production
/// request is a 403.
///
/// Both directions are asserted, because a knob that only ever widens is a
/// guard that has been removed rather than configured.
#[tokio::test]
async fn the_host_header_is_checked_against_the_configured_hostnames() {
    let (auth, _, _) = two_tenant_authenticator("hosts").await;
    let server = TestServer::start(auth).await;

    let foreign = server
        .post_with_host("mundu.example.com", ACME_TOKEN, tools_list_body())
        .await;
    assert_eq!(
        foreign.status(),
        reqwest::StatusCode::FORBIDDEN,
        "an unconfigured Host should be refused by default"
    );

    let (auth, _, _) = two_tenant_authenticator("hosts-allowed").await;
    let configured = TestServer::start_with_hosts(auth, &["mundu.example.com"]).await;

    let accepted = configured
        .post_with_host("mundu.example.com", ACME_TOKEN, tools_list_body())
        .await;
    assert!(
        accepted.status().is_success(),
        "a configured Host should be served, got {}",
        accepted.status()
    );
}

/// **MRTR, over the wire.** A contested name comes back as an
/// `input_required` with a question and a sealed `requestState`; the client
/// answers it by repeating the same `tools/call` with the answer and the state
/// echoed back, and the second call finishes the write.
///
/// Driven as two independent HTTP requests on purpose. That is what a client
/// actually sends, and it is the property the design turns on: nothing is
/// remembered between them, so the retry could as easily have landed on another
/// instance.
#[tokio::test]
async fn a_contested_name_is_asked_about_and_the_answer_completes_the_write() {
    let (auth, tenant, _) = two_tenant_authenticator("mrtr").await;
    let server = TestServer::start(auth).await;
    let store = store().await;

    let contested = "Distributed Tracing Pipeline";
    let seeded = seed_neighbour_of(&store, &tenant, contested, "Observability Stack").await;
    let arguments = json!({ "nodes": [{ "display_name": contested }] });

    let asked = server
        .call(
            ACME_TOKEN,
            tools_call_body_from_an_answering_client("upsert_node", arguments.clone()),
        )
        .await;

    assert_eq!(
        asked["resultType"], "input_required",
        "a contested name should be asked about, got {asked}"
    );
    let questions = asked["inputRequests"]
        .as_object()
        .unwrap_or_else(|| panic!("an input_required has to say what it is asking: {asked}"));
    assert_eq!(questions.len(), 1, "one question for one contested name");

    let key = questions
        .keys()
        .next()
        .expect("the question is filed under a key the answer uses")
        .clone();
    let question = &questions[&key];
    assert_eq!(question["method"], "elicitation/create");
    assert!(
        question["params"]["message"]
            .as_str()
            .is_some_and(|message| message.contains(contested) && message.contains(&seeded)),
        "the question should name what was asked about and what it might be: {question}"
    );

    let state = asked["requestState"]
        .as_str()
        .unwrap_or_else(|| panic!("the progress has to travel with the client: {asked}"))
        .to_string();
    assert!(state.starts_with("rs1."), "the state should be sealed");

    let mut retry = tools_call_body_from_an_answering_client("upsert_node", arguments);
    retry["params"]["inputResponses"] =
        json!({ key: { "action": "accept", "content": { "node_id": seeded } } });
    retry["params"]["requestState"] = json!(state);

    let answered = server.call(ACME_TOKEN, retry).await;
    let reports = items(&answered);

    assert_eq!(
        reports[0]["node_id"],
        json!(seeded),
        "the name should land on the entity the caller picked: {answered}"
    );
    assert_eq!(reports[0]["matched"], json!(true));

    let (aliases,): (Vec<String>,) =
        sqlx::query_as("SELECT aliases FROM node WHERE tenant_id = $1 AND node_id = $2")
            .bind(&tenant)
            .bind(&seeded)
            .fetch_one(store.pool())
            .await
            .expect("the chosen node should be readable");
    assert_eq!(
        aliases,
        vec![contested.to_string()],
        "the chosen node has to learn the name it was chosen for"
    );
    assert_eq!(
        nodes_in(&store, &tenant).await,
        1,
        "answering must not also mint the duplicate the question existed to avoid"
    );
}
