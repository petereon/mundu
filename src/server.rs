//! The HTTP transport: an axum router with the MCP service mounted at `/mcp`.

use std::borrow::Cow;
use std::sync::Arc;

use axum::Router;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ListToolsResult, PaginatedRequestParams,
    ProtocolVersion, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::StreamableHttpService;
use rmcp::transport::streamable_http_server::StreamableHttpServerConfig;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::session::never::NeverSessionManager;
use rmcp::{ErrorData, RoleServer, ServerHandler};

use crate::auth::{Authenticator, Principal};
use crate::config::ProtocolCompat;
use crate::embed::Embedder;
use crate::error::AuthError;
use crate::mcp::MunduTools;
use crate::store::Store;

/// The protocol revision this server is built around.
///
/// Deliberately not "the newest one `rmcp` knows" and deliberately alone in the
/// default list. Everything below it depends on per-request `_meta` — the
/// protocol version, and the client identity that becomes the provenance
/// recorded against every write — and a client negotiated down sends none of
/// it. Under [`ProtocolCompat::ModernOnly`] that client is refused, because a
/// silently unattributed edge is worse than a connection that did not happen.
const MODERN_PROTOCOL_VERSIONS: &[ProtocolVersion] = &[ProtocolVersion::V_2026_07_28];

/// The revisions [`ProtocolCompat::DualEra`] adds, and where the list stops.
///
/// `2026-07-28` stays **first**, so it is still what this server prefers and
/// still what a capable client negotiates. The two behind it are additions, not
/// a replacement.
///
/// - **`2025-11-25`** is the one proven necessary. It is `rmcp`'s own
///   `ProtocolVersion::LATEST`, so every client built on the same SDK this
///   server uses — the Devin CLI among them — asks for exactly this and
///   nothing else.
/// - **`2025-06-18`** is where the list stops, and the reason is mechanical
///   rather than a guess about who ships what. It is the oldest revision that
///   defines the `MCP-Protocol-Version` HTTP header, so it is the oldest one in
///   which a request *states* which era it belongs to. Below it there is no
///   such header, and `rmcp`'s transport falls back to assuming `2025-03-26`
///   for anything that arrives without one — meaning the era of the request
///   would be a guess, and a request that declared nothing at all would be
///   served. `2025-03-26` and `2024-11-05` are therefore left out on purpose:
///   they are not "older clients", they are "no statement at all".
const DUAL_ERA_PROTOCOL_VERSIONS: &[ProtocolVersion] = &[
    ProtocolVersion::V_2026_07_28,
    ProtocolVersion::V_2025_11_25,
    ProtocolVersion::V_2025_06_18,
];

/// What a [`ProtocolCompat`] means to the transport, in one place.
///
/// Three settings that have to agree, which is why they are read off one value
/// rather than set at three call sites: the advertised version list decides
/// what a client may negotiate, and the two flags decide whether what it then
/// sends is served or refused. A list that admits `2025-11-25` with the modern
/// flags still on is the failure this whole change exists to fix — the client
/// connects and every subsequent call is a `400`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransportPolicy {
    /// Everything [`ServerHandler::supported_protocol_versions`] advertises,
    /// most-preferred first.
    pub protocol_versions: &'static [ProtocolVersion],
    /// Whether a legacy request is given a session.
    ///
    /// **On under `dual-era`, and this is the trade.** A legacy client states
    /// its identity once, in the `initialize` handshake, and a session is the
    /// only thing that remembers it — `RequestContext::client_info()` reads it
    /// back out of the peer's handshake info. Off, the same call is served
    /// statelessly and `rmcp` synthesises a peer with
    /// `Implementation::default()`, so `client_info()` yields `rmcp/<the
    /// version of rmcp this binary was compiled against>` for *every* legacy
    /// client — a fabricated writer, not an absent one. Measured, not inferred:
    /// see `tests/server_http.rs`.
    ///
    /// The cost is instance affinity. Sessions live in this process, so a
    /// legacy client pinned to nothing gets `404 Not Found: Session not found`
    /// the first time a load balancer sends it elsewhere. Modern requests are
    /// unaffected — `rmcp` routes them statelessly whatever this flag says.
    pub legacy_session_mode: bool,
    /// Whether a stateless request is refused for carrying no per-request
    /// protocol metadata.
    ///
    /// Must be off wherever legacy requests are served: there is no `_meta` in
    /// a `2025-11-25` request to require, and requiring it is precisely what
    /// turns a legacy client's every call into
    /// `-32020 Missing MCP-Protocol-Version header`.
    pub stateless_protocol_metadata_required: bool,
}

impl TransportPolicy {
    /// The one mapping from a deployment's answer to the transport's behaviour.
    pub const fn for_compat(compat: ProtocolCompat) -> Self {
        match compat {
            ProtocolCompat::ModernOnly => Self {
                protocol_versions: MODERN_PROTOCOL_VERSIONS,
                // Sessions are gone in 2026-07-28. Leaving the legacy path on
                // would let a client mint one and start depending on instance
                // affinity.
                legacy_session_mode: false,
                stateless_protocol_metadata_required: true,
            },
            ProtocolCompat::DualEra => Self {
                protocol_versions: DUAL_ERA_PROTOCOL_VERSIONS,
                legacy_session_mode: true,
                stateless_protocol_metadata_required: false,
            },
        }
    }
}

/// The path the MCP endpoint is served on.
pub const MCP_PATH: &str = "/mcp";

/// Everything a request needs that outlives the request.
///
/// `Clone` and cheap to clone, because it has to be: `StreamableHttpService`
/// runs its `service_factory` once **per request**, so anything expensive to
/// build has to already exist behind a handle. A `PgPool` clone is a handle to
/// the same pool; an `Arc<dyn Embedder>` clone is a pointer bump.
///
/// Nothing in here is per-caller or per-conversation. That is the property that
/// makes horizontal scaling work under `2026-07-28`: with no session to pin a
/// client to an instance, two consecutive requests from one client can land on
/// two different processes, and neither may be holding state the other needs.
#[derive(Clone)]
pub struct AppState {
    tools: MunduTools,
    auth: Arc<Authenticator>,
    allowed_hosts: Vec<String>,
    compat: ProtocolCompat,
}

impl AppState {
    pub fn new(store: Store, embedder: Arc<dyn Embedder>, auth: Arc<Authenticator>) -> Self {
        Self {
            tools: MunduTools::new(store, embedder),
            auth,
            allowed_hosts: crate::config::DEFAULT_ALLOWED_HOSTS
                .map(str::to_string)
                .to_vec(),
            compat: ProtocolCompat::default(),
        }
    }

    /// Widens — or restates — which protocol revisions this server answers.
    ///
    /// A builder rather than an argument to [`AppState::new`] for the same
    /// reason [`AppState::with_allowed_hosts`] is one: the default is the safe
    /// answer, and every deployment that leaves this alone gets it. Calling it
    /// with [`ProtocolCompat::DualEra`] is what turns sessions on.
    pub fn with_protocol_compat(mut self, compat: ProtocolCompat) -> Self {
        self.compat = compat;
        self
    }

    /// Turns on mid-call disambiguation, sealing the `requestState` its rounds
    /// carry under `key`.
    ///
    /// Separate from [`AppState::new`] because it is optional, and optional in
    /// a specific way: without a key this server never asks, which is the same
    /// behaviour on one instance and on twenty. The key itself must be the
    /// *same* on all of them — the retry that answers a question is a fresh
    /// request that a load balancer may send anywhere, and a key that differed
    /// per process would refuse answers it had just asked for.
    pub fn with_request_state_key(mut self, key: &[u8]) -> Self {
        // Cloned rather than moved out: `MunduTools` is a pair of handles, so
        // this is two atomic increments, and it keeps the builder on the tool
        // surface itself where the doc comment about the key belongs.
        self.tools = self.tools.clone().sealing_state_with(key);
        self
    }

    /// Names the hostnames this server answers to, replacing the loopback
    /// default.
    ///
    /// Needed by any deployment reached through a load balancer, where the
    /// `Host` is the public name rather than the address the socket is bound
    /// to. Separate from [`AppState::new`] so that widening the guard is always
    /// something a deployment did on purpose.
    pub fn with_allowed_hosts(mut self, allowed_hosts: Vec<String>) -> Self {
        self.allowed_hosts = allowed_hosts;
        self
    }
}

/// Where a request's [`Principal`] comes from.
///
/// One variant per transport, and no third one, because "somewhere else" is
/// exactly the hole this enum exists to close. A [`Principal`] reaches a tool
/// through one of these two paths or it does not reach one at all.
#[derive(Clone)]
enum Caller {
    /// HTTP: whatever the authenticating middleware attached to *this* request.
    /// There is no fallback — a request that arrived without one is refused.
    PerRequest,
    /// stdio: one principal for the life of the process, resolved once at
    /// startup from the configured authenticator.
    Fixed(Principal),
}

/// The tool surface, with the caller attached.
///
/// Exists because of a seam between two extension maps. The middleware puts a
/// [`Principal`] into the *HTTP* request's extensions; `rmcp` carries those
/// through as a nested [`http::request::Parts`] inside the MCP request context,
/// while the tool adapters' `Extension<Principal>` extractor reads the MCP
/// context's own extensions. Something has to lift the principal from one to
/// the other, and this is it.
///
/// Doing it here rather than in each adapter means there is one place to audit,
/// and it is the same place that decides what happens when there is no
/// principal to lift: the call is refused.
#[derive(Clone)]
pub struct AuthenticatedTools {
    tools: MunduTools,
    caller: Caller,
    /// Carried rather than read from a constant, because
    /// [`ServerHandler::supported_protocol_versions`] is what `rmcp` negotiates
    /// against and it has to give the same answer as the transport the handler
    /// is mounted behind.
    policy: TransportPolicy,
}

impl AuthenticatedTools {
    /// The tool surface as served over HTTP, taking its caller from each
    /// request.
    fn per_request(tools: MunduTools, policy: TransportPolicy) -> Self {
        Self {
            tools,
            caller: Caller::PerRequest,
            policy,
        }
    }

    /// The tool surface as served over stdio, where every call is the same
    /// caller because there is only ever one.
    ///
    /// Takes the compatibility setting too, so a developer driving this server
    /// from a local client negotiates the same revisions the deployment does.
    /// The trade the setting names does not arise here — stdio is one process
    /// and one connection, so there is no load balancer to be sticky behind —
    /// but a local run that spoke a different protocol than production would
    /// be a difference discovered at the worst possible time.
    pub fn for_principal(tools: MunduTools, principal: Principal, compat: ProtocolCompat) -> Self {
        Self {
            tools,
            caller: Caller::Fixed(principal),
            policy: TransportPolicy::for_compat(compat),
        }
    }

    /// Lifts the caller into the place the tool adapters read it from.
    ///
    /// Fails closed. An HTTP request with no principal in its parts is a
    /// request that did not pass the middleware, which should be impossible —
    /// the middleware is layered over the whole route — so it is treated as a
    /// server fault and refused rather than served with a guess.
    fn attach_caller(&self, context: &mut RequestContext<RoleServer>) -> Result<(), ErrorData> {
        let principal = match &self.caller {
            Caller::Fixed(principal) => principal.clone(),
            Caller::PerRequest => context
                .extensions
                .get::<http::request::Parts>()
                .and_then(|parts| parts.extensions.get::<Principal>())
                .cloned()
                .ok_or_else(|| {
                    tracing::error!(
                        "a tool call reached the handler with no principal attached; \
                         refusing it — this means the authenticating layer was bypassed"
                    );
                    ErrorData::internal_error(
                        "this memory could not identify the caller and will not act for one \
                         it cannot name",
                        None,
                    )
                })?,
        };

        context.extensions.insert(principal);
        Ok(())
    }
}

impl ServerHandler for AuthenticatedTools {
    fn get_info(&self) -> ServerInfo {
        let mut info = self.tools.get_info();
        info.protocol_version = ProtocolVersion::V_2026_07_28;
        info
    }

    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        Cow::Borrowed(self.policy.protocol_versions)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        mut context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        self.attach_caller(&mut context)?;
        self.tools.call_tool(request, context).await
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        // No principal: the tools and their schemas are the same for every
        // tenant, and listing them tells a caller nothing about anyone's data.
        self.tools.list_tools(request, context).await
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.tools.get_tool(name)
    }
}

/// The router: `/mcp`, behind the authenticating middleware.
///
/// Two shapes, one per [`ProtocolCompat`], differing only in which session
/// manager is mounted. That has to be decided here rather than by a flag,
/// because the session manager is a *type*: `NeverSessionManager` cannot mint
/// the session a legacy handshake asks for — it answers every call with
/// "session management is not supported" — so turning `legacy_session_mode` on
/// over it would turn a legacy `initialize` into a 500 rather than a session.
pub fn router(state: AppState) -> Router {
    let policy = TransportPolicy::for_compat(state.compat);
    let tools = state.tools.clone();

    // Runs per request (per session, under the legacy lifecycle). It captures
    // `tools`, a clone-cheap handle, and clones it again here — so the pool and
    // the embedder are shared while nothing per-request survives past the
    // response.
    let factory = move || Ok(AuthenticatedTools::per_request(tools.clone(), policy));

    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(policy.legacy_session_mode)
        // A modern request carries its own protocol version and client
        // identity, so a simple call is a simple request/response and does not
        // need a stream to answer it. Read only on the stateless path, so a
        // legacy request still gets the `text/event-stream` its era expects.
        .with_json_response(true)
        .with_stateless_protocol_metadata_required(policy.stateless_protocol_metadata_required)
        .with_allowed_hosts(state.allowed_hosts.clone());

    // Only one arm ever runs, so `factory` and `config` are moved once.
    let routed = match state.compat {
        ProtocolCompat::ModernOnly => Router::new().route_service(
            MCP_PATH,
            StreamableHttpService::new(factory, Arc::new(NeverSessionManager::default()), config),
        ),
        ProtocolCompat::DualEra => Router::new().route_service(
            MCP_PATH,
            StreamableHttpService::new(factory, Arc::new(LocalSessionManager::default()), config),
        ),
    };

    routed
        // `route_layer`, not `layer`: this runs only for requests that matched
        // a route above, so an unauthenticated probe of a path that does not
        // exist gets a 404 rather than a 401 that confirms nothing. Every route
        // the middleware must cover has to be registered before this line.
        .route_layer(axum::middleware::from_fn_with_state(
            state.auth.clone(),
            authenticate,
        ))
        // Outermost, so a request refused by the middleware is still counted.
        //
        // `new_for_http` records the method, path, status and latency — and
        // deliberately no headers, which is the only reason this is safe to
        // switch on: every request here carries an `Authorization` header, and
        // a layer that logged headers would write bearer tokens to disk on
        // every call. Do not replace this with a configuration that does.
        .layer(tower_http::trace::TraceLayer::new_for_http())
}

/// Resolves the caller and puts them where the tools look.
///
/// This is the only thing in the server that produces a [`Principal`]. A
/// request that does not get one here does not reach a tool: it is answered
/// with 401 from this function, and the tool adapters' `Extension<Principal>`
/// extractor fails the call if anything ever gets past.
async fn authenticate(
    State(auth): State<Arc<Authenticator>>,
    mut request: Request,
    next: Next,
) -> Response {
    let header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());

    match auth.authenticate(header) {
        Ok(principal) => {
            request.extensions_mut().insert(principal);
            next.run(request).await
        }
        Err(error) => unauthorized(&error),
    }
}

/// Why `--stdio` could not start.
#[derive(Debug, thiserror::Error)]
#[error(
    "refusing to serve over stdio under an authenticator that expects a credential: stdio has \
     no Authorization header and no handshake to carry one, so every call would have to be \
     attributed to a tenant nobody configured. Run with MUNDU_AUTH_MODE=disabled and \
     MUNDU_INSECURE_NO_AUTH=1 for local development, or drop --stdio and serve over HTTP"
)]
pub struct StdioNeedsACredential;

/// The one caller every stdio session acts as.
///
/// Resolved from the configured [`Authenticator`] rather than assumed, and
/// resolved by *asking* it — `authenticate(None)` is exactly the question
/// "who is this, given no credential?", and only `Disabled` has an answer.
/// Deriving it this way means a new authenticator variant cannot accidentally
/// acquire a stdio bypass: if it needs a credential, it refuses here for free.
///
/// The refusal is deliberate. A server told to check credentials, run over a
/// transport that cannot supply one, has no safe interpretation — inventing a
/// tenant writes somebody's memory under a name nobody chose, and quietly
/// downgrading to `disabled` would let a command-line flag undo the
/// deployment's authentication.
pub fn stdio_principal(auth: &Authenticator) -> Result<Principal, StdioNeedsACredential> {
    auth.authenticate(None).map_err(|_| StdioNeedsACredential)
}

/// A 401 that tells a client how to authenticate and nothing else.
///
/// The `WWW-Authenticate` header is what makes this actionable — a client that
/// gets a bare 401 cannot tell "wrong credential" from "wrong URL". The `error`
/// parameter distinguishes *no* credential from a bad one, per RFC 6750 §3.
///
/// Nothing about the credential itself is echoed, logged or reflected.
fn unauthorized(error: &AuthError) -> Response {
    let challenge = match error {
        AuthError::MissingCredentials => "Bearer",
        AuthError::MalformedCredentials => r#"Bearer error="invalid_request""#,
        AuthError::InvalidCredentials => r#"Bearer error="invalid_token""#,
    };

    tracing::debug!(reason = %error, "refused an unauthenticated request");

    (
        axum::http::StatusCode::UNAUTHORIZED,
        [(axum::http::header::WWW_AUTHENTICATE, challenge)],
        axum::Json(serde_json::json!({ "error": error.to_string() })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use jsonwebtoken::jwk::JwkSet;

    use super::*;
    use crate::auth::{JwtConfig, hash_token};

    /// The default is one revision, and it is the modern one. Everything the
    /// stateless design assumes — per-request provenance, no instance
    /// affinity — follows from there being nothing else in this list.
    #[test]
    fn modern_only_advertises_the_modern_revision_and_nothing_else() {
        let policy = TransportPolicy::for_compat(ProtocolCompat::ModernOnly);

        assert_eq!(
            policy.protocol_versions,
            &[ProtocolVersion::V_2026_07_28],
            "widening the default list is the one change this setting exists to avoid"
        );
    }

    /// The modern revision stays first — it is still what this server prefers
    /// and still what a capable client gets — with the legacy revisions behind
    /// it. `2025-11-25` is the one proven necessary: it is `rmcp`'s own
    /// `LATEST`, so every client built on this same SDK asks for it.
    #[test]
    fn dual_era_advertises_the_legacy_revisions_beside_the_modern_one() {
        let policy = TransportPolicy::for_compat(ProtocolCompat::DualEra);

        assert_eq!(
            policy.protocol_versions,
            &[
                ProtocolVersion::V_2026_07_28,
                ProtocolVersion::V_2025_11_25,
                ProtocolVersion::V_2025_06_18,
            ],
        );
        assert_eq!(
            policy.protocol_versions[0],
            ProtocolVersion::V_2026_07_28,
            "the modern revision has to stay the server's own preference"
        );
        assert!(
            !policy
                .protocol_versions
                .contains(&ProtocolVersion::V_2025_03_26),
            "2025-03-26 predates the MCP-Protocol-Version header, so it is the revision \
             rmcp guesses when a request declares nothing; serving it would make declaring \
             nothing the way in"
        );
    }

    /// The two flags that decide whether a legacy request is served at all, and
    /// what it costs. They move together and only under `dual-era`: sessions on
    /// means instance affinity, and the metadata requirement off means a
    /// request without per-request `_meta` is no longer refused at the
    /// transport.
    #[test]
    fn dual_era_turns_on_the_legacy_lifecycle_that_modern_only_turns_off() {
        let modern = TransportPolicy::for_compat(ProtocolCompat::ModernOnly);
        assert!(
            !modern.legacy_session_mode,
            "sessions are what the stateless deployment does not have"
        );
        assert!(
            modern.stateless_protocol_metadata_required,
            "a modern-only server refuses a request that carries no provenance"
        );

        let dual = TransportPolicy::for_compat(ProtocolCompat::DualEra);
        assert!(
            dual.legacy_session_mode,
            "a legacy client's identity exists only in the handshake, so it needs the \
             session that remembers it"
        );
        assert!(
            !dual.stateless_protocol_metadata_required,
            "a legacy request has no per-request _meta to require"
        );
    }

    /// stdio has no headers, so the dev tenant configured for `disabled` mode
    /// is the whole of the caller's identity. Resolving it once at startup is
    /// what lets the tool adapters keep reading a `Principal` they did not
    /// have to construct.
    #[test]
    fn stdio_under_disabled_auth_serves_the_configured_dev_tenant() {
        let principal = stdio_principal(&Authenticator::Disabled {
            tenant: "workbench".to_string(),
        })
        .expect("disabled auth needs no credential and should resolve");

        assert_eq!(principal.tenant, "workbench");
    }

    /// The refusal that matters. A static-token or JWT deployment expects a
    /// credential on every request, and stdio has no way to carry one — there
    /// is no header, and there is no handshake under `2026-07-28` that could
    /// stand in for it.
    ///
    /// The alternatives are all worse than not starting: inventing a tenant
    /// would write somebody's memory under a name nobody configured, and
    /// silently downgrading to `disabled` would turn a locked-down
    /// configuration into an open one because of a command-line flag.
    #[test]
    fn stdio_refuses_to_run_under_an_authenticator_that_needs_a_credential() {
        let static_token = Authenticator::StaticToken {
            tokens: HashMap::from([(
                hash_token("0123456789abcdef0123456789abcdef"),
                "acme".to_string(),
            )]),
        };
        let jwt = Authenticator::Jwt {
            config: JwtConfig {
                issuer: "https://issuer.test/".to_string(),
                audience: "https://mundu.test/mcp".to_string(),
                jwks_url: "https://issuer.test/jwks.json".to_string(),
                tenant_claim: "tenant".to_string(),
            },
            keys: JwkSet::default(),
        };

        for auth in [static_token, jwt] {
            let err = stdio_principal(&auth)
                .expect_err("stdio cannot present a credential and must not pretend to");

            assert!(
                err.to_string().contains("--stdio"),
                "the refusal should name the flag that caused it, got: {err}"
            );
        }
    }
}
