//! The `mundu` binary: configuration, startup checks, and one of two
//! transports.
//!
//! Everything that can be wrong with a deployment is checked here, before the
//! listener binds. A server that starts and then refuses every request is
//! harder to diagnose than one that never started and said why.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use mundu::auth::Authenticator;
use mundu::config::{Config, ProtocolCompat};
#[cfg(feature = "candle-embedder")]
use mundu::embed::CandleEmbedder;
#[cfg(not(feature = "candle-embedder"))]
use mundu::embed::DeterministicEmbedder;
use mundu::embed::Embedder;
use mundu::mcp::MunduTools;
use mundu::server::{AppState, AuthenticatedTools, stdio_principal};
use mundu::store::Store;
use mundu::store::consolidate::ConsolidationWorker;

/// How long to wait for the IdP's key set before giving up at startup.
///
/// Short on purpose. An unreachable IdP is a failed boot, and a failed boot
/// that takes ninety seconds looks like a hang to whatever is supervising it.
const JWKS_FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let stdio = std::env::args().any(|argument| argument == "--stdio");

    let mut config = Config::from_env().context("could not read the configuration")?;
    // Before anything binds, connects or fetches. Every check in here is one
    // whose failure mode is a server that runs but should not.
    config
        .validate()
        .context("the configuration is unsafe to serve")?;

    // The JWKS is fetched once, here, rather than lazily on the first request.
    // `authenticator_from` leaves the key set empty and an empty key set
    // validates nothing, so the alternative is a server that boots healthy and
    // rejects every token until somebody notices.
    fetch_jwks_if_needed(&mut config.auth).await?;

    let store = Store::connect(&config.database_url)
        .await
        .context("could not connect to the database")?;
    store
        .migrate()
        .await
        .context("could not apply migrations")?;

    let embedder = embedder().await?;
    tracing::info!(
        model = embedder.model_id(),
        dimensions = embedder.dimensions(),
        "embedder ready"
    );

    start_consolidation(&store, &config);

    announce_protocol_compat(config.protocol_compat, config.bind);

    if stdio {
        serve_stdio(
            store,
            embedder,
            &config.auth,
            config.request_state_key.as_deref(),
            config.protocol_compat,
        )
        .await
    } else {
        serve_http(store, embedder, config).await
    }
}

/// Says which protocol era this process serves, and what that costs.
///
/// Loud on both settings, because both are decisions. The narrow one turns
/// clients away that a reader may be about to go looking for; the wide one
/// reintroduces the server-side state the rest of this architecture is built
/// around not having, and the symptom when that bites is a `404` from a
/// sibling instance rather than anything that names sessions.
///
/// Not a `Config::validate` refusal, and that was weighed. The hazard is real —
/// a legacy session lives in *this* process and nowhere else — but the bind
/// address is a poor witness for it: every container binds `0.0.0.0`, including
/// the single-replica deployment that is the main reason to want `dual-era` at
/// all, while the deployment that actually breaks is two replicas that may both
/// be on loopback behind a local proxy. A refusal keyed on that signal would
/// block the safe case and miss the unsafe one. So it warns, and names the
/// exact string an operator will see when it goes wrong.
fn announce_protocol_compat(compat: ProtocolCompat, bind: std::net::SocketAddr) {
    match compat {
        ProtocolCompat::ModernOnly => tracing::info!(
            protocol_compat = compat.as_str(),
            "speaking MCP 2026-07-28 only: no sessions, and every request must carry its own \
             protocol version and client identity in _meta. Clients negotiated below \
             2026-07-28 — which today is most of them, including anything built on rmcp's \
             own default — will handshake and then fail every call. Set \
             MUNDU_PROTOCOL_COMPAT=dual-era to serve them, and read what it costs first"
        ),
        ProtocolCompat::DualEra => {
            tracing::warn!(
                protocol_compat = compat.as_str(),
                "serving legacy MCP clients (2025-11-25, 2025-06-18) beside 2026-07-28. This \
                 turns server-side sessions back on for legacy requests only: a legacy client \
                 handshakes once and every later call of its must reach THIS process. Behind a \
                 load balancer without sticky routing it will get `404 Not Found: Session not \
                 found` from the next instance. Modern clients are unaffected and are still \
                 served statelessly"
            );
            tracing::warn!(
                protocol_compat = compat.as_str(),
                "legacy requests are no longer refused for carrying no per-request _meta, so \
                 the transport can no longer prove a caller declared itself. Writer provenance \
                 for a legacy client comes from its initialize handshake instead — which is \
                 why the session above is on rather than off; sessionless, rmcp would report \
                 every legacy writer as the rmcp version this binary was built against"
            );
            if !bind.ip().is_loopback() {
                tracing::warn!(
                    %bind,
                    "dual-era on a non-loopback listener: if more than one instance answers on \
                     this address, configure the load balancer to pin a client to an instance \
                     for the life of its Mcp-Session-Id, or legacy clients will break on their \
                     second request"
                );
            }
        }
    }
}

/// Starts the background worker that merges duplicate entities.
///
/// **Started on every instance, and that is deliberate.** The server is
/// stateless behind a load balancer with nothing shared but the database, so
/// there is nowhere else to put it: an instance nominated to run it would be an
/// instance the others depend on, which is the thing this architecture does not
/// have. Concurrency is settled in Postgres instead — each tenant is
/// consolidated under an advisory lock, and the instances that lose it skip.
///
/// It is started here rather than inside `serve_http`, because it is not a
/// property of the transport: a `--stdio` process pointed at a shared database
/// is as capable of consolidating it as an HTTP one, and a duplicate merged
/// there is merged for everybody.
///
/// The task is spawned and its handle dropped. It has no shutdown of its own
/// and needs none: everything it does is committed a merge at a time, so a
/// process killed mid-run leaves a consistent graph, an audit trail for every
/// merge that completed, and — because the advisory lock is session-scoped —
/// no lock behind it.
fn start_consolidation(store: &Store, config: &Config) {
    if !config.consolidation.enabled {
        tracing::info!(
            "consolidation is switched off; duplicate entities will accumulate until \
             something merges them"
        );
        return;
    }

    ConsolidationWorker::new(
        store.clone(),
        config.consolidation.policy.clone(),
        config.consolidation.tick,
    )
    .spawn();
}

/// The MCP endpoint over HTTP, which is how this server is meant to run.
async fn serve_http(store: Store, embedder: Arc<dyn Embedder>, config: Config) -> Result<()> {
    let mut state = AppState::new(store, embedder, Arc::new(config.auth))
        .with_allowed_hosts(config.allowed_hosts.clone())
        .with_protocol_compat(config.protocol_compat);

    // Asking a client which entity an ambiguous name meant needs somewhere to
    // keep the progress between the two rounds, and the only place that
    // survives a retry landing on another instance is a sealed blob in the
    // client's hands. No key, no asking — said out loud, because the failure it
    // prevents is invisible: a key invented per process opens its own states
    // perfectly and refuses every one from a sibling.
    match config.request_state_key.as_deref() {
        Some(key) => state = state.with_request_state_key(key.as_bytes()),
        None => tracing::warn!(
            "MUNDU_REQUEST_STATE_KEY is not set, so upsert_node will not ask a client which \
             entity an ambiguous name meant; it will mint a node and report the near misses \
             instead. Set the same key on every instance to turn asking on"
        ),
    }

    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .with_context(|| format!("could not bind {}", config.bind))?;
    let bound = listener.local_addr().unwrap_or(config.bind).to_string();

    tracing::info!(
        bind = %bound,
        path = mundu::server::MCP_PATH,
        allowed_hosts = ?config.allowed_hosts,
        "serving MCP over HTTP"
    );

    axum::serve(listener, mundu::server::router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("the HTTP server stopped unexpectedly")
}

/// The MCP endpoint over stdio, for pointing a local client at a development
/// database.
///
/// There is no credential on this path and no way to supply one, so the caller
/// is resolved once from the configured authenticator — which refuses unless
/// that authenticator is one that needs no credential.
async fn serve_stdio(
    store: Store,
    embedder: Arc<dyn Embedder>,
    auth: &Authenticator,
    request_state_key: Option<&str>,
    compat: ProtocolCompat,
) -> Result<()> {
    use rmcp::ServiceExt;

    let principal = stdio_principal(auth)?;
    tracing::info!(tenant = %principal.tenant, "serving MCP over stdio");

    // The same rule as over HTTP, and on purpose. A single process could seal
    // its states with a key it invented, but then a deployment would ask
    // questions locally and silently stop asking them in production, which is
    // the kind of difference nobody discovers until it matters.
    let mut tools = MunduTools::new(store, embedder);
    if let Some(key) = request_state_key {
        tools = tools.sealing_state_with(key.as_bytes());
    }

    let service = AuthenticatedTools::for_principal(tools, principal, compat)
        .serve(rmcp::transport::stdio())
        .await
        .context("could not start the stdio transport")?;

    service
        .waiting()
        .await
        .context("the stdio session failed")?;
    Ok(())
}

/// Replaces the empty key set `authenticator_from` leaves behind with the one
/// the IdP is publishing.
///
/// A failure here stops the boot. The server would technically run without the
/// keys — and reject every token, since nothing validates against an empty set
/// — but "up and refusing everyone" is a far more expensive way to discover a
/// typo in `MUNDU_JWKS_URL` than "did not start".
///
/// `Config::validate` has already refused a plaintext JWKS URL by this point,
/// so the fetch below cannot be reading its trust root off an unprotected
/// connection.
async fn fetch_jwks_if_needed(auth: &mut Authenticator) -> Result<()> {
    let Authenticator::Jwt { config, keys } = auth else {
        return Ok(());
    };

    tracing::info!(jwks_url = %config.jwks_url, "fetching the issuer's key set");

    let fetched: jsonwebtoken::jwk::JwkSet = reqwest::Client::builder()
        .timeout(JWKS_FETCH_TIMEOUT)
        .build()
        .context("could not build an HTTP client for the JWKS fetch")?
        .get(&config.jwks_url)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .with_context(|| format!("could not fetch the JWKS from {}", config.jwks_url))?
        .json()
        .await
        .with_context(|| format!("{} did not return a JWKS", config.jwks_url))?;

    // An empty key set validates nothing, so booting with one is booting a
    // server that will reject every caller. Say so now.
    if fetched.keys.is_empty() {
        bail!(
            "{} published an empty key set; no token could ever validate against it",
            config.jwks_url
        );
    }

    tracing::info!(keys = fetched.keys.len(), "key set loaded");
    *keys = fetched;
    Ok(())
}

/// The embedder this build was compiled with.
///
/// One function, so swapping the deterministic embedder for the real one is a
/// change to this seam and nothing else.
///
/// With `candle-embedder` on — the default — this is the real model, and a
/// failure to load it stops the boot. **It must never fall back to the
/// deterministic embedder.** That fake produces hash-derived vectors with no
/// semantic content whatsoever, so a server running on it answers every search
/// with confident, well-formed nonsense and writes vectors that are worthless
/// to every later query. A boot that fails on a bad `HF_HOME` costs minutes; a
/// boot that quietly succeeds on the fake costs however long it takes somebody
/// to notice that recall is indistinguishable from random, plus a re-embedding
/// of everything written in between.
///
/// Without the feature the fake is all there is, so the log line above and the
/// warning below both name it.
async fn embedder() -> Result<Arc<dyn Embedder>> {
    #[cfg(feature = "candle-embedder")]
    {
        let embedder = CandleEmbedder::load().await.context(
            "could not load the embedding model; refusing to start rather than fall back to \
             the synthetic embedder, which would return plausible-looking nonsense",
        )?;
        tracing::info!(device = ?embedder.device(), "embedding locally with candle");
        Ok(Arc::new(embedder))
    }

    #[cfg(not(feature = "candle-embedder"))]
    {
        tracing::warn!(
            "built without the `candle-embedder` feature: embeddings are hash-derived and \
             carry no meaning, so every search result is arbitrary"
        );
        Ok(Arc::new(DeterministicEmbedder::new()))
    }
}

/// Structured logs, filtered by `RUST_LOG`, written to **stderr**.
///
/// stderr is not a style choice. Under `--stdio`, stdout *is* the protocol:
/// every byte written there is read by the client as JSON-RPC, so a single log
/// line on stdout is a parse error in the middle of a session. It is also the
/// most easily missed bug of its kind, because a server that logs to stdout
/// looks perfect when a human drives it by hand and fails only against a real
/// client. One writer for both transports means the HTTP path cannot drift
/// into the habit either.
///
/// **Nothing here may log a credential.** No token, no `Authorization` header
/// value, and no `proof_excerpt`: proof text is other tenants' stored data and
/// belongs in the database, not in an operator's log aggregator. The code that
/// handles all three logs around them rather than through them — the auth
/// middleware records why a request was refused, never what it presented.
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};

    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}

/// Resolves on SIGTERM or SIGINT so in-flight requests finish.
///
/// SIGTERM is the one that matters: it is what a container runtime sends
/// before it kills the process, and answering it is the difference between
/// draining cleanly during a rolling deploy and dropping whatever was in
/// flight. SIGINT is for a developer's Ctrl-C.
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut terminate = signal(SignalKind::terminate()).expect("a process can listen for SIGTERM");
    let mut interrupt = signal(SignalKind::interrupt()).expect("a process can listen for SIGINT");

    let signal = tokio::select! {
        _ = terminate.recv() => "SIGTERM",
        _ = interrupt.recv() => "SIGINT",
    };

    tracing::info!(%signal, "shutting down; finishing in-flight requests");
}
