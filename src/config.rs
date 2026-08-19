use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use jsonwebtoken::jwk::JwkSet;

use crate::auth::{Authenticator, DEFAULT_TENANT_CLAIM, JwtConfig, hash_token};
use crate::error::ConfigError;
use crate::store::consolidate::ConsolidationPolicy;

#[derive(Debug, Clone)]
pub struct Config {
    pub bind: SocketAddr,
    pub database_url: String,
    /// Hostnames this server answers to, checked against the `Host` header.
    ///
    /// A guard against DNS rebinding, which is why it defaults to loopback: a
    /// page in a browser can be made to resolve a name it controls to
    /// `127.0.0.1` and talk to a local server, and the `Host` header is what
    /// gives that away. A deployment reached by its real name has to say so.
    pub allowed_hosts: Vec<String>,
    pub auth: Authenticator,
    /// The key that seals the `requestState` a mid-call disambiguation carries
    /// between its rounds, or `None` to not ask at all.
    ///
    /// **Deliberately not defaulted.** The client answers a question by
    /// repeating its `tools/call` with the state echoed back, and under
    /// `2026-07-28` nothing pins that retry to the instance that asked — so a
    /// key minted per process would open the state exactly as long as the
    /// deployment had one replica and start refusing valid answers the day it
    /// had two, which is the sticky-session failure this whole architecture
    /// exists to avoid. Unset therefore means *do not ask*: `upsert_node`
    /// resolves a contested name by minting a node and reporting the near
    /// misses, exactly as it did before MRTR, on one instance or on twenty.
    pub request_state_key: Option<String>,
    /// When and how often this instance consolidates duplicate entities.
    pub consolidation: ConsolidationSettings,
    /// Which protocol revisions this deployment will speak.
    ///
    /// **Defaults to [`ProtocolCompat::ModernOnly`], and that is the whole
    /// point of it being a setting.** Widening it is not free — see the type —
    /// so it is something a deployment does on purpose, having read what it
    /// costs, rather than something that happens because the field exists.
    pub protocol_compat: ProtocolCompat,
}

/// How far back in MCP revisions this deployment is willing to speak.
///
/// The narrow answer is the default and stays the default. Everything under
/// `2026-07-28` is a different lifecycle — a handshake instead of per-request
/// `_meta` — and serving it means keeping server-side session state, which is
/// exactly the thing this server was built without so that any instance can
/// answer any request.
///
/// The wide answer exists because the narrow one makes the server unreachable
/// by the clients that actually ship. Verified against the Devin CLI, which
/// speaks `2025-11-25` through the same `rmcp` 3.1.2 this server uses: it
/// handshakes, is negotiated up to a version it does not implement, and then
/// every call fails on a header it never sends. "Connected, and nothing works"
/// is a worse outcome than either a clean refusal or a served legacy client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProtocolCompat {
    /// `2026-07-28` only: no sessions, no legacy lifecycle, per-request
    /// provenance required on every call. What every deployment gets unless it
    /// says otherwise.
    #[default]
    ModernOnly,
    /// `2026-07-28` **and** the legacy revisions, each on its own lifecycle.
    ///
    /// Modern requests are still served statelessly and still carry their own
    /// `_meta`; legacy requests get a session, because a session is the only
    /// place a legacy client's identity exists. See
    /// [`crate::server::TransportPolicy`] for what that buys and what it costs.
    DualEra,
}

impl ProtocolCompat {
    /// The spelling a deployment writes in `MUNDU_PROTOCOL_COMPAT`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModernOnly => "modern-only",
            Self::DualEra => "dual-era",
        }
    }
}

/// The background consolidation worker's schedule.
///
/// Separate from [`ConsolidationPolicy`], which is the *schedule*, because
/// `enabled` and `tick` are properties of this process rather than of the
/// tenant: turning the worker off on one instance leaves the schedule intact
/// and lets every other instance keep it.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsolidationSettings {
    /// **On by default.** The deployment asked for automatic consolidation, and
    /// a duplicate-merging worker that has to be switched on is a worker that
    /// stays off. It is off-switchable because a merge is still a semi-
    /// destructive operation, and an operator watching one go wrong needs to be
    /// able to stop it in an environment variable rather than in a deploy.
    pub enabled: bool,
    pub policy: ConsolidationPolicy,
    /// How often the worker wakes to ask which tenants are due. Not the
    /// schedule — see [`ConsolidationPolicy`] — only how finely it can be
    /// observed.
    pub tick: Duration,
}

/// Five minutes: fine enough that a four-hour bound is honoured to within two
/// percent of itself, coarse enough that the cost of asking is nothing.
const DEFAULT_CONSOLIDATION_TICK: Duration = Duration::from_secs(300);

impl Default for ConsolidationSettings {
    fn default() -> Self {
        Self {
            // Off unless asked for. The worker merges entities automatically
            // and unsupervised, and an over-merge cannot be undone from the
            // data kept -- only from the audit trail, by someone who noticed.
            // A capability that rewrites the graph on a timer is not something
            // a deployment should acquire by saying nothing.
            enabled: false,
            policy: ConsolidationPolicy::default(),
            tick: DEFAULT_CONSOLIDATION_TICK,
        }
    }
}

/// Where the listener goes when the deployment does not say.
///
/// Loopback, not `0.0.0.0`. Binding to every interface by default would put a
/// freshly started server on the network before anybody decided it should be
/// there, and the mistake is silent — it looks exactly like a working server.
const DEFAULT_BIND: &str = "127.0.0.1:8080";

/// The `Host` values accepted when the deployment names none.
///
/// Loopback only, matching `rmcp`'s own default. A server that has not been
/// told which hostnames it answers to should not answer to all of them.
pub const DEFAULT_ALLOWED_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "::1"];

impl Config {
    /// Reads the whole configuration from the process environment.
    ///
    /// Does not validate: see [`Config::validate`], which `main` calls before
    /// binding. Keeping them apart means there is one place that decides
    /// whether a configuration is safe to serve, rather than one per loader.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// The body of [`Config::from_env`], over an arbitrary lookup.
    ///
    /// Takes a closure for the same reason [`authenticator_from`] does: tests
    /// exercise it without mutating the process environment, which is global
    /// state shared by every test running in parallel.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let bind = get("MUNDU_BIND").unwrap_or_else(|| DEFAULT_BIND.to_string());

        Ok(Self {
            bind: bind.parse().map_err(|source| ConfigError::Invalid {
                var: "MUNDU_BIND",
                reason: format!("`{bind}` is not a `host:port` address: {source}"),
            })?,
            // No default. A guessed database URL is a deployment that writes
            // its memory somewhere nobody is reading it from.
            database_url: get("MUNDU_DATABASE_URL")
                .ok_or(ConfigError::Missing("MUNDU_DATABASE_URL"))?,
            allowed_hosts: get("MUNDU_ALLOWED_HOSTS")
                .map(|raw| {
                    raw.split(',')
                        .map(str::trim)
                        .filter(|host| !host.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_else(|| DEFAULT_ALLOWED_HOSTS.map(str::to_string).to_vec()),
            request_state_key: request_state_key_from(&get)?,
            consolidation: consolidation_from(&get)?,
            protocol_compat: protocol_compat_from(&get)?,
            auth: authenticator_from(get)?,
        })
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if matches!(self.auth, Authenticator::Disabled { .. }) && !self.bind.ip().is_loopback() {
            return Err(ConfigError::InsecureBind {
                bind: self.bind.to_string(),
            });
        }
        if let Authenticator::Jwt { config, .. } = &self.auth {
            validate_jwks_url(&config.jwks_url)?;
        }
        validate_consolidation(&self.consolidation)?;
        Ok(())
    }
}

/// Reads which protocol era this deployment serves.
///
/// Absent means [`ProtocolCompat::ModernOnly`], because that is the decision
/// the rest of the design was made under. Anything else is refused rather than
/// read as the narrow answer: a deployment that meant to widen the server and
/// mistyped the value would otherwise get exactly the symptom it was trying to
/// fix, from a variable that looks set.
fn protocol_compat_from(
    get: impl Fn(&str) -> Option<String>,
) -> Result<ProtocolCompat, ConfigError> {
    match get("MUNDU_PROTOCOL_COMPAT").as_deref() {
        None => Ok(ProtocolCompat::default()),
        Some("modern-only") => Ok(ProtocolCompat::ModernOnly),
        Some("dual-era") => Ok(ProtocolCompat::DualEra),
        Some(other) => Err(ConfigError::Invalid {
            var: "MUNDU_PROTOCOL_COMPAT",
            reason: format!("expected `modern-only` or `dual-era`, got `{other}`"),
        }),
    }
}

/// Reads the consolidation schedule.
///
/// Every value has a default that a deployment which says nothing gets, because
/// the deployment asked for this to run automatically. What it may not do is
/// say something unparseable: a mistyped interval that silently fell back to
/// four hours would be a schedule nobody chose and nobody could see.
fn consolidation_from(
    get: impl Fn(&str) -> Option<String>,
) -> Result<ConsolidationSettings, ConfigError> {
    let defaults = ConsolidationSettings::default();

    let enabled = match get("MUNDU_CONSOLIDATION").as_deref() {
        None => defaults.enabled,
        Some("on" | "1" | "true") => true,
        Some("off" | "0" | "false") => false,
        Some(other) => {
            return Err(ConfigError::Invalid {
                var: "MUNDU_CONSOLIDATION",
                reason: format!("expected `on` or `off`, got `{other}`"),
            });
        }
    };

    let growth_percent = match get("MUNDU_CONSOLIDATION_GROWTH_PERCENT") {
        None => defaults.policy.growth_percent,
        Some(raw) => {
            let percent = raw.parse::<f64>().map_err(|source| ConfigError::Invalid {
                var: "MUNDU_CONSOLIDATION_GROWTH_PERCENT",
                reason: format!("`{raw}` is not a number: {source}"),
            })?;
            if !percent.is_finite() || percent < 0.0 {
                return Err(ConfigError::Invalid {
                    var: "MUNDU_CONSOLIDATION_GROWTH_PERCENT",
                    reason: format!("`{raw}` is not a percentage a graph can grow by"),
                });
            }
            percent
        }
    };

    let max_interval = positive_seconds(
        &get,
        "MUNDU_CONSOLIDATION_MAX_INTERVAL_SECS",
        defaults.policy.max_interval.to_std().unwrap_or_default(),
    )?;
    let tick = positive_seconds(
        &get,
        "MUNDU_CONSOLIDATION_TICK_SECS",
        DEFAULT_CONSOLIDATION_TICK,
    )?;

    Ok(ConsolidationSettings {
        enabled,
        policy: ConsolidationPolicy {
            growth_percent,
            max_interval: chrono::Duration::from_std(max_interval).map_err(|source| {
                ConfigError::Invalid {
                    var: "MUNDU_CONSOLIDATION_MAX_INTERVAL_SECS",
                    reason: format!("that many seconds is not an interval: {source}"),
                }
            })?,
        },
        tick,
    })
}

/// A duration in whole seconds, which must be at least one.
///
/// Zero is refused rather than treated as "as often as possible": as an
/// interval it would consolidate every tenant on every tick forever, and as a
/// tick it would spin.
fn positive_seconds(
    get: impl Fn(&str) -> Option<String>,
    var: &'static str,
    default: Duration,
) -> Result<Duration, ConfigError> {
    let Some(raw) = get(var) else {
        return Ok(default);
    };

    let seconds = raw.parse::<u64>().map_err(|source| ConfigError::Invalid {
        var,
        reason: format!("`{raw}` is not a whole number of seconds: {source}"),
    })?;

    if seconds == 0 {
        return Err(ConfigError::Invalid {
            var,
            reason: "zero seconds is not an interval; set MUNDU_CONSOLIDATION=off to turn \
                     the worker off instead"
                .to_string(),
        });
    }

    Ok(Duration::from_secs(seconds))
}

/// Refuses a schedule that cannot keep the promise it makes.
///
/// A worker that wakes less often than the bound it is enforcing does not
/// enforce it: the deployment would have configured "at least every four
/// hours" and got "at least every tick". That is a configuration whose failure
/// is invisible — the worker runs, the log looks healthy, and the bound is
/// quietly whatever the tick is — which is exactly the kind of thing this
/// server refuses to boot with.
fn validate_consolidation(settings: &ConsolidationSettings) -> Result<(), ConfigError> {
    if !settings.enabled {
        return Ok(());
    }

    let interval = settings
        .policy
        .max_interval
        .to_std()
        .unwrap_or(Duration::ZERO);

    if settings.tick > interval {
        return Err(ConfigError::Invalid {
            var: "MUNDU_CONSOLIDATION_TICK_SECS",
            reason: format!(
                "the worker wakes every {}s but is meant to consolidate every tenant at \
                 least every {}s; it cannot honour a bound it sleeps through",
                settings.tick.as_secs(),
                interval.as_secs()
            ),
        });
    }

    Ok(())
}

/// The shortest signing key this server will seal a `requestState` with.
///
/// The same bar as a static token, for the same reason: the key is the only
/// thing between a client and a `requestState` of its own making, and such a
/// state names the node a surface form binds to — and therefore the node every
/// edge later written against that name attaches to. A key short enough to
/// guess is not a key.
const MIN_REQUEST_STATE_KEY_LEN: usize = 32;

/// Reads the disambiguation signing key, refusing a guessable one.
///
/// Absent is a supported answer and means *do not ask*, which is documented on
/// [`Config::request_state_key`]. What is not supported is a key so short that
/// sealing the state proves nothing.
fn request_state_key_from(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Option<String>, ConfigError> {
    let Some(key) = get("MUNDU_REQUEST_STATE_KEY") else {
        return Ok(None);
    };

    if key.len() < MIN_REQUEST_STATE_KEY_LEN {
        return Err(ConfigError::Invalid {
            var: "MUNDU_REQUEST_STATE_KEY",
            reason: format!(
                "the key is {} bytes; minimum is {MIN_REQUEST_STATE_KEY_LEN}. It signs the \
                 disambiguation state a client echoes back, and a state a caller could forge \
                 names whichever node it likes",
                key.len()
            ),
        });
    }

    Ok(Some(key))
}

/// Refuses a JWKS endpoint that could be substituted in transit.
///
/// The key set is the entire root of trust: a caller who can replace it can
/// mint a token for any tenant and have it validate. Over plaintext `http` that
/// is one on-path attacker away, so the scheme is checked at startup and a
/// misconfigured deployment fails to boot rather than serving requests it
/// cannot vouch for.
///
/// Loopback is the one exception. There is no network segment to sit on, and an
/// attacker who is already running code on the host has strictly better options
/// than forging a JWKS. Refusing it would only push local development towards
/// disabling authentication altogether.
fn validate_jwks_url(raw: &str) -> Result<(), ConfigError> {
    let url = reqwest::Url::parse(raw).map_err(|source| ConfigError::Invalid {
        var: "MUNDU_JWKS_URL",
        reason: format!("`{raw}` is not a URL: {source}"),
    })?;

    match url.scheme() {
        "https" => Ok(()),
        "http" if is_loopback_host(&url) => Ok(()),
        "http" => Err(ConfigError::Invalid {
            var: "MUNDU_JWKS_URL",
            reason: format!(
                "refusing to fetch the JWKS from `{raw}` over plaintext http; the key set is \
                 this server's entire root of trust, so anyone on the network path could \
                 replace it and have every token they mint validate. Use https, or a loopback \
                 host for local development"
            ),
        }),
        other => Err(ConfigError::Invalid {
            var: "MUNDU_JWKS_URL",
            reason: format!("`{other}` is not a scheme the JWKS can be fetched over"),
        }),
    }
}

/// Whether a URL's host is unreachable from off the machine.
///
/// `localhost` is included by name because it is defined to resolve to a
/// loopback address (RFC 6761 §6.3) and is what a developer actually types.
fn is_loopback_host(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    // `host_str` brackets IPv6 literals, which `IpAddr` does not parse.
    let host = host.trim_start_matches('[').trim_end_matches(']');

    if let Ok(address) = host.parse::<std::net::IpAddr>() {
        return address.is_loopback();
    }
    host.eq_ignore_ascii_case("localhost")
        || host
            .rsplit_once('.')
            .is_some_and(|(_, last)| last.eq_ignore_ascii_case("localhost"))
}

/// Builds an [`Authenticator`] from configuration values.
///
/// Takes a lookup closure rather than reading the process environment directly,
/// so tests exercise it without mutating global state.
pub fn authenticator_from(
    get: impl Fn(&str) -> Option<String>,
) -> Result<Authenticator, ConfigError> {
    match get("MUNDU_AUTH_MODE").as_deref().unwrap_or("jwt") {
        "disabled" | "none" => {
            if get("MUNDU_INSECURE_NO_AUTH").as_deref() != Some("1") {
                return Err(ConfigError::UnacknowledgedInsecureAuth);
            }
            Ok(Authenticator::Disabled {
                tenant: get("MUNDU_DEV_TENANT").unwrap_or_else(|| "local".to_string()),
            })
        }
        "token" | "static" => {
            let raw =
                get("MUNDU_STATIC_TOKENS").ok_or(ConfigError::Missing("MUNDU_STATIC_TOKENS"))?;
            Ok(Authenticator::StaticToken {
                tokens: parse_static_tokens(&raw)?,
            })
        }
        "jwt" => Ok(Authenticator::Jwt {
            config: JwtConfig {
                issuer: get("MUNDU_JWT_ISSUER").ok_or(ConfigError::Missing("MUNDU_JWT_ISSUER"))?,
                audience: get("MUNDU_JWT_AUDIENCE")
                    .ok_or(ConfigError::Missing("MUNDU_JWT_AUDIENCE"))?,
                jwks_url: get("MUNDU_JWKS_URL").ok_or(ConfigError::Missing("MUNDU_JWKS_URL"))?,
                tenant_claim: get("MUNDU_TENANT_CLAIM")
                    .unwrap_or_else(|| DEFAULT_TENANT_CLAIM.to_string()),
            },
            // Empty until the JWKS is fetched at startup. Nothing validates
            // against an empty key set, so the window before that fetch fails
            // closed rather than open.
            keys: JwkSet::default(),
        }),
        other => Err(ConfigError::Invalid {
            var: "MUNDU_AUTH_MODE",
            reason: format!("unknown mode `{other}`"),
        }),
    }
}

/// Parses comma-separated `tenant:token` pairs into a digest-keyed lookup.
fn parse_static_tokens(raw: &str) -> Result<HashMap<[u8; 32], String>, ConfigError> {
    // A static token is the whole credential, so a guessable one is equivalent
    // to disabling authentication.
    const MIN_TOKEN_LEN: usize = 32;

    let mut tokens = HashMap::new();
    for entry in raw.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let (tenant, token) = entry.split_once(':').ok_or_else(|| ConfigError::Invalid {
            var: "MUNDU_STATIC_TOKENS",
            reason: "expected comma-separated `tenant:token` pairs".to_string(),
        })?;
        if token.len() < MIN_TOKEN_LEN {
            return Err(ConfigError::Invalid {
                var: "MUNDU_STATIC_TOKENS",
                reason: format!(
                    "token for tenant `{tenant}` is {} chars; minimum is {MIN_TOKEN_LEN}",
                    token.len()
                ),
            });
        }
        tokens.insert(hash_token(token), tenant.to_string());
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ISSUER: &str = "https://issuer.test/";
    const AUDIENCE: &str = "https://mundu.test/mcp";
    const JWKS_URL: &str = "https://issuer.test/jwks.json";

    /// The three variables `jwt` mode cannot start without.
    const REQUIRED_JWT_VARS: [(&str, &str); 3] = [
        ("MUNDU_JWT_ISSUER", ISSUER),
        ("MUNDU_JWT_AUDIENCE", AUDIENCE),
        ("MUNDU_JWKS_URL", JWKS_URL),
    ];

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key: &str| map.get(key).cloned()
    }

    #[test]
    fn disabled_auth_is_rejected_on_a_public_listener() {
        let config = Config {
            bind: "0.0.0.0:8080".parse().unwrap(),
            database_url: "postgres://mundu:mundu@localhost:55432/mundu".to_string(),
            allowed_hosts: DEFAULT_ALLOWED_HOSTS.map(str::to_string).to_vec(),
            auth: Authenticator::Disabled {
                tenant: "local".into(),
            },
            request_state_key: None,
            consolidation: ConsolidationSettings::default(),
            protocol_compat: ProtocolCompat::default(),
        };

        assert!(matches!(
            config.validate(),
            Err(ConfigError::InsecureBind { .. })
        ));
    }

    #[test]
    fn disabled_auth_requires_explicit_insecure_acknowledgement() {
        let err = authenticator_from(env_of(&[("MUNDU_AUTH_MODE", "disabled")]))
            .expect_err("disabling auth without acknowledgement should be refused");

        assert!(
            matches!(err, ConfigError::UnacknowledgedInsecureAuth),
            "expected UnacknowledgedInsecureAuth, got {err:?}"
        );
    }

    #[test]
    fn static_token_mode_builds_an_authenticator_that_accepts_its_token() {
        let auth = authenticator_from(env_of(&[
            ("MUNDU_AUTH_MODE", "token"),
            (
                "MUNDU_STATIC_TOKENS",
                "acme:0123456789abcdef0123456789abcdef",
            ),
        ]))
        .expect("token mode should parse");

        let principal = auth
            .authenticate(Some("Bearer 0123456789abcdef0123456789abcdef"))
            .expect("the configured token should authenticate");

        assert_eq!(principal.tenant, "acme");
    }

    #[test]
    fn jwt_mode_carries_its_configuration_and_defaults_the_tenant_claim() {
        // No MUNDU_AUTH_MODE: `jwt` is the default, so an otherwise complete
        // deployment must not have to name it.
        let auth = authenticator_from(env_of(&REQUIRED_JWT_VARS))
            .expect("jwt mode should build from a complete configuration");

        let Authenticator::Jwt { config, keys } = auth else {
            panic!("expected jwt mode, got {auth:?}");
        };
        assert_eq!(config.issuer, ISSUER);
        assert_eq!(config.audience, AUDIENCE);
        assert_eq!(config.jwks_url, JWKS_URL);
        assert_eq!(config.tenant_claim, "tenant");
        // The JWKS is fetched at startup, not here, so the authenticator starts
        // out trusting nothing.
        assert!(
            keys.keys.is_empty(),
            "authenticator_from must not have fetched any keys"
        );
    }

    /// A blank issuer, audience or JWKS URL is not a usable resource-server
    /// configuration; starting with one silently would leave the audience check
    /// comparing against an empty string.
    #[test]
    fn jwt_mode_refuses_to_start_without_issuer_audience_or_jwks_url() {
        for (omitted, _) in REQUIRED_JWT_VARS {
            let present: Vec<_> = REQUIRED_JWT_VARS
                .iter()
                .filter(|(var, _)| *var != omitted)
                .copied()
                .collect();

            let Err(err) = authenticator_from(env_of(&present)) else {
                panic!("{omitted} must be required");
            };

            assert!(
                matches!(err, ConfigError::Missing(var) if var == omitted),
                "expected Missing({omitted}), got {err:?}"
            );
        }
    }

    /// IdPs disagree on where the tenant lives — `org_id`, `tid`, a namespaced
    /// URI — so the claim name has to be deployment-configurable.
    #[test]
    fn the_tenant_claim_name_is_configurable() {
        let mut env: Vec<_> = REQUIRED_JWT_VARS.to_vec();
        env.push(("MUNDU_TENANT_CLAIM", "org_id"));

        let auth = authenticator_from(env_of(&env))
            .expect("jwt mode should build from a complete configuration");

        let Authenticator::Jwt { config, .. } = auth else {
            panic!("expected jwt mode, got {auth:?}");
        };
        assert_eq!(config.tenant_claim, "org_id");
    }

    const REQUEST_STATE_KEY: &str = "a-request-state-key-of-32-bytes!!";

    /// The smallest environment that should produce a usable server.
    const MINIMAL_ENV: [(&str, &str); 3] = [
        ("MUNDU_DATABASE_URL", "postgres://mundu:mundu@db/mundu"),
        ("MUNDU_AUTH_MODE", "token"),
        (
            "MUNDU_STATIC_TOKENS",
            "acme:0123456789abcdef0123456789abcdef",
        ),
    ];

    #[test]
    fn the_loader_reads_the_bind_address_the_database_url_and_the_authenticator() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push(("MUNDU_BIND", "0.0.0.0:9000"));

        let config = Config::from_lookup(env_of(&env)).expect("a complete environment should load");

        assert_eq!(config.bind, "0.0.0.0:9000".parse::<SocketAddr>().unwrap());
        assert_eq!(config.database_url, "postgres://mundu:mundu@db/mundu");
        assert_eq!(
            config
                .auth
                .authenticate(Some("Bearer 0123456789abcdef0123456789abcdef"))
                .expect("the configured token should authenticate")
                .tenant,
            "acme"
        );
    }

    /// An unset listener defaults to loopback. A server that guessed `0.0.0.0`
    /// would be reachable from the network the moment it started, before
    /// anybody decided it should be.
    #[test]
    fn an_unset_bind_address_defaults_to_loopback() {
        let config =
            Config::from_lookup(env_of(&MINIMAL_ENV)).expect("bind should not be required");

        assert!(
            config.bind.ip().is_loopback(),
            "expected a loopback default, got {}",
            config.bind
        );
    }

    /// There is no sensible default database: guessing one means a deployment
    /// that quietly writes to the wrong graph, or to a local one that is empty.
    #[test]
    fn the_database_url_is_required() {
        let without: Vec<_> = MINIMAL_ENV
            .iter()
            .filter(|(var, _)| *var != "MUNDU_DATABASE_URL")
            .copied()
            .collect();

        let err = Config::from_lookup(env_of(&without))
            .expect_err("a server with no database is not a server");

        assert!(
            matches!(err, ConfigError::Missing("MUNDU_DATABASE_URL")),
            "expected Missing(MUNDU_DATABASE_URL), got {err:?}"
        );
    }

    #[test]
    fn an_unparseable_bind_address_is_refused() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push(("MUNDU_BIND", "not-an-address"));

        let err = Config::from_lookup(env_of(&env))
            .expect_err("a bind address that cannot be parsed must not start the server");

        assert!(
            matches!(err, ConfigError::Invalid { var, .. } if var == "MUNDU_BIND"),
            "expected Invalid(MUNDU_BIND), got {err:?}"
        );
    }

    /// `rmcp` refuses any `Host` it does not recognise, defaulting to loopback
    /// only — a DNS-rebinding guard aimed at servers running on a laptop. This
    /// server is designed to sit behind a round-robin load balancer, where the
    /// `Host` is the deployment's public name, so the list has to be
    /// configurable or every production request is a 403.
    #[test]
    fn the_allowed_hosts_are_configurable_for_a_deployment_behind_a_load_balancer() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push((
            "MUNDU_ALLOWED_HOSTS",
            "mundu.example.com, mundu.internal:80",
        ));

        let config = Config::from_lookup(env_of(&env)).expect("a complete environment should load");

        assert_eq!(
            config.allowed_hosts,
            vec!["mundu.example.com", "mundu.internal:80"],
            "each entry should be trimmed and kept in order"
        );
    }

    /// The default stays narrow. A deployment that has not thought about which
    /// hostnames it answers to should not be answering to all of them.
    #[test]
    fn the_allowed_hosts_default_to_loopback_only() {
        let config = Config::from_lookup(env_of(&MINIMAL_ENV)).expect("this environment parses");

        assert_eq!(config.allowed_hosts, vec!["localhost", "127.0.0.1", "::1"]);
    }

    /// The key that seals a disambiguation's `requestState` comes from the
    /// deployment, because every instance that could serve the retry has to
    /// have the same one. A per-process key would work until the day somebody
    /// ran a second replica.
    #[test]
    fn the_request_state_signing_key_is_read_from_the_environment() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push(("MUNDU_REQUEST_STATE_KEY", REQUEST_STATE_KEY));

        let config = Config::from_lookup(env_of(&env)).expect("a complete environment should load");

        assert_eq!(config.request_state_key.as_deref(), Some(REQUEST_STATE_KEY));
    }

    /// Unset is a supported deployment, not a defaulted one: the server does
    /// not ask questions it cannot seal the answers to, and behaves as it did
    /// before it could ask. Inventing a key here is the one thing that must not
    /// happen — it would work perfectly on one instance and refuse every valid
    /// answer on two.
    #[test]
    fn an_unset_request_state_key_is_not_invented() {
        let config = Config::from_lookup(env_of(&MINIMAL_ENV)).expect("this environment parses");

        assert!(
            config.request_state_key.is_none(),
            "a signing key nobody configured must not be conjured up"
        );
    }

    /// The key is the only thing standing between a client and a `requestState`
    /// of its own choosing, which names the node every later edge binds to. A
    /// guessable one is no key at all.
    #[test]
    fn a_short_request_state_key_is_refused() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push(("MUNDU_REQUEST_STATE_KEY", "too-short"));

        let err = Config::from_lookup(env_of(&env))
            .expect_err("a guessable signing key must not start the server");

        assert!(
            matches!(err, ConfigError::Invalid { var, .. } if var == "MUNDU_REQUEST_STATE_KEY"),
            "expected Invalid(MUNDU_REQUEST_STATE_KEY), got {err:?}"
        );
    }

    /// The loader builds; it does not vet. `validate` is a separate call so
    /// there is exactly one place that decides whether a configuration is safe
    /// to serve, and `main` cannot forget it by loading through a shortcut.
    #[test]
    fn the_loader_does_not_validate_what_it_loads() {
        let env = [
            ("MUNDU_BIND", "0.0.0.0:9000"),
            ("MUNDU_DATABASE_URL", "postgres://mundu:mundu@db/mundu"),
            ("MUNDU_AUTH_MODE", "disabled"),
            ("MUNDU_INSECURE_NO_AUTH", "1"),
        ];

        let config = Config::from_lookup(env_of(&env)).expect("this environment parses");

        assert!(
            matches!(config.validate(), Err(ConfigError::InsecureBind { .. })),
            "a public listener with auth disabled must be caught by validate"
        );
    }

    fn jwt_config_with(jwks_url: &str) -> Config {
        Config {
            bind: "0.0.0.0:8080".parse().unwrap(),
            database_url: "postgres://mundu:mundu@localhost:55432/mundu".to_string(),
            allowed_hosts: DEFAULT_ALLOWED_HOSTS.map(str::to_string).to_vec(),
            auth: Authenticator::Jwt {
                config: JwtConfig {
                    issuer: ISSUER.to_string(),
                    audience: AUDIENCE.to_string(),
                    jwks_url: jwks_url.to_string(),
                    tenant_claim: DEFAULT_TENANT_CLAIM.to_string(),
                },
                keys: JwkSet::default(),
            },
            request_state_key: None,
            consolidation: ConsolidationSettings::default(),
            protocol_compat: ProtocolCompat::default(),
        }
    }

    /// The JWKS *is* the trust root. Fetched over plaintext, anyone on the path
    /// can substitute their own key set, and from that moment every token they
    /// mint validates. This has to fail at boot, not at the first request.
    #[test]
    fn a_plaintext_jwks_url_on_a_remote_host_is_refused() {
        for url in [
            "http://issuer.test/.well-known/jwks.json",
            "http://10.0.0.7:8080/jwks.json",
            "http://[2001:db8::1]/jwks.json",
        ] {
            let err = jwt_config_with(url)
                .validate()
                .expect_err("a plaintext JWKS endpoint must not be trusted: {url}");

            assert!(
                matches!(err, ConfigError::Invalid { var, .. } if var == "MUNDU_JWKS_URL"),
                "expected Invalid(MUNDU_JWKS_URL) for {url}, got {err:?}"
            );
        }
    }

    /// TLS is what makes the key set attributable to the issuer, so `https` is
    /// always fine — and a loopback `http` endpoint has no network path for an
    /// attacker to sit on, which is what makes local development workable.
    #[test]
    fn an_https_or_loopback_jwks_url_is_accepted() {
        for url in [
            "https://issuer.test/.well-known/jwks.json",
            "http://localhost:8080/jwks.json",
            "http://127.0.0.1:8080/jwks.json",
            "http://[::1]:8080/jwks.json",
        ] {
            assert!(
                jwt_config_with(url).validate().is_ok(),
                "{url} should be an acceptable JWKS endpoint"
            );
        }
    }

    /// A URL the fetcher cannot parse is not a URL it can be trusted to fetch
    /// from either, so it is refused rather than deferred to reqwest.
    #[test]
    fn an_unparseable_jwks_url_is_refused() {
        let err = jwt_config_with("not a url")
            .validate()
            .expect_err("an unparseable JWKS URL must not start the server");

        assert!(
            matches!(err, ConfigError::Invalid { var, .. } if var == "MUNDU_JWKS_URL"),
            "expected Invalid(MUNDU_JWKS_URL), got {err:?}"
        );
    }

    #[test]
    fn a_short_static_token_is_refused() {
        let err = authenticator_from(env_of(&[
            ("MUNDU_AUTH_MODE", "token"),
            ("MUNDU_STATIC_TOKENS", "acme:hunter2"),
        ]))
        .expect_err("a guessable token must not be accepted as a credential");

        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected Invalid, got {err:?}"
        );
    }

    /// Consolidation merges entities automatically and unsupervised, so it is
    /// off unless a deployment asks for it. The bounds it would run at are
    /// still the documented ones; only the switch is off.
    #[test]
    fn consolidation_is_off_by_default_at_ten_percent_and_four_hours() {
        let config = Config::from_lookup(env_of(&MINIMAL_ENV)).expect("this environment parses");

        assert!(
            !config.consolidation.enabled,
            "merging entities unsupervised is opt-in"
        );
        assert!((config.consolidation.policy.growth_percent - 10.0).abs() < f64::EPSILON);
        assert_eq!(
            config.consolidation.policy.max_interval,
            chrono::Duration::hours(4)
        );
        assert_eq!(config.consolidation.tick, Duration::from_secs(300));
        config.validate().expect("the defaults must be servable");
    }

    #[test]
    fn both_consolidation_bounds_and_the_tick_are_configurable() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push(("MUNDU_CONSOLIDATION_GROWTH_PERCENT", "2.5"));
        env.push(("MUNDU_CONSOLIDATION_MAX_INTERVAL_SECS", "1800"));
        env.push(("MUNDU_CONSOLIDATION_TICK_SECS", "30"));

        let config = Config::from_lookup(env_of(&env)).expect("this environment parses");

        assert!((config.consolidation.policy.growth_percent - 2.5).abs() < f64::EPSILON);
        assert_eq!(
            config.consolidation.policy.max_interval,
            chrono::Duration::seconds(1800)
        );
        assert_eq!(config.consolidation.tick, Duration::from_secs(30));
        config
            .validate()
            .expect("a tighter schedule is still servable");
    }

    /// A merge is reversible and audited, but it is still semi-destructive. An
    /// operator watching one go wrong has to be able to stop it in an
    /// environment variable rather than in a deploy.
    #[test]
    fn consolidation_can_be_switched_off() {
        for value in ["off", "0", "false"] {
            let mut env: Vec<_> = MINIMAL_ENV.to_vec();
            env.push(("MUNDU_CONSOLIDATION", value));

            let config = Config::from_lookup(env_of(&env)).expect("this environment parses");
            assert!(!config.consolidation.enabled, "{value} should turn it off");
        }

        for value in ["on", "1", "true"] {
            let mut env: Vec<_> = MINIMAL_ENV.to_vec();
            env.push(("MUNDU_CONSOLIDATION", value));

            let config = Config::from_lookup(env_of(&env)).expect("this environment parses");
            assert!(config.consolidation.enabled, "{value} should turn it on");
        }
    }

    /// A mistyped switch must not silently mean "on". Half the point of the
    /// switch is being able to turn the worker off in a hurry, and a typo that
    /// left it running would be discovered by whatever it merged next.
    #[test]
    fn an_unrecognised_consolidation_switch_is_refused() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push(("MUNDU_CONSOLIDATION", "disabled"));

        let err = Config::from_lookup(env_of(&env))
            .expect_err("`disabled` is not one of the two answers");

        assert!(
            matches!(err, ConfigError::Invalid { var, .. } if var == "MUNDU_CONSOLIDATION"),
            "expected Invalid(MUNDU_CONSOLIDATION), got {err:?}"
        );
    }

    #[test]
    fn an_unusable_consolidation_bound_is_refused_rather_than_defaulted() {
        for (var, value) in [
            ("MUNDU_CONSOLIDATION_GROWTH_PERCENT", "quite a lot"),
            ("MUNDU_CONSOLIDATION_GROWTH_PERCENT", "-5"),
            ("MUNDU_CONSOLIDATION_MAX_INTERVAL_SECS", "4h"),
            ("MUNDU_CONSOLIDATION_MAX_INTERVAL_SECS", "0"),
            ("MUNDU_CONSOLIDATION_TICK_SECS", "0"),
        ] {
            let mut env: Vec<_> = MINIMAL_ENV.to_vec();
            env.push((var, value));

            let err = Config::from_lookup(env_of(&env))
                .expect_err("a bound that cannot be parsed must not load");

            assert!(
                matches!(&err, ConfigError::Invalid { var: named, .. } if *named == var),
                "expected Invalid({var}), got {err:?}"
            );
        }
    }

    /// The failure this refusal exists for is invisible: the worker runs, the
    /// log looks healthy, and the four-hour bound the deployment configured is
    /// quietly whatever the tick is.
    #[test]
    fn a_worker_that_sleeps_through_its_own_bound_is_refused_at_startup() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        // Explicitly on: the rule under test is about a worker that runs, and
        // consolidation is opt-in, so the default would skip the check.
        env.push(("MUNDU_CONSOLIDATION", "on"));
        env.push(("MUNDU_CONSOLIDATION_MAX_INTERVAL_SECS", "60"));
        env.push(("MUNDU_CONSOLIDATION_TICK_SECS", "600"));

        let config = Config::from_lookup(env_of(&env)).expect("the loader does not validate");
        let err = config
            .validate()
            .expect_err("a tick longer than the bound cannot honour it");

        assert!(
            matches!(err, ConfigError::Invalid { var, .. } if var == "MUNDU_CONSOLIDATION_TICK_SECS"),
            "expected Invalid(MUNDU_CONSOLIDATION_TICK_SECS), got {err:?}"
        );
    }

    /// Modern-only is the deployment nobody asked for anything. Serving a
    /// legacy client is a decision with a cost — a session, sticky routing —
    /// and a default that quietly took it would undo the reasoning the
    /// stateless design rests on.
    #[test]
    fn the_protocol_compatibility_defaults_to_modern_only() {
        let config = Config::from_lookup(env_of(&MINIMAL_ENV)).expect("this environment parses");

        assert_eq!(config.protocol_compat, ProtocolCompat::ModernOnly);
    }

    /// The escape hatch. A deployment whose clients are the ones that exist
    /// rather than the ones the spec describes has to be able to say so.
    #[test]
    fn the_protocol_compatibility_can_be_widened_to_dual_era() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push(("MUNDU_PROTOCOL_COMPAT", "dual-era"));

        let config = Config::from_lookup(env_of(&env)).expect("this environment parses");

        assert_eq!(config.protocol_compat, ProtocolCompat::DualEra);
    }

    /// A typo must not silently mean the narrow answer either. An operator who
    /// wrote `dual_era` and got a server that still refuses every client they
    /// own would look everywhere except at the underscore.
    #[test]
    fn an_unrecognised_protocol_compatibility_is_refused() {
        for value in ["dual_era", "legacy", "on", ""] {
            let mut env: Vec<_> = MINIMAL_ENV.to_vec();
            env.push(("MUNDU_PROTOCOL_COMPAT", value));

            let err = Config::from_lookup(env_of(&env))
                .expect_err("a value that is neither `modern-only` nor `dual-era` must not load");

            assert!(
                matches!(err, ConfigError::Invalid { var, .. } if var == "MUNDU_PROTOCOL_COMPAT"),
                "expected Invalid(MUNDU_PROTOCOL_COMPAT) for `{value}`, got {err:?}"
            );
        }
    }

    /// A worker that is off has no bound to sleep through, so the schedule
    /// around it is not held to one. An operator turning consolidation off must
    /// not also have to fix a tick that has stopped mattering.
    #[test]
    fn a_switched_off_worker_is_not_held_to_a_schedule_it_will_not_run() {
        let mut env: Vec<_> = MINIMAL_ENV.to_vec();
        env.push(("MUNDU_CONSOLIDATION", "off"));
        env.push(("MUNDU_CONSOLIDATION_MAX_INTERVAL_SECS", "60"));
        env.push(("MUNDU_CONSOLIDATION_TICK_SECS", "600"));

        Config::from_lookup(env_of(&env))
            .expect("this environment parses")
            .validate()
            .expect("an off worker cannot fail to honour a bound");
    }
}
