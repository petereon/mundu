use std::collections::HashMap;

use jsonwebtoken::jwk::JwkSet;
use jsonwebtoken::{Algorithm, DecodingKey, Validation};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::AuthError;

/// Claim carrying the tenant when the deployment does not name one explicitly.
pub const DEFAULT_TENANT_CLAIM: &str = "tenant";

/// Everything needed to validate a bearer token from an external IdP, minus the
/// keys themselves.
#[derive(Debug, Clone)]
pub struct JwtConfig {
    pub issuer: String,
    /// The identifier this server is known by at the IdP. A token whose `aud`
    /// is anything else was minted for a different resource server and must not
    /// be honoured here.
    pub audience: String,
    pub jwks_url: String,
    pub tenant_claim: String,
}

/// The authenticated caller behind a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    pub tenant: String,
    pub subject: Option<String>,
}

/// Deriving `Debug` is safe here: only SHA-256 digests of tokens are retained,
/// never the tokens themselves, and the JWKS holds public keys only, so a
/// debug-formatted config cannot leak a credential into a log line.
#[derive(Debug, Clone)]
pub enum Authenticator {
    Disabled { tenant: String },
    StaticToken { tokens: HashMap<[u8; 32], String> },
    Jwt { config: JwtConfig, keys: JwkSet },
}

/// Hashes a bearer token for constant-time-safe lookup.
pub fn hash_token(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

impl Authenticator {
    /// Resolves the caller from a raw `Authorization` header value.
    pub fn authenticate(&self, header: Option<&str>) -> Result<Principal, AuthError> {
        match self {
            // Local-only mode: the listener is pinned to loopback by
            // `Config::validate`, so there is no credential to check.
            Self::Disabled { tenant } => Ok(Principal {
                tenant: tenant.clone(),
                subject: None,
            }),
            Self::StaticToken { tokens } => {
                let tenant = tokens
                    .get(&hash_token(bearer_token(header)?))
                    .ok_or(AuthError::InvalidCredentials)?;
                Ok(Principal {
                    tenant: tenant.clone(),
                    subject: None,
                })
            }
            Self::Jwt { config, keys } => validate_jwt(bearer_token(header)?, config, keys),
        }
    }
}

/// Extracts the credential from an `Authorization` header value.
///
/// `strip_prefix` rather than `trim_start_matches`: the latter strips a prefix
/// repeatedly, so `Bearer Bearer <token>` would collapse to a valid credential.
fn bearer_token(header: Option<&str>) -> Result<&str, AuthError> {
    header
        .ok_or(AuthError::MissingCredentials)?
        .strip_prefix("Bearer ")
        .ok_or(AuthError::MalformedCredentials)
}

/// Only `sub` is named: the tenant claim is deployment-configurable, so it can
/// only be looked up by name at runtime.
#[derive(Deserialize)]
struct JwtClaims {
    sub: Option<String>,
    #[serde(flatten)]
    other: serde_json::Map<String, serde_json::Value>,
}

/// Validates an RS256 bearer token against an already-fetched key set.
///
/// Takes the `JwkSet` by reference rather than fetching it, so the trust
/// decision is a pure function of its inputs and stays testable offline.
fn validate_jwt(token: &str, config: &JwtConfig, keys: &JwkSet) -> Result<Principal, AuthError> {
    // A `kid` is required rather than trying every published key in turn, which
    // would let an unauthenticated caller spend one RSA verification per key in
    // the set on every request.
    let kid = jsonwebtoken::decode_header(token)
        .map_err(|_| AuthError::MalformedCredentials)?
        .kid
        .ok_or(AuthError::InvalidCredentials)?;
    let jwk = keys.find(&kid).ok_or(AuthError::InvalidCredentials)?;
    let key = DecodingKey::from_jwk(jwk).map_err(|_| AuthError::InvalidCredentials)?;

    // Pinned to RS256 rather than derived from the token header, which is
    // attacker-controlled: `alg: none` or an HS256 token keyed on the public
    // JWKS material would otherwise verify.
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_audience(&[&config.audience]);
    validation.set_issuer(&[&config.issuer]);
    // Required, not merely checked: `jsonwebtoken` skips a comparison it has no
    // claim for, so an omitted `aud` or `iss` would otherwise sail past the
    // calls above.
    validation.set_required_spec_claims(&["exp", "aud", "iss"]);
    // `nbf` is not validated, so the default 60s leeway can only ever extend the
    // life of an already-expired token. A resource server should not honour a
    // credential the IdP has already retired.
    validation.leeway = 0;

    let claims = jsonwebtoken::decode::<JwtClaims>(token, &key, &validation)
        .map_err(|_| AuthError::InvalidCredentials)?
        .claims;

    // Absent tenant claim is a rejection, never a fallback: defaulting would
    // silently file one IdP's users under some other tenant's memory.
    let tenant = claims
        .other
        .get(&config.tenant_claim)
        .and_then(serde_json::Value::as_str)
        .ok_or(AuthError::InvalidCredentials)?;

    // A blank claim is rejected for a different reason than a missing one, and
    // a sharper one. The empty tenant is not "unspecified" in this server, it
    // is the scope holding the curated relationship vocabulary that *every*
    // tenant reads. An IdP mapping that yields `tenant: ""` — a user with the
    // attribute unset, a claim template with a missing substitution — would
    // therefore be write access to shared state rather than an isolated
    // mistake. `StoreError::BlankTenant` refuses the write; this refuses the
    // credential, so no such `Principal` is ever constructed.
    //
    // Trimmed before the check: a tenant of `" "` is not the curated scope, but
    // it is nobody's real tenant either, and treating whitespace as a name
    // would give one typo its own private graph.
    if tenant.trim().is_empty() {
        return Err(AuthError::InvalidCredentials);
    }

    Ok(Principal {
        tenant: tenant.to_string(),
        subject: claims.sub,
    })
}

#[cfg(test)]
mod tests {
    use aws_lc_rs::encoding::{AsDer, Pkcs8V1Der};
    use jsonwebtoken::jwk::{Jwk, JwkSet};
    use jsonwebtoken::{Algorithm, EncodingKey, Header};
    use serde_json::json;

    use super::*;

    const ACME_TOKEN: &str = "0123456789abcdef0123456789abcdef";

    const ISSUER: &str = "https://issuer.test/";
    const AUDIENCE: &str = "https://mundu.test/mcp";
    const KEY_ID: &str = "test-signing-key";

    /// A throwaway RSA keypair minted for a single test, exposed both as the
    /// signing key and as the public JWK an IdP would publish. Generated at
    /// runtime so no private key material is ever committed.
    struct TestKey {
        encoding: EncodingKey,
        jwk: Jwk,
    }

    fn generate_key(kid: &str) -> TestKey {
        let pair = aws_lc_rs::rsa::KeyPair::generate(aws_lc_rs::rsa::KeySize::Rsa2048)
            .expect("RSA keygen should succeed");
        let der: Pkcs8V1Der<'static> = pair.as_der().expect("key should encode as PKCS#8 DER");
        // Via PEM because `EncodingKey::from_rsa_der` wants the bare PKCS#1
        // `RSAPrivateKey`, while aws-lc-rs only exports PKCS#8.
        let key_pem = pem::encode(&pem::Pem::new("PRIVATE KEY", der.as_ref()));
        let encoding =
            EncodingKey::from_rsa_pem(key_pem.as_bytes()).expect("PKCS#8 PEM should parse");

        let mut jwk =
            Jwk::from_encoding_key(&encoding, Algorithm::RS256).expect("JWK should derive");
        jwk.common.key_id = Some(kid.to_string());

        TestKey { encoding, jwk }
    }

    /// Seconds since the epoch, offset by `delta`. Tokens are minted relative to
    /// the wall clock so expiry tests do not depend on a frozen time source.
    fn timestamp(delta: i64) -> i64 {
        chrono::Utc::now().timestamp() + delta
    }

    fn sign(key: &TestKey, claims: serde_json::Value) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = key.jwk.common.key_id.clone();
        jsonwebtoken::encode(&header, &claims, &key.encoding).expect("test token should sign")
    }

    /// The claim set a correctly configured IdP would mint for this server.
    fn valid_claims() -> serde_json::Value {
        json!({
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": timestamp(3600),
            "sub": "user-1",
            "tenant": "acme",
        })
    }

    fn jwt_authenticator(published: &TestKey) -> Authenticator {
        Authenticator::Jwt {
            config: JwtConfig {
                issuer: ISSUER.to_string(),
                audience: AUDIENCE.to_string(),
                jwks_url: "https://issuer.test/.well-known/jwks.json".to_string(),
                tenant_claim: DEFAULT_TENANT_CLAIM.to_string(),
            },
            keys: JwkSet {
                keys: vec![published.jwk.clone()],
            },
        }
    }

    #[test]
    fn a_jwt_from_the_configured_issuer_and_audience_resolves_its_tenant() {
        let key = generate_key(KEY_ID);
        let token = sign(&key, valid_claims());

        let principal = jwt_authenticator(&key)
            .authenticate(Some(&format!("Bearer {token}")))
            .expect("a correctly minted token should authenticate");

        assert_eq!(principal.tenant, "acme");
        assert_eq!(principal.subject.as_deref(), Some("user-1"));
    }

    /// The confused-deputy case, and the reason `aud` is not optional: the token
    /// below is signed by the right key and issued by the right IdP, but it was
    /// minted for a different resource server. Honouring it would turn every
    /// token that IdP issues, for any service, into a key to this graph.
    #[test]
    fn a_token_minted_for_another_resource_server_is_rejected() {
        let key = generate_key(KEY_ID);
        let mut claims = valid_claims();
        claims["aud"] = json!("https://billing.test/api");

        let token = sign(&key, claims);
        let result = jwt_authenticator(&key).authenticate(Some(&format!("Bearer {token}")));

        assert!(
            result.is_err(),
            "a token whose audience is another service must not authenticate, got {result:?}"
        );
    }

    /// Checking `aud` is not enough on its own: an omitted claim has nothing to
    /// compare against, so `aud` has to be *required* as well as checked, or an
    /// audience-less token walks straight past the check above.
    #[test]
    fn a_token_with_no_audience_claim_is_rejected() {
        let key = generate_key(KEY_ID);
        let mut claims = valid_claims();
        claims.as_object_mut().unwrap().remove("aud");

        let token = sign(&key, claims);
        let result = jwt_authenticator(&key).authenticate(Some(&format!("Bearer {token}")));

        assert!(
            result.is_err(),
            "a token that names no audience must not authenticate, got {result:?}"
        );
    }

    /// `iss` pins which IdP the JWKS is trusted to speak for. Without it, a
    /// second issuer whose keys happen to be reachable at the same JWKS URL —
    /// or one added to it later — would be treated as authoritative.
    #[test]
    fn a_token_from_another_issuer_is_rejected() {
        let key = generate_key(KEY_ID);
        let mut claims = valid_claims();
        claims["iss"] = json!("https://other-issuer.test/");

        let token = sign(&key, claims);
        let result = jwt_authenticator(&key).authenticate(Some(&format!("Bearer {token}")));

        assert!(
            result.is_err(),
            "a token from an unconfigured issuer must not authenticate, got {result:?}"
        );
    }

    /// Expired half a minute ago — inside `jsonwebtoken`'s default 60s leeway,
    /// which must not be relied on. `exp` is how a revoked session eventually
    /// stops working, so a stale token is not "close enough".
    #[test]
    fn an_expired_token_is_rejected() {
        let key = generate_key(KEY_ID);
        let mut claims = valid_claims();
        claims["exp"] = json!(timestamp(-30));

        let token = sign(&key, claims);
        let result = jwt_authenticator(&key).authenticate(Some(&format!("Bearer {token}")));

        assert!(
            result.is_err(),
            "an expired token must not authenticate, got {result:?}"
        );
    }

    /// Every fact this server stores is filed under the tenant, so a token that
    /// does not name one has no safe interpretation. Falling back to a default
    /// would quietly merge one IdP's users into some other tenant's graph.
    #[test]
    fn a_valid_token_without_a_tenant_claim_is_rejected() {
        let key = generate_key(KEY_ID);
        let mut claims = valid_claims();
        claims.as_object_mut().unwrap().remove("tenant");

        let token = sign(&key, claims);
        let result = jwt_authenticator(&key).authenticate(Some(&format!("Bearer {token}")));

        assert!(
            result.is_err(),
            "a token carrying no tenant must not authenticate, got {result:?}"
        );
    }

    /// An empty tenant claim is not "no tenant", it is *the* tenant the store
    /// keeps the curated relationship vocabulary under — the one scope every
    /// other tenant reads. An IdP that maps some user to `tenant: ""` would
    /// hand that user write access to shared state.
    ///
    /// [`StoreError::BlankTenant`](crate::error::StoreError::BlankTenant)
    /// already refuses the write; this is the same refusal one layer earlier,
    /// so a blank claim never becomes a [`Principal`] at all.
    #[test]
    fn a_token_whose_tenant_claim_is_empty_is_rejected() {
        let key = generate_key(KEY_ID);

        for blank in ["", "   ", "\t\n"] {
            let mut claims = valid_claims();
            claims["tenant"] = json!(blank);

            let token = sign(&key, claims);
            let result = jwt_authenticator(&key).authenticate(Some(&format!("Bearer {token}")));

            assert!(
                result.is_err(),
                "a token whose tenant claim is {blank:?} must not authenticate, got {result:?}"
            );
        }
    }

    /// The claim name configured in `JwtConfig` is the one actually read, not
    /// just the one stored.
    #[test]
    fn the_tenant_is_read_from_the_configured_claim_name() {
        let key = generate_key(KEY_ID);
        let mut claims = valid_claims();
        claims.as_object_mut().unwrap().remove("tenant");
        claims["org_id"] = json!("acme");

        let Authenticator::Jwt { mut config, keys } = jwt_authenticator(&key) else {
            unreachable!("jwt_authenticator builds the Jwt variant");
        };
        config.tenant_claim = "org_id".to_string();

        let token = sign(&key, claims);
        let principal = Authenticator::Jwt { config, keys }
            .authenticate(Some(&format!("Bearer {token}")))
            .expect("the configured claim should supply the tenant");

        assert_eq!(principal.tenant, "acme");
    }

    /// The JWKS is the only root of trust. A token that is well-formed in every
    /// other respect but signed by a key the IdP never published is forged.
    #[test]
    fn a_token_signed_by_an_unpublished_key_is_rejected() {
        let published = generate_key(KEY_ID);
        // Same `kid`, so the server cannot reject on key lookup alone and has to
        // actually verify the signature.
        let attacker = generate_key(KEY_ID);

        let token = sign(&attacker, valid_claims());
        let result = jwt_authenticator(&published).authenticate(Some(&format!("Bearer {token}")));

        assert!(
            result.is_err(),
            "a token signed by a key outside the JWKS must not authenticate, got {result:?}"
        );
    }

    fn static_authenticator() -> Authenticator {
        let mut tokens = HashMap::new();
        tokens.insert(hash_token(ACME_TOKEN), "acme".to_string());
        Authenticator::StaticToken { tokens }
    }

    #[test]
    fn a_valid_static_token_resolves_to_its_tenant() {
        let principal = static_authenticator()
            .authenticate(Some(&format!("Bearer {ACME_TOKEN}")))
            .expect("valid token should authenticate");

        assert_eq!(principal.tenant, "acme");
    }

    #[test]
    fn a_missing_header_is_reported_as_missing_not_invalid() {
        let err = static_authenticator()
            .authenticate(None)
            .expect_err("no header should not authenticate");

        assert!(
            matches!(err, AuthError::MissingCredentials),
            "expected MissingCredentials, got {err:?}"
        );
    }

    #[test]
    fn a_non_bearer_scheme_is_reported_as_malformed() {
        let err = static_authenticator()
            .authenticate(Some(&format!("Basic {ACME_TOKEN}")))
            .expect_err("a non-Bearer scheme should not authenticate");

        assert!(
            matches!(err, AuthError::MalformedCredentials),
            "expected MalformedCredentials, got {err:?}"
        );
    }

    /// `trim_start_matches` strips a prefix repeatedly, so `Bearer Bearer <tok>`
    /// would collapse to a valid token. Only a single prefix may be removed.
    #[test]
    fn a_repeated_bearer_prefix_does_not_authenticate() {
        let result =
            static_authenticator().authenticate(Some(&format!("Bearer Bearer {ACME_TOKEN}")));

        assert!(
            result.is_err(),
            "a doubled Bearer prefix must not resolve to a tenant"
        );
    }

    #[test]
    fn disabled_auth_resolves_the_configured_tenant_without_a_header() {
        let auth = Authenticator::Disabled {
            tenant: "local".to_string(),
        };

        let principal = auth
            .authenticate(None)
            .expect("disabled auth should accept an unauthenticated request");

        assert_eq!(principal.tenant, "local");
    }
}
