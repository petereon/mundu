use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("{0} is not set")]
    Missing(&'static str),
    #[error("{var} is invalid: {reason}")]
    Invalid { var: &'static str, reason: String },
    #[error(
        "refusing to start: authentication is disabled but the listener is bound to {bind}, \
         which is not a loopback address"
    )]
    InsecureBind { bind: String },
    #[error(
        "refusing to start: MUNDU_AUTH_MODE=disabled requires MUNDU_INSECURE_NO_AUTH=1 to be set \
         explicitly, so that unauthenticated operation is always a deliberate act"
    )]
    UnacknowledgedInsecureAuth,
}

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("missing Authorization header")]
    MissingCredentials,
    #[error("malformed Authorization header; expected `Bearer <token>`")]
    MalformedCredentials,
    #[error("invalid credentials")]
    InvalidCredentials,
}

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedding model {model} is unavailable: {reason}")]
    ModelUnavailable { model: String, reason: String },
    #[error("failed to tokenize input for embedding: {0}")]
    Tokenization(String),
    #[error("embedding inference failed: {0}")]
    Inference(String),
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("could not connect to the database: {source}")]
    Connect {
        #[source]
        source: sqlx::Error,
    },
    #[error("database query failed: {0}")]
    Query(#[from] sqlx::Error),
    #[error("could not apply migrations: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),
    #[error("could not embed node names: {0}")]
    Embed(#[from] EmbedError),
    #[error(
        "node `{node_id}` in tenant `{tenant}` is part of a merged_into cycle; \
         resolution cannot pick a successor"
    )]
    MergeCycle { tenant: String, node_id: String },
    #[error(
        "refusing to merge node `{node_id}` in tenant `{tenant}` into itself; the schema's \
         node_merge_not_self CHECK would refuse the write anyway, and a constraint violation \
         tells a caller nothing about what it asked for"
    )]
    SelfMerge { tenant: String, node_id: String },
    #[error(
        "tenant `{tenant}` has no node `{node_id}` to merge as the {role}; a merge names two \
         ids the server itself minted, so a miss is a spelling or a staleness problem rather \
         than a semantic one"
    )]
    UnknownMergeNode {
        tenant: String,
        node_id: String,
        /// `loser` or `winner`. A caller told only that one of two ids is
        /// unknown cannot tell whether it mistyped the node it is retiring or
        /// the one it is keeping, and those are opposite mistakes.
        role: &'static str,
    },
    #[error(
        "node `{node_id}` in tenant `{tenant}` was already merged into `{merged_into}`; \
         retiring it again would rewrite that redirect and strand the facts that moved with \
         it. Merge `{merged_into}` instead, if that is still what you meant"
    )]
    AlreadyMerged {
        tenant: String,
        node_id: String,
        merged_into: String,
    },
    #[error(
        "input {index} has a blank display_name; the server mints every node_id from it, \
         so there is nothing to name the entity after"
    )]
    BlankDisplayName { index: usize },
    #[error(
        "relationship type `{raw}` has no usable characters; registering it would put an \
         empty name in the vocabulary that every other blank type would then share"
    )]
    BlankRelationshipType { raw: String },
    #[error(
        "refusing to act for a caller with no tenant; the empty tenant is the curated \
         vocabulary that every tenant reads, so a blank credential claim would write there"
    )]
    BlankTenant,
    #[error(
        "input {index} sets valid_from to {valid_from}, more than {tolerance_seconds}s past the \
         server's clock; valid_from is the key supersession orders on, so a future-dated fact \
         would win every conflict until that date passes and no later fact could displace it"
    )]
    FutureValidFrom {
        index: usize,
        valid_from: chrono::DateTime<chrono::Utc>,
        tolerance_seconds: i64,
    },
    #[error(
        "utility_score {utility_score} is outside [-1, 1]; the scale runs from -1 (misleading) \
         to 1 (critical), and a score off it would move salience by an amount nothing in the \
         ranking band can absorb"
    )]
    UtilityScoreOutOfRange { utility_score: f32 },
    #[error(
        "`{argument}` carries {received} items and this server takes at most {limit} in one \
         call; embedding the array is bounded per pass but the work still scales with its \
         length. Nothing was written: split the {received} into batches of {limit} or fewer \
         and send them as separate calls"
    )]
    BatchTooLarge {
        argument: &'static str,
        received: usize,
        limit: usize,
    },
    #[error(
        "input {index} has a {field} of {chars} characters and this server takes at most \
         {limit}; the length of the longest text in a batch is the other half of what one \
         call costs, and past this point the text is no longer a name or a quotation. \
         Nothing was written: shorten it and send the batch again"
    )]
    TextTooLong {
        index: usize,
        field: &'static str,
        chars: usize,
        limit: usize,
    },
    #[error(
        "input {index} has a proof_excerpt of {chars} usable characters and at least {minimum} \
         are needed; a fact is recorded here only with the words that justify it, so an empty \
         or near-empty excerpt is a bare assertion wearing the costume of an evidenced one — \
         and it would be embedded into a proof_vector of essentially nothing, which then ranks \
         against every later search. Quote what was actually said or written"
    )]
    UnevidencedProofExcerpt {
        index: usize,
        chars: usize,
        minimum: usize,
    },
}
