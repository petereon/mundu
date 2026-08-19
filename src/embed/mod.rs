//! Text embedding, behind a trait so the store never depends on a model.

#[cfg(feature = "candle-embedder")]
pub mod candle;
pub mod deterministic;

use async_trait::async_trait;

use crate::error::EmbedError;

#[cfg(feature = "candle-embedder")]
pub use candle::CandleEmbedder;
pub use deterministic::DeterministicEmbedder;

/// The width every embedder in this server must produce, fixed by the
/// `VECTOR(768)` columns in the schema. A model of any other width cannot be
/// swapped in without a migration.
pub const EMBEDDING_DIMENSIONS: usize = 768;

/// The instruction `bge-base-en-v1.5` expects in front of a search string, and
/// only a search string. Stored text must be embedded without it.
///
/// Lives on the trait's module rather than inside one implementation so that
/// every embedder — including the test fake — is asymmetric in the same way. A
/// fake that ignored the prefix would let a caller embed a query as a document
/// and still pass its tests.
pub const QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";

/// Turns text into vectors for similarity search.
///
/// Queries and documents are embedded through **separate methods on purpose**.
/// The production model, `bge-base-en-v1.5`, is asymmetric: it expects an
/// instruction prefix on the query side and none on the document side. Omitting
/// the prefix does not fail, it quietly degrades ranking, so the split exists to
/// make the mistake unrepresentable rather than merely documented. A single
/// method with a `is_query: bool` flag would put the decision back at every call
/// site, which is the failure mode this design removes.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Identifies the model that produced a vector. Persisted alongside every
    /// embedding so a model change is detectable rather than silently mixing
    /// incomparable vector spaces in one index.
    fn model_id(&self) -> &str;

    /// The width of the vectors this embedder produces.
    fn dimensions(&self) -> usize;

    /// Embeds stored text — relationship proof excerpts and entity names — with
    /// no instruction prefix. Returns one vector per input, in input order.
    ///
    /// One call, one result, in the caller's order: how many times an
    /// implementation runs its model to get there is its own business.
    /// `CandleEmbedder` splits a large batch into several bounded forward
    /// passes so that peak memory is a property of the model rather than of how
    /// much the caller asked for; nothing about that reaches this signature,
    /// and nothing about it may reorder, deduplicate or drop a vector. Callers
    /// zip the result against their own list by position, so a batch that came
    /// back rearranged would file every fact under somebody else's embedding
    /// and never fail.
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Embeds search text, applying the model's query-side instruction prefix.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}
