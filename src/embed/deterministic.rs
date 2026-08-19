//! A fake embedder that derives vectors from a hash of the input.
//!
//! Every later task's tests resolve entities and rank proof excerpts through
//! this type, so its output is treated as a fixed specification rather than an
//! implementation detail: changing the derivation moves similarity scores under
//! tests that never mention embeddings.

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use super::{EMBEDDING_DIMENSIONS, Embedder, QUERY_INSTRUCTION};
use crate::error::EmbedError;

/// Deliberately not a plausible model name. Every embedding is stored beside
/// the id of the model that produced it, so this string is what makes a fake
/// that escaped into a real deployment visible in the data.
pub const MODEL_ID: &str = "deterministic-768";

/// A stand-in [`Embedder`] for tests: no model, no weights, no network, and the
/// same vector for the same text on every machine.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicEmbedder;

impl DeterministicEmbedder {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Embedder for DeterministicEmbedder {
    fn model_id(&self) -> &str {
        MODEL_ID
    }

    fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|text| vector_for(text)).collect())
    }

    /// Prefixed exactly as the real model requires, so a caller that reaches for
    /// the wrong method gets a visibly wrong vector here rather than a quietly
    /// worse ranking in production.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(vector_for(&format!("{QUERY_INSTRUCTION}{text}")))
    }
}

/// A SHA-256 digest is 32 bytes, read as eight little-endian `u32` words.
const WORDS_PER_BLOCK: usize = 8;

/// How many digests are needed to fill one vector.
const BLOCKS: u32 = 96;

const _: () = assert!(BLOCKS as usize * WORDS_PER_BLOCK == EMBEDDING_DIMENSIONS);

/// Expands `text` into a unit-length 768-dimensional vector.
///
/// ```text
/// seed          = SHA256(text)
/// block b       = SHA256(seed || u32_le(b))        for b in 0..96
/// raw[8b + w]   = u32_le(block b, word w) / u32::MAX * 2.0 - 1.0
/// vector        = raw / ||raw||_2
/// ```
///
/// SHA-256 rather than [`std::hash::Hash`]: `RandomState` reseeds per process,
/// so the default hasher would produce a vector that is stable within one test
/// run and different in the next, which is exactly the kind of failure that
/// reads as flakiness somewhere else entirely.
///
/// Intermediates are `f64` and narrowed only at the end, so the result is a
/// property of the bytes rather than of accumulated rounding.
fn vector_for(text: &str) -> Vec<f32> {
    let seed: [u8; 32] = Sha256::digest(text.as_bytes()).into();

    let mut raw = Vec::with_capacity(EMBEDDING_DIMENSIONS);
    for block in 0..BLOCKS {
        let mut hasher = Sha256::new();
        hasher.update(seed);
        hasher.update(block.to_le_bytes());
        let digest: [u8; 32] = hasher.finalize().into();

        for word in digest.chunks_exact(4) {
            let bits = u32::from_le_bytes(word.try_into().expect("chunks_exact(4) yields 4 bytes"));
            // Spread over [-1, 1] so the components have both signs and no
            // single coordinate dominates the direction.
            raw.push(f64::from(bits) / f64::from(u32::MAX) * 2.0 - 1.0);
        }
    }

    normalise(&raw)
}

/// Scales `raw` to unit length, because the schema indexes these columns with
/// `vector_cosine_ops`.
fn normalise(raw: &[f64]) -> Vec<f32> {
    let norm = raw.iter().map(|value| value * value).sum::<f64>().sqrt();

    if !norm.is_finite() || norm <= 0.0 {
        // Unreachable for SHA-256 output — it would take 768 words landing on
        // exactly the midpoint — but the unit-length contract has to hold for
        // every input rather than for almost every input, and a silently
        // zero-length vector would sort arbitrarily under cosine distance.
        let mut fallback = vec![0.0; raw.len()];
        if let Some(first) = fallback.first_mut() {
            *first = 1.0;
        }
        return fallback;
    }

    raw.iter().map(|value| (value / norm) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schema pins `VECTOR(768)`, so an embedder that reports one width and
    /// produces another would only fail at INSERT time, deep inside a write.
    ///
    /// Exercised through `&dyn Embedder` because the store is handed one: if the
    /// trait ever stopped being object-safe, this is where it surfaces rather
    /// than in the middle of a task that only wanted to write a query.
    #[tokio::test]
    async fn the_embedder_reports_768_dimensions_and_produces_vectors_of_that_width() {
        let embedder: &dyn Embedder = &DeterministicEmbedder::new();

        assert_eq!(embedder.dimensions(), 768);

        let documents = embedder
            .embed_documents(&["a stored fact".to_string()])
            .await
            .expect("the deterministic embedder cannot fail");
        assert_eq!(documents[0].len(), 768);

        let query = embedder
            .embed_query("a stored fact")
            .await
            .expect("the deterministic embedder cannot fail");
        assert_eq!(query.len(), 768);
    }

    /// Every vector is persisted next to the id of the model that produced it.
    /// If this fake is ever wired up outside a test, that column is the signal,
    /// so the name has to be one no real model would answer to.
    #[tokio::test]
    async fn the_model_id_is_obviously_synthetic() {
        assert_eq!(DeterministicEmbedder::new().model_id(), "deterministic-768");
    }

    /// Computed in f64 so the assertion measures the vector, not the error of
    /// the measurement.
    fn l2_norm(vector: &[f32]) -> f64 {
        vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt()
    }

    /// The index is `vector_cosine_ops`, which assumes unit length. A vector of
    /// any other magnitude still indexes and still returns rows, just in a
    /// subtly wrong order, so nothing downstream would report the mistake.
    #[tokio::test]
    async fn every_vector_is_l2_normalised_to_unit_length() {
        let embedder = DeterministicEmbedder::new();

        let documents = embedder
            .embed_documents(&[
                "Ada Lovelace".to_string(),
                String::new(),
                "a much longer excerpt, quoted verbatim from the source".to_string(),
            ])
            .await
            .expect("the deterministic embedder cannot fail");

        for (text_index, vector) in documents.iter().enumerate() {
            let norm = l2_norm(vector);
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "document {text_index} has L2 norm {norm}, expected 1.0"
            );
        }

        let query = embedder
            .embed_query("Ada Lovelace")
            .await
            .expect("the deterministic embedder cannot fail");
        let norm = l2_norm(&query);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "query vector has L2 norm {norm}, expected 1.0"
        );
    }

    /// Entity resolution re-embeds a name it has already stored and compares the
    /// two. If that comparison were not exact, a node would fail to match itself
    /// and the graph would fork a duplicate.
    #[tokio::test]
    async fn the_same_text_embeds_to_the_same_vector_every_time() {
        let embedder = DeterministicEmbedder::new();
        let text = "Grace Hopper".to_string();

        let first = embedder
            .embed_documents(std::slice::from_ref(&text))
            .await
            .expect("the deterministic embedder cannot fail");
        let second = embedder
            .embed_documents(std::slice::from_ref(&text))
            .await
            .expect("the deterministic embedder cannot fail");

        assert_eq!(first, second);

        let first_query = embedder
            .embed_query(&text)
            .await
            .expect("the deterministic embedder cannot fail");
        let second_query = embedder
            .embed_query(&text)
            .await
            .expect("the deterministic embedder cannot fail");

        assert_eq!(first_query, second_query);
    }

    /// Both vectors are unit length, so the dot product is the cosine.
    fn cosine(left: &[f32], right: &[f32]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(l, r)| f64::from(*l) * f64::from(*r))
            .sum()
    }

    /// Not merely `!=`: a derivation that mapped every input into the same
    /// narrow cone would still be unequal element-wise while making every entity
    /// look like every other one, and entity resolution matches on a similarity
    /// threshold. The near-miss pairs are the interesting ones — a hash that
    /// only mixes some of its input would keep `Ada Lovelace` and `ada lovelace`
    /// adjacent.
    #[tokio::test]
    async fn different_texts_embed_to_meaningfully_different_vectors() {
        let texts = [
            "Ada Lovelace".to_string(),
            "Grace Hopper".to_string(),
            "ada lovelace".to_string(),
            "Ada Lovelace ".to_string(),
        ];

        let vectors = DeterministicEmbedder::new()
            .embed_documents(&texts)
            .await
            .expect("the deterministic embedder cannot fail");

        for left in 0..texts.len() {
            for right in (left + 1)..texts.len() {
                assert!(
                    vectors[left] != vectors[right],
                    "{:?} and {:?} embed identically",
                    texts[left],
                    texts[right]
                );

                let similarity = cosine(&vectors[left], &vectors[right]);
                assert!(
                    similarity.abs() < 0.2,
                    "{:?} and {:?} have cosine {similarity}, which is not meaningfully apart",
                    texts[left],
                    texts[right]
                );
            }
        }
    }

    /// Callers zip the returned vectors back against their own input list by
    /// position, so a reordered, deduplicated or short batch silently files one
    /// entity's embedding under another's name. The batch is deliberately
    /// unsorted and contains a repeat: sorting or deduplicating internally is
    /// the plausible optimisation, and both are caught here.
    #[tokio::test]
    async fn a_batch_returns_one_vector_per_input_in_input_order() {
        let embedder = DeterministicEmbedder::new();
        let texts = [
            "zebra".to_string(),
            "apple".to_string(),
            "zebra".to_string(),
            "mango".to_string(),
        ];

        let batch = embedder
            .embed_documents(&texts)
            .await
            .expect("the deterministic embedder cannot fail");

        assert_eq!(batch.len(), texts.len());

        for (index, text) in texts.iter().enumerate() {
            let alone = embedder
                .embed_documents(std::slice::from_ref(text))
                .await
                .expect("the deterministic embedder cannot fail");

            assert_eq!(
                batch[index], alone[0],
                "position {index} does not hold the embedding of {text:?}"
            );
        }
    }

    /// The same guarantee at the sizes where the other implementation could
    /// lose it.
    ///
    /// `CandleEmbedder` runs a batch as several bounded forward passes and
    /// stitches the results back together, so for it a batch one row past a
    /// pass, or exactly two passes with nothing left over, is where an
    /// off-by-one would put every vector against the wrong text. This embedder
    /// has no model and nothing to split, which is precisely why the sizes are
    /// worth pinning here as well: the contract belongs to the trait rather
    /// than to one implementation of it, and a fake that is only ever asked for
    /// four vectors cannot hold a later one to it.
    #[tokio::test]
    async fn a_batch_of_any_size_returns_one_vector_per_input_in_input_order() {
        let embedder = DeterministicEmbedder::new();

        for count in [0, 1, 2, 7, 8, 9, 16, 17, 64, 65, 129] {
            let texts: Vec<String> = (0..count)
                .map(|index| format!("entity number {index}"))
                .collect();

            let batch = embedder
                .embed_documents(&texts)
                .await
                .expect("the deterministic embedder cannot fail");

            assert_eq!(
                batch.len(),
                count,
                "a batch of {count} came back as {} vectors",
                batch.len()
            );
            for (index, text) in texts.iter().enumerate() {
                let alone = embedder
                    .embed_documents(std::slice::from_ref(text))
                    .await
                    .expect("the deterministic embedder cannot fail");

                assert_eq!(
                    batch[index], alone[0],
                    "in a batch of {count}, position {index} does not hold the embedding of \
                     {text:?}"
                );
            }
        }
    }

    /// The production model is asymmetric: queries carry an instruction prefix,
    /// documents do not. Getting that backwards degrades ranking without
    /// erroring, so the fake has to be asymmetric too — otherwise a later task
    /// could embed a search string with `embed_documents`, pass every test, and
    /// only lose recall against the real model.
    #[tokio::test]
    async fn a_query_and_a_document_of_the_same_text_embed_differently() {
        let embedder = DeterministicEmbedder::new();
        let text = "who maintains the billing service";

        let document = embedder
            .embed_documents(&[text.to_string()])
            .await
            .expect("the deterministic embedder cannot fail");
        let query = embedder
            .embed_query(text)
            .await
            .expect("the deterministic embedder cannot fail");

        assert!(
            query != document[0],
            "embed_query applied no prefix, so calling the wrong method is undetectable"
        );

        let similarity = cosine(&query, &document[0]);
        assert!(
            similarity.abs() < 0.2,
            "query and document vectors have cosine {similarity}; the prefix barely moved the vector"
        );
    }

    /// A write that touches no new text still calls the embedder. An empty batch
    /// is an ordinary case, not an error: the real implementation runs a forward
    /// pass, and a zero-length one has to be short-circuited rather than handed
    /// to the model, which would fail on an empty tensor.
    #[tokio::test]
    async fn an_empty_batch_embeds_to_an_empty_result() {
        let vectors = DeterministicEmbedder::new()
            .embed_documents(&[])
            .await
            .expect("an empty batch is not an error");

        assert!(vectors.is_empty());
    }

    /// Later tasks assert on similarity scores produced by this embedder, so the
    /// vectors have to be reproducible across processes and across runs, not
    /// merely within one. A per-process seed — `HashMap`'s `RandomState` being
    /// the easy mistake — satisfies every other test in this file and fails only
    /// this one.
    ///
    /// The expected values come from an independent implementation of the
    /// derivation documented on `vector_for`, so this pins the output to the
    /// specification rather than to whatever the code currently computes.
    #[tokio::test]
    async fn the_vector_for_a_given_text_is_fixed_across_processes() {
        const EXPECTED_HEAD: [f32; 8] = [
            0.030_845_331,
            -0.002_930_846_3,
            -0.027_276_956,
            -0.001_495_907_4,
            0.054_683_663,
            -0.010_683_852,
            -0.045_287_143,
            -0.028_393_723,
        ];

        let documents = DeterministicEmbedder::new()
            .embed_documents(&["mundu".to_string()])
            .await
            .expect("the deterministic embedder cannot fail");

        for (index, expected) in EXPECTED_HEAD.iter().enumerate() {
            let actual = documents[0][index];
            assert!(
                (actual - expected).abs() < 1e-6,
                "element {index} is {actual}, expected {expected}"
            );
        }
    }
}
