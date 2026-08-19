//! The production [`Embedder`]: `BAAI/bge-base-en-v1.5` run locally with
//! [candle].
//!
//! Weights, tokenizer and config come from the HuggingFace hub on first use and
//! live in the ambient HF cache afterwards, so a warm process starts without
//! touching the network. Nothing about the model is bundled into the binary and
//! no weights belong in this repository.
//!
//! [candle]: https://github.com/huggingface/candle

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use super::{EMBEDDING_DIMENSIONS, Embedder, QUERY_INSTRUCTION};
use crate::error::EmbedError;

/// The hub id of the model, and the string persisted next to every vector it
/// produces. Distinguishable at a glance from `deterministic-768`, which is the
/// point: the model id column is how a fake that escaped into a deployment is
/// found.
pub const MODEL_ID: &str = "BAAI/bge-base-en-v1.5";

const MODEL_OWNER: &str = "BAAI";
const MODEL_NAME: &str = "bge-base-en-v1.5";

/// Pinned to a commit rather than `main`.
///
/// Every stored vector is tagged with [`MODEL_ID`] alone, so if the branch
/// moved under us two deployments would write mutually incomparable vectors
/// under the same name and nothing would say so. A commit id makes a weight
/// change a diff in this file.
const MODEL_REVISION: &str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a";

const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
/// safetensors rather than `pytorch_model.bin`: the pickle format executes
/// arbitrary code while unpickling, and this file is fetched over the network.
const WEIGHTS_FILE: &str = "model.safetensors";

/// The production [`Embedder`].
///
/// Cloning is cheap — the weights sit behind an [`Arc`] — so the server holds
/// one instance and shares it across requests.
#[derive(Clone)]
pub struct CandleEmbedder {
    inner: Arc<Bert>,
}

/// The loaded model, tokenizer and device, kept together because a forward pass
/// needs all three and mixing devices between them is a runtime error.
struct Bert {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    /// Where the tokenizer truncates, and so the widest a row can ever be.
    ///
    /// Read off the tokenizer rather than the config so it is the number that
    /// actually governs the batch, not a second copy of it that could drift.
    max_positions: usize,
}

impl fmt::Debug for CandleEmbedder {
    /// Hand-written because neither [`BertModel`] nor [`Tokenizer`] is `Debug`,
    /// and dumping several hundred megabytes of weights would not help anyone
    /// if they were.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CandleEmbedder")
            .field("model", &MODEL_ID)
            .field("revision", &MODEL_REVISION)
            .field("device", &self.inner.device)
            .field("dimensions", &EMBEDDING_DIMENSIONS)
            .finish()
    }
}

impl CandleEmbedder {
    /// Fetches the model from the hub (or the local HF cache) and loads it.
    ///
    /// Slow and network-dependent the first time — the weights are around
    /// 440 MB — and fast afterwards. Callers run this once, at startup, so that
    /// a broken model is a failed boot rather than a failure on the first write.
    pub async fn load() -> Result<Self, EmbedError> {
        let client = hf_hub::HFClient::new().map_err(|error| {
            unavailable(format!("could not reach the HuggingFace hub: {error}"))
        })?;
        let repository = client.model(MODEL_OWNER, MODEL_NAME);

        let mut paths = Vec::with_capacity(3);
        for filename in [CONFIG_FILE, TOKENIZER_FILE, WEIGHTS_FILE] {
            paths.push(fetch(&repository, filename).await?);
        }
        let [config_path, tokenizer_path, weights_path] = paths
            .try_into()
            .expect("three filenames were requested, so three paths came back");

        // Blocking file work — a 440 MB mmap and twelve layers of tensor
        // construction — kept off the async worker it was called from.
        tokio::task::spawn_blocking(move || {
            Self::from_files(&config_path, &tokenizer_path, &weights_path)
        })
        .await
        .map_err(|error| unavailable(format!("the model load task did not finish: {error}")))?
    }

    /// Builds the embedder from files already on disk.
    ///
    /// Split out from [`Self::load`] so that everything after the download is
    /// exercised by the same code path whether the files came from the hub or
    /// from a cache hit.
    fn from_files(
        config_path: &Path,
        tokenizer_path: &Path,
        weights_path: &Path,
    ) -> Result<Self, EmbedError> {
        let config_json = std::fs::read_to_string(config_path).map_err(|error| {
            unavailable(format!("could not read {}: {error}", config_path.display()))
        })?;
        let config: Config = serde_json::from_str(&config_json).map_err(|error| {
            unavailable(format!(
                "could not parse {}: {error}",
                config_path.display()
            ))
        })?;
        check_dimensions(&config)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|error| {
            unavailable(format!(
                "could not load {}: {error}",
                tokenizer_path.display()
            ))
        })?;
        configure_tokenizer(&mut tokenizer, &config)?;

        let device = select_device();
        // SAFETY: `from_mmaped_safetensors` maps the file into this process. The
        // mapping is only unsound if the file is mutated underneath us; it lives
        // in the content-addressed HF cache, where a new revision is a new path
        // rather than an in-place rewrite.
        let variables =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device) }
                .map_err(|error| {
                    unavailable(format!(
                        "could not map the weights at {}: {error}",
                        weights_path.display()
                    ))
                })?;
        let model = BertModel::load(variables, &config).map_err(|error| {
            unavailable(format!(
                "could not build the model from the weights: {error}"
            ))
        })?;

        Ok(Self::from_parts(model, tokenizer, device))
    }

    fn from_parts(model: BertModel, tokenizer: Tokenizer, device: Device) -> Self {
        // [`configure_tokenizer`] always sets truncation, so the fallback is
        // unreachable. It is the conservative reading anyway: without a known
        // ceiling a text of any length could be as wide as the whole budget,
        // which plans it a pass of its own rather than assuming it is short.
        let max_positions = tokenizer
            .get_truncation()
            .map_or(MAX_CHUNK_POSITIONS, |truncation| truncation.max_length);

        Self {
            inner: Arc::new(Bert {
                model,
                tokenizer,
                device,
                max_positions,
            }),
        }
    }

    /// The device the forward pass runs on. Logged at startup so a build that
    /// silently fell back from the GPU is visible.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }
}

#[async_trait]
impl Embedder for CandleEmbedder {
    fn model_id(&self) -> &str {
        MODEL_ID
    }

    fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        // Short-circuited before anything touches the model: a zero-row tensor
        // is not something BERT can be asked to run, and a write that happens to
        // introduce no new text is an ordinary case rather than an error.
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let bert = Arc::clone(&self.inner);
        let owned = texts.to_vec();
        tokio::task::spawn_blocking(move || bert.forward(&owned))
            .await
            .map_err(|error| {
                EmbedError::Inference(format!("the embedding task did not finish: {error}"))
            })?
    }

    /// Applies [`QUERY_INSTRUCTION`] and then embeds the result as ordinary
    /// text. bge is asymmetric: the prefix belongs on the query side only, and
    /// leaving it off ranks worse without failing, which is why it is applied
    /// here rather than at any call site.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut vectors = self.embed_documents(&[query_text(text)]).await?;
        vectors.pop().ok_or_else(|| {
            EmbedError::Inference("the model returned no vector for the query".to_string())
        })
    }
}

impl Bert {
    /// Embeds the whole batch, in as many bounded forward passes as it takes.
    ///
    /// Returns one vector per input, in input order — callers zip the result
    /// back against their own list by position, so reordering or deduplicating
    /// here would file one entity's embedding under another's name.
    ///
    /// One call was one pass until the memory that cost was measured. The
    /// original reason for a single pass was to avoid paying per-item overhead
    /// on a batched write, and that still holds: [`MAX_CHUNK_POSITIONS`] is a
    /// budget over positions rather than items, so a batch of names — which is
    /// what most writes are — is one pass exactly as before, and only a batch
    /// big enough to be worth splitting gets split.
    fn forward(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        in_bounded_passes(texts, self.max_positions, |chunk| self.one_pass(chunk))
    }

    /// One tokenisation and one forward pass over one chunk.
    ///
    /// Tokenising per chunk rather than per batch is not an accident of where
    /// the loop sits: padding is to the longest row in the call, so a single
    /// long proof excerpt pads only its own chunk to 512 instead of dragging
    /// all 64 rows of a batch out to the ceiling with it.
    fn one_pass(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|error| EmbedError::Tokenization(error.to_string()))?;

        let mut ids = Vec::with_capacity(encodings.len());
        let mut masks = Vec::with_capacity(encodings.len());
        for encoding in &encodings {
            ids.push(Tensor::new(encoding.get_ids(), &self.device).map_err(inference)?);
            masks
                .push(Tensor::new(encoding.get_attention_mask(), &self.device).map_err(inference)?);
        }

        let input_ids = Tensor::stack(&ids, 0).map_err(inference)?;
        let attention_mask = Tensor::stack(&masks, 0).map_err(inference)?;
        // bge encodes single sentences, so every token belongs to segment 0.
        let token_type_ids = input_ids.zeros_like().map_err(inference)?;

        // The mask is handed to the model, not applied afterwards. `cls_pool`
        // reads position 0 and nothing else, so this call is the only place
        // padding is excluded: without the mask every padded position would
        // attend into `[CLS]` and the padding would arrive inside the pooled
        // vector rather than beside it.
        let hidden_states = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(inference)?;

        let pooled = cls_pool(&hidden_states)?;
        let rows: Vec<Vec<f32>> = pooled.to_vec2::<f32>().map_err(inference)?;
        Ok(rows.iter().map(|row| normalise(row)).collect())
    }
}

/// The most token positions one forward pass may cover: rows times the padded
/// width of the widest row in the chunk.
///
/// A pass costs memory by the rectangle it runs over, not by the number of
/// texts in it, and the expensive term is attention — `rows x heads x width x
/// width` — so the same rectangle costs more the longer its rows are. The
/// tokenizer truncates at 512 positions, which makes a chunk of full-length
/// rows the worst a given budget can do, and that is the case measured. On an
/// M2 Pro, 64 texts of 8000 characters (every one of them at the ceiling),
/// peak RSS of the whole process against 854 MiB idle:
///
/// | rows per pass | peak RSS | wall clock |
/// |---------------|----------|------------|
/// | 64 (one pass) | 5675 MiB |     83.6 s |
/// | 32            | 3184 MiB |     73.1 s |
/// | 16            | 1897 MiB |     68.0 s |
/// | 8             | 1266 MiB |     69.4 s |
/// | 4             |  854 MiB |     63.3 s |
///
/// 4096 is 8 rows at the ceiling: it leaves a 2 GB container room for the
/// 440 MB of weights, the runtime and a second request in flight, and it is
/// where the curve stops costing 80 MiB a row and starts disappearing into the
/// floor. That same batch through this constant, end to end, measures
/// **1174 MiB against 5675 MiB before, and 63.2 s against 83.6 s** — chunking
/// is not a throughput tax here, because a 64 x 512 rectangle spills every
/// cache in the machine. Two runs of it landed 5% apart, at 1111 MiB / 59.0 s
/// and 1174 MiB / 63.2 s, so read the figures in this file as the shape of the
/// curve rather than as constants.
///
/// The cost, where there is one, is per-pass overhead on short text — around
/// 35 ms a pass — and avoiding it is why the budget counts positions and not
/// items. Measured at 64 items, before against after: 100 characters each is
/// 40 rows a pass and 1.70 s against 1.63 s, which is no cost at all; 200
/// characters each is 20 rows, four passes, and 2.47 s against 2.63 s, or
/// +6.5%; 2000 characters each is 31.5 s against 38.8 s, or 19% faster. A fixed
/// item count low enough to bound the table above would have paid that overhead
/// on every batch instead: four items a pass costs +38% at 100 characters and
/// saves nothing, because short text was never the problem.
const MAX_CHUNK_POSITIONS: usize = 4_096;

/// An upper bound on the positions `text` will occupy once tokenised.
///
/// Deliberately an over-estimate rather than a tokenisation. No script produces
/// more wordpieces than it has characters — the dense end, CJK, measures 1.00
/// characters a token, and English 5.4 — so a character count plus `[CLS]` and
/// `[SEP]` cannot be short, whatever the caller writes. Being wrong high costs
/// a few more passes than the batch strictly needed; being wrong low would cost
/// the memory bound, which is the entire point of the budget. Counting the
/// tokens properly would mean tokenising every batch twice, since each pass
/// already tokenises its own chunk to pad it to its own width, and it would buy
/// about 1.4x more English text per pass.
///
/// `ceiling` is where the tokenizer truncates, so no text can cost more than
/// that however long it is.
fn estimated_positions(text: &str, ceiling: usize) -> usize {
    text.chars().count().saturating_add(2).min(ceiling)
}

/// How many texts, taken from the front of `texts`, fit in one forward pass.
///
/// Never zero for a non-empty batch: a single text wider than the whole budget
/// is a pass of its own, because one row is as small as a pass can be made and
/// the tokenizer truncates it at `ceiling` regardless. Returning zero there
/// would leave the caller's loop with nothing consumed and nothing to do about
/// it.
fn pass_width(texts: &[String], ceiling: usize) -> usize {
    let mut widest = 0;
    for (index, text) in texts.iter().enumerate() {
        // Padding is to the longest row in the chunk, so admitting this text
        // widens every row already in it.
        let widened = widest.max(estimated_positions(text, ceiling));
        if index > 0 && (index + 1).saturating_mul(widened) > MAX_CHUNK_POSITIONS {
            return index;
        }
        widest = widened;
    }
    texts.len()
}

/// Runs `pass` over consecutive slices of `texts`, each inside
/// [`MAX_CHUNK_POSITIONS`], and concatenates what comes back.
///
/// The whole point of this function is that it cannot reorder anything. It
/// walks the batch front to back, hands each pass a slice of it, and appends —
/// there is no index arithmetic to get wrong, no map keyed by text (which would
/// silently deduplicate the two identical aliases a real batch often carries),
/// and no place where a pass could be skipped and leave the ones after it
/// shifted onto the wrong input. Callers zip these vectors back against their
/// own list by position, so an order this function did not preserve would file
/// every later fact under somebody else's embedding, and nothing downstream
/// would fail.
///
/// The one thing a pass could still do to break that is return the wrong number
/// of vectors, which would shift the rest of the batch by however many it was
/// short. That is checked rather than trusted.
fn in_bounded_passes<Pass>(
    texts: &[String],
    ceiling: usize,
    mut pass: Pass,
) -> Result<Vec<Vec<f32>>, EmbedError>
where
    Pass: FnMut(&[String]) -> Result<Vec<Vec<f32>>, EmbedError>,
{
    let mut vectors = Vec::with_capacity(texts.len());
    let mut remaining = texts;

    while !remaining.is_empty() {
        // Never zero for a non-empty batch, so the batch always shrinks.
        let (chunk, rest) = remaining.split_at(pass_width(remaining, ceiling));
        let embedded = pass(chunk)?;

        if embedded.len() != chunk.len() {
            return Err(EmbedError::Inference(format!(
                "a forward pass over {} texts returned {} vectors; the rest of the batch \
                 would come back filed under the wrong text",
                chunk.len(),
                embedded.len()
            )));
        }

        vectors.extend(embedded);
        remaining = rest;
    }

    Ok(vectors)
}

/// Resolves one file of the model, from the local HF cache if it is there and
/// from the hub if it is not.
///
/// The cache is tried first *explicitly*, rather than left to the hub client to
/// notice. A warm start is then a pair of `stat` calls: no request, no
/// conditional revalidation, and — the part that actually bites — no per-file
/// cache lock. Several `mundu` processes starting at once on the same machine
/// is normal (a rolling restart, or an integration suite spawning the binary
/// three times), and processes queued on a download lock time that lock out and
/// fail to boot for a reason that has nothing to do with the model.
///
/// Only a genuine cache miss reaches the network, and it says so first, because
/// a silent two-minute pause at startup reads as a hang.
async fn fetch<T: hf_hub::RepoType>(
    repository: &hf_hub::HFRepository<T>,
    filename: &str,
) -> Result<std::path::PathBuf, EmbedError> {
    let cached = repository
        .download_file()
        .filename(filename)
        .revision(MODEL_REVISION)
        .local_files_only(true)
        .send()
        .await;

    match cached {
        Ok(path) => return Ok(path),
        // The one error that means "not here yet" rather than "something is
        // wrong". Anything else — an unreadable cache, a poisoned entry — is
        // reported rather than papered over with a download.
        Err(hf_hub::HFError::LocalEntryNotFound { .. }) => {}
        Err(error) => {
            return Err(unavailable(format!(
                "could not read {filename} from the local model cache: {error}"
            )));
        }
    }

    tracing::info!(
        model = MODEL_ID,
        revision = MODEL_REVISION,
        file = filename,
        "not in the local cache; downloading from the HuggingFace hub"
    );

    repository
        .download_file()
        .filename(filename)
        .revision(MODEL_REVISION)
        .send()
        .await
        .map_err(|error| {
            unavailable(format!(
                "could not fetch {filename} at revision {MODEL_REVISION}: {error}"
            ))
        })
}

/// Prepends the model's query-side instruction.
///
/// Reads [`QUERY_INSTRUCTION`] from the trait's module rather than repeating the
/// string, so the fake and the real model stay asymmetric in exactly the same
/// way and a test written against one holds for the other.
fn query_text(text: &str) -> String {
    format!("{QUERY_INSTRUCTION}{text}")
}

/// Refuses a model whose hidden size is not the width the schema stores.
///
/// Checked at construction. The alternative is discovering it as
/// `expected 768 dimensions, not N` from Postgres on the first write, halfway
/// through a transaction that was supposed to record somebody's memory.
fn check_dimensions(config: &Config) -> Result<(), EmbedError> {
    if config.hidden_size != EMBEDDING_DIMENSIONS {
        return Err(unavailable(format!(
            "the model has a hidden size of {}, but every vector column in the schema is \
             VECTOR({EMBEDDING_DIMENSIONS}); a model of another width needs a migration, not a \
             swap",
            config.hidden_size
        )));
    }
    Ok(())
}

/// Pads to the longest member of each batch and truncates at the model's
/// position limit.
///
/// Padding is what makes a single forward pass over mixed-length inputs
/// possible; the attention mask is what keeps it from changing the answer. Both
/// are set here, together, because either alone is a bug.
///
/// The direction is pinned rather than left to the default, because
/// [`cls_pool`] reads position 0 and nothing else. Padding on the left would
/// put `[PAD]` there, and every vector in the store would become an embedding
/// of the pad token — identical, unit length, and completely uninformative.
fn configure_tokenizer(tokenizer: &mut Tokenizer, config: &Config) -> Result<(), EmbedError> {
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        // Load-bearing under `[CLS]` pooling; see above.
        direction: PaddingDirection::Right,
        pad_id: u32::try_from(config.pad_token_id).map_err(|_| {
            unavailable(format!(
                "the model declares pad_token_id {}, which is not a token id",
                config.pad_token_id
            ))
        })?,
        ..PaddingParams::default()
    }));

    // Longer input does not error, it indexes past the position embeddings and
    // fails inside the forward pass, which reads as a model bug rather than as
    // an over-long proof excerpt.
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: config.max_position_embeddings,
            ..TruncationParams::default()
        }))
        .map_err(|error| unavailable(format!("could not limit the input length: {error}")))?;

    Ok(())
}

/// Takes the `[CLS]` position — index 0 — of each row of the last hidden state.
///
/// This is bge's own pooling rather than a choice made here. The model card
/// ships `1_Pooling/config.json` with `pooling_mode_cls_token: true`, and
/// BAAI's FlagEmbedding reads `last_hidden_state[:, 0]`; `[CLS]` is the
/// position the contrastive fine-tuning optimised, so it is the only one whose
/// geometry the training actually shaped.
///
/// Averaging the positions instead is the dangerous kind of wrong: the vectors
/// still come out the right width, still normalise, and still rank — just
/// worse. Measured on the relevance test below, `[CLS]` separates the relevant
/// sentence from the unrelated one by 0.536 against a masked mean's 0.463.
/// The two are also not interchangeable after the fact, since text stored under
/// one pooling and queried under the other ranks badly, so this function is a
/// re-embedding of the whole store rather than an ordinary edit.
///
/// The attention mask is absent here, and that is the point rather than an
/// omission. Only position 0 is read, so no padded position can be averaged in
/// at this step; padding is excluded one layer earlier, where the mask is
/// handed to the model, and if it were dropped there the padding would reach
/// `[CLS]` through attention instead. Position 0 is the real `[CLS]` token only
/// because [`configure_tokenizer`] pads on the right.
fn cls_pool(hidden_states: &Tensor) -> Result<Tensor, EmbedError> {
    hidden_states
        // `[batch, positions, features]` to `[batch, 1, features]` to
        // `[batch, features]`.
        .narrow(1, 0, 1)
        .map_err(inference)?
        .squeeze(1)
        .map_err(inference)?
        // `narrow` hands back a view that still strides over the whole
        // sequence; reading it out as rows of `f32` needs them laid end to end.
        .contiguous()
        .map_err(inference)
}

/// Scales a pooled vector to unit length, because the schema indexes these
/// columns with `vector_cosine_ops`.
///
/// Accumulated in `f64` so the result is a property of the pooled vector rather
/// than of the order the squares were added in.
fn normalise(pooled: &[f32]) -> Vec<f32> {
    let norm = pooled
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();

    if !norm.is_finite() || norm <= 0.0 {
        // Unreachable for a trained model, whose hidden states are never
        // identically zero. Guarded anyway: a zero-length vector is not merely
        // wrong under cosine distance, it has no ordering at all, so it would
        // return an arbitrary set of neighbours rather than none.
        let mut fallback = vec![0.0; pooled.len()];
        if let Some(first) = fallback.first_mut() {
            *first = 1.0;
        }
        return fallback;
    }

    pooled
        .iter()
        .map(|value| (f64::from(*value) / norm) as f32)
        .collect()
}

/// Prefers the GPU when this build asked for one, and says so when it cannot
/// have it.
///
/// A silent fall back to the CPU is only a latency problem, but an unexplained
/// one wastes an afternoon, so the warning is not optional.
fn select_device() -> Device {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => return device,
            Err(error) => {
                tracing::warn!(%error, "metal was requested but is unavailable; embedding on the CPU");
            }
        }
    }
    Device::Cpu
}

/// The model could not be obtained or built. Always carries [`MODEL_ID`], so the
/// message names which model failed rather than just that one did.
fn unavailable(reason: impl fmt::Display) -> EmbedError {
    EmbedError::ModelUnavailable {
        model: MODEL_ID.to_string(),
        reason: reason.to_string(),
    }
}

/// A tensor operation or the forward pass failed. candle reports shape, dtype
/// and device mismatches through one error type, and all of them are inference
/// faults from this module's point of view.
fn inference(error: candle_core::Error) -> EmbedError {
    EmbedError::Inference(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every vector is stored beside the id of the model that produced it. If
    /// this one ever equalled the fake's, the column that distinguishes real
    /// embeddings from synthetic ones would stop distinguishing them.
    #[test]
    fn the_model_id_names_the_real_model() {
        assert_eq!(MODEL_ID, "BAAI/bge-base-en-v1.5");
        assert_ne!(MODEL_ID, crate::embed::deterministic::MODEL_ID);
    }

    /// The prefix is the difference between bge ranking well and ranking
    /// plausibly. Asserted against the shared constant, not a copy of the
    /// string, so a change to the trait's module cannot leave this
    /// implementation behind.
    #[test]
    fn a_query_carries_the_shared_instruction_prefix() {
        let prefixed = query_text("who maintains the billing service");

        assert_eq!(
            prefixed,
            format!("{QUERY_INSTRUCTION}who maintains the billing service")
        );
        assert!(prefixed.starts_with(QUERY_INSTRUCTION));
        assert!(prefixed.ends_with("who maintains the billing service"));
    }

    /// The width is fixed by `VECTOR(768)` in the schema, so a model of any
    /// other size is a migration rather than a swap.
    #[test]
    fn a_model_of_the_wrong_width_is_refused_at_construction() {
        let config = Config {
            hidden_size: 1024,
            ..Config::default()
        };

        let error = check_dimensions(&config).expect_err("1024 is not 768");

        assert!(
            matches!(&error, EmbedError::ModelUnavailable { model, .. } if model == MODEL_ID),
            "expected ModelUnavailable naming the model, got {error:?}"
        );
        let message = error.to_string();
        assert!(message.contains("1024"), "{message}");
        assert!(message.contains("768"), "{message}");
    }

    #[test]
    fn a_model_of_the_right_width_is_accepted() {
        let config = Config::default();
        assert_eq!(config.hidden_size, EMBEDDING_DIMENSIONS);
        check_dimensions(&config).expect("the default BERT base config is 768 wide");
    }

    /// candle reports a bad shape as an ordinary `Err`. It has to arrive as
    /// `Inference` rather than as an unwrap: a malformed batch is a request that
    /// fails, not a process that dies.
    #[test]
    fn a_tensor_failure_is_reported_as_an_inference_error() {
        let no_positions =
            Tensor::zeros((2, 0, 4), DTYPE, &Device::Cpu).expect("a zero tensor is constructible");

        let error = cls_pool(&no_positions).expect_err("a row of no positions has no [CLS]");

        assert!(
            matches!(error, EmbedError::Inference(_)),
            "expected Inference, got {error:?}"
        );
    }

    /// A `[batch, 3 positions, 4 features]` hidden state. Position 0 stands in
    /// for `[CLS]`, and positions 1 and 2 for whatever followed it — real
    /// tokens, padding, or one then the other.
    fn cpu_tensor<const BATCH: usize>(values: &[[[f32; 4]; 3]; BATCH]) -> Tensor {
        Tensor::new(values, &Device::Cpu).expect("a literal tensor is constructible")
    }

    /// The heart of this module, checkable without a single weight.
    ///
    /// Every row's pooled vector is its position 0 verbatim. The later
    /// positions are chosen so that any averaging shows up immediately: row 0
    /// would mean to `[2, 3, 4, 5]` over its two real tokens and row 1 to
    /// `[1, 1, 1, 0]` over all three, neither of which is position 0.
    #[test]
    fn pooling_takes_the_cls_position_of_each_row() {
        let hidden_states = cpu_tensor(&[
            [
                [1.0, 2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0, 6.0],
                [1000.0, 1000.0, 1000.0, 1000.0],
            ],
            [
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ],
        ]);

        let pooled = cls_pool(&hidden_states)
            .expect("the shape is poolable")
            .to_vec2::<f32>()
            .expect("the pooled tensor is two-dimensional");

        assert_eq!(pooled.len(), 2);
        for (index, (actual, expected)) in pooled[0].iter().zip([1.0, 2.0, 3.0, 4.0]).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "row 0 element {index} is {actual}, expected {expected}; something other \
                 than the [CLS] position reached the output"
            );
        }
        for (index, (actual, expected)) in pooled[1].iter().zip([3.0, 0.0, 0.0, 0.0]).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "row 1 element {index} is {actual}, expected {expected}; something other \
                 than the [CLS] position reached the output"
            );
        }
    }

    /// The property that outlives the change of pooling: whatever the model
    /// leaves in the padded tail must not reach the output vector.
    ///
    /// The two rows here are the same two real tokens followed by wildly
    /// different rubbish, which is what a padded row looks like coming out of a
    /// transformer — the padded positions hold *something*, just nothing that
    /// belongs to this input. Any pooling that reads past position 0 fails
    /// this, which is what makes it worth keeping now that the arithmetic no
    /// longer touches the mask.
    #[test]
    fn what_sits_under_the_padding_does_not_change_the_result() {
        let with_zeros = cpu_tensor(&[[
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0, 0.0],
        ]]);
        let with_noise = cpu_tensor(&[[
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [-7.5, 91.0, 0.25, -1e3],
        ]]);

        let pooled_zeros = cls_pool(&with_zeros)
            .expect("the shape is poolable")
            .to_vec2::<f32>()
            .expect("the pooled tensor is two-dimensional");
        let pooled_noise = cls_pool(&with_noise)
            .expect("the shape is poolable")
            .to_vec2::<f32>()
            .expect("the pooled tensor is two-dimensional");

        assert_eq!(
            pooled_zeros, pooled_noise,
            "the padded tail changed the pooled vector, so something past [CLS] is being read"
        );
    }

    /// The half of the padding story that pooling cannot check.
    ///
    /// `[CLS]` pooling reads position 0 unconditionally, so two things have to
    /// be true of every batch before that is the right position to read: the
    /// padding goes on the *end*, and the mask marks it, so the model keeps it
    /// out of attention. Left padding would silently embed `[PAD]` for every
    /// short input, and an unmarked pad would leak into `[CLS]` through the
    /// transformer. Neither shows up as an error anywhere else.
    #[test]
    fn padding_is_appended_and_marked_so_position_zero_is_always_the_real_first_token() {
        let config = Config {
            vocab_size: 8,
            max_position_embeddings: 32,
            ..Config::default()
        };
        let mut tokenizer =
            Tokenizer::from_bytes(STUB_TOKENIZER).expect("the inline tokenizer is well formed");
        configure_tokenizer(&mut tokenizer, &config).expect("the stub config is usable");

        let encodings = tokenizer
            .encode_batch(vec!["alpha", "alpha beta gamma"], true)
            .expect("the stub vocabulary covers these words");

        assert_eq!(
            encodings[0].get_attention_mask(),
            [1, 0, 0],
            "the short input was not right-padded, so position 0 is not its first token"
        );
        assert_eq!(encodings[1].get_attention_mask(), [1, 1, 1]);
        for (index, encoding) in encodings.iter().enumerate() {
            assert_eq!(
                encoding.get_attention_mask().first(),
                Some(&1),
                "row {index} masks out its own position 0, which is the one [CLS] pooling reads"
            );
        }
    }

    #[test]
    fn normalising_a_zero_vector_yields_a_unit_vector_rather_than_zeros() {
        let normalised = normalise(&[0.0; 8]);

        let norm: f64 = normalised
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "a degenerate vector normalised to length {norm}, which has no ordering under \
             cosine distance"
        );
    }

    #[test]
    fn normalising_scales_to_unit_length_without_changing_direction() {
        let normalised = normalise(&[3.0, 4.0]);

        assert!((normalised[0] - 0.6).abs() < 1e-6, "{normalised:?}");
        assert!((normalised[1] - 0.8).abs() < 1e-6, "{normalised:?}");
    }

    /// The ceiling the real tokenizer truncates at. Written out here so the
    /// planning tests below describe the model in production rather than the
    /// 32-position stub further down.
    const REAL_POSITIONS: usize = 512;

    /// The widths of every pass a batch would be split into, in order.
    ///
    /// Walks the planner exactly as [`in_bounded_passes`] does, so a plan that
    /// could never finish the batch — a pass of nothing — fails here rather
    /// than hanging the suite.
    fn plan(texts: &[String], ceiling: usize) -> Vec<usize> {
        let mut widths = Vec::new();
        let mut remaining = texts;
        while !remaining.is_empty() {
            let width = pass_width(remaining, ceiling);
            assert!(
                width > 0,
                "a pass of no texts never consumes the batch, so this plan would not terminate"
            );
            widths.push(width);
            remaining = &remaining[width..];
        }
        widths
    }

    /// The batch this task exists for: 64 proof excerpts long enough to reach
    /// the tokenizer's ceiling are a 64 x 512 rectangle of positions, sixteen
    /// times the budget, measured at 5.7 GiB of peak RSS in one pass. It has to
    /// come apart, and every piece has to be inside the budget.
    #[test]
    fn a_batch_of_full_length_texts_is_split_into_bounded_passes() {
        let texts = vec!["x".repeat(8_000); 64];

        let widths = plan(&texts, REAL_POSITIONS);

        assert!(
            widths.len() > 1,
            "the whole batch is still one pass, so peak memory is still the caller's to choose"
        );
        assert_eq!(
            widths.iter().sum::<usize>(),
            texts.len(),
            "the plan does not account for every text exactly once: {widths:?}"
        );
        for width in &widths {
            assert!(
                width * REAL_POSITIONS <= MAX_CHUNK_POSITIONS,
                "a pass of {width} rows at the ceiling is {} positions, over the \
                 {MAX_CHUNK_POSITIONS} the budget allows",
                width * REAL_POSITIONS
            );
        }
    }

    /// The reason the budget counts positions rather than items.
    ///
    /// Entity names are a few words. Sixty-four of them are a 64 x 22
    /// rectangle — a rounding error against the budget — so they stay in one
    /// pass and pay no per-pass overhead at all. A fixed item count small
    /// enough to bound the batch above would have split this one eight or
    /// sixteen ways for no memory saved: measured at 64 x 100 characters, four
    /// items a pass costs 38% wall clock and not one byte of peak RSS.
    #[test]
    fn a_batch_of_short_texts_stays_in_one_pass() {
        let texts: Vec<String> = (0..64)
            .map(|index| format!("entity number {index}"))
            .collect();

        assert_eq!(plan(&texts, REAL_POSITIONS), vec![texts.len()]);
    }

    /// One row is as narrow as a pass can be made. A text of its own over the
    /// budget is embedded anyway — the tokenizer truncates it at the ceiling —
    /// rather than refused or, worse, planned into a pass of no rows that the
    /// loop would never get past.
    #[test]
    fn one_text_over_the_budget_is_a_pass_of_its_own() {
        let huge = "x".repeat(MAX_CHUNK_POSITIONS * 4);

        assert_eq!(plan(std::slice::from_ref(&huge), usize::MAX), vec![1]);
        assert_eq!(plan(&[huge.clone(), huge], usize::MAX), vec![1, 1]);
    }

    #[test]
    fn an_empty_batch_plans_no_passes() {
        assert!(plan(&[], REAL_POSITIONS).is_empty());
    }

    /// Texts that say where they came from: position `n` is the number `n`,
    /// zero-padded to `chars` so that every text costs the same and every pass
    /// is therefore the same width.
    ///
    /// A vector of `[n]` is what a pass over them should hand back, so a batch
    /// whose vectors were shifted, reversed, deduplicated or dropped reads off
    /// as plainly as a wrong number.
    fn numbered(count: usize, chars: usize) -> Vec<String> {
        (0..count)
            .map(|index| format!("{index:0>chars$}"))
            .collect()
    }

    /// The vector at position `n` of a pass over [`numbered`] texts.
    fn number_in(text: &str) -> Vec<f32> {
        vec![
            text.parse::<f32>()
                .expect("a numbered text parses back to its position"),
        ]
    }

    /// The failure this whole task has to not cause.
    ///
    /// Chunking is the one change that can hand back the right number of
    /// vectors, all of them real, all of them the wrong way round. Nothing
    /// errors, every write succeeds, and every fact in the store is filed under
    /// somebody else's embedding — so search returns confident, wrong answers
    /// and no test that only counts vectors would notice.
    ///
    /// Checked at every size around a boundary rather than at one of them,
    /// because an off-by-one in the loop shows up at exactly one width: a batch
    /// one short of a pass, one over it, and a batch of two and three whole
    /// passes with nothing left over.
    #[test]
    fn every_vector_comes_back_beside_the_text_it_was_made_from() {
        let width = pass_width(&numbered(1_000, 64), REAL_POSITIONS);
        assert!(
            width > 1 && width < 1_000,
            "the texts have to span several passes for this test to mean anything, not {width}"
        );

        for count in [
            1,
            width - 1,
            width,
            width + 1,
            2 * width,
            2 * width + 1,
            3 * width,
        ] {
            let texts = numbered(count, 64);
            let mut widths = Vec::new();

            let vectors = in_bounded_passes(&texts, REAL_POSITIONS, |chunk| {
                widths.push(chunk.len());
                Ok(chunk.iter().map(|text| number_in(text)).collect())
            })
            .expect("a pass that cannot fail did not fail");

            assert_eq!(
                vectors.len(),
                count,
                "a batch of {count} came back as {} vectors",
                vectors.len()
            );
            for (index, vector) in vectors.iter().enumerate() {
                let expected = index as f32;
                assert_eq!(
                    vector[0], expected,
                    "in a batch of {count}, position {index} holds the vector for text \
                     {} — every fact after this one is filed under the wrong embedding",
                    vector[0]
                );
            }
            assert_eq!(widths.iter().sum::<usize>(), count, "{widths:?}");
        }
    }

    /// The same property when the passes are not all the same size, which is
    /// what a real batch looks like: a long proof excerpt next to an entity
    /// name. The uneven plan is the interesting one, because a loop that
    /// tracked its position by pass number rather than by rows consumed would
    /// still pass the even case above.
    #[test]
    fn an_uneven_plan_still_returns_every_vector_in_input_order() {
        let texts: Vec<String> = (0..40)
            .map(|index| {
                let chars = if index < 10 { 900 } else { 20 };
                format!("{index:0>chars$}")
            })
            .collect();
        let widths = plan(&texts, REAL_POSITIONS);
        let distinct: std::collections::HashSet<_> = widths.iter().collect();
        assert!(
            widths.len() > 1 && distinct.len() > 1,
            "the plan is not uneven, so this test is the previous one: {widths:?}"
        );

        let vectors = in_bounded_passes(&texts, REAL_POSITIONS, |chunk| {
            Ok(chunk.iter().map(|text| number_in(text)).collect())
        })
        .expect("a pass that cannot fail did not fail");

        assert_eq!(vectors.len(), texts.len());
        for (index, vector) in vectors.iter().enumerate() {
            assert_eq!(vector[0], index as f32, "position {index} of {widths:?}");
        }
    }

    /// A pass that hands back a different number of vectors than it was given
    /// texts does not merely lose one: every vector after it slides onto the
    /// wrong text. There is nothing sensible to return, so the batch fails.
    #[test]
    fn a_pass_that_returns_the_wrong_number_of_vectors_fails_the_batch() {
        let texts = numbered(8, 64);

        let error = in_bounded_passes(&texts, REAL_POSITIONS, |chunk| {
            Ok(vec![vec![0.0]; chunk.len() - 1])
        })
        .expect_err("a short pass cannot be zipped back against its input");

        assert!(
            matches!(error, EmbedError::Inference(_)),
            "expected Inference, got {error:?}"
        );
    }

    #[test]
    fn an_empty_batch_runs_no_passes_at_all() {
        let vectors = in_bounded_passes(&[], REAL_POSITIONS, |_| {
            panic!("the model was handed a batch of nothing")
        })
        .expect("an empty batch is not an error");

        assert!(vectors.is_empty());
    }

    /// A BERT with the right shape and no learned weights. Loads in
    /// milliseconds, needs no network, and produces meaningless vectors — which
    /// is exactly enough to check the plumbing around the model without
    /// pretending to check the model.
    fn offline_stub() -> CandleEmbedder {
        let config = Config {
            vocab_size: 8,
            hidden_size: EMBEDDING_DIMENSIONS,
            num_hidden_layers: 1,
            num_attention_heads: 12,
            intermediate_size: 32,
            max_position_embeddings: 32,
            ..Config::default()
        };

        let device = Device::Cpu;
        let variables = VarBuilder::zeros(DTYPE, &device);
        let model = BertModel::load(variables, &config).expect("a zero-weight BERT loads");

        let mut tokenizer =
            Tokenizer::from_bytes(STUB_TOKENIZER).expect("the inline tokenizer is well formed");
        configure_tokenizer(&mut tokenizer, &config).expect("the stub config is usable");

        CandleEmbedder::from_parts(model, tokenizer, device)
    }

    /// A whitespace word-level vocabulary, enough to turn the test sentences
    /// below into ids.
    const STUB_TOKENIZER: &[u8] = br#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4},
            "unk_token": "[UNK]"
        }
    }"#;

    /// A write that introduces no new text still calls the embedder. The model
    /// cannot be handed a zero-row tensor, so the short circuit is load-bearing
    /// rather than an optimisation.
    #[tokio::test]
    async fn an_empty_batch_embeds_to_an_empty_result_without_running_the_model() {
        let vectors = offline_stub()
            .embed_documents(&[])
            .await
            .expect("an empty batch is not an error");

        assert!(vectors.is_empty());
    }

    /// Structure only — the stub has no weights, so the vectors mean nothing.
    /// What it does prove is that a batch comes back with one row per input at
    /// the declared width, which is the contract the store relies on.
    #[tokio::test]
    async fn a_batch_returns_one_vector_of_the_declared_width_per_input() {
        let embedder = offline_stub();
        let texts = [
            "alpha".to_string(),
            "alpha beta gamma".to_string(),
            "beta".to_string(),
        ];

        let vectors = embedder
            .embed_documents(&texts)
            .await
            .expect("the stub model runs");

        assert_eq!(vectors.len(), texts.len());
        for (index, vector) in vectors.iter().enumerate() {
            assert_eq!(
                vector.len(),
                embedder.dimensions(),
                "vector {index} is not the declared width"
            );
        }
    }

    #[tokio::test]
    async fn the_embedder_reports_the_schema_width_through_the_trait() {
        let embedder = offline_stub();
        let embedder: &dyn Embedder = &embedder;

        assert_eq!(embedder.dimensions(), EMBEDDING_DIMENSIONS);
        assert_eq!(embedder.model_id(), MODEL_ID);
    }

    // ----------------------------------------------------------------------
    // Tests below here download roughly 440 MB of weights on their first run
    // and are therefore ignored by default. `cargo test --features
    // candle-embedder -- --ignored` runs them.
    //
    // They are the only tests in this file that can say anything about
    // *meaning*. Everything above would pass just as happily against an
    // embedder that returned noise.
    // ----------------------------------------------------------------------

    /// Loaded once for the whole test binary.
    ///
    /// These tests run in parallel, and `load` on a cold cache takes a lock on
    /// the HF cache entry for as long as the 440 MB download lasts — four
    /// concurrent loads time that lock out and fail for a reason that has
    /// nothing to do with embedding. One shared instance also keeps the suite
    /// to a single download and a single mmap.
    static EMBEDDER: tokio::sync::OnceCell<CandleEmbedder> = tokio::sync::OnceCell::const_new();

    async fn downloaded_embedder() -> &'static CandleEmbedder {
        EMBEDDER
            .get_or_init(|| async {
                CandleEmbedder::load()
                    .await
                    .expect("the model downloads and loads")
            })
            .await
    }

    fn l2_norm(vector: &[f32]) -> f64 {
        vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt()
    }

    /// Both vectors are unit length, so the dot product is the cosine.
    fn cosine(left: &[f32], right: &[f32]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(l, r)| f64::from(*l) * f64::from(*r))
            .sum()
    }

    #[tokio::test]
    #[ignore = "downloads ~440 MB of model weights"]
    async fn the_real_model_produces_768_dimensional_unit_vectors() {
        let embedder = downloaded_embedder().await;

        let documents = embedder
            .embed_documents(&[
                "Ada Lovelace wrote the first algorithm.".to_string(),
                String::new(),
            ])
            .await
            .expect("the model runs");

        for (index, vector) in documents.iter().enumerate() {
            assert_eq!(vector.len(), EMBEDDING_DIMENSIONS, "document {index}");
            let norm = l2_norm(vector);
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "document {index} has L2 norm {norm}, expected 1.0"
            );
        }

        let query = embedder.embed_query("who wrote the first algorithm").await;
        let query = query.expect("the model runs");
        assert_eq!(query.len(), EMBEDDING_DIMENSIONS);
        let norm = l2_norm(&query);
        assert!((norm - 1.0).abs() < 1e-4, "query L2 norm is {norm}");
    }

    /// The reason this task exists.
    ///
    /// Every other test in this file passes against a hash-based fake. This one
    /// does not: it asserts that the vectors carry meaning, by requiring a
    /// question to land nearer its answer than a sentence on an unrelated topic.
    /// Without it a green suite would say nothing at all about retrieval.
    #[tokio::test]
    #[ignore = "downloads ~440 MB of model weights"]
    async fn a_query_is_closer_to_a_relevant_sentence_than_to_an_unrelated_one() {
        let embedder = downloaded_embedder().await;

        let documents = [
            "The billing service is maintained by the payments team.".to_string(),
            "Emperor penguins breed on the sea ice of Antarctica during winter.".to_string(),
        ];
        let vectors = embedder
            .embed_documents(&documents)
            .await
            .expect("the model runs");
        let query = embedder
            .embed_query("who maintains the billing service")
            .await
            .expect("the model runs");

        let relevant = cosine(&query, &vectors[0]);
        let unrelated = cosine(&query, &vectors[1]);

        assert!(
            relevant > unrelated,
            "the query is {relevant} from the relevant sentence and {unrelated} from the \
             unrelated one; the vectors carry no meaning"
        );
        // Not merely ordered: a model that mapped everything into one narrow
        // cone would satisfy `>` by a rounding error and rank no better than
        // chance in practice.
        assert!(
            relevant - unrelated > 0.1,
            "the margin between the relevant and unrelated sentence is only \
             {}; that is not a usable ranking signal",
            relevant - unrelated
        );
    }

    /// Catches the padding bug that the offline pooling tests describe, this
    /// time end to end and against real weights.
    ///
    /// The batch mixes a two-word input with a long one, so the short input is
    /// heavily padded. If the mask were ignored anywhere between tokenisation
    /// and pooling, the batched vector for the short input would drift away from
    /// the one it gets on its own, where there is no padding at all.
    #[tokio::test]
    #[ignore = "downloads ~440 MB of model weights"]
    async fn batching_preserves_order_and_matches_one_at_a_time_embedding() {
        let embedder = downloaded_embedder().await;

        let texts = [
            "zebra".to_string(),
            "The payments team took over the billing service after the reorganisation \
             in March, and now owns its on-call rotation as well."
                .to_string(),
            "zebra".to_string(),
            "mango".to_string(),
        ];

        let batched = embedder
            .embed_documents(&texts)
            .await
            .expect("the model runs");
        assert_eq!(batched.len(), texts.len());

        for (index, text) in texts.iter().enumerate() {
            let alone = embedder
                .embed_documents(std::slice::from_ref(text))
                .await
                .expect("the model runs");

            let similarity = cosine(&batched[index], &alone[0]);
            assert!(
                similarity > 0.9999,
                "position {index} has cosine {similarity} against the same text embedded \
                 alone; padding is leaking into the pooled vector"
            );

            let largest = batched[index]
                .iter()
                .zip(&alone[0])
                .map(|(batched, alone)| f64::from((batched - alone).abs()))
                .fold(0.0_f64, f64::max);
            assert!(
                largest < 1e-3,
                "position {index} differs from the same text embedded alone by up to \
                 {largest}"
            );
        }

        // Duplicated inputs at 0 and 2, distinct text at 3: deduplicating or
        // sorting internally would still return four vectors, just not these.
        assert_eq!(batched[0], batched[2]);
        assert!(cosine(&batched[0], &batched[3]) < 0.99);
    }

    /// The same property one layer down, against real weights, for a batch too
    /// big to be one forward pass.
    ///
    /// The test above never leaves a single pass, so it cannot see the loop
    /// that stitches several of them back together — and that loop is the one
    /// place in this crate where every vector can come back real, unit length,
    /// the right width, in the right quantity, and against the wrong text. A
    /// batch one row past a boundary and a batch of exactly two full passes are
    /// where an off-by-one lands.
    ///
    /// The inputs are 600 characters of which nearly all are spaces. The
    /// planner reads the characters, so they are wide enough to be given the
    /// narrowest pass the budget allows; the tokenizer discards the spaces, so
    /// the model still only sees a handful of wordpieces a row. A real
    /// 600-character excerpt is around 110 of them, and forty passes of that in
    /// an unoptimised test build is minutes of CPU spent on a property that has
    /// nothing to do with how long the rows are.
    #[tokio::test]
    #[ignore = "downloads ~440 MB of model weights"]
    async fn a_batch_spanning_several_passes_keeps_every_vector_with_its_own_text() {
        let embedder = downloaded_embedder().await;
        assert_eq!(
            embedder.inner.max_positions, REAL_POSITIONS,
            "the budget is reasoned about in rows of at most {REAL_POSITIONS} positions, but \
             this model truncates somewhere else"
        );

        let padded = |index: usize| format!("proof {index} of the migration{}", " ".repeat(600));
        let width = pass_width(&vec![padded(0); 64], REAL_POSITIONS);
        assert!(
            width > 1 && width < 32,
            "a pass of {width} rows leaves nothing to cross or too much to embed"
        );

        let mut texts: Vec<String> = (0..2 * width).map(padded).collect();
        // The same text in two different passes. A cache or a deduplication
        // keyed on the text would still return the right number of vectors
        // here, and the assertions below would still hold — but a batch that
        // came back one short because of one would not.
        texts[2 * width - 1] = texts[0].clone();

        let mut alone = Vec::with_capacity(texts.len());
        for text in &texts {
            let single = embedder
                .embed_documents(std::slice::from_ref(text))
                .await
                .expect("the model runs");
            alone.push(single.into_iter().next().expect("one text, one vector"));
        }

        for count in [width + 1, 2 * width] {
            let batch = &texts[..count];
            let widths = plan(batch, REAL_POSITIONS);
            assert!(
                widths.len() > 1,
                "a batch of {count} is still one pass, so this test proves nothing: {widths:?}"
            );

            let batched = embedder
                .embed_documents(batch)
                .await
                .expect("the model runs");

            assert_eq!(
                batched.len(),
                count,
                "a batch of {count} came back as {} vectors",
                batched.len()
            );
            for index in 0..count {
                let similarity = cosine(&batched[index], &alone[index]);
                assert!(
                    similarity > 0.9999,
                    "in a batch of {count} split as {widths:?}, position {index} has cosine \
                     {similarity} against its own text embedded alone; the passes have been \
                     stitched back together in the wrong order"
                );
                let largest = batched[index]
                    .iter()
                    .zip(&alone[index])
                    .map(|(batched, alone)| f64::from((batched - alone).abs()))
                    .fold(0.0_f64, f64::max);
                assert!(
                    largest < 1e-3,
                    "in a batch of {count}, position {index} differs from the same text \
                     embedded alone by up to {largest}"
                );
            }

            // Adjacent positions hold different texts, so a loop that repeated
            // one pass's vectors for every pass would not get this far.
            assert!(cosine(&batched[0], &batched[1]) < 0.9999);
        }
    }

    /// The prefix has to reach the model, not merely the format string.
    #[tokio::test]
    #[ignore = "downloads ~440 MB of model weights"]
    async fn a_query_and_a_document_of_the_same_text_embed_differently() {
        let embedder = downloaded_embedder().await;
        let text = "who maintains the billing service";

        let document = embedder
            .embed_documents(&[text.to_string()])
            .await
            .expect("the model runs");
        let query = embedder.embed_query(text).await.expect("the model runs");

        assert!(
            query != document[0],
            "embed_query applied no prefix, so bge is being used symmetrically"
        );
        // bge's instruction is a nudge, not a different sentence, so the two
        // vectors stay close. The bound only has to exclude "identical".
        let similarity = cosine(&query, &document[0]);
        assert!(
            similarity < 0.999,
            "query and document vectors have cosine {similarity}; the prefix barely moved \
             the vector"
        );
    }
}
