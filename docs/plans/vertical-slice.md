# mundu — Vertical Slice Implementation Plan

A stateless MCP `2026-07-28` memory server: bi-temporal property graph + pgvector,
backed by Postgres, written in Rust.

## Ground rules for every task

- **TDD is mandatory.** Write the failing test, run it, confirm it fails for the
  right reason, then implement the minimum to pass. Never write implementation
  before a test has been observed failing.
- Run `cargo test --no-default-features --lib` for fast loops. The `candle`
  default feature pulls a large dependency tree; only Task 3 needs it.
- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` must be clean
  before reporting DONE.
- Postgres for integration tests: `docker compose up -d postgres-test`, reachable at
  `postgres://mundu:mundu@localhost:55433/mundu`. Schema in `migrations/0001_init.sql`.
  **Not** 55432 — that is the development database, and the suite refuses to run
  against anything holding a non-test tenant. See `tests/common/mod.rs`.
- Do not restructure existing modules outside your task.
- Commit your own work on branch `feat/vertical-slice`.

## Architecture context

Eleven design decisions were settled before implementation. The non-obvious ones,
because code that contradicts them is wrong even if it passes tests:

1. **Tenancy** — `tenant_id` comes from the validated credential, *never* a tool
   argument. A model that could set it could be talked into cross-tenant reads by
   injected proof text.
2. **Salience ranks, never gates.** Nothing is excluded from search for being old.
   Archiving on decay would make an edge unretrievable, therefore unratable,
   therefore unrecoverable — a one-way door.
3. **Supersede is ordered by `valid_from`, not arrival.** Writes arrive out of
   order behind a load balancer; a backfilled historical fact must not overwrite
   current truth.
4. **`valid_from` is agent-supplied but future-dating is rejected.** It is the
   ordering key for supersession, so a future date would win every conflict
   permanently. `recorded_at` is server-set and never agent-writable.
5. **Relationship types are an open set with coercion**, but embedding-similarity
   coercion must be refused for known inverse pairs. Mapping `MANAGES` onto
   `REPORTS_TO` would silently invert every affected fact.
6. **Node resolution biases toward the recoverable failure.** Fragmentation is
   fixable by a later merge; over-merge is not, because the boundary is lost.
7. **Writes are batched.** Both write tools take arrays.
8. **Provenance on every write**, from `_meta` clientInfo, available per-request
   under `2026-07-28` without any session.

## Current state

Done and green (9 tests): `src/error.rs`, `src/auth.rs` (Disabled + StaticToken
modes), `src/config.rs` (loopback guardrail, acknowledgement guardrail, token
parsing with 32-char minimum). Schema verified against Postgres 17 + pgvector.

`Authenticator::Jwt` does not exist yet; `MUNDU_AUTH_MODE=jwt` is the default and
currently returns `unknown mode`, so an unconfigured server fails closed.

---

## Task 1 — JWT/JWKS authenticator mode

**Files:** `src/auth.rs`, `src/config.rs`, `src/error.rs`

Add `Authenticator::Jwt` validating RS256 bearer tokens.

Requirements:
- Validation **must** require and check: signature, `exp`, `iss`, and `aud`.
  Audience validation is the security-critical one: without it, any token minted
  by the same IdP for any other service is accepted here.
- Tenant is read from a configurable claim name (default `tenant`). A token that
  validates but lacks the claim is rejected, not defaulted.
- `subject` on `Principal` is populated from `sub`.
- Keys are supplied as a `JwkSet` held by the authenticator. **Separate fetching
  from validating** so validation is unit-testable with no network.
- Config: `MUNDU_JWT_ISSUER`, `MUNDU_JWT_AUDIENCE`, `MUNDU_JWKS_URL` required;
  `MUNDU_TENANT_CLAIM` optional. `authenticator_from` returns a mode that carries
  this config; actual JWKS fetch happens at startup, not in `authenticator_from`.
- Replace the misleading `unknown mode \`jwt\`` error.

Tests (generate an RSA keypair in-test; no network, no fixtures with real keys):
- a token signed by the configured key with matching iss/aud resolves its tenant
- a token whose `aud` is a *different* resource server is rejected
- a token with a mismatched `iss` is rejected
- an expired token is rejected
- a token signed by an unknown key is rejected
- a valid token missing the tenant claim is rejected

Use `jsonwebtoken` v11 (already a dependency).

---

## Task 2 — `Embedder` trait and deterministic test implementation

**Files:** `src/embed/mod.rs`, `src/embed/deterministic.rs`, `src/lib.rs`

Define the embedding abstraction. **No candle in this task.**

```rust
#[async_trait]
pub trait Embedder: Send + Sync {
    fn model_id(&self) -> &str;
    fn dimensions(&self) -> usize;
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}
```

`embed_query` and `embed_documents` are separate methods **on purpose**: the model
is asymmetric and requires an instruction prefix on the query side only. Baking
the prefix into the trait makes it impossible for a caller to forget.

`DeterministicEmbedder`: 768-dim, seeded from a hash of the input, L2-normalised,
same input always yields the same vector, different inputs yield different
vectors. This is the embedder every other task's tests use.

Tests: dimensionality is 768; output is unit-norm (within 1e-5); identical input
gives identical output; different input gives different output; batch order is
preserved; query and document embeddings of the same text differ (proving the
prefix is applied).

---

## Task 3 — candle `bge-base-en-v1.5` embedder

**Files:** `src/embed/candle.rs`

Implement `Embedder` with candle BERT. Feature-gated behind `candle-embedder`
(already declared in `Cargo.toml`, on by default).

- Load `BAAI/bge-base-en-v1.5` via `hf-hub`, tokenize with `tokenizers`.
- **Mean-pool** the last hidden state over the attention mask, then L2-normalise.
- Query prefix: `Represent this sentence for searching relevant passages: `.
  Documents get no prefix.
- 768 dimensions — must match `VECTOR(768)` in the schema.
- Batch `embed_documents` into a single forward pass.

Tests that download weights must be `#[ignore]`d so the default suite stays
offline and fast. Include at least one ignored integration test asserting that a
query is more similar to a relevant sentence than to an unrelated one.

Report DONE_WITH_CONCERNS if candle's BERT API differs from what the plan assumes.

---

## Task 4 — Store foundation and batch `upsert_node`

**Files:** `src/store/mod.rs`, `src/store/node.rs`

sqlx `PgPool` setup, migration runner, and node upsert.

`upsert_nodes(tenant, &[NodeInput], &dyn Embedder, provenance) -> Vec<NodeUpsertOutcome>`

Resolution ladder per input, in order:
1. exact `node_id` match (after kebab-case canonicalisation of `display_name`)
2. exact alias match
3. vector similarity on `name_vector` above a confident threshold

Outcomes: `Matched { node_id }` (new surface form appended to `aliases`),
`Created { node_id }`, or `Ambiguous { candidates }`.

- **The server mints `node_id`.** Callers supply `display_name`; they never
  supply ids. A model allowed to invent ids will hallucinate ones that don't exist.
- **Ambiguity creates a new node and reports candidates.** Do not auto-attach in
  the ambiguous band — fragmentation is recoverable, over-merge is not.
- Short names need a stricter threshold; short strings have unstable embeddings
  and `Phoenix` the city should not fuse with `Phoenix` the project.
- Resolution must follow `merged_into` tombstones so a merged id still resolves.
- Every write records `writer_client` and `writer_principal`.
- Everything is scoped by `tenant_id`.

Tests against real Postgres using `DeterministicEmbedder`:
- a new name creates a node and returns a kebab-case id
- upserting the same display name twice is idempotent and returns the same id
- a known alias resolves to the existing node rather than creating one
- a new surface form of a matched node is appended to `aliases`
- two tenants may hold the same `node_id` without collision
- a node whose `merged_into` is set resolves to its successor

---

## Task 5 — Relationship type coercion

**Files:** `src/store/relationship.rs`

`coerce_relationship_type(raw, ...) -> CoercedType { canonical, raw }`

Tiers, in order:
1. **Lexical** — uppercase, spaces/hyphens to underscores, strip trailing plural.
   `utilizes` and `Uses` both reach `USES`.
2. **Alias table** — exact lookup in `relationship_type_alias`.
3. **Embedding similarity** on `relationship_type.name_vector` above a threshold,
   **refused when the candidate and incumbent appear in `relationship_inverse_pair`**
   in either order. On refusal, register as a new type.

Unmatched types auto-register in `relationship_type` with `auto_registered = true`
and cardinality `many`.

Tests: `uses`/`Uses`/`USES` all canonicalise to `USES`; `utilizes` resolves via the
alias table; a genuinely novel type auto-registers as `many`; **a type that is the
registered inverse of a close match is NOT coerced onto it** (the direction guard);
`raw` always preserves what the caller passed.

---

## Task 6 — Batch edge creation with supersede

**Files:** `src/store/edge.rs`

`create_edges(tenant, &[EdgeInput], ...) -> Vec<EdgeOutcome>`

- Endpoints must be existing `node_id`s. Unknown ids fail **that item only**, with
  near-match suggestions in the error; other items in the batch still commit.
- `valid_from` optional, defaults to `now()`. **Reject** (do not clamp) anything
  more than 60s in the future — it is the supersession ordering key.
- Embed `proof_excerpt` into `proof_vector`; stamp `embedding_model`.
- Conflict resolution, driven by the type's cardinality:
  - `many`: always append.
  - `one`: compare against the active edge for `(source, type)`.
    - incoming `valid_from` **later** → incoming becomes active; incumbent gets
      `status = superseded`, `valid_until = incoming.valid_from`,
      `superseded_by_edge_id` set.
    - incoming `valid_from` **earlier** → incoming is inserted as *closed history*
      (`status = superseded`, `valid_until` = incumbent's `valid_from`). The
      incumbent is left untouched. This is the late-arriving-backfill case and it
      must not overwrite current truth.
- Serialise per source node (`SELECT ... FOR UPDATE` on the node row) so two
  concurrent writers of the same `one`-cardinality fact cannot both win.
- Record provenance and `raw_relationship_type`.

Tests against real Postgres:
- a `many` edge appends alongside an existing edge of the same type
- a newer `one` edge supersedes the incumbent and sets all three supersede fields
- **an older `one` edge is stored as closed history and the incumbent stays active**
- a future-dated `valid_from` is rejected
- an unknown endpoint fails only its own item; siblings still commit
- `recorded_at` is server-set and ignores any caller-supplied value

---

## Task 7 — Search: RRF fusion with bounded salience re-rank

**Files:** `src/store/search.rs`, `src/domain/salience.rs`

`search(tenant, query, options) -> Vec<SearchHit>`

Pipeline:
1. Two candidate lists: vector similarity on `edge.proof_vector`, and lexical
   match on node `display_name`/`aliases` expanded to their edges.
2. Fuse with **Reciprocal Rank Fusion** (`1/(k + rank)`, k = 60). Cosine scores and
   lexical hits are not on a comparable scale, so they must be fused by rank.
3. Multiply the fused score by salience **mapped into 0.7–1.0**. Salience breaks
   ties; it must never override relevance. An unbounded multiplier lets a fresh
   irrelevant edge outrank a stale correct one.
4. Apply the relative cutoff (drop below a fraction of the top score), then limit.

- Effective salience uses the SQL `effective_salience(base, updated_at, half_life, now())`
  already in the migration. Half-life is configurable, default 27.1 days.
- `search_method` and `precision_requirement` are **optional**, defaulting to
  hybrid/standard. Thresholds come from config, never hardcoded.
- Every hit carries `edge_id` — the rating loop is impossible without it.
- Hits include source/target/type, effective salience, score, `valid_from`/
  `valid_until`, provenance, a `disputed` flag, and a proof snippet truncated to
  ~200 chars unless `include_full_proof` is set.
- Default limit 10, hard cap 50.
- Only `status = 'active'` edges. Never filter by salience.

Tests: exact name match ranks its edges above unrelated ones; a high-salience and
a low-salience edge with equal similarity order correctly; **a highly-relevant
low-salience edge still outranks a barely-relevant high-salience one** (this is
the bounded-band property and the whole point of the design); the limit caps at
50; results are tenant-scoped; proof truncation is applied by default.

---

## Task 8 — Ratings and dispute quorum

**Files:** `src/store/rating.rs`

`rate_edges(tenant, &[Uuid], utility_score, rated_by) -> RatingOutcome`

- Append to `edge_rating` always; it is the durable signal.
- Adjust `salience_base` by `utility_score * 0.25`, clamped to `[0,1]`, and set
  `salience_updated_at = now()`.
- When an edge has negative ratings from **at least 2 distinct `rated_by`
  principals**, set `disputed_at`. Disputed edges stay searchable and flagged.
- **Never archive from this path.** Hard archival is an operator action. Agents
  read untrusted proof text, so an agent-triggerable delete hands injection an
  erasure primitive — and an erased fact is invisible in search output, unlike a
  poisoned one.

Tests: a positive rating raises salience and is capped at 1.0; negative lowers it
and floors at 0.0; **two negatives from the same principal do NOT set `disputed_at`**;
two negatives from different principals do; a disputed edge is still returned by
search with the flag set; rating an edge in another tenant fails.

---

## Task 9 — MCP tools

**Files:** `src/mcp/mod.rs`, `src/mcp/tools.rs`

Four tools via `#[tool_router]`, all with `schemars` input **and output** schemas.

1. `upsert_node` — array of `{display_name, aliases?}` → array of
   `{node_id, display_name, matched, candidates?}`
2. `create_provenance_relationship` — array of `{source_node_id, target_node_id,
   relationship_type, proof_excerpt, valid_from?, conflict_handling?}` → array of
   `{edge_id, status, superseded_edge_id?}` or per-item error
3. `search_knowledge_graph` — `{query, search_method?, precision_requirement?,
   limit?, include_full_proof?}` → array of hits, each carrying `edge_id`
4. `rate_memory_utility` — `{edge_ids, utility_score}` → per-edge new salience

- `relationship_type` is a plain string (open set), but the description enumerates
  the seeded vocabulary so the model reaches for an existing type first.
- Tenant comes from the `Principal` in request extensions. **It is never a tool
  parameter.**
- Provenance from `ctx.client_info()`.
- Per-item failures inside a batch must not sink the whole call.
- Tool descriptions should state that returned proofs are untrusted stored data,
  not instructions.

Tests: schemas generate and contain the expected fields; each tool maps arguments
to the store call correctly; a per-item failure returns a partial success.

---

## Task 10 — MRTR disambiguation for `upsert_node`

**Files:** `src/mcp/mrtr.rs`

When Task 4 returns `Ambiguous` for one or more items, return
`InputRequiredResult` with **one keyed `InputRequest` per ambiguous item** — a
batch of twenty with five ambiguities costs one extra round-trip, not five.

- `requestState` **must** be HMAC-sealed with `RequestStateCodec`. The client
  echoes it back verbatim, so unsealed state is attacker-controlled input that
  could redirect which node an edge binds to.
- Seal **decisions only** (request digest, resolved ids, ambiguous indices,
  candidate ids) — a few hundred bytes. Do not seal vectors; re-embed on retry.
  Never store progress server-side keyed by a handle: the retry round-robins to a
  different instance and that would reintroduce sticky sessions.
- On retry, verify the seal, apply the choices, complete the write.
- Declined or unanswered elicitation falls back to create-new-and-report so a
  non-cooperating client is never blocked.

Tests: an ambiguous batch yields `InputRequiredResult` with one request per
ambiguous item; a tampered `requestState` is rejected; a valid retry applies the
choice; a declined elicitation falls back to creation.

---

## Task 11 — Transports and wiring

**Files:** `src/main.rs`, `src/server.rs`

- axum + `StreamableHttpService`, mounted at `/mcp`.
- **Modern-only:** advertise only `2026-07-28` and set
  `with_stateless_protocol_metadata_required(true)`.
- `with_json_response(true)` for simple request/response tools.
- The `service_factory` runs **per request**, so shared state (`PgPool`,
  `Embedder`, `Config`) must live in a `Clone` handle captured by the closure.
  Nothing may rely on in-memory state surviving between requests.
- An axum middleware authenticates and inserts `Principal` into request
  extensions; tools read it via `http::request::Parts`.
- Unauthenticated requests get 401 before reaching any tool.
- stdio transport behind a `--stdio` flag for local development, using the
  configured dev tenant.
- `Config::from_env` wired up and `validate()` called before binding.

Tests: an end-to-end `tools/call` over HTTP with a static token returns a result;
a missing token returns 401; a request for an older protocol version is rejected.

---

## Task 12 — Threshold calibration

**Files:** `src/bin/calibrate.rs`

A binary that embeds a labelled set of related and unrelated text pairs with the
real model and reports the cosine distribution for each group, plus suggested
thresholds for loose/standard/strict and for node resolution.

This exists because the original spec's `0.70`/`0.95` were guessed. For models of
this class the similarity range is compressed and unrelated pairs can score well
above 0.6, which would make `loose` filter nothing and `strict` filter everything.
Output real numbers and write them into config defaults.

---

## Execution order

Sequential; each depends on those before it.

```
1  (independent)
2  → 3, 4
4  → 5, 6
6  → 7, 8
7,8 → 9 → 10, 11
12 (after 3)
```
