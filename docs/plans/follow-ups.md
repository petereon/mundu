# mundu — follow-ups after the vertical slice

The slice is complete and green (260 tests), MRTR included. These are the items the
work surfaced, in the order I would take them. Each states the evidence, because
in every case the evidence is what changed the answer.

Measurements below come from `cargo run --features candle-embedder --bin calibrate`
over 422 labelled pairs against the real `bge-base-en-v1.5` model.

---

## Task 10 — MRTR disambiguation (resume)

**Status:** attempted, parked incomplete on branch `wip/task-10-mrtr` (commit
`2221036`), 112 passing / 1 failing with several test binaries not compiling.

**Restart rather than merge.** The WIP rewrote 284 lines of `src/store/node.rs`,
and that file has since gained the lexical guard (`names_contradict`, wired at
the tier-3 merge site). A merge would conflict in exactly the region both
changes touch.

Two things from the WIP are worth carrying over rather than rediscovering:

- Its module doc in `src/mcp/disambiguation.rs` correctly derives every
  constraint that matters — one round for the whole batch, nothing remembered
  between rounds, `requestState` as untrusted input, decisions never vectors.
- Its `node.rs` split into `resolve_nodes` (decide, don't write) and
  `upsert_nodes_choosing` (write, honouring the caller's choices) is the right
  architecture. MRTR needs resolution and commitment to be separable, because
  round one resolves and asks, and round two commits.

Full requirements are in the Task 10 section of `vertical-slice.md`.

---

## Task 14 — the short-name threshold, and why it exists — **DONE**

**Decided.** The length split survives, but only at the confident bar, and the
number was re-derived. `CONFIDENT_SIMILARITY_SHORT` 0.97 → **0.93**;
`AMBIGUOUS_SIMILARITY_SHORT` (0.90) **removed** — both regimes now report
candidates from `AMBIGUOUS_SIMILARITY` = 0.82. `SHORT_NAME_CHARS` stays 12 with
a rewritten doc. `CONFIDENT_SIMILARITY` and `AMBIGUOUS_SIMILARITY` untouched.

**The stated rationale was false and the doc now says so.** Negatives do not
climb as names shorten (1–6: max 0.9115; 7–12: 0.8977; 13–20: **0.9451**), so
the old comment had the risk pointing the wrong way.

**What replaced it.** `names_contradict` needs a word on each side the other
lacks, so a one-token name can never be vetoed by it — provable from the guard,
not inferred from the corpus. That makes the two regimes structurally unlike,
and the sweep added to `calibrate.rs` measures how unlike:

```
                         worst negative        worst the guard CANNOT veto
long  (> 12 chars)   0.9451 Acme Corp/Holdings   0.7181 Internal Tools/External Partners
short (<= 12 chars)  0.9115 v1.2.0/v1.3.0        0.8977 Phoenix/Phoenix, AZ
```

At 0.92 a long name sits **+0.2019** above the highest negative its guard cannot
see. No value in `[0, 1]` gives a short name that margin. The split is therefore
not a matter of degree, and matching the margins is not available.

**How 0.93 was derived.** From the guard's blind spot, which is *one token plus
a qualifier* — the guard reads the extra word as elaboration and allows it. That
band holds `Phoenix`/`Phoenix, AZ` (0.8977) and `Barcelona`/`Barcelona FC`
(0.8534) as negatives against `AWS`/`Amazon AWS` (0.9294) and `Acme`/`ACME Inc`
(0.9259) as true matches — 0.0282 apart, same lexical shape, indistinguishable
to every other signal the resolver has. The bar goes above the whole band rather
than through it. 0.92 would cut through the middle of it.

**The trade, from the sweep:**

```
bar    same merged (of 14)   negatives merged (of 30)   margin   unguarded
0.92        4                        0                  +0.0085   +0.0223
0.93        2                        0                  +0.0185   +0.0323
0.94        1                        0                  +0.0285   +0.0423
0.95        0                        0                  +0.0385   +0.0523
0.97        0                        0                  +0.0585   +0.0723   (was here)
```

0.93 admits **2 of 14** true short matches — `GPT-4`/`GPT4` (0.9409) and
`Postgres`/`PostgreSQL` (0.9334) — at **+0.0185** over the worst observed short
negative and **+0.0323** over the worst one the guard cannot veto. Not 0.94,
which costs `Postgres`/`PostgreSQL` and buys nothing: both bars are bounded by
the same unguardable negative and both clear it.

**Why the candidate floor came down with it.** The guard is a veto at the *merge
site* and is never consulted about the candidate list, so the asymmetry that
justifies splitting the confident bar says nothing about the floor. Neither does
the model: unrelated short pairs measure 0.3631–0.5071 against 0.3205–0.4589 for
long ones. The old 0.90 left 10 of 14 true short matches as duplicates with *no
candidate list* — silent fragmentation, which is the failure mode the strict bar
is supposed to be trading down to. At 0.82 the short regime reports 4 of 30
negatives, against 7 of 22 for long names at the same floor: less chatty, not
more.

**What did not change, and the residual risk.** 30 short negatives and 14 short
same-pairs is a thin corpus, and 0.93 is defended by 0.0185 of margin on it. The
mitigation is that the pairs 0.93 admits are the ones whose *shape* the guard's
silence endorses, not merely the ones that scored high. `SHORT_NAME_CHARS` is
still a character count standing in for a token count — lossy, but lossy toward
duplicates rather than fusions; making it a token count is a real refinement this
corpus cannot adjudicate, because no short negative reaches 0.92 under either
rule.

**Where this disagrees with the instrument, deliberately.** The Recommendations
block still reads `CONFIDENT_SIMILARITY_SHORT current 0.93 recommended 0.92 TOO
HIGH`. That verdict is guard-blind by construction — `no_false_merge_threshold`
knows only the cosines, so it rounds up past the worst negative and stops. It
cannot see that the negative it stops above (`v1.2.0`/`v1.3.0`) is one the guard
vetoes anyway, nor that the one the guard *cannot* veto sits 0.0282 under the
nearest true match of the same shape. The 0.01 gap is that reasoning, and the
sweep section is where it is visible.

**Instrument added.** `calibrate.rs` gained a "confident bar, swept" section:
per regime, every two-decimal candidate bar with true matches and false merges
counted twice — cosine alone and after `names_contradict` — plus the margin to
the worst negative and to the worst *unguardable* negative. The trade above is
read off it rather than asserted.

---

## Task 15 — mechanical inverse-pair detection — **DONE**

**Decided.** The direction guard now has two halves and tier 3 must clear both:
`relationship_inverse_pair` as before, and `are_mechanical_inverses`, a rule
about spelling that needs no table. Neither replaces the other and neither is a
superset of the other. `CONFIDENT_SIMILARITY` is **untouched at 0.95** — this
task closes the hole so that lowering it is a decision somebody can take safely,
not the decision itself. No migration: the fix is a code rule, and the eight
curated rows are still exactly right for what they cover.

**The patterns, and why only these two.**

- **`X` / `X_BY` — the passive agent.** All four unguarded pairs the calibration
  found are this shape. The stem under the suffix is compared against the other
  name, through one inflection on each side (`MENTORS` ↔ `MENTORED`), because
  the pairs that actually occur are inflected differently on the two halves. The
  suffix on its own is *not* evidence: `AUTHORED_BY` is a seeded canonical type,
  and a rule that read the suffix alone would refuse every coercion onto it —
  `WRITTEN_BY`, `CREATED_BY`, `PENNED_BY` included.

  The inflections undone are `S`/`ES` (through the same `plural_stem` the exact
  tiers use, so the one rule about English `S` lives in one place), `ED`/`D`, and
  `ING`. The last is not a fifth shape, it is the same verb: `MENTORING` was a
  false negative until it was added, and a false negative here is the reversal.
  Nothing is stripped below three remaining letters, which is what keeps `USED`
  minus `ED` from matching the `US` of an unrelated name — and incidentally
  drops exactly the short irregulars (`USING`, `GOING`, `BEING`) whose stems
  would be junk.
- **`HAS_X` / `X_OF` — the possessive.** `A HAS_PART B` and `B PART_OF A` are one
  fact from opposite ends, and the shape is productive: `HAS_OWNER`/`OWNER_OF`,
  `HAS_PARENT`/`PARENT_OF`. Both ends are required, because `_OF` alone is the
  commonest ending in this vocabulary (`MEMBER_OF`, `PART_OF`, `COMPONENT_OF`)
  and says nothing about direction by itself.

  This one contradicts the task note above, which listed `PART_OF`/`HAS_PART`
  and `MEMBER_OF`/`HAS_MEMBER` among the pairs no morphology finds. They are
  mechanical, and the rule derives both. The two that genuinely are not are
  `REPORTS_TO`/`MANAGES` and `LOCATED_IN`/`CONTAINS` (plus
  `DEPENDS_ON`/`REQUIRED_BY`), which is still four rows' worth of reason to keep
  the table.

Nothing else was added. A third pattern was considered and rejected: stripping a
leading copula, so that `IS_PART_OF` reaches `HAS_PART`. It is a tier-1
*normalisation* gap — `IS_PART_OF` should fold onto `PART_OF` before any of this
runs — and fixing it there would give the inverse rule the pair for free, where
fixing it here would teach the guard more English than the normaliser knows.

**Which way the rule leans, deliberately.** Towards refusing. A false positive
refuses a coercion that should have happened: the name registers as its own
type, an operator sees two words for one relation in a small table and writes
one alias row. A false negative folds a relation onto its own opposite and
reverses every fact written with it behind a `proof_excerpt` that still reads
correctly. On the calibration corpus the lean costs nothing measurable — all 14
`Same`-labelled relationship pairs pass through untouched, and structurally they
must, because the rule never looks at two names unless one carries `_BY` or one
carries `HAS_` and the other `_OF`.

**Coverage, measured.** `calibrate.rs` gained an inverse-coverage section: per
measured pair, whether the table catches it, the rule catches it, or nothing
does.

```
    table + rule   5   AUTHORED_BY/AUTHORED, OWNS/OWNED_BY, PART_OF/HAS_PART,
                       USES/USED_BY, MEMBER_OF/HAS_MEMBER
    table only     3   DEPENDS_ON/REQUIRED_BY, LOCATED_IN/CONTAINS,
                       REPORTS_TO/MANAGES
    rule only      4   MENTORS/MENTORED_BY, MANAGES/MANAGED_BY,
                       FUNDS/FUNDED_BY, SUPERSEDES/SUPERSEDED_BY
    NOTHING        0

  12 of 12 refused by at least one half. The table alone reaches 8; the rule
  alone reaches 9 — neither is a superset of the other, which is why both run.
```

The same coverage is asserted as a test that needs no model, so a corpus pair
that falls out of both halves fails CI rather than being noticed in a report.

**Is 0.87 safe now? Yes, and it was not before.** On this corpus nothing is
unrefused, so no bar in `[0, 1]` reverses a fact and 0.87 is not the binding
constraint at all. Three things belong with that answer.

*How thin the margin is without the guard.* 0.87 is not an independent number:
`no_false_merge_threshold` produced it by rounding up past the highest pair the
bar must reject, and that pair is an inverse — `AUTHORED_BY`/`AUTHORED` at
0.8692. So the bar clears the highest measured inverse by **0.0008**, which is
inside the noise of the instrument. Arithmetically the guard is still belt and
braces at 0.87; practically it is the only thing standing between the bar and a
reversal, and any re-measurement that moves 0.8692 up by a thousandth makes that
literal. Before this task, four of the pairs in that band had nothing behind
them at all.

*What 0.87 buys.* 4 of the 14 true synonyms, against 1 at 0.95 — the interleaving
is why it is only 4, and why chasing more of them with a lower bar walks straight
into the inverse population.

*What the rule still cannot see.* It is morphology, so irregular participles
(`PAYS`/`PAID_BY`) and suppletive pairs nobody has curated go straight through.
Those are table rows and always will be. 12 measured pairs is a measurement, not
a proof about English, and the honest form of "the hole is closed" is "closed
over everything anyone has measured, by two independent mechanisms, with a test
that fails if a thirteenth pair escapes both".

**Why the guard can never be retired in favour of a better number**, now stated
in the module doc where a reader will meet it: the inverse and synonym
populations are *interleaved*, not adjacent. `AUTHORED_BY`/`AUTHORED` at 0.8692
outscores 10 of the 14 true synonyms. A bar high enough to exclude every inverse
excludes most real synonyms with it, which is the fragmentation the module
exists to prevent. Direction is not a matter of degree.

---

## Task 16 — `precision_requirement` outside hybrid search — **DONE**

**Decided.** Say what the knob does, and make what it does sayable. The tool
description, the `precision_requirement` argument and the `PrecisionRequirement`
enum now all state that the setting applies to `hybrid` search only and is
ignored under `semantic` and `lexical`; `Store::search` stops applying the
relative cutoff when only one candidate list came back, so "ignored" is exactly
what happens rather than nearly. No second meaning for the word, no new
constant, no absolute floor. `RRF_K`, `CANDIDATE_DEPTH`, `CUTOFF_FRACTION_*`,
`MAX_LIMIT` and the salience band are untouched, and hybrid ranking is unchanged
wherever both halves returned something — a hybrid search whose lexical half
matched no node was already a single-list search wearing the other name, and is
now treated as one.

**The finding was right, and its arithmetic was one setting off.** Reproduced
before anything was changed: five edges in one tenant, the query
`Alice Whitfield`, all three precisions. `Semantic` returned the same five ids
at Relaxed, Standard and Strict; `Lexical` the same two. `Hybrid` over that same
corpus returned five at Relaxed and two at Strict — so the knob does bite, and
only there. But the rank bound is not "roughly 114" at every setting. With
`best = 1/61` the three floors solve to

```
Relaxed   0.10  floor 0.001639  keeps rank 550 → capped at CANDIDATE_DEPTH = 200
Standard  0.35  floor 0.005738  keeps rank 114
Strict    0.60  floor 0.009836  keeps rank  41
```

and **41 is inside `MAX_LIMIT = 50`, not past it.** The parameter was therefore
not uniformly inert. Pinned as a failing test first — 55 edges, `Semantic`,
`limit = 50` — Strict returned 41 ids and Relaxed 50, the 41 a prefix of the 50.
Nine hits dropped by `61 / 0.6 - 60`: a number that is a property of `RRF_K` and
of nothing about the query, the corpus, or the forty-second hit. Change `RRF_K`
and the same word would name a different depth on the same data.

**So the choice was not between inert and meaningful, but between inert and
arbitrary.** Documenting "applies to hybrid only" and leaving the rank-41 cut in
place would have published a false sentence. Documenting the truth would have
meant writing "and also truncates a single-index search at 41" into a schema a
model reads before every call, which is precisely the knob-that-costs-tokens the
task set out to remove. The branch is what makes the short sentence true, and it
is the whole of the behaviour change.

**Why not give it a single-list meaning (option 1).** There is nothing to build
one out of. The only per-hit signal a lone list carries beyond rank is the raw
score, and that regime was measured rather than assumed: answers span
0.5791–0.7808, same-topic non-answers reach 0.7010, unrelated pairs 0.2185–
0.5055. The populations overlap, so a floor tight enough to exclude the
non-answers (0.71) loses 3 of 10 real answers. Depth is not a substitute either
— a "precision" that returned fewer hits is `limit` under a second name, and two
knobs on one quantity are a pair a caller can set against each other.

**Why not remove it for single-list modes (option 3).** It needs `oneOf` or
`if`/`then` in a published input schema, which model clients read worse than a
flat object, and it converts "ignored" into "rejected". `MAX_LIMIT` already
settles that trade here: *clamped rather than refused, because an error teaches
a model nothing it can act on.*

**Where the wrong number came from.** The structural NOTE in `calibrate.rs`
claimed the single-list Strict bound was deeper than the hard cap, which is how
the analysis reached "all three settings return identical results" without
qualification. It now states the bound against the cap it is measured against,
and records that search no longer applies the cutoff there at all.

**Tests.** In `tests/search.rs`,
`precision_changes_nothing_a_single_list_search_can_show` — a characterisation
test, green the first time it ran, whose hybrid half is what stops it being
vacuous — and `a_single_list_search_keeps_a_tail_no_fusion_ever_measured`, the
failing test that drove the branch. In `src/store/search.rs`,
`the_cutoff_on_a_single_list_solves_to_a_fixed_rank_whatever_the_corpus` pins the
200/114/41 arithmetic and `only_two_rankings_can_disagree` pins the predicate. In
`tests/mcp_schema.rs`, `precision_says_it_applies_to_hybrid_search_and_nowhere_else`
holds all three published descriptions — blurb, argument and enum — to the
behaviour, so the promise cannot go stale on the next change.

**Superseded in part by the `RRF_K` fix below.** The 200/114/41 arithmetic was a
property of `RRF_K = 60`, exactly as this section said it would be. At
`RRF_K = 12` the same three settings solve to 200/74/9, the guard is unchanged,
and the argument for it is stronger rather than weaker.

---

## Task 19 — `RRF_K` was inherited, not derived — **DONE**

**The defect.** The salience band exists so that salience breaks ties and never
overturns relevance. That is not a property of the band alone: it holds only
while fusion spreads relevance across a result page by more than the band is
wide, `1 / 0.7 = 1.429x`. At `RRF_K = 60` it did not. Rank 1 to rank 10 spanned
`70/61 = 1.148x`, so across the *default* window

```
rank  1, salience 0.10 = 0.02393
rank 10, salience 1.00 = 0.02857   <- fresh junk outranks the stale right answer
rank 20, salience 1.00 = 0.02500   <- still inverted
```

The guarantee the band was introduced to provide was false throughout the page
every caller gets. 60 is the constant from the original RRF paper (Cormack,
Clarke & Buettcher, 2009), tuned for fusing TREC runs of thousands of documents;
a tenant's memory graph has tens of edges, and the number was carried over
rather than worked out.

**Why the existing test never caught it.**
`relevance_beyond_the_bands_width_can_never_be_overturned` asserts an
*implication* — if relevance differs by more than the band, salience cannot flip
it — which is true as arithmetic whatever `RRF_K` is. It never asked whether
fusion produces a gap that wide inside a window anyone sees. A correct test of
the wrong thing. It now reads `RRF_K` for its baseline instead of the literal
`0.016_393` (which was `1/61`, a stale copy of the old constant) and says in its
doc that it is only half the property.

**Derived, not guessed.** An RRF score is `1/(k + rank)` and never reads the
corpus, so this is solvable rather than measurable. With `L = DEFAULT_LIMIT` and
`F = SALIENCE_BAND_FLOOR`:

1. *Relevance must outspread the band across a page.* `(k + L)/(k + 1) > 1/F`,
   i.e. `k < (L·F − 1)/(1 − F)` = **20**. The window is the default and not
   `MAX_LIMIT` because the spread grows with the window: the short page is the
   binding constraint and satisfying it satisfies every longer one.
2. *Agreement must still beat being first in one list.* An edge both lists rank
   `r` scores `2/(k + r)` against `1/(k + 1)`, so agreement wins while
   `r < k + 2`; requiring that across the page gives `k > L − 2` = **8**.

`8 < k < 20`, and **12** is the integer maximising the smaller of the two
relative margins — the page outspreads the band by 1.185x, agreement outscores a
list-leader by 1.182x, against a real-valued optimum of 12.04. Both bounds are
18% away, which is the most either can be. The band's reach is rank 6: a fully
salient hit climbs over a completely faded one up to six places above it and no
further — inside the page, not across it.

**The band did not move, and here is why not.** Narrowing to `[0.9, 1.0]` also
satisfies (1) and was the other candidate. Rejected twice over: against
`RRF_K = 60` it clears the spread by 3.3%, one change to `DEFAULT_LIMIT` from
being false again, and it leaves the reported `score` exactly as flat as it
found it; against a corrected `RRF_K` it drops the band's reach from 6 ranks to
2, which is a re-ranker that cannot re-rank. Moving `k` fixes the inversion *and*
the flatness; narrowing the band fixes neither properly.

**The cutoff fractions had to move with it, and this was not optional.** A
fraction of the best fused score is a bound on rank, and the rank it solves to
scales with `k`. Left at 0.35, `Standard` — the *default* — would have kept a
single-list hit only to rank 6, and `the_limit_is_capped_however_much_a_caller_asks_for`
(60 corroborated edges, `limit = 500`) would have returned about 42 of the 50 it
must. No `k` rescues it: keeping 0.35 non-binding over that page needs `k > 30`,
and constraint (1) needs `k < 20`. So they were re-solved to name the same things
they named before:

```
                   k = 60            k = 12
Relaxed    0.10 -> 0.03   single-list to rank 204 (past CANDIDATE_DEPTH), as before
Standard   0.35 -> 0.15   single-list to rank 31, corroborated to 74  (was 27 / 114)
Strict     0.60 -> 0.60   no single-list hit at all; agreement to rank 9 (was 41)
```

`Standard`'s depth has to sit strictly between `DEFAULT_LIMIT` and `MAX_LIMIT` —
shallower and the default precision docks a page nobody asked it to, deeper and
it cannot bite at all — and 31 is near the middle of that range and within four
ranks of what the setting meant before. `Strict` keeps its number: the property
that makes it mean "the two lists must agree" is `f > 0.5`, and `f > 0.5` also
forces its agreement depth below `k + 2`, which is exactly the rank at which
agreement stops outscoring a list-leader. The tightest setting keeping only
agreements that beat a leader is the coherent reading of that coupling.

**The single-list guard is more necessary than it was, not less.** In single-list
mode the three settings now solve to 200/74/9. `Strict` at rank 9 is *inside*
`DEFAULT_LIMIT = 10`: without `lists_can_disagree`, a semantic-only search would
come back a hit short for a caller who asked for nothing at all, where at
`RRF_K = 60` only a caller asking for more than 41 could notice. Task 16's branch
was right and is now load-bearing at the default limit.

**The demo's relevance miss is not a ranking defect and is unchanged.**
Reproduced against the real `bge-base-en-v1.5` and the live `acme` tenant. The
query `who is Alice's manager?` returns, at `RRF_K = 60`, exactly the scores the
demo reported — `0.03278 / 0.03226 / 0.03174`, which is `2/61, 2/62, 2/63` — and
at `RRF_K = 12` returns `0.15385 / 0.14286 / 0.13333` in the *same order*.
Fusion is not the culprit, because **both candidate lists independently rank the
wrong edge first**:

- The *semantic* half puts `MEMBER_OF` ("Alice has worked at Acme Corporation
  for six years") above `REPORTS_TO` ("As of June, Alice reports directly to the
  Acme CTO, not to Bob"). That is a property of the embedding model and no value
  of `k` can reorder it.
- The *lexical* half puts it first too, but for no reason at all. All three
  candidate edges hang off the one matched node, so `exact`, `terms` and
  `name_length` tie across every row and the ORDER BY falls through to
  `edge.edge_id` — a UUID sort. `6879…` < `81fb…` < `ddc4…` and that is the
  whole of the ranking.

The two lists then agreed, so the wrong answer was scored as *corroborated*. The
ranking is doing exactly what it is told; what it is told is wrong. Not fixed
here — one anecdote does not justify contorting the ranking — but see the note
below, because the lexical degeneracy is a real and general problem.

**Tests.** In `src/store/search.rs`:
`the_fusion_outspreads_the_band_across_the_default_page` is the regression test
this task was missing — it fuses an actual `DEFAULT_LIMIT`-long page through
`reciprocal_rank_fusion` and asserts the observed spread beats the observed band
width, referencing the constants rather than restating them, so it fails if
`RRF_K` rises, if the band widens, or if `DEFAULT_LIMIT` drops below the band's
reach. Its RED against `RRF_K = 60` read *"relevance spreads only 1.1475x from
rank 1 to rank 10 while salience can move a score 1.4286x"*.
`agreement_anywhere_on_the_default_page_beats_being_first_in_one_list` pins the
other end of the derived interval and
`the_band_reaches_into_the_page_without_reaching_across_it` pins that the band
still does something (it also failed at 60, with reach = the whole page).
`the_cutoff_on_a_single_list_solves_to_a_fixed_rank_whatever_the_corpus` now pins
200/74/9. In `src/bin/calibrate.rs`,
`the_documented_standard_cutoff_really_does_fall_at_rank_31` additionally asserts
the depth lies between the default limit and the cap, so the next person to move
`RRF_K` is told what the number has to be rather than only what it was.

---

## Task 20 — modern-only made the server unreachable by every client that exists — **DONE**

**The defect, found against a real client.** `SUPPORTED_PROTOCOL_VERSIONS =
[V_2026_07_28]` with `legacy_session_mode(false)` and
`stateless_protocol_metadata_required(true)` is not a server that refuses old
clients. It is a server that *accepts* them and then fails every call.
Reproduced against the Devin CLI, which speaks `2025-11-25` through **the same
rmcp 3.1.2 this server uses**, and again from a throwaway rmcp client driven
through the default `serve()` lifecycle:

```
[probe] connecting as real-rmcp-client/1.2.3 requesting protocolVersion=2025-11-25
[probe] peer_info = ... protocol_version: ProtocolVersion("2026-07-28") ...
[probe] tools/list FAILED: Mcp error: -32602: Invalid params: request _meta is
        missing or has malformed required fields:
        io.modelcontextprotocol/protocolVersion,
        io.modelcontextprotocol/clientCapabilities
```

and with raw `curl`, the shape the report named:

```
initialize  -> HTTP 200 {"result":{"protocolVersion":"2026-07-28", ...}}  (no mcp-session-id)
tools/call  -> HTTP 400 {"error":{"code":-32020,"message":"Missing MCP-Protocol-Version
               header for request requiring per-request protocol metadata"}}
```

The handshake is answered with a version the client never asked for and cannot
speak, so it believes it is connected. `2025-11-25` is `rmcp`'s own
`ProtocolVersion::LATEST`, which is why this is not a niche case: it is the
default of every client built on this SDK.

**The fix is an escape hatch, not a reversal.** `MUNDU_PROTOCOL_COMPAT`, values
`modern-only` (**the unchanged default**) and `dual-era`. Nothing about a
deployment that says nothing changes; all 310 existing tests pass untouched,
including the three in `tests/server_http.rs` that assert modern-only refusals.

**What `dual-era` advertises, and where the list stops.**
`[2026-07-28, 2025-11-25, 2025-06-18]`, modern first so it stays this server's
preference. `2025-11-25` is the one proven necessary, above. `2025-06-18` is the
floor for a mechanical reason rather than a guess about who ships what: it is
the oldest revision that defines the `MCP-Protocol-Version` header, so it is the
oldest one in which a request *states* which era it belongs to. Below it `rmcp`
assumes `2025-03-26` for anything arriving without a header — the era would be a
guess, and a request that declared nothing at all would be served. `2025-03-26`
and `2024-11-05` are therefore not "older clients", they are "no statement", and
they are left out. (Those two plus the three above are the whole of what
`ProtocolVersion` exposes in rmcp 3.1.2.)

**`legacy_session_mode(true)`, and the measurement that decided it.** The doc
comment says `client_info()` falls back "only for legacy sessions", which reads
as *sessionless legacy writes lose their attribution*. That is not what happens,
and the truth is worse. Measured both ways against Postgres, same client
handshaking as `legacy-probe/9.9.9`:

```
legacy_session_mode(false)  writer_client = "rmcp/3.1.2"          <- fabricated
legacy_session_mode(true)   writer_client = "legacy-probe/9.9.9"  <- the handshake
```

Sessionless, `StreamableHttpService::peer_info_for_stateless_request` builds a
peer with `Implementation::default()`, which is `from_build_env()` **of the rmcp
crate** — so `client_info()` yields `rmcp/<the rmcp version this binary was
built against>` for every legacy client alike. Not a gap in the audit trail: a
plausible, uniform, false name in it, indistinguishable from a genuine
rmcp-based writer. Against a stated defence of "which client wrote this", that
is the one outcome worse than nothing, so the session is on.

Confirmed live on nodes *and* edges, over `curl` with no `_meta` anywhere:

```
        node_id        |      writer_client
-----------------------+--------------------------
 curl-legacy-factory   | curl-legacy-client/0.0.1
 curl-legacy-widget    | curl-legacy-client/0.0.1
 modern-under-dual-era | modern-curl-client/7.7.7
 rmcp-probe-entity     | real-rmcp-client/1.2.3
```

**The cost, stated plainly.** A legacy session lives in one process. Behind a
load balancer without sticky routing, a legacy client's second request gets
`404 Not Found: Session not found`. Modern clients are unaffected — `rmcp`
routes a `2026-07-28` request statelessly whatever the flag says, verified by
the modern row above coming back as JSON with `resultType: complete` while the
legacy rows came back over SSE with a session id.

**Why `Config::validate` does not refuse it.** Considered and rejected on the
signal, not on the risk. The hazard is real, but the bind address is a poor
witness for it: every container binds `0.0.0.0`, including the single-replica
deployment that is the main reason to want `dual-era`, while the deployment that
actually breaks is two replicas that may both be on loopback behind a local
proxy. A refusal keyed on `!bind.is_loopback()` would block the safe case and
miss the unsafe one. It is also a *loud* failure — a 404 naming the session, not
a silent one — which is the line every other refusal in `validate` is drawn on.
So startup warns instead, twice, and a third time when the listener is
non-loopback, naming the exact string an operator will see.

**Where the session manager had to change too.** `legacy_session_mode(true)` is
not a flag `NeverSessionManager` can honour — it answers `create_session` with
"session management is not supported" — so `router` mounts
`LocalSessionManager` under `dual-era` and `NeverSessionManager` under
`modern-only`. Per-request authentication is unaffected: `rmcp` injects the
current request's `http::request::Parts` into every message, session or not, so
the `Principal` is still re-derived from the `Authorization` header on each
call rather than remembered from the handshake. A stolen session id without a
credential still gets a 401.

**Still true afterwards, and worth knowing.** `serverInfo` in every handshake
reads `{"name":"rmcp","version":"3.1.2"}`, because `MunduTools::get_info()`
leaves `Implementation::default()` in place. Harmless — nothing keys off it —
but this server does not identify itself by name to any client, which is a
one-line fix somebody should take.

---

## Smaller notes, not worth their own task

- **Overlapping history intervals.** Two backfills into the same gap both close
  against the current active edge, so their `valid_until` intervals overlap. The
  current fact is always right; only the history chain is imprecise. Repairing
  it is an interval pass over rows that are already history.
- ~~**Blank `proof_excerpt` is accepted.**~~ **Done (Task 17.)** Refused in the
  store, in `create_edges`, beside the blank-relationship-type and future-dated
  guards rather than at the MCP boundary — the store is reachable by callers the
  MCP layer is not, and this is where every analogous guard already sits. The
  excerpt is trimmed and then measured: `MIN_PROOF_EXCERPT_CHARS = 3`. That
  floor is deliberately one that cannot bite. "Is this evidence?" is not a
  question a length can answer, and a bar set high enough to filter weak
  evidence would refuse honest writes in languages that say more per character
  than English — so the number sits below the shortest real quotation in the
  densest script the tokenizer reads (CJK, measured at 1.00 chars/token), and
  rules out only the degenerate case: a single character, or a full stop,
  standing in for a quotation. The schema carries `minLength`, which is
  ergonomics only — it counts spaces, and the store trims before it counts.
- ~~**Unbounded batch size.**~~ **Done (Task 17.)** `MAX_BATCH_ITEMS = 64` on
  `nodes`, `relationships`, `edge_ids` and one node's `aliases`, plus
  `MAX_DISPLAY_NAME_CHARS = 200` and `MAX_PROOF_EXCERPT_CHARS = 2000` on the
  individual texts. Enforced in the store — where `embed_documents` is actually
  called — and published as `maxItems`/`maxLength` so a model reads the limit
  before it generates rather than after it is refused. Measured on the real
  `bge-base-en-v1.5` rather than guessed: at the tokenizer's 512-position
  ceiling the forward pass costs **77 MB of peak RSS per item** (64 items:
  5.8 GB peak against a 0.9 GB idle process, 88 s of CPU), and the cost is
  linear in items, so the thousand-item call nothing refused before was ~77 GB.
  64 is where three unrelated arguments meet: a conversation yields a few dozen
  facts, it is above `MAX_LIMIT = 50` so a whole search's results can still be
  rated in one call, and it bounds the pass.
- ~~**Chunk inside the embedder.**~~ **Done (Task 18.)** `embed_documents` now
  runs a batch as however many forward passes it takes, each inside
  `MAX_CHUNK_POSITIONS = 4096` token positions — rows times the padded width of
  the widest row, which is what a pass actually costs. Measured at 64 items of
  8000 characters, every one of them at the tokenizer's 512-position ceiling:
  **peak RSS 5675 MiB before, 1174 MiB after** (4.8x), and the call got *faster*
  rather than slower, 83.6 s to 63.2 s, because a 64 x 512 rectangle spills
  every cache in the machine. Peak memory is now a property of the model instead
  of the caller's batch size, and `MAX_BATCH_ITEMS` stays exactly as it was: the
  cap bounds how much work one request may ask for, chunking bounds the memory
  inside it.
- **Why the budget counts positions and not items.** Short text is cheap — 64
  entity names are a 64 x 22 rectangle, a rounding error against the budget — so
  a batch of names stays in one pass and pays nothing at all, while a fixed item
  count small enough to bound the worst case above would have split *every*
  batch instead. Measured: four items a pass costs +38% wall clock at 64 x 100
  characters and saves not one byte of peak RSS, because short text was never
  the problem. The only measured cost of the budget as chosen is +6.5% at
  64 x 200 characters, which is four passes where there was one; 64 x 2000 came
  out 19% faster and 64 x 8000 29% faster.
- **The positions are estimated, not counted.** Characters plus two, capped at
  the truncation ceiling. No script produces more wordpieces than it has
  characters — the dense end, CJK, measures 1.00 characters a token, English
  5.4 — so the estimate can only be high, which costs a few more passes than
  English prose strictly needs and can never cost the memory bound. Tokenising
  first to get the exact figure would mean tokenising every batch twice, since
  each pass already tokenises its own chunk to pad it to its own width. If the
  extra passes ever matter, that is the knob: exact token counts would fit
  around 1.4x more English text into the same pass.
- **`calibrate` still chunks its own corpus** at `EMBED_BATCH = 16`. Redundant
  now rather than wrong — the embedder would split the batch anyway — and left
  alone deliberately.
- **A character cap cannot bound a token count.** Measured: English prose
  tokenizes at 5.4 chars/token, CJK at 1.00. So a 2000-character English proof
  is ~370 tokens and is embedded whole, while a 2000-character CJK proof is
  ~2000 tokens and the tokenizer silently keeps the first 512. That truncation
  predates Task 17 and is unchanged by it — the text is stored whole and only
  the vector is short — but it means the same cap buys different guarantees in
  different scripts. Fixing it properly means measuring tokens rather than
  characters, which puts the tokenizer on the write path.
- **Model switching needs a re-embed script.** Source text is retained so vectors
  are always reproducible, but nothing automates it, and `embedding_model` is
  recorded per row precisely so a mixed state is detectable.
- **A known test flake.** A pool-acquire timeout can fail integration tests when
  the database has just cold-started; the retry passes. Each test binary builds
  its own pool and they run in parallel.
- ~~**The lexical list can be a UUID sort wearing a ranking's clothes.**~~
  **Done (Task 20.)** Fixed by letting a list *say* it cannot rank, rather than
  by finding it something else to rank on. `reciprocal_rank_fusion` no longer
  takes `Vec<Uuid>` and counts positions — a position cannot express a tie — but
  `Vec<RankedEdge>`, where each list states its own rank. Both candidate queries
  state theirs with SQL's `RANK() OVER`: **competition ranking**, ties sharing a
  rank and the next distinct value skipping. The lexical window runs over the
  meaningful keys *only*, deliberately excluding `edge.edge_id`, so edges
  sharing a matched node are paid identically and the semantic list decides
  between them. `edge_id` keeps both of its other jobs — which rows the depth
  limit keeps and what order they are emitted in — so one dataset is still one
  answer on every replica; it just no longer reaches the rank *value* fusion
  scores.
- **`RRF_K` is unchanged at 12, and the derivation was re-checked rather than
  assumed.** Skipping ranks look as though they could run past the end of a
  list — 1, 1, 3, 4, 4, 6 reaches 6 in six rows — but they cannot: competition
  rank is one plus the number of rows strictly ahead, at most every row above a
  row is strictly ahead of it, so no row ranks deeper than its own position and
  the deepest rank in a list of `n` is `n`. That is exactly the `1..=L` the two
  inequalities were solved over. Inequality 2 (agreement beats a list leader
  anywhere on the page) only gets easier, because every `1/(k + rank)` is now at
  least the `1/(k + position)` it was solved with. Inequality 1 (the page
  outspreads the band) is stated over one *consistent ordering of distinct
  hits*, which ties are not: a tie compresses the spread by making two hits
  equally relevant, and ordering equally relevant hits is the band's job — the
  same reading that already covered two lists disagreeing, which flattens a page
  further than any tie does. `competition_ranking_never_ranks_a_row_deeper_than_its_position`
  checks it over every tie pattern a default page can hold.
- **The semantic half had the same flaw and got the same treatment.** Far
  smaller: a cosine is a property of the edge, so that list ties only when two
  edges embed to the same point — two facts quoting one sentence as their proof.
  It cost nothing to close and "a position is not a rank" is easier to keep true
  without an exception. The window sits *outside* the vector query's `LIMIT`,
  in a subselect: window functions are evaluated before `LIMIT`, so writing it
  beside the `ORDER BY` would sort every active edge in the tenant and throw
  away the HNSW scan `VECTOR_RECALL_SESSION_SETUP` exists to guarantee.
- **A per-edge lexical signal was measured and left out.** The obvious next
  move — count query terms in the edge's own `proof_excerpt` and
  `relationship_type`, so edges sharing a node can be told apart on their own
  merits — was measured against the live `acme` tenant over eight natural
  queries, fusing the real `bge-base-en-v1.5` semantic ranks. Competition
  ranking alone took precision@1 from **3/8 to 5/8**. Adding the per-edge count
  on top separated a lexical tie in **6 of 8** queries and changed the final
  order in **0 of 8**: it bought nothing. It is not a neutral nothing, either.
  It reads `proof_excerpt`, which is the text `proof_vector` embeds, so every
  time it fired it reproduced the semantic list's own opinion — RRF pays for
  *independent* agreement, and two readings of one sentence are not two
  opinions. Worse, on `what database does Phoenix use?` it demolished a clean
  0.1538/0.1484 separation into an exact 0.1484/0.1484 tie that then had to be
  broken by `edge_id`, which is the coin flip the task removed, moved one level
  down. Substring matching also misfires: `work` matches `worked at`, so
  `what does Alice work on?` gets employment promoted over the project. If a
  per-edge signal is revisited it should be one the semantic half cannot see —
  `relationship_type` alone is the only such component here, and it fires only
  when a query happens to use the same word as a SCREAMING_SNAKE identifier an
  agent coined.
- **Two hits tied on relevance can still swap between two searches.** Older than
  Task 20 and surfaced by it, because exact relevance ties are now the normal
  case for a single-node query. `effective_salience` returns `REAL`; one ULP at
  0.99999 is about a fifth of a second of decay at the 27.1-day half-life, so
  two edges written milliseconds apart round to the same `f32` most of the time
  and to adjacent ones for a few milliseconds in every fifth of a second — and
  in that window they swap. It is bounded (one place, between two hits the query
  genuinely cannot separate) and it needs a migration to fix properly, since the
  decay curve must not be implemented twice. `precision_changes_nothing_a_single_list_search_can_show`
  pins the salience of its two tied edges so it is not measuring this.
- **The score field is comparable within a response and not across two.** Still
  true, and still documented on `SearchHit::score`, but the numbers a model sees
  are far more informative than they were: adjacent ranks now differ by 7.7%
  rather than 1.6%. Nothing depends on this, and nothing should.
