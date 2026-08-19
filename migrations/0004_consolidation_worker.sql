-- mundu: the state and the record of the consolidation worker.
--
-- Entity resolution is biased toward minting a duplicate in three places, and
-- every one of those was justified by the same sentence: fragmentation is
-- recoverable by a later merge, over-merge is not. 0003 built the later merge
-- and made it reversible. This is the thing that calls it without being asked.
--
-- That is a delicate thing to automate, so what the automation may do is
-- deliberately narrower than what an operator may do, and everything it does is
-- written down here rather than only logged. Three tables:
--
--   consolidation_state      one row per tenant: when the last run finished and
--                            what the graph measured then. The two trigger
--                            bounds are computed against it.
--   consolidation_run        one row per run. What fired it, what it saw, how
--                            long it took, which verifier judged it.
--   consolidation_decision   one row per candidate pair the run considered,
--                            merged or declined, with the reason.
--
-- The decision table is the point. A worker that merges entities and leaves
-- only a log line behind is a worker whose behaviour an operator can only
-- reconstruct from whatever log retention happens to be. "What did the worker
-- do last night, and what did it nearly do" has to be a query.
--
-- Nothing here is agent-reachable. Consolidation is an operator and server
-- concern with no MCP tool in front of it, for the reason 0003 gives about
-- merging: an agent-triggerable merge is an erasure primitive handed to prompt
-- injection, and one that runs on a timer is worse, not better.

-- ---------------------------------------------------------------------------
-- Per-tenant trigger state
-- ---------------------------------------------------------------------------

-- Per tenant rather than global, and the choice matters. Growth is something a
-- tenant does: one busy customer writing ten thousand nodes says nothing
-- whatsoever about whether anybody else's graph has accumulated duplicates. A
-- global counter would make every tenant's consolidation schedule a function of
-- the noisiest tenant's write rate — work done for tenants that needed none,
-- and, worse, a quiet tenant's own 40% growth invisible underneath it.
CREATE TABLE consolidation_state (
    tenant_id   TEXT        PRIMARY KEY,

    -- When the last run *finished*, taken from the clock the run was given
    -- rather than from `now()`. The interval bound is measured against this in
    -- the application, so both ends of the comparison have to come from the
    -- same clock: the server and the database are not the same machine, and
    -- 0003's suite already had to be corrected for exactly that skew.
    last_run_at TIMESTAMPTZ NOT NULL,

    -- What the tenant measured when that run started, not when it ended. Work
    -- written *during* a run has not been consolidated, so it counts toward the
    -- next trigger rather than being absorbed by the run that could not see it.
    --
    -- Both counts are over every row, tombstones and superseded edges included.
    -- Nothing in this schema is ever deleted, so counted this way they are
    -- monotonically non-decreasing and "growth" cannot come out negative — and
    -- a merge, which retires a node without removing its row, does not silently
    -- reset a tenant's baseline downward.
    node_count  BIGINT      NOT NULL,
    edge_count  BIGINT      NOT NULL,

    CONSTRAINT consolidation_state_counts_nonnegative
        CHECK (node_count >= 0 AND edge_count >= 0)
);

-- ---------------------------------------------------------------------------
-- One row per run
-- ---------------------------------------------------------------------------

-- Which bound fired. `first_run` is its own value rather than being folded into
-- `growth`: a tenant nobody has consolidated before has no baseline to have
-- grown from, and recording that as growth would claim a measurement that was
-- never taken.
CREATE TYPE consolidation_trigger AS ENUM ('first_run', 'growth', 'interval');

CREATE TABLE consolidation_run (
    run_id            UUID        PRIMARY KEY,
    tenant_id         TEXT        NOT NULL,

    triggered_by      consolidation_trigger NOT NULL,

    -- Which verifier judged this run's pairs. Recorded because the verifier is
    -- a seam: today it is the lexical guard the inline resolver already uses,
    -- and if an LLM verifier is ever added, every decision below has to remain
    -- attributable to the thing that actually made it.
    verifier          TEXT        NOT NULL,

    -- Both from the clock the run was given, for the reason `last_run_at`
    -- gives. `finished_at` is set even when the run merged nothing.
    started_at        TIMESTAMPTZ NOT NULL,
    finished_at       TIMESTAMPTZ NOT NULL,

    -- The counts this run measured, so the next run's growth arithmetic is
    -- auditable rather than merely repeatable.
    node_count        BIGINT      NOT NULL,
    edge_count        BIGINT      NOT NULL,

    -- How many live nodes the run actually looked at, and whether that was all
    -- of them. A run is bounded so it cannot starve the request path, so on a
    -- large tenant it sees the newest slice; `scan_truncated` is how an
    -- operator tells "found nothing" from "did not look everywhere".
    nodes_scanned     INTEGER     NOT NULL,
    scan_truncated    BOOLEAN     NOT NULL,

    pairs_considered  INTEGER     NOT NULL,
    pairs_merged      INTEGER     NOT NULL,

    CONSTRAINT consolidation_run_span CHECK (finished_at >= started_at),
    CONSTRAINT consolidation_run_tallies CHECK (
        nodes_scanned >= 0
        AND pairs_considered >= 0
        AND pairs_merged >= 0
        AND pairs_merged <= pairs_considered
    )
);

-- The operator's question is "what happened on this tenant, most recently
-- first", so that is the index.
CREATE INDEX consolidation_run_tenant_idx
    ON consolidation_run (tenant_id, started_at DESC);

-- ---------------------------------------------------------------------------
-- One row per candidate pair considered
-- ---------------------------------------------------------------------------

CREATE TYPE consolidation_verdict AS ENUM ('merged', 'declined');

-- Everything the run looked at, not only what it acted on. A declined pair is
-- the more interesting row of the two: it is the worker saying "these two look
-- like one entity and I would not fuse them", which is the whole safety
-- argument made visible. It is also what an operator needs to find the merges
-- the worker is deliberately not making, and to decide them by hand.
CREATE TABLE consolidation_decision (
    run_id        UUID        NOT NULL
        REFERENCES consolidation_run (run_id) ON DELETE CASCADE,
    tenant_id     TEXT        NOT NULL,

    -- Ordered by node_id, so one pair is one row however the scan reached it.
    -- Which of the two won a merge is not recorded here; it is in `node_merge`,
    -- reachable through `merge_id`, and duplicating it would give an unmerge
    -- two sources of truth about the same fact.
    node_a_id     TEXT        NOT NULL,
    node_b_id     TEXT        NOT NULL,

    -- The cosine the run measured, and the bar that cosine had to clear. Both,
    -- because the bar moves with the names: a pair with a short name on either
    -- side is held to the stricter of the two thresholds, and a row recording
    -- only "0.925, declined" would not say why.
    similarity    DOUBLE PRECISION NOT NULL,
    bar           DOUBLE PRECISION NOT NULL,

    verdict       consolidation_verdict NOT NULL,

    -- A short stable code, not prose. TEXT rather than an enum because the
    -- verifier is a seam and a later verifier will have reasons this one cannot
    -- name; a closed type here would make adding one a migration, and a
    -- migration is exactly the wrong amount of friction on writing down why
    -- something was refused.
    reason        TEXT        NOT NULL,

    -- The `node_merge` row this decision produced, for a merged pair.
    --
    -- Deliberately not a foreign key. This table is an operational log of what
    -- the worker saw and decided; `node_merge` is the graph's own audit trail.
    -- Pointing one at the other would make the graph's records undeletable by
    -- anything that does not also understand the log, and would tie the
    -- retention of the two together when they answer different questions.
    merge_id      UUID,

    PRIMARY KEY (run_id, node_a_id, node_b_id),
    CONSTRAINT consolidation_decision_pair_ordered CHECK (node_a_id < node_b_id),
    CONSTRAINT consolidation_decision_merge_id_iff_merged CHECK (
        (verdict = 'merged') = (merge_id IS NOT NULL)
    )
);

CREATE INDEX consolidation_decision_tenant_idx
    ON consolidation_decision (tenant_id, verdict);
