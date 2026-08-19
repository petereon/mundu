-- mundu: the audit trail a merge leaves behind.
--
-- `merged_into` has been in the schema since 0001 and was read in three places
-- — resolution follows it, search and suggestion exclude it — but nothing ever
-- wrote it. Every deliberate bias in entity resolution toward minting a
-- duplicate rather than fusing two entities was justified by the same sentence:
-- fragmentation is recoverable by a later merge, over-merge is not. There was
-- no later merge, so the recoverable error was not recoverable and duplicates
-- accumulated with no remedy.
--
-- These two tables are what makes the remedy reversible, and reversibility is
-- the precondition for anything doing it unattended. A merge re-points edges,
-- retires some of them, and unions two alias lists; all three are lossy in the
-- same way, in that the previous value is simply overwritten. So the previous
-- value is written down here first.
--
-- Nothing in this file is agent-reachable. Merging is a store-layer operation
-- with no MCP tool in front of it, for the reason `rating.rs` gives about
-- archival: everything in `proof_excerpt` is text some other agent wrote, and a
-- semi-destructive operation an agent can trigger is an erasure primitive
-- handed to prompt injection.

-- ---------------------------------------------------------------------------
-- One row per merge
-- ---------------------------------------------------------------------------

CREATE TABLE node_merge (
    merge_id            UUID        PRIMARY KEY,
    tenant_id           TEXT        NOT NULL,

    -- Which node lost and which won. Both are foreign keys, because neither is
    -- ever deleted: the loser survives as the tombstone this row explains.
    loser_node_id       TEXT        NOT NULL,
    winner_node_id      TEXT        NOT NULL,

    -- The loser's names as they stood at merge time. `display_name` and
    -- `aliases` are not moved off the loser — a merge adds, it does not strip —
    -- so these are recoverable from the tombstone today. They are recorded
    -- anyway, because a second merge onto the same loser, or an operator's
    -- edit, would make the tombstone's row a later reading rather than a record
    -- of what this merge acted on.
    loser_display_name  TEXT        NOT NULL,
    loser_aliases       TEXT[]      NOT NULL DEFAULT '{}',

    -- Exactly the surface forms this merge appended to the winner: the loser's
    -- names minus whatever the winner already answered to. The difference
    -- between this and `loser_aliases` is the whole point — an undo that
    -- removed every name the loser had would strip names the winner carried
    -- before the merge and would have kept after it.
    aliases_added       TEXT[]      NOT NULL DEFAULT '{}',

    -- When, and on whose authority. `merged_at` is server-set. The writer
    -- columns mirror the pair `node` and `edge` already carry, and come from
    -- the originating request rather than from an argument.
    merged_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    merged_by_client    TEXT,
    merged_by_principal TEXT,

    FOREIGN KEY (tenant_id, loser_node_id)  REFERENCES node (tenant_id, node_id),
    FOREIGN KEY (tenant_id, winner_node_id) REFERENCES node (tenant_id, node_id),
    -- The same rule `node_merge_not_self` states on `node`, restated here so a
    -- nonsensical audit row is unrepresentable rather than merely unwritten.
    CONSTRAINT node_merge_audit_not_self CHECK (loser_node_id <> winner_node_id)
);

CREATE INDEX node_merge_loser_idx  ON node_merge (tenant_id, loser_node_id);
CREATE INDEX node_merge_winner_idx ON node_merge (tenant_id, winner_node_id);

-- ---------------------------------------------------------------------------
-- One row per edge the merge changed
-- ---------------------------------------------------------------------------

-- The prior values, never the new ones. The new ones are readable from `edge`;
-- these are the only copy of what was there before.
--
-- Every column a merge can write is here, not only the endpoints. Re-pointing
-- can put two edges that were legally active on two different source nodes onto
-- one, which breaks the `one`-cardinality invariant `create_edges` maintains, so
-- the merge resolves that contest the same way the write path does — and the
-- edge it retires may be one it never re-pointed, because that edge already hung
-- off the winner. Recording only the endpoints would leave exactly that row
-- unrestorable.
CREATE TABLE node_merge_edge (
    merge_id                       UUID        NOT NULL
        REFERENCES node_merge (merge_id) ON DELETE CASCADE,
    tenant_id                      TEXT        NOT NULL,
    edge_id                        UUID        NOT NULL,

    previous_source_node_id        TEXT        NOT NULL,
    previous_target_node_id        TEXT        NOT NULL,
    previous_status                edge_status NOT NULL,
    previous_valid_until           TIMESTAMPTZ,
    previous_superseded_by_edge_id UUID,

    PRIMARY KEY (merge_id, edge_id),
    FOREIGN KEY (tenant_id, edge_id) REFERENCES edge (tenant_id, edge_id)
);

CREATE INDEX node_merge_edge_edge_idx ON node_merge_edge (tenant_id, edge_id);
