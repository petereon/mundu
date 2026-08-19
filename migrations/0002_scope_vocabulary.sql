-- mundu: give the relationship vocabulary a tenant scope.
--
-- `relationship_type` was the one agent-writable table in this schema with no
-- tenant_id. Agents may coin types, coined types auto-register, and every
-- tenant then read the result: a verb derived from untrusted text in one
-- tenant's conversation became part of every other tenant's vocabulary, and the
-- table grew without bound on any single tenant's say-so.
--
-- The vocabulary is now in two halves, distinguished by tenant_id:
--
--   ''        the curated set. Seeded here, operator-maintained, readable by
--             every tenant and written by none: coercion registers into the
--             caller's own scope, and refuses outright to act for a caller
--             whose tenant is empty, so '' is not a scope any agent can be in.
--   <tenant>  that tenant's private vocabulary. Auto-registration lands here,
--             and nothing outside the tenant can see it.
--
-- Coercion reads `tenant_id IN ('', <caller>)` and prefers the curated row when
-- a name exists in both, so the curated vocabulary stays authoritative while a
-- tenant can still say something new.

-- ---------------------------------------------------------------------------
-- The vocabulary itself
-- ---------------------------------------------------------------------------

-- DEFAULT '' rather than a backfill: the ten rows already in this table are the
-- curated set, and '' is exactly where they belong.
--
-- embedding_model mirrors the column `node` has carried since 0001. Without it,
-- two models' vectors sit in one column and nothing can tell them apart, while
-- cosines between them are meaningless numbers that still sort — a wrong answer
-- with no symptom. Nullable, because the seeded rows have no vector yet: which
-- model embeds them is a runtime choice, so the server backfills both columns
-- together on first use.
ALTER TABLE relationship_type
    ADD COLUMN tenant_id       TEXT NOT NULL DEFAULT '',
    ADD COLUMN embedding_model TEXT;

-- Every vector already in this column was written when there was nowhere to
-- record what produced it, so its provenance is not merely unknown, it is
-- unknowable. Coercion now refuses to compare a vector it cannot attribute, so
-- leaving these in place would make each of those rows permanently unreachable
-- by similarity while still looking populated. Discarding them costs nothing:
-- the text they were derived from is the row's own name, and the server
-- re-embeds any row it finds empty. On a fresh database this updates nothing.
UPDATE relationship_type SET name_vector = NULL WHERE name_vector IS NOT NULL;

-- Both foreign keys point at the primary key that is about to change, so they
-- come off first and go back on below against the composite key.
ALTER TABLE edge                    DROP CONSTRAINT edge_relationship_type_fkey;
ALTER TABLE relationship_type_alias DROP CONSTRAINT relationship_type_alias_canonical_fkey;

-- Scope first in the key, so one tenant's vocabulary is one contiguous range.
ALTER TABLE relationship_type DROP CONSTRAINT relationship_type_pkey;
ALTER TABLE relationship_type ADD  CONSTRAINT relationship_type_pkey
    PRIMARY KEY (tenant_id, name);

-- The lookup coercion actually performs: one name, two scopes
-- (`name = $2 AND tenant_id IN ('', $1)`). The primary key is tenant-first and
-- would have to be probed twice by prefix; this answers it as two adjacent
-- index entries.
CREATE INDEX relationship_type_name_scope_idx ON relationship_type (name, tenant_id);

-- ---------------------------------------------------------------------------
-- Aliases: curated, and only ever curated
-- ---------------------------------------------------------------------------

-- Aliases are operator-written — there is no agent-reachable path that inserts
-- one — and they exist to route a synonym onto the canonical vocabulary. So the
-- referencing half of the composite key is not data, it is the constant ''.
--
-- GENERATED rather than DEFAULT '' precisely because it is not data: a default
-- can be overridden by naming the column in an INSERT, and this one must not be
-- overridable at all. Postgres refuses any write to it, so "an alias resolves to
-- a curated type" is enforced by the database rather than asserted in a comment.
-- A tenant-private alias is then unrepresentable, which is the property wanted:
-- an alias that could reach a private type would leak that type's name to
-- whoever triggered it.
ALTER TABLE relationship_type_alias
    ADD COLUMN canonical_tenant TEXT NOT NULL GENERATED ALWAYS AS ('') STORED;

ALTER TABLE relationship_type_alias
    ADD CONSTRAINT relationship_type_alias_canonical_fkey
    FOREIGN KEY (canonical_tenant, canonical)
        REFERENCES relationship_type (tenant_id, name) ON DELETE CASCADE;

-- `relationship_inverse_pair` needs no change. It has no foreign key into
-- relationship_type — see 0001 — because it deliberately records pairs whose
-- other half is not registered yet: MANAGES is in it precisely so that the
-- first agent to write MANAGES is not folded onto REPORTS_TO. It is curated,
-- global, and read only by the tier-3 direction guard, which compares names.

-- ---------------------------------------------------------------------------
-- Edges
-- ---------------------------------------------------------------------------

-- An edge in tenant T may legitimately reference a curated type or one of T's
-- own, so `relationship_type` alone no longer identifies a row. Carrying the
-- scope beside it keeps the reference a real foreign key rather than a name the
-- application promises to have checked: this is a memory store, and the one
-- thing it must not do is quietly hold facts that point at nothing.
--
-- DEFAULT '' costs one small column and matches the common case, a curated type.
ALTER TABLE edge
    ADD COLUMN relationship_type_tenant TEXT NOT NULL DEFAULT '';

ALTER TABLE edge
    ADD CONSTRAINT edge_relationship_type_fkey
    FOREIGN KEY (relationship_type_tenant, relationship_type)
        REFERENCES relationship_type (tenant_id, name);
