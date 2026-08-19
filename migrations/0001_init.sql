-- mundu: initial schema.
--
-- Vector columns are 768-wide to match bge-base-en-v1.5. The source text for
-- every vector is retained (proof_excerpt, display_name), so vectors are always
-- reproducible: switching models is an ALTER plus a scripted re-embed, and the
-- embedding_model column below makes mixed-model state detectable rather than
-- silently comparing vectors across incompatible spaces.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Relationship ontology
-- ---------------------------------------------------------------------------

-- Cardinality drives conflict resolution: a `one` type may have at most one
-- active edge per (source, type), so a newer fact supersedes the incumbent. A
-- `many` type never supersedes implicitly, because "A USES B" and "A USES C"
-- are both legitimately true at once.
CREATE TYPE relationship_cardinality AS ENUM ('one', 'many');

-- The type set is open: agents may introduce new types, which the server
-- registers on first use. Coercion to an existing equivalent happens before
-- insert, so this table stays the canonical vocabulary.
CREATE TABLE relationship_type (
    name          TEXT PRIMARY KEY,
    cardinality   relationship_cardinality NOT NULL DEFAULT 'many',
    description   TEXT        NOT NULL DEFAULT '',
    name_vector   VECTOR(768),
    auto_registered BOOLEAN   NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO relationship_type (name, cardinality, description) VALUES
    ('USES',        'many', 'Subject makes use of the object'),
    ('DEPENDS_ON',  'many', 'Subject requires the object to function'),
    ('REPORTS_TO',  'one',  'Subject reports to exactly one object at a time'),
    ('OWNS',        'many', 'Subject owns the object'),
    ('MEMBER_OF',   'many', 'Subject belongs to the object'),
    ('PART_OF',     'one',  'Subject is a component of exactly one whole'),
    ('AUTHORED_BY', 'many', 'Subject was created by the object'),
    ('LOCATED_IN',  'one',  'Subject resides in exactly one location'),
    ('PREFERS',     'many', 'Subject has a stated preference for the object'),
    ('RELATED_TO',  'many', 'Untyped association; use a precise type when possible');

-- Tier 2 of type coercion: exact synonym lookups after lexical normalization.
CREATE TABLE relationship_type_alias (
    alias     TEXT PRIMARY KEY,
    canonical TEXT NOT NULL REFERENCES relationship_type (name) ON DELETE CASCADE
);

INSERT INTO relationship_type_alias (alias, canonical) VALUES
    ('UTILIZES',      'USES'),
    ('USING',         'USES'),
    ('CONSUMES',      'USES'),
    ('REQUIRES',      'DEPENDS_ON'),
    ('NEEDS',         'DEPENDS_ON'),
    ('DEPENDS_UPON',  'DEPENDS_ON'),
    ('BELONGS_TO',    'MEMBER_OF'),
    ('COMPONENT_OF',  'PART_OF'),
    ('CREATED_BY',    'AUTHORED_BY'),
    ('WRITTEN_BY',    'AUTHORED_BY'),
    ('BASED_IN',      'LOCATED_IN'),
    ('LIKES',         'PREFERS');

-- Direction guard for tier 3 (embedding-similarity coercion).
--
-- Inverse relations sit close together in embedding space but mean the
-- opposite thing. Coercing MANAGES onto REPORTS_TO would not merely blur the
-- ontology, it would invert the direction of every affected fact while leaving
-- a proof_excerpt that still reads correctly. Any similarity match whose
-- candidate appears here paired with the incumbent is refused, and the incoming
-- type is registered as new instead.
CREATE TABLE relationship_inverse_pair (
    type_a TEXT NOT NULL,
    type_b TEXT NOT NULL,
    PRIMARY KEY (type_a, type_b)
);

INSERT INTO relationship_inverse_pair (type_a, type_b) VALUES
    ('REPORTS_TO',  'MANAGES'),
    ('OWNS',        'OWNED_BY'),
    ('DEPENDS_ON',  'REQUIRED_BY'),
    ('PART_OF',     'HAS_PART'),
    ('MEMBER_OF',   'HAS_MEMBER'),
    ('AUTHORED_BY', 'AUTHORED'),
    ('USES',        'USED_BY'),
    ('LOCATED_IN',  'CONTAINS');

-- ---------------------------------------------------------------------------
-- Entities
-- ---------------------------------------------------------------------------

CREATE TYPE edge_status AS ENUM ('active', 'superseded', 'archived');

-- tenant_id scopes every read and write. It is derived from a validated access
-- token claim and is never a tool argument: if a model could set it, untrusted
-- proof text could talk it into reading another tenant's memory.
CREATE TABLE node (
    tenant_id       TEXT        NOT NULL,
    -- Server-minted from display_name. Agents never supply ids; they receive
    -- them from upsert_node and thread them back as explicit handles.
    node_id         TEXT        NOT NULL,
    display_name    TEXT        NOT NULL,
    aliases         TEXT[]      NOT NULL DEFAULT '{}',
    name_vector     VECTOR(768),
    embedding_model TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Writer provenance, taken from _meta clientInfo on the originating
    -- request. Available per-request under 2026-07-28 without any session.
    writer_client   TEXT,
    writer_principal TEXT,

    -- Tombstone redirect. Entity resolution must never delete a node: agents
    -- hold node_ids across stateless calls, so a deleted id would dangle in an
    -- agent's context. Merged nodes stay resolvable.
    merged_into     TEXT,

    PRIMARY KEY (tenant_id, node_id),
    CONSTRAINT node_merge_not_self CHECK (merged_into IS DISTINCT FROM node_id),
    FOREIGN KEY (tenant_id, merged_into) REFERENCES node (tenant_id, node_id)
);

CREATE INDEX node_aliases_idx ON node USING GIN (aliases);
CREATE INDEX node_display_name_idx ON node (tenant_id, lower(display_name));
CREATE INDEX node_name_vector_idx ON node
    USING hnsw (name_vector vector_cosine_ops)
    WHERE merged_into IS NULL;

CREATE TABLE edge (
    tenant_id             TEXT        NOT NULL,
    edge_id               UUID        NOT NULL,
    source_node_id        TEXT        NOT NULL,
    target_node_id        TEXT        NOT NULL,

    relationship_type     TEXT        NOT NULL REFERENCES relationship_type (name),
    -- Exactly what the agent wrote, before normalization or coercion. Without
    -- it, a bad coercion is undebuggable and irreversible.
    raw_relationship_type TEXT        NOT NULL,

    proof_excerpt         TEXT        NOT NULL,
    proof_vector          VECTOR(768) NOT NULL,
    embedding_model       TEXT        NOT NULL,

    -- Lazy salience: value at last write plus when that write happened, decayed
    -- in the query. No nightly full-table UPDATE, no leader election.
    --
    -- Salience ranks but never gates. Nothing is excluded from search for being
    -- old, because archival would remove an edge from the vector index, making
    -- it unretrievable, therefore unratable, therefore unable to recover - a
    -- one-way door. Low salience sinks in ordering and can be rehabilitated by
    -- a single positive rating.
    salience_base         REAL        NOT NULL DEFAULT 1.0,
    salience_updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Bi-temporal. valid_* is when the fact held in the world; recorded_at is
    -- when we came to believe it. valid_from is agent-supplied but rejected if
    -- it is meaningfully in the future: it orders supersession, so a future
    -- date would win every conflict permanently. recorded_at is server-set and
    -- never agent-writable.
    valid_from            TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until           TIMESTAMPTZ,
    recorded_at           TIMESTAMPTZ NOT NULL DEFAULT now(),

    status                edge_status NOT NULL DEFAULT 'active',
    superseded_by_edge_id UUID,

    -- Soft dispute. Set once negative ratings reach quorum from distinct
    -- principals. A disputed edge is still returned by search, demoted below
    -- the normal salience band and flagged in output, so the agent can judge it
    -- from proof and provenance.
    --
    -- Hard archival is deliberately NOT reachable from here. Everything in
    -- proof_excerpt is untrusted text, so an agent-triggerable delete would
    -- hand injection an erasure primitive - and an erased fact is invisible in
    -- search output, unlike a poisoned one. Archiving is an operator action.
    disputed_at           TIMESTAMPTZ,

    writer_client         TEXT,
    writer_principal      TEXT,

    PRIMARY KEY (tenant_id, edge_id),
    FOREIGN KEY (tenant_id, source_node_id) REFERENCES node (tenant_id, node_id),
    FOREIGN KEY (tenant_id, target_node_id) REFERENCES node (tenant_id, node_id),
    FOREIGN KEY (tenant_id, superseded_by_edge_id) REFERENCES edge (tenant_id, edge_id),
    CONSTRAINT edge_salience_range CHECK (salience_base >= 0.0 AND salience_base <= 1.0),
    CONSTRAINT edge_valid_range CHECK (valid_until IS NULL OR valid_until >= valid_from)
);

CREATE INDEX edge_source_idx ON edge (tenant_id, source_node_id) WHERE status = 'active';
CREATE INDEX edge_target_idx ON edge (tenant_id, target_node_id) WHERE status = 'active';
CREATE INDEX edge_type_idx ON edge (tenant_id, relationship_type) WHERE status = 'active';
CREATE INDEX edge_writer_idx ON edge (tenant_id, writer_client);
CREATE INDEX edge_disputed_idx ON edge (tenant_id) WHERE disputed_at IS NOT NULL;
CREATE INDEX edge_valid_from_idx ON edge (tenant_id, source_node_id, relationship_type, valid_from)
    WHERE status = 'active';

CREATE INDEX edge_proof_vector_idx ON edge
    USING hnsw (proof_vector vector_cosine_ops)
    WHERE status = 'active';

-- Append-only feedback log. Ratings mutate edge.salience_base, but the raw
-- signal is retained so salience can be recomputed if the decay model changes.
CREATE TABLE edge_rating (
    rating_id     BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id     TEXT        NOT NULL,
    edge_id       UUID        NOT NULL,
    utility_score REAL        NOT NULL,
    rated_by      TEXT,
    rated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (tenant_id, edge_id) REFERENCES edge (tenant_id, edge_id),
    CONSTRAINT edge_rating_range CHECK (utility_score >= -1.0 AND utility_score <= 1.0)
);

CREATE INDEX edge_rating_edge_idx ON edge_rating (tenant_id, edge_id);

-- Effective salience under exponential decay. Takes the evaluation instant as
-- an argument rather than calling now() internally, so it stays IMMUTABLE and
-- usable in index and constraint contexts.
CREATE FUNCTION effective_salience(
    base           REAL,
    updated_at     TIMESTAMPTZ,
    half_life_days REAL,
    evaluated_at   TIMESTAMPTZ
) RETURNS REAL
LANGUAGE SQL
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT (
        base * POWER(
            0.5::DOUBLE PRECISION,
            GREATEST(EXTRACT(EPOCH FROM (evaluated_at - updated_at)), 0.0) / 86400.0
                / GREATEST(half_life_days, 0.000001)
        )
    )::REAL
$$;
