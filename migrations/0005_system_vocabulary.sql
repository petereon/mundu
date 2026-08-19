-- Relationship types for describing a computer system.
--
-- The original ten describe people and organisations: who reports to whom, who
-- owns what, who wrote it. Used against a real architecture graph they run out
-- almost immediately -- in the corpus that motivated this migration, RELATED_TO
-- (the untyped fallback, whose own description says "use a precise type when
-- possible") accounted for 30 of 100 facts. Every one of the types below was
-- chosen by reading those thirty and asking what relation the proof was
-- actually asserting, rather than by imagining what a systems vocabulary ought
-- to contain.
--
-- All are `many`. Cardinality `one` supersedes the previous fact on write, so a
-- wrong `one` silently destroys a fact while a wrong `many` leaves a visible
-- duplicate -- and every relation here is honestly one-to-many anyway: a
-- service publishes to several topics and has several failure modes.
--
-- The vocabulary is deliberately small. Relationship-type embeddings interleave
-- (the calibration found the closest inverse pair outscoring ten of fourteen
-- true synonyms), so near-synonyms make coercion collide rather than sharpen
-- it. CALLS was considered and rejected for sitting too close to DEPENDS_ON for
-- an agent to choose between them consistently.

INSERT INTO relationship_type (tenant_id, name, cardinality, description) VALUES
    -- Traffic and call path
    ('', 'PROXIES_TO',        'many', 'Subject forwards traffic to the object, which serves it'),
    ('', 'AUTHENTICATES_WITH','many', 'Subject proves its identity to the object before being served'),

    -- Data flow. Asynchronous movement is invisible in a DEPENDS_ON graph, and
    -- it is where the incidents in this corpus concentrate.
    ('', 'PUBLISHES_TO',      'many', 'Subject writes messages or events to the object'),
    ('', 'CONSUMES_FROM',     'many', 'Subject reads messages or events from the object'),
    ('', 'PERSISTS_TO',       'many', 'Subject stores durable state in the object'),

    -- Topology
    ('', 'DEPLOYED_ON',       'many', 'Subject runs on the object: a host, cluster or region'),

    -- Observability
    ('', 'EMITS_METRICS_TO',  'many', 'Subject sends metrics or telemetry to the object'),

    -- Operational knowledge. Not generic architecture: a memory used during an
    -- incident needs to answer "what breaks here" and "who wrote that down",
    -- and neither is expressible with the original ten.
    ('', 'FAILURE_MODE_OF',   'many', 'Subject is a known way the object fails'),
    ('', 'INCIDENT_ON',       'many', 'Subject is an incident, ticket or alert raised against the object'),
    ('', 'DOCUMENTS',         'many', 'Subject describes or explains the object'),

    -- Lifecycle
    ('', 'SUPERSEDES',        'many', 'Subject replaces the object, which is no longer the way it is done');

-- Phrasings an agent reaches for. Tier 2 is exact-match on the alias, so these
-- cost nothing at query time and stop a coined synonym fragmenting the
-- vocabulary. Kept narrow: an alias claims a phrase globally, so a generic one
-- ("writes to") would hijack a phrase that legitimately means something else.
INSERT INTO relationship_type_alias (alias, canonical) VALUES
    ('FORWARDS_TO',      'PROXIES_TO'),
    ('ROUTES_TO',        'PROXIES_TO'),
    ('PRODUCES_TO',      'PUBLISHES_TO'),
    ('SUBSCRIBES_TO',    'CONSUMES_FROM'),
    ('READS_FROM',       'CONSUMES_FROM'),
    ('STORES_STATE_IN',  'PERSISTS_TO'),
    ('RUNS_ON',          'DEPLOYED_ON'),
    ('HOSTED_ON',        'DEPLOYED_ON'),
    ('SENDS_METRICS_TO', 'EMITS_METRICS_TO'),
    ('REPLACES',         'SUPERSEDES'),
    ('DESCRIBES',        'DOCUMENTS');

-- Direction guard.
--
-- Only the pairs the mechanical rule cannot derive. `are_mechanical_inverses`
-- already covers X / X_BY (SUPERSEDES / SUPERSEDED_BY, PROXIES_TO / PROXIED_BY,
-- DOCUMENTS / DOCUMENTED_BY) and HAS_X / X_OF (FAILURE_MODE_OF /
-- HAS_FAILURE_MODE), so restating those here would be duplication that can
-- drift.
--
-- PUBLISHES_TO / CONSUMES_FROM are not inverses in the graph sense -- both run
-- from a service to a topic -- but coercing one onto the other reverses what
-- the fact means, turning a producer into a consumer, which is the exact
-- failure this table exists to prevent. They are also close in embedding space,
-- being two short phrases about moving messages.
INSERT INTO relationship_inverse_pair (type_a, type_b) VALUES
    ('PUBLISHES_TO',  'CONSUMES_FROM'),
    ('DEPLOYED_ON',   'HOSTS'),
    ('PERSISTS_TO',   'STORES_FOR'),
    ('EMITS_METRICS_TO', 'SCRAPES'),
    ('INCIDENT_ON',   'HAD_INCIDENT');

-- Repair, not schema.
--
-- `backfill_missing_name_vectors` fills any row whose `name_vector` is NULL,
-- and it does so for the *global* scope with whatever embedder is running. One
-- test run against the hash-based DeterministicEmbedder therefore stamps the
-- shared vocabulary with vectors no real deployment can compare against --
-- tier 3 filters on `embedding_model`, by design, because a cosine between two
-- models means nothing. The rows are then never NULL again, so the real
-- embedder never gets a turn, and semantic coercion is silently dead for every
-- canonical type.
--
-- Clearing them puts the rows back in the state migration 0001 left them in, so
-- the next real embedder refills them. A fresh database matches nothing here.
UPDATE relationship_type
   SET name_vector = NULL, embedding_model = NULL
 WHERE tenant_id = ''
   AND embedding_model IS DISTINCT FROM NULL
   AND embedding_model <> 'BAAI/bge-base-en-v1.5';
