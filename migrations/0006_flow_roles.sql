-- Where a flow enters and leaves the system.
--
-- A "flow" here is a named end-to-end path -- keep-alive, registration,
-- telemetry-upload -- that crosses several services. The graph already records
-- which services a flow needs, as `flow DEPENDS_ON service`, and that is also
-- the honest way to say "this service caters to that flow": the same fact from
-- the other end. A second type for the reverse direction would duplicate
-- ninety-eight edges and deepen a fragmentation already visible in the data,
-- where `commsdk DEPENDS_ON asset-delivery-flow` sits beside
-- `asset-delivery-flow DEPENDS_ON commsdk`.
--
-- What `DEPENDS_ON` cannot say is *which end*. Thirteen existing edges already
-- carry it in prose -- their proofs open "Originates here." or "Terminates
-- here." -- and the flow documents establish it for the rest in their
-- step-by-step sections. Promoting it from prose to a type makes the triage
-- question answerable: when a flow breaks, where does it start and where is it
-- supposed to end.
--
-- Ownership is deliberately not modelled. Nothing records it -- not the graph,
-- not the thirteen flow documents, which have no owner or responsible-service
-- section. Minting an OWNS edge per flow would mean writing a proof excerpt
-- that no source says, and a fabricated proof is worse than a missing fact
-- because it reads as evidence.
--
-- Both `many`: a service originates several flows, and a flow can legitimately
-- have more than one entry point.
INSERT INTO relationship_type (tenant_id, name, cardinality, description) VALUES
    ('', 'ORIGINATES', 'many', 'A flow starts at the subject: the object''s first step happens here'),
    ('', 'TERMINATES', 'many', 'A flow ends at the subject: the object''s last step happens here')
ON CONFLICT (tenant_id, name) DO NOTHING;

INSERT INTO relationship_type_alias (alias, canonical) VALUES
    ('ENTRY_POINT_FOR', 'ORIGINATES'),
    ('STARTS',          'ORIGINATES'),
    ('EXIT_POINT_FOR',  'TERMINATES'),
    ('ENDS',            'TERMINATES')
ON CONFLICT (alias) DO NOTHING;

-- Not inverses of each other in the graph sense -- both run service -> flow --
-- but coercing one onto the other reverses what the fact means, turning the
-- place a flow begins into the place it ends. They are two short phrases about
-- the ends of a path, so they will sit close in embedding space, which is
-- exactly the case this table exists for. Same reasoning as
-- PUBLISHES_TO / CONSUMES_FROM in 0005.
INSERT INTO relationship_inverse_pair (type_a, type_b) VALUES
    ('ORIGINATES', 'TERMINATES')
ON CONFLICT (type_a, type_b) DO NOTHING;
