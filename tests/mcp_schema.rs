//! Tests over the published tool schemas.
//!
//! No database and no MCP session: these read the JSON a client actually
//! receives from `tools/list`, which is generated at compile time from the
//! input and output types.
//!
//! The centrepiece is [`no_input_schema_anywhere_mentions_a_tenant`]. Every
//! other property here is a usability one; that one is the security boundary
//! this server is built around, so it is asserted against the generated JSON
//! rather than by reading the structs.

use mundu::mcp::MunduTools;
use serde_json::Value;

/// The five tools exactly as a client would list them.
fn published() -> Vec<rmcp::model::Tool> {
    MunduTools::tool_router().list_all()
}

fn tool(name: &str) -> rmcp::model::Tool {
    published()
        .into_iter()
        .find(|tool| tool.name == name)
        .unwrap_or_else(|| panic!("`{name}` should be published"))
}

fn input_schema(name: &str) -> Value {
    Value::Object((*tool(name).input_schema).clone())
}

fn output_schema(name: &str) -> Value {
    Value::Object(
        (*tool(name)
            .output_schema
            .unwrap_or_else(|| panic!("`{name}` should publish an output schema")))
        .clone(),
    )
}

/// The name of every tool this server publishes.
///
/// Hard-coded on purpose, and cross-checked against the router below: the
/// security test walks whatever the router lists, and this is what stops a
/// silently empty router from making it vacuous.
const TOOL_NAMES: [&str; 5] = [
    "create_provenance_relationship",
    "explore_entity",
    "rate_memory_utility",
    "search_knowledge_graph",
    "upsert_node",
];

/// Collects every property name at every depth of a JSON Schema.
///
/// Walks `properties`, `$defs`, `items`, `oneOf`/`anyOf`/`allOf`, and anything
/// else nested, because a field can be introduced at any of them — a tenant
/// smuggled into a `$defs` entry reached from a `oneOf` branch is a tenant a
/// model can set, and a check that only read the top level would miss it.
fn property_names(schema: &Value) -> Vec<String> {
    let mut names = Vec::new();
    collect_property_names(schema, &mut names);
    names
}

fn collect_property_names(value: &Value, names: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if key == "properties"
                    && let Value::Object(properties) = child
                {
                    names.extend(properties.keys().cloned());
                }
                collect_property_names(child, names);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_property_names(item, names);
            }
        }
        _ => {}
    }
}

/// Every tool is published with both schemas, and the set of tools is the set
/// this suite believes it is checking.
#[test]
fn every_tool_publishes_an_input_and_an_output_schema() {
    let mut names: Vec<String> = published()
        .iter()
        .map(|tool| tool.name.to_string())
        .collect();
    names.sort();

    assert_eq!(
        names, TOOL_NAMES,
        "the router should publish exactly the five tools this suite checks"
    );

    for name in TOOL_NAMES {
        let tool = tool(name);
        assert_eq!(
            tool.input_schema.get("type").and_then(Value::as_str),
            Some("object"),
            "`{name}` must take an object, as MCP requires of every inputSchema"
        );
        assert!(
            tool.output_schema.is_some(),
            "`{name}` must publish an output schema; a client that cannot \
             validate structuredContent has to fall back to parsing prose"
        );
        assert!(
            tool.description
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty()),
            "`{name}` must describe itself"
        );
    }
}

/// The arguments each tool takes reach the wire under the names the plan gives
/// them.
#[test]
fn the_input_schemas_name_the_arguments_each_tool_takes() {
    for (name, expected) in [
        ("upsert_node", vec!["nodes", "display_name", "aliases"]),
        (
            "create_provenance_relationship",
            vec![
                "relationships",
                "source_node_id",
                "target_node_id",
                "relationship_type",
                "proof_excerpt",
                "valid_from",
            ],
        ),
        (
            "search_knowledge_graph",
            vec![
                "query",
                "search_method",
                "precision_requirement",
                "limit",
                "include_full_proof",
            ],
        ),
        ("rate_memory_utility", vec!["edge_ids", "utility_score"]),
        ("explore_entity", vec!["node", "relationship_type", "depth"]),
    ] {
        let found = property_names(&input_schema(name));
        for field in expected {
            assert!(
                found.iter().any(|property| property == field),
                "`{name}` should take `{field}`; its schema has {found:?}"
            );
        }
    }
}

/// Every hit carries the handle the rating loop consumes, and the loop's other
/// half names it too. Asserted on the published schema because that is what a
/// model plans against: an `edge_id` the server returns but does not advertise
/// is one a model will not think to thread back.
#[test]
fn the_output_schemas_name_what_each_tool_returns() {
    for (name, expected) in [
        (
            "upsert_node",
            vec!["node_id", "display_name", "matched", "candidates"],
        ),
        (
            "create_provenance_relationship",
            vec![
                "edge_id",
                "outcome",
                "superseded_edge_id",
                "valid_until",
                "suggestions",
            ],
        ),
        (
            "search_knowledge_graph",
            vec![
                "edge_id",
                "source_node_id",
                "target_node_id",
                "relationship_type",
                "effective_salience",
                "score",
                "valid_from",
                "valid_until",
                "disputed",
                "writer_client",
                "writer_principal",
                "proof",
            ],
        ),
        ("rate_memory_utility", vec!["edge_id", "salience", "reason"]),
        (
            "explore_entity",
            vec![
                "matches",
                "node_id",
                "display_name",
                "aliases",
                "degree",
                "matched_by",
                "similarity",
                "edges",
                "edges_withheld",
                "neighbours",
                "edge_id",
                "relationship_type",
                "proof",
                "proof_truncated",
                "salience",
                "disputed",
                "valid_from",
                "relationship_types",
            ],
        ),
    ] {
        let found = property_names(&output_schema(name));
        for field in expected {
            assert!(
                found.iter().any(|property| property == field),
                "`{name}` should return `{field}`; its schema has {found:?}"
            );
        }
    }
}

/// **The security regression test.**
///
/// The tenant is the whole isolation boundary, and everything a model reads
/// before filling in these arguments — `proof_excerpt` text written by other
/// agents — is untrusted. A tenant argument would therefore be a tenant that
/// injected text can set, which turns one poisoned memory into a read of
/// somebody else's.
///
/// So it is asserted three ways, over whatever the router publishes rather than
/// over a hand-written list of tools:
///
/// 1. no property, at any depth of any input schema, is named for a tenant;
/// 2. the word does not appear *anywhere* in the serialized input schema —
///    not in a `$ref`, an `enum`, a `const`, a `patternProperties` key or a
///    description — which catches shapes the structural walk does not model;
/// 3. the walk is not vacuous: it finds fields that are genuinely there,
///    including ones only reachable through `$defs` and `oneOf`.
#[test]
fn no_input_schema_anywhere_mentions_a_tenant() {
    let published = published();
    assert_eq!(
        published.len(),
        TOOL_NAMES.len(),
        "an empty or truncated router would make this test pass by doing nothing"
    );

    // (3) The walk descends into `$defs`: `display_name` and `proof_excerpt`
    // live on types referenced from the top-level object, never on it.
    assert!(
        property_names(&input_schema("upsert_node"))
            .iter()
            .any(|property| property == "display_name"),
        "the property walk must reach into $defs, or it proves nothing"
    );
    assert!(
        property_names(&input_schema("create_provenance_relationship"))
            .iter()
            .any(|property| property == "proof_excerpt"),
        "the property walk must reach into $defs, or it proves nothing"
    );

    for tool in published {
        let schema = Value::Object((*tool.input_schema).clone());

        // (1) Structural: no argument is named for a tenant, at any depth.
        for property in property_names(&schema) {
            let folded = property.to_lowercase().replace(['_', '-'], "");
            assert!(
                !folded.contains("tenant"),
                "`{}` takes an argument named `{property}`; the tenant comes \
                 from the credential and from nowhere else",
                tool.name
            );
        }

        // (2) Textual: the concept does not appear in the schema at all.
        let serialized = serde_json::to_string(&schema)
            .expect("a schema should serialize")
            .to_lowercase();
        assert!(
            !serialized.contains("tenant"),
            "`{}` mentions a tenant somewhere in its input schema: {serialized}",
            tool.name
        );
    }
}

/// A model should be able to ask a question without first making a retrieval
/// engineering decision, so the knobs are optional in the schema — not merely
/// defaulted in the code, which a client reading the schema cannot see.
#[test]
fn the_search_knobs_are_optional_and_only_the_query_is_required() {
    let schema = input_schema("search_knowledge_graph");
    let required: Vec<&str> = schema["required"]
        .as_array()
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    assert_eq!(
        required,
        vec!["query"],
        "only the question itself is required"
    );
}

/// **A knob a model can set, will reason about, and that does nothing, is worse
/// than no knob.**
///
/// `precision_requirement` moves a cutoff taken on the spread between two fused
/// candidate rankings. A `semantic`- or `lexical`-only search produces one
/// ranking and therefore no spread, so the store does not apply the cutoff at
/// all — see `precision_changes_nothing_a_single_list_search_can_show` in the
/// search suite, which pins the behaviour this text describes.
///
/// A setting that is quietly inert costs a model tokens to reason about and
/// invites it to believe a result set was filtered when it was not, and the
/// only place it can learn otherwise before spending a call is here.
///
/// Asserted in all three places a client might surface — the tool blurb, the
/// argument, and the enum the argument refers to — for the reason the
/// relationship vocabulary test gives: clients surface some of these and not
/// others, and the one a model reads is the one that has to say it.
#[test]
fn precision_says_it_applies_to_hybrid_search_and_nowhere_else() {
    let tool = tool("search_knowledge_graph");
    let schema = Value::Object((*tool.input_schema).clone());

    let described = |value: &Value, what: &str| -> String {
        value["description"]
            .as_str()
            .unwrap_or_else(|| panic!("{what} should describe itself: {value}"))
            .to_lowercase()
    };

    for (place, text) in [
        (
            "the tool description",
            tool.description.unwrap_or_default().to_lowercase(),
        ),
        (
            "the `precision_requirement` argument",
            described(
                &schema["properties"]["precision_requirement"],
                "precision_requirement",
            ),
        ),
        (
            "the `PrecisionRequirement` enum",
            described(&schema["$defs"]["PrecisionRequirement"], "the enum"),
        ),
    ] {
        assert!(
            text.contains("hybrid"),
            "{place} should name the one search method precision applies to; \
             it reads: {text}"
        );
        assert!(
            text.contains("semantic") && text.contains("lexical"),
            "{place} should name the methods it does nothing for, because those \
             are the ones a model would otherwise set it alongside; it reads: {text}"
        );
        assert!(
            text.contains("ignore"),
            "{place} should say plainly that the setting is ignored there, not \
             merely that it is weaker; it reads: {text}"
        );
    }
}

/// The vocabulary is an open set, so `relationship_type` is a plain string —
/// but a vocabulary nobody is shown grows a `WORKS_FOR`, an `EMPLOYED_BY` and a
/// `REPORTS_TO` for one relation. The seeded types are therefore named in the
/// text a model actually reads.
#[test]
fn the_seeded_relationship_vocabulary_is_named_where_a_model_will_read_it() {
    let tool = tool("create_provenance_relationship");
    let schema = Value::Object((*tool.input_schema).clone());
    let described = tool.description.unwrap_or_default().to_string();

    // On the argument itself, not only in the tool blurb: some clients surface
    // one and not the other, and a vocabulary a model never sees is a
    // vocabulary that grows a synonym per agent.
    let on_the_argument =
        schema["$defs"]["RelationshipSpec"]["properties"]["relationship_type"]["description"]
            .as_str()
            .expect("relationship_type should describe itself")
            .to_string();

    for seeded in [
        "USES",
        "DEPENDS_ON",
        "REPORTS_TO",
        "OWNS",
        "MEMBER_OF",
        "PART_OF",
        "AUTHORED_BY",
        "LOCATED_IN",
        "PREFERS",
        "RELATED_TO",
    ] {
        assert!(
            described.contains(seeded),
            "the tool description should offer `{seeded}` before a model coins a synonym"
        );
        assert!(
            on_the_argument.contains(seeded),
            "the `relationship_type` argument should offer `{seeded}` too; \
             it reads: {on_the_argument}"
        );
    }

    // Open set, not an enum: an unrecognised type is coerced server-side onto
    // the closest existing one, so a closed enum would refuse novel relations
    // the graph is perfectly able to record.
    let relationship_type = &schema["$defs"]["RelationshipSpec"]["properties"]["relationship_type"];
    assert_eq!(relationship_type["type"], "string");
    assert!(
        relationship_type.get("enum").is_none() && relationship_type.get("const").is_none(),
        "the vocabulary is an open set: {relationship_type}"
    );
}

/// Memory written by other agents is read back into this one's context, so the
/// tools that return it are a prompt-injection propagation channel. The tool
/// description is one of the few places a server can say so.
#[test]
fn the_tools_that_return_stored_text_say_it_is_untrusted() {
    for name in [
        "search_knowledge_graph",
        "create_provenance_relationship",
        "explore_entity",
    ] {
        let description = tool(name).description.unwrap_or_default().to_lowercase();

        assert!(
            description.contains("untrusted"),
            "`{name}` should tell the model that stored proof text is untrusted \
             data rather than instructions; it says: {description}"
        );
        assert!(
            description.contains("instruction"),
            "`{name}` should say what not to do with it; it says: {description}"
        );
    }
}

/// Belt and braces for the schema check: even if a client sends a tenant
/// anyway, there is nowhere for it to land.
#[test]
fn a_tenant_smuggled_into_the_arguments_is_dropped_on_the_floor() {
    use mundu::mcp::tools::UpsertNodeInput;

    let smuggled: UpsertNodeInput = serde_json::from_value(serde_json::json!({
        "tenant": "someone-elses-memory",
        "tenant_id": "someone-elses-memory",
        "nodes": [{"display_name": "Alice Whitfield"}],
    }))
    .expect("extra fields are ignored rather than rejected");

    let clean: UpsertNodeInput = serde_json::from_value(serde_json::json!({
        "nodes": [{"display_name": "Alice Whitfield"}],
    }))
    .expect("the same arguments without them should parse");

    assert_eq!(
        smuggled, clean,
        "a tenant in the arguments must not survive deserialization in any form"
    );
}

/// **The guards, as a model sees them before it generates anything.**
///
/// Enforcement lives in the store, where every caller reaches it and where the
/// sibling guards already are. That is the actual defence. This is the other
/// half: a constraint a model reads at generation time costs a round trip
/// nothing, and a limit that exists only in an error message is a limit every
/// caller discovers by tripping over it.
///
/// Asserted against the store's own constants rather than against numbers
/// written out here, so the schema and the code that enforces it cannot drift
/// apart — which is the failure mode a published constraint invites.
#[test]
fn the_input_schemas_carry_the_batch_and_length_limits() {
    use mundu::store::{
        MAX_BATCH_ITEMS, MAX_DISPLAY_NAME_CHARS, MAX_PROOF_EXCERPT_CHARS, MIN_PROOF_EXCERPT_CHARS,
    };

    let upsert = input_schema("upsert_node");
    assert_eq!(
        upsert["properties"]["nodes"]["maxItems"].as_u64(),
        Some(MAX_BATCH_ITEMS as u64),
        "`nodes` should advertise its cap: {upsert}"
    );

    let node_spec = &upsert["$defs"]["NodeSpec"]["properties"];
    assert_eq!(
        node_spec["display_name"]["maxLength"].as_u64(),
        Some(MAX_DISPLAY_NAME_CHARS as u64),
        "`display_name` should advertise its cap: {upsert}"
    );
    assert_eq!(
        node_spec["aliases"]["maxItems"].as_u64(),
        Some(MAX_BATCH_ITEMS as u64),
        "one entity's alias list is an array argument too: {upsert}"
    );
    assert_eq!(
        node_spec["aliases"]["items"]["maxLength"].as_u64(),
        Some(MAX_DISPLAY_NAME_CHARS as u64),
        "an alias is a name and gets a name's budget: {upsert}"
    );

    let relationships = input_schema("create_provenance_relationship");
    assert_eq!(
        relationships["properties"]["relationships"]["maxItems"].as_u64(),
        Some(MAX_BATCH_ITEMS as u64),
        "`relationships` should advertise its cap: {relationships}"
    );

    let proof = &relationships["$defs"]["RelationshipSpec"]["properties"]["proof_excerpt"];
    assert_eq!(
        proof["maxLength"].as_u64(),
        Some(MAX_PROOF_EXCERPT_CHARS as u64),
        "`proof_excerpt` should advertise its cap: {proof}"
    );
    assert_eq!(
        proof["minLength"].as_u64(),
        Some(MIN_PROOF_EXCERPT_CHARS as u64),
        "`proof_excerpt` should advertise that a fact needs justifying: {proof}"
    );

    let ratings = input_schema("rate_memory_utility");
    assert_eq!(
        ratings["properties"]["edge_ids"]["maxItems"].as_u64(),
        Some(MAX_BATCH_ITEMS as u64),
        "`edge_ids` should advertise its cap: {ratings}"
    );

    // The prose beside each constraint names the same number. Some clients
    // surface a field's description and not its keywords, and a description
    // that has drifted from the constraint is worse than one that says nothing.
    for (schema, path, limit) in [
        (&upsert, vec!["properties", "nodes"], MAX_BATCH_ITEMS),
        (
            &upsert,
            vec!["$defs", "NodeSpec", "properties", "display_name"],
            MAX_DISPLAY_NAME_CHARS,
        ),
        (
            &relationships,
            vec!["properties", "relationships"],
            MAX_BATCH_ITEMS,
        ),
        (
            &relationships,
            vec!["$defs", "RelationshipSpec", "properties", "proof_excerpt"],
            MAX_PROOF_EXCERPT_CHARS,
        ),
        (&ratings, vec!["properties", "edge_ids"], MAX_BATCH_ITEMS),
    ] {
        let field = path.iter().fold(schema, |value, key| &value[key]);
        let described = field["description"].as_str().unwrap_or_default();

        assert!(
            described.contains(&limit.to_string()),
            "the description of {path:?} should name its limit of {limit}, it reads: {described}"
        );
    }
}

/// The schema is ergonomics; the code is the guard. A `minLength` counts
/// spaces, so a client that satisfies the published constraint can still send
/// something that is not evidence — and the store still refuses it.
///
/// Asserted here rather than left implied, because the temptation with a
/// published constraint is to treat it as having done the work.
#[test]
fn the_published_minimum_is_weaker_than_the_guard_behind_it() {
    use mundu::mcp::tools::RelationshipSpec;
    use mundu::store::MIN_PROOF_EXCERPT_CHARS;

    let spaces = " ".repeat(MIN_PROOF_EXCERPT_CHARS);
    let parsed: RelationshipSpec = serde_json::from_value(serde_json::json!({
        "source_node_id": "alice",
        "target_node_id": "bob",
        "relationship_type": "REPORTS_TO",
        "proof_excerpt": spaces,
    }))
    .expect("a string of spaces satisfies minLength and parses");

    assert_eq!(
        parsed.proof_excerpt.trim().chars().count(),
        0,
        "which is exactly why the store trims before it counts"
    );
}

/// Every published `outputSchema` is a JSON object at the top level.
///
/// `2026-07-28` relaxed this — SEP-2106 permits any JSON type, so a bare
/// `"type": "array"` is legal there — but every earlier revision requires an
/// object, and this server serves those too under `dual-era`. Claude Code
/// 2.1.220 connects and then refuses the whole tool list over it:
///
/// ```text
/// mundu: http://127.0.0.1:18200/mcp (HTTP) - ! Connected - tools fetch failed
///        - Invalid input: expected "object" (at tools.0.outputSchema.type)
/// ```
///
/// An object wrapper is legal in *every* revision, so the shape is
/// unconditional rather than negotiated: making the handshake era-aware while
/// leaving the payload modern-only is what produced this bug in the first
/// place.
///
/// Driven off the router rather than a list of names, so a fifth tool cannot
/// reintroduce it.
#[test]
fn every_output_schema_is_an_object_because_older_clients_reject_anything_else() {
    for tool in published() {
        let schema = output_schema(&tool.name);

        assert_eq!(
            schema.get("type").and_then(Value::as_str),
            Some("object"),
            "`{}` publishes a top-level `{}` output schema; a client older than \
             2026-07-28 rejects the entire tools/list over it, so the tool \
             becomes unreachable rather than merely unvalidated",
            tool.name,
            schema
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("(absent)"),
        );
    }
}

/// **The fifth tool's answer is an object, and its array is behind a name.**
///
/// The generic test above walks the router and would catch this too. This one
/// names the tool anyway, because the temptation with a traversal is exactly
/// the shape that broke `tools/list` before: the natural return value is a list
/// of matches, `schemars` describes a bare `Vec` as `"type": "array"`, and
/// every revision before `2026-07-28` rejects the whole tool list over it —
/// taking the other four tools down with it.
#[test]
fn explore_entity_publishes_an_object_wrapping_its_matches() {
    let schema = output_schema("explore_entity");

    assert_eq!(
        schema["type"], "object",
        "a bare array outputSchema is illegal before 2026-07-28: {schema}"
    );
    assert_eq!(
        schema["properties"]["matches"]["type"], "array",
        "the matches are an array behind a named field, as the other four are: {schema}"
    );
    assert_eq!(
        schema["required"].as_array().map(|required| required.len()),
        Some(1),
        "and the field is always there, so a client can read `matches` without \
         checking whether it exists: {schema}"
    );
}

/// A traversal should cost one argument to ask for.
///
/// `relationship_type` and `depth` are refinements of the same walk, not modes
/// of it, so a caller that knows only which entity it means can ask without
/// making two more decisions. The depth's ceiling is published rather than
/// discovered by tripping over it, and it is asserted against the store's own
/// constant so the schema and the clamp cannot drift apart.
#[test]
fn only_the_entity_is_required_and_the_depth_publishes_its_ceiling() {
    use mundu::store::MAX_DISPLAY_NAME_CHARS;
    use mundu::store::explore::MAX_DEPTH;

    let schema = input_schema("explore_entity");
    let required: Vec<&str> = schema["required"]
        .as_array()
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    assert_eq!(
        required,
        vec!["node"],
        "only the entity is required: {schema}"
    );
    assert_eq!(
        schema["properties"]["node"]["maxLength"].as_u64(),
        Some(MAX_DISPLAY_NAME_CHARS as u64),
        "a name is a name and gets a name's budget: {schema}"
    );

    let depth = &schema["properties"]["depth"];
    assert_eq!(depth["minimum"].as_u64(), Some(1), "{depth}");
    assert_eq!(
        depth["maximum"].as_u64(),
        Some(u64::from(MAX_DEPTH)),
        "the published ceiling is the one the store clamps to: {depth}"
    );
}

/// **A second hop that returns something other than what a first hop returns
/// has to say so before it is called.**
///
/// `depth: 2` does not return the neighbours' facts; it returns each neighbour
/// summarised. A model that expected facts and got counts will either call
/// again for each neighbour — which is the cost the summary exists to avoid —
/// or report that the memory holds nothing about them. The only place it can
/// learn otherwise before spending a call is the schema.
///
/// Asserted in both places a client might surface, for the reason the
/// `precision_requirement` test gives: clients show some of these and not
/// others, and the one a model reads is the one that has to say it.
#[test]
fn depth_two_says_the_second_hop_is_summarised_before_a_model_asks_for_it() {
    let tool = tool("explore_entity");
    let schema = Value::Object((*tool.input_schema).clone());

    for (place, text) in [
        (
            "the tool description",
            tool.description.unwrap_or_default().to_lowercase(),
        ),
        (
            "the `depth` argument",
            schema["properties"]["depth"]["description"]
                .as_str()
                .expect("depth should describe itself")
                .to_lowercase(),
        ),
    ] {
        assert!(
            text.contains("summaris"),
            "{place} should say the second hop comes back summarised; it reads: {text}"
        );
        assert!(
            text.contains("neighbour"),
            "{place} should say what is summarised; it reads: {text}"
        );
    }
}
