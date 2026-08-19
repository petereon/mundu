//! The stdio transport, exercised by running the actual binary.
//!
//! Spawning the real process rather than calling into the library is the point:
//! stdio's failure modes are all about the process's own streams — what goes to
//! stdout, what goes to stderr, and what the exit status says — and none of
//! those exist inside a unit test.
//!
//! Bring the test database up with `docker compose up -d postgres-test`.

mod common;

use std::io::Write;
use std::process::{Command, Stdio};

/// The URL the spawned process is given, checked before it is handed over.
///
/// These tests write through a real binary rather than through a `Store`, so
/// the guard has to run on this side of the spawn: once the child has the URL
/// it is out of reach.
fn database_url() -> String {
    common::database_url_for_subprocess()
}

/// Runs the binary with `--stdio`, feeds it `input`, and returns
/// `(stdout, stderr, success)` once it exits.
fn run_stdio(env: &[(&str, &str)], input: &str) -> (String, String, bool) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mundu"));
    command
        .arg("--stdio")
        .env_clear()
        // `env_clear` and then only what is needed, so a variable that happens
        // to be set in the developer's shell cannot change what is tested.
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("MUNDU_DATABASE_URL", database_url())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    for (key, value) in env {
        command.env(key, value);
    }

    let mut child = command.spawn().expect("the binary should be runnable");
    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(input.as_bytes())
        .expect("writing to the child should succeed");

    let output = child.wait_with_output().expect("the child should exit");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.success(),
    )
}

/// A `2026-07-28` request, which carries its own protocol metadata because
/// there is no handshake to carry it for them.
fn request(id: u32, method: &str, params: serde_json::Value) -> String {
    let mut params = params;
    params["_meta"] = serde_json::json!({
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": { "name": "mundu-stdio-test", "version": "0.0.0" },
    });

    format!(
        "{}\n",
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        })
    )
}

const DEV_ENV: [(&str, &str); 3] = [
    ("MUNDU_AUTH_MODE", "disabled"),
    ("MUNDU_INSECURE_NO_AUTH", "1"),
    ("MUNDU_DEV_TENANT", "test-stdio-dev"),
];

/// **stdout is the protocol.** Over stdio there is no second channel: every
/// byte the process writes to stdout is read by the client as JSON-RPC. A log
/// line there is not untidy, it is a parse error in the middle of a session,
/// and it is the single easiest way to build a stdio server that works when you
/// try it by hand and fails against a real client.
///
/// So: every line of stdout must be a JSON-RPC message, and the logs must be on
/// stderr where they belong.
#[tokio::test]
async fn stdio_keeps_its_logs_off_the_protocol_stream() {
    let (stdout, stderr, _) = run_stdio(&DEV_ENV, &request(1, "tools/list", serde_json::json!({})));

    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
        assert!(
            parsed.is_ok_and(|message| message["jsonrpc"] == "2.0"),
            "every line on stdout must be a JSON-RPC message, or a client's parser \
             breaks mid-session. Found:\n{line}"
        );
    }

    assert!(
        stderr.contains("serving MCP over stdio"),
        "the startup log should be on stderr, got:\n{stderr}"
    );
}

/// The tools have to be reachable over this transport too, and reachable as the
/// configured dev tenant — the whole reason `--stdio` exists is pointing a
/// local client at a development graph.
#[tokio::test]
async fn stdio_serves_the_tools_as_the_configured_dev_tenant() {
    let input = format!(
        "{}{}",
        request(1, "tools/list", serde_json::json!({})),
        request(
            2,
            "tools/call",
            serde_json::json!({
                "name": "upsert_node",
                "arguments": { "nodes": [{ "display_name": "Stdio Subject" }] },
            }),
        ),
    );

    let (stdout, stderr, _) = run_stdio(&DEV_ENV, &input);

    let replies: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    let listed = replies
        .iter()
        .find(|reply| reply["id"] == 1)
        .unwrap_or_else(|| panic!("tools/list should be answered; stdout was:\n{stdout}"));
    assert_eq!(
        listed["result"]["tools"]
            .as_array()
            .map(|tools| tools.len()),
        Some(5),
        "all five tools should be listed over stdio"
    );

    let called = replies
        .iter()
        .find(|reply| reply["id"] == 2)
        .unwrap_or_else(|| panic!("tools/call should be answered; stdout was:\n{stdout}"));
    assert_eq!(
        called["result"]["structuredContent"]["nodes"][0]["node_id"], "stdio-subject",
        "the tool should have run and minted an id, got {called}"
    );

    assert!(
        stderr.contains("test-stdio-dev"),
        "stdio should say which tenant it is acting as, got:\n{stderr}"
    );
}

/// stdio has no `Authorization` header and no handshake that could carry one,
/// so a deployment configured to check credentials cannot be served this way.
/// Refusing to start is the only safe answer: inventing a tenant would write
/// somebody's memory under a name nobody configured, and silently dropping to
/// unauthenticated operation would let a command-line flag undo the
/// deployment's authentication.
#[tokio::test]
async fn stdio_refuses_to_start_when_the_deployment_expects_a_credential() {
    let (stdout, stderr, success) = run_stdio(
        &[
            ("MUNDU_AUTH_MODE", "token"),
            (
                "MUNDU_STATIC_TOKENS",
                "acme:0123456789abcdef0123456789abcdef",
            ),
        ],
        &request(1, "tools/list", serde_json::json!({})),
    );

    assert!(!success, "the process should exit non-zero, got:\n{stderr}");
    assert!(
        stderr.contains("--stdio"),
        "the refusal should name the flag that caused it, got:\n{stderr}"
    );
    assert!(
        !stdout.contains("\"result\""),
        "nothing should have been served, got:\n{stdout}"
    );
}
