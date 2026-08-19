//! A read-only browser view of the graph, for an operator looking at it.
//!
//! Deliberately a second binary rather than routes on the MCP server. It reads
//! *every* tenant, which is the one thing the MCP server spends its whole
//! authentication and tenancy design preventing — putting that on the same
//! listener would mean one routing mistake exposes it. Here the blast radius is
//! a process that binds loopback and issues nothing but `SELECT`.
//!
//! It also needs no embedder. Nothing here searches, so there is no query to
//! embed, and the binary starts immediately instead of loading ~900MB of model.
//!
//! ```text
//! MUNDU_DATABASE_URL=postgres://mundu:mundu@localhost:55432/mundu cargo run --bin viz
//! ```

use std::net::{IpAddr, SocketAddr};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row, postgres::PgPoolOptions};

/// The page and the vendored library are compiled in, so the binary runs from
/// any directory and cannot be served a file the repository does not hold.
const INDEX_HTML: &str = include_str!("../../static/index.html");
const D3_JS: &str = include_str!("../../static/d3.min.js");

/// Where the viewer listens unless told otherwise. Loopback is not a default
/// here so much as the whole security model: see the module docs.
const DEFAULT_BIND: &str = "127.0.0.1:18300";

/// One entity, and how many facts touch it.
///
/// `degree` is computed rather than stored so the page can size a node by how
/// connected it is without a second round trip.
#[derive(Debug, Serialize)]
struct VizNode {
    id: String,
    label: String,
    degree: i64,
    aliases: Vec<String>,
}

/// One fact, carrying enough to explain itself when clicked.
#[derive(Debug, Serialize)]
struct VizLink {
    edge_id: String,
    source: String,
    target: String,
    relationship_type: String,
    proof: String,
    salience: f64,
    disputed: bool,
    status: String,
    valid_from: String,
    valid_until: Option<String>,
    writer_client: Option<String>,
}

#[derive(Debug, Serialize)]
struct Graph {
    tenant: String,
    nodes: Vec<VizNode>,
    links: Vec<VizLink>,
}

/// A tenant, and enough of a count to tell whether it is worth opening.
#[derive(Debug, Serialize)]
struct Tenant {
    tenant_id: String,
    nodes: i64,
    edges: i64,
}

#[derive(Debug, Deserialize)]
struct GraphQuery {
    tenant: String,
    /// Include superseded and archived facts. The graph is bi-temporal, so the
    /// history is real data rather than clutter — but it is off by default,
    /// because the current picture is what a reader almost always wants.
    #[serde(default)]
    history: bool,
}

/// Anything that goes wrong is the operator's to read, so it is reported as
/// text rather than swallowed into an empty graph.
struct VizError(String);

impl From<sqlx::Error> for VizError {
    fn from(error: sqlx::Error) -> Self {
        Self(error.to_string())
    }
}

impl IntoResponse for VizError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let database_url = std::env::var("MUNDU_DATABASE_URL")
        .map_err(|_| "MUNDU_DATABASE_URL is required: point it at the database to read")?;
    let bind: SocketAddr = std::env::var("MUNDU_VIZ_BIND")
        .unwrap_or_else(|_| DEFAULT_BIND.to_string())
        .parse()?;

    // The viewer answers for every tenant with no credential at all, so it
    // refuses to listen anywhere a credential would be needed. This is the one
    // guard standing between "operator tool" and "the whole graph, publicly".
    //
    // The exception exists for containers, where the check would otherwise be
    // worse than useless: a process binding 127.0.0.1 inside its own network
    // namespace is reachable by nothing, not even the operator who started it,
    // so the only way to serve a browser is to bind 0.0.0.0 and confine the
    // exposure when publishing the port. Setting this says the boundary is
    // real and lives elsewhere; it does not remove the need for one.
    let confined_elsewhere = std::env::var("MUNDU_VIZ_ALLOW_NON_LOOPBACK")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !is_loopback(bind.ip()) && !confined_elsewhere {
        return Err(format!(
            "refusing to bind {bind}: this viewer reads every tenant and asks for no \
             credential, so it serves loopback only. Tunnel to it rather than exposing \
             it. In a container, where binding loopback would make it reachable by \
             nobody, publish the port to the host's loopback only \
             (`ports: [\"127.0.0.1:18301:8080\"]`) and set \
             MUNDU_VIZ_ALLOW_NON_LOOPBACK=1"
        )
        .into());
    }

    if confined_elsewhere && !is_loopback(bind.ip()) {
        tracing::warn!(
            %bind,
            "listening beyond loopback because MUNDU_VIZ_ALLOW_NON_LOOPBACK is set. \
             Nothing here asks for a credential and every tenant is readable, so \
             whatever publishes this port is now the only thing deciding who can read \
             the whole graph"
        );
    }

    // Few connections: a browser makes a handful of requests and the server
    // must never be the reason the MCP server cannot get a connection.
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&database_url)
        .await?;

    let app = Router::new()
        .route("/", get(index))
        .route("/d3.min.js", get(d3))
        .route("/api/tenants", get(tenants))
        .route("/api/graph", get(graph))
        .with_state(pool);

    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(%bind, "graph viewer ready — open http://{bind}/ in a browser");
    axum::serve(listener, app).await?;

    Ok(())
}

/// `SocketAddr::ip().is_loopback()` misses IPv4-mapped IPv6 (`::ffff:127.0.0.1`),
/// which is exactly what a dual-stack bind produces.
fn is_loopback(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => v4.is_loopback(),
            None => v6.is_loopback(),
        },
    }
}

async fn index() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        INDEX_HTML,
    )
}

async fn d3() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        D3_JS,
    )
}

/// Every tenant holding anything, busiest first.
async fn tenants(State(pool): State<PgPool>) -> Result<Json<Vec<Tenant>>, VizError> {
    let rows = sqlx::query(
        "SELECT t.tenant_id, \
                (SELECT count(*) FROM node n \
                  WHERE n.tenant_id = t.tenant_id AND n.merged_into IS NULL) AS nodes, \
                (SELECT count(*) FROM edge e \
                  WHERE e.tenant_id = t.tenant_id AND e.status = 'active') AS edges \
           FROM (SELECT DISTINCT tenant_id FROM node) AS t \
          ORDER BY 2 DESC, 1",
    )
    .fetch_all(&pool)
    .await?;

    Ok(Json(
        rows.into_iter()
            .map(|row| Tenant {
                tenant_id: row.get("tenant_id"),
                nodes: row.get("nodes"),
                edges: row.get("edges"),
            })
            .collect(),
    ))
}

/// One tenant's graph.
///
/// Tombstoned nodes are always excluded: a merged entity is not a second thing
/// in the world, and drawing it would show a duplicate the store has already
/// resolved away.
async fn graph(
    State(pool): State<PgPool>,
    Query(query): Query<GraphQuery>,
) -> Result<Json<Graph>, VizError> {
    // One statement, with `history` bound rather than interpolated. sqlx lints
    // format!()-built SQL, and the lint is right: the fix is not to audit the
    // interpolation but to remove it, since the only thing varying is a flag.
    let link_rows = sqlx::query(
        "SELECT edge.edge_id::text AS edge_id, edge.source_node_id, edge.target_node_id, \
                edge.relationship_type, edge.proof_excerpt, \
                effective_salience(edge.salience_base, edge.salience_updated_at, $2, now()) \
                    ::double precision AS salience, \
                (edge.disputed_at IS NOT NULL) AS disputed, edge.status::text AS status, \
                edge.valid_from, edge.valid_until, edge.writer_client \
           FROM edge \
          WHERE edge.tenant_id = $1 \
            AND ($3 OR edge.status = 'active') \
          ORDER BY edge.valid_from DESC \
          LIMIT 2000",
    )
    .bind(&query.tenant)
    .bind(mundu::store::search::SALIENCE_HALF_LIFE_DAYS)
    .bind(query.history)
    .fetch_all(&pool)
    .await?;

    let links: Vec<VizLink> = link_rows
        .into_iter()
        .map(|row| VizLink {
            edge_id: row.get("edge_id"),
            source: row.get("source_node_id"),
            target: row.get("target_node_id"),
            relationship_type: row.get("relationship_type"),
            proof: row.get("proof_excerpt"),
            salience: row.get("salience"),
            disputed: row.get("disputed"),
            status: row.get("status"),
            valid_from: row
                .get::<chrono::DateTime<chrono::Utc>, _>("valid_from")
                .to_rfc3339(),
            valid_until: row
                .get::<Option<chrono::DateTime<chrono::Utc>>, _>("valid_until")
                .map(|at| at.to_rfc3339()),
            writer_client: row.get("writer_client"),
        })
        .collect();

    let node_rows = sqlx::query(
        "SELECT node.node_id, node.display_name, node.aliases, \
                (SELECT count(*) FROM edge \
                  WHERE edge.tenant_id = node.tenant_id \
                    AND edge.status = 'active' \
                    AND node.node_id IN (edge.source_node_id, edge.target_node_id)) AS degree \
           FROM node \
          WHERE node.tenant_id = $1 AND node.merged_into IS NULL \
          ORDER BY 4 DESC \
          LIMIT 2000",
    )
    .bind(&query.tenant)
    .fetch_all(&pool)
    .await?;

    let nodes: Vec<VizNode> = node_rows
        .into_iter()
        .map(|row| VizNode {
            id: row.get("node_id"),
            label: row.get("display_name"),
            degree: row.get("degree"),
            aliases: row.get("aliases"),
        })
        .collect();

    Ok(Json(Graph {
        tenant: query.tenant,
        nodes,
        links,
    }))
}
