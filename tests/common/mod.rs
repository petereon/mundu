//! What every integration test needs before it can touch Postgres: which
//! database to use, and a refusal to use the wrong one.
//!
//! # Why this is a module and not a shared pool
//!
//! The eleven test binaries used to carry an identical copy of `store()`, and
//! the obvious tidy-up — one pool in a `static`, built once — is the one thing
//! that must not happen here. A sqlx pool is bound to the reactor that created
//! it, so a pool first built inside test A fails inside test B the moment A's
//! runtime shuts down, with `A Tokio 1.x context was found, but it is being
//! shutdown`: a failure that reads as a database problem and is not one.
//!
//! So [`store()`] still builds a fresh pool per call, exactly as the copies it
//! replaces did. What is shared is the *code*, not the connection. There is no
//! `static` in this file and there should not be one.
//!
//! # Why the guard
//!
//! This suite is destructive by construction. `fresh_tenant` deletes whole
//! tenants so an assertion means the same thing on the hundredth run as on the
//! first, and the relationship-coercion tests deliberately write into
//! `tenant_id = ''` — the curated global vocabulary every tenant reads — to
//! exercise cross-scope behaviour. Pointed at a database somebody is also
//! using for real, that is not untidiness, it is data loss, and two of its
//! forms are silent: junk relationship types become live coercion targets for
//! real writes, and `backfill_missing_name_vectors` stamps the canonical types
//! with whatever embedder the tests were run under, which tier-3 coercion then
//! filters out by design.
//!
//! Moving the default to a throwaway database fixes the accident. It does not
//! fix the mistake, because `DATABASE_URL` still overrides it. [`store()`]
//! therefore asks the database what it holds before writing to it, and refuses
//! if the answer is anything but test data.

// Each test binary compiles this module separately and uses the part of it that
// it needs -- `server_stdio` never builds a `Store`, and nothing but
// `server_stdio` spawns a subprocess. Without this, every binary warns about
// the half it does not call, and the suite is built under `-D warnings`.
#![allow(dead_code)]

use mundu::store::Store;

/// The database the suite runs against unless `DATABASE_URL` says otherwise:
/// the `postgres-test` service in `docker-compose.yml`, on tmpfs, holding
/// nothing anybody wants.
///
/// Deliberately **not** 55432. That is the development database, and it is the
/// one this suite must never be pointed at.
pub const DEFAULT_DATABASE_URL: &str = "postgres://mundu:mundu@localhost:55433/mundu";

/// What a tenant id has to start with for [`store()`] to accept the database
/// holding it.
///
/// Every `fresh_tenant` in the suite mints `test-<something>`, and the
/// authenticated ones (`test-http-acme-…`, `test-stdio-dev`) do too, so this is
/// a description of what the tests already do rather than a new rule they have
/// to be made to follow.
pub const TEST_TENANT_PREFIX: &str = "test-";

/// Where to connect, honouring the `DATABASE_URL` override so CI can redirect
/// the suite at a database it stood up itself.
pub fn database_url() -> String {
    std::env::var("DATABASE_URL").unwrap_or_else(|_| DEFAULT_DATABASE_URL.to_string())
}

/// A [`Store`] on a pool of its own, guarded and migrated.
///
/// The order is the point: connect, *then* refuse, *then* migrate. Connecting
/// runs two `SET`s and nothing else, so a refusal here happens before this
/// process has written a single row — the guard fires early rather than
/// complaining after the damage.
pub async fn store() -> Store {
    let url = database_url();
    let store = connect(&url).await;

    store
        .migrate()
        .await
        .unwrap_or_else(|error| panic!("could not apply migrations to {url}: {error}"));

    store
}

/// The URL to hand a spawned `mundu` process, checked first.
///
/// `server_stdio` runs the real binary from a synchronous helper, so it cannot
/// `await` the guard the way [`store()`] does — and blocking on it in place
/// would panic, because that helper is called from inside a `#[tokio::test]`
/// runtime. A scratch thread with a runtime of its own sidesteps both: it
/// borrows nothing from the caller's reactor, and everything it builds is
/// dropped before it returns.
pub fn database_url_for_subprocess() -> String {
    let url = database_url();

    let checked = url.clone();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("building a runtime for the database guard should succeed")
            .block_on(async move {
                // Dropped at the end of this block, with the runtime, before
                // the caller gets its URL back.
                let _guarded = connect(&checked).await;
            });
    })
    .join()
    // The guard panics, and this is the thread it panics on. Re-raising the
    // original payload puts the refusal in front of the developer instead of
    // `Any { .. }`.
    .unwrap_or_else(|payload| std::panic::resume_unwind(payload));

    url
}

/// Opens a pool and refuses to hand it back if the database holds real data.
///
/// Not public: there is no route to a connection in this suite that skips the
/// check.
async fn connect(url: &str) -> Store {
    let store = Store::connect(url).await.unwrap_or_else(|error| {
        panic!(
            "\n\
             the test database is not running. Start it with:\n\
             \n\
             \x20   docker compose up -d postgres-test\n\
             \n\
             These are integration tests against a real Postgres, so there is \
             nothing\n\
             to run without one.\n\
             \n\
             If you have the *development* database up on 55432, that is not a \
             substitute\n\
             and pointing DATABASE_URL at it will be refused: this suite deletes \
             tenants\n\
             and rewrites the shared relationship vocabulary. `postgres-test` is a \
             second\n\
             server on 55433 that exists to be thrown away.\n\
             \n\
             Tried: {}\n\
             Postgres said: {error}\n",
            redacted(url)
        )
    });

    refuse_unless_test_database(&store, url).await;

    store
}

/// Panics unless every tenant in `node` is a test tenant.
///
/// # Cost, and why this runs per call
///
/// One catalog lookup and one aggregate over a table that, on the database
/// this is supposed to be talking to, holds a few dozen rows. Every caller
/// already runs `migrate()` — an advisory lock plus a scan of the applied-
/// migrations table — immediately afterwards, which costs strictly more. The
/// saving from running it once per process is therefore unmeasurable, and
/// buying it would mean caching the result in a `static`, which is the shape
/// this file's header explains the suite has to avoid. Per call also means the
/// check cannot be outflanked by a test that opens a second connection.
///
/// # Why `node`
///
/// It is the table real data lands in and the one everything else references:
/// an `edge` needs two of them, a rating needs an edge. `relationship_type` is
/// no use as a signal — the curated global rows are seeded by migration `0005`,
/// so they are present in an empty test database too.
///
/// # Why there is no escape hatch
///
/// An environment variable that switches this off would reintroduce exactly the
/// hole it exists to close: the failure being prevented is somebody exporting
/// `DATABASE_URL` without thinking, and somebody who would do that would export
/// the override too. Nor is there a legitimate need for one. Wanting to run the
/// suite against production-shaped data is a real wish, and the way to have it
/// is to restore a dump into the *test* database, where the tenants are still
/// test tenants and nothing is lost when the suite deletes them. Wanting to run
/// it against data that matters is not a use case, it is the accident.
async fn refuse_unless_test_database(store: &Store, url: &str) {
    // A database that has never been migrated has no `node` table, and an empty
    // database is a perfectly legitimate target -- it is what `docker compose up
    // -d postgres-test` produces. Ask before selecting, so "nothing here yet"
    // reads as consent rather than as an error about a missing relation.
    let migrated: bool = sqlx::query_scalar("SELECT to_regclass('node') IS NOT NULL")
        .fetch_one(store.pool())
        .await
        .unwrap_or_else(|error| {
            panic!(
                "could not inspect {} before using it: {error}",
                redacted(url)
            )
        });

    if !migrated {
        return;
    }

    let offenders: Vec<(String, i64)> = sqlx::query_as(
        "SELECT tenant_id, count(*) FROM node \
         WHERE tenant_id NOT LIKE $1 \
         GROUP BY tenant_id \
         ORDER BY count(*) DESC, tenant_id",
    )
    .bind(format!("{TEST_TENANT_PREFIX}%"))
    .fetch_all(store.pool())
    .await
    .unwrap_or_else(|error| {
        panic!(
            "could not inspect {} before using it: {error}",
            redacted(url)
        )
    });

    if offenders.is_empty() {
        return;
    }

    let total: i64 = offenders.iter().map(|(_, count)| count).sum();
    let listed = offenders
        .iter()
        .map(|(tenant, count)| format!("      {tenant}: {count}"))
        .collect::<Vec<_>>()
        .join("\n");

    let pointed_here = if std::env::var("DATABASE_URL").is_ok() {
        "DATABASE_URL is set, which is what pointed the suite here."
    } else {
        "DATABASE_URL is unset, so this came from the compiled-in default -- \
         which should be the test database, and is not."
    };

    panic!(
        "\n\
         refusing to run the integration suite against {}\n\
         \n\
         It holds {total} node(s) under {} tenant(s) that are not test tenants:\n\
         \n\
         {listed}\n\
         \n\
         A test tenant's id starts with `{TEST_TENANT_PREFIX}`; anything else is \
         somebody's data.\n\
         This suite deletes whole tenants to keep its assertions repeatable, and \
         writes\n\
         into `tenant_id = ''` -- the curated global vocabulary every tenant reads \
         -- to\n\
         exercise cross-scope coercion. Running it here would not clutter this \
         database,\n\
         it would corrupt it, and two of the ways are silent.\n\
         \n\
         Nothing has been written. Start the throwaway database instead:\n\
         \n\
         \x20   docker compose up -d postgres-test\n\
         \n\
         It listens on 55433 and is the default. {pointed_here}\n",
        redacted(url),
        offenders.len(),
    );
}

/// `postgres://user:secret@host/db` without the secret.
///
/// The guard fires precisely when someone has aimed the suite at a database it
/// does not know, which is the one case where the URL in hand might carry a
/// password worth keeping out of a CI log.
fn redacted(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    let (authority, path) = match rest.find('/') {
        Some(cut) => (&rest[..cut], &rest[cut..]),
        None => (rest, ""),
    };
    let Some((userinfo, host)) = authority.rsplit_once('@') else {
        return url.to_string();
    };

    match userinfo.split_once(':') {
        Some((user, _)) => format!("{scheme}://{user}:***@{host}{path}"),
        None => url.to_string(),
    }
}
