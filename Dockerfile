# Two stages, because the build tree is enormous and none of it is needed to
# run: candle and aws-lc-rs both compile C, and the release artefacts alone run
# to gigabytes.

FROM rust:1.95-bookworm AS builder

# aws-lc-rs needs cmake and a C toolchain; the rust image already carries gcc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependencies first, against a stub, so editing src/ does not recompile
# candle. This layer is the expensive one and it changes only when the manifest
# does.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && echo 'fn main() {}' > src/main.rs \
    && echo '' > src/lib.rs \
    && cargo build --release --quiet 2>/dev/null || true \
    && rm -rf src

COPY src ./src
COPY migrations ./migrations
# viz.rs embeds the page and the vendored d3 with include_str!, so the build
# stage needs them. Neither reaches the server image: that stage copies one
# binary and nothing else.
COPY static ./static

# `migrations/` is embedded at compile time by sqlx::migrate!, so the runtime
# image does not need the directory. Touch main.rs so the stub above cannot be
# mistaken for current.
# Only the server. The graph viewer is a separate binary that binds loopback,
# reads every tenant and asks for no credential -- an operator tool, deliberately
# not something a shipped image can be talked into serving. Building just this
# bin also keeps the static assets it embeds out of the build context.
RUN touch src/main.rs src/lib.rs \
    && cargo build --release --bin mundu \
    && strip target/release/mundu

FROM debian:bookworm-slim AS runtime

# ca-certificates is not optional: the embedder fetches model weights from the
# HuggingFace hub over TLS on a cache miss, and a JWKS fetch is HTTPS too.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mundu

COPY --from=builder /build/target/release/mundu /usr/local/bin/mundu

USER mundu
WORKDIR /home/mundu

# Where candle looks for model weights. Mount a cache here to avoid paying the
# ~440MB download on every container start; the image deliberately does not
# bake the weights in.
ENV HF_HOME=/home/mundu/.cache/huggingface

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/mundu"]


# ---------------------------------------------------------------------------
# The graph viewer, as its own image.
#
# Deliberately not a second binary inside the server image. The viewer reads
# every tenant and asks for no credential, so the useful property is that the
# thing exposed to the network has no code path that can serve it -- and that
# property only holds if the binary is absent, not merely unreferenced.
#
# It carries no embedder: nothing here searches, so there is no query to embed.
# That is why this image needs no model cache and starts immediately, where the
# server spends ~20s loading weights.

FROM builder AS viz-builder

RUN cargo build --release --no-default-features --bin viz \
    && strip target/release/viz

FROM debian:bookworm-slim AS viz

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 mundu

COPY --from=viz-builder /build/target/release/viz /usr/local/bin/viz

USER mundu
WORKDIR /home/mundu

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/viz"]
