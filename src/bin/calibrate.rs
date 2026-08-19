//! Measures what the real model actually does, so the similarity thresholds
//! this server ranks and merges on can be set from data rather than from
//! reasoning about embedding spaces in the abstract.
//!
//! Every threshold in `src/store` was a guess. The guesses were *argued for* —
//! that models of the `bge-base` class compress their similarity range, so a
//! figure which sounds strict in the abstract merges half the graph — but an
//! argument is not a measurement, and the argument could be wrong in either
//! direction. This binary embeds a labelled corpus with
//! `BAAI/bge-base-en-v1.5`, reports the cosine distribution of each regime, and
//! says for every constant whether the number in the code is defensible, too
//! high or too low.
//!
//! It changes nothing. Moving a threshold is a deliberate decision that belongs
//! in its own commit, taken by somebody looking at this output.
//!
//! ```text
//! cargo run --features candle-embedder --bin calibrate
//! cargo run --features candle-embedder --bin calibrate -- --json
//! ```
//!
//! No database is needed or opened: every number here comes out of the model
//! and the corpus below.
//!
//! # What this instrument can and cannot measure
//!
//! Cosine thresholds — the node resolver's four, and relationship coercion's
//! one — are directly measurable, and are measured.
//!
//! The search tuning block is not all of a kind. `CUTOFF_FRACTION_*` apply to
//! *fused RRF scores*, which are a function of rank alone: no cosine corpus can
//! calibrate them, and treating them as cosines would be the same category
//! error the thresholds themselves were guessed with. What is reported for
//! those is arithmetic on `RRF_K`, labelled as such, plus the one thing the
//! corpus does bear on — how far down the semantic list a true answer actually
//! falls.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use mundu::embed::{CandleEmbedder, Embedder};
use mundu::store::node::{
    AMBIGUOUS_SIMILARITY, CONFIDENT_SIMILARITY, CONFIDENT_SIMILARITY_SHORT, SHORT_NAME_CHARS,
    names_contradict,
};
use mundu::store::relationship::{
    CONFIDENT_SIMILARITY as TYPE_CONFIDENT_SIMILARITY, are_mechanical_inverses,
};
use mundu::store::search::{
    CANDIDATE_DEPTH, CUTOFF_FRACTION_RELAXED, CUTOFF_FRACTION_STANDARD, CUTOFF_FRACTION_STRICT,
    DEFAULT_LIMIT, MAX_LIMIT, RRF_K, SALIENCE_BAND_FLOOR, SALIENCE_BAND_SPAN,
};

// ---------------------------------------------------------------------------
// The corpus
//
// Data, not code. Every pair is one line a reader can disagree with in
// isolation, which is the only way a labelled corpus stays honest: a label
// buried in a loop body is a label nobody will ever audit.
// ---------------------------------------------------------------------------

/// What a pair is, and therefore what a threshold is supposed to do with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Label {
    /// The two texts denote the same thing. A threshold must admit this pair,
    /// or the graph fragments.
    Same,
    /// Related, and not the same. The hard negatives — same domain, adjacent
    /// meaning, often one word apart. A threshold must reject these, or the
    /// resolver silently fuses distinct entities.
    Related,
    /// Nothing in common. These measure the *floor* of the space, which is the
    /// number the compressed-range argument is really about.
    Unrelated,
    /// Opposite directions of one relation. Rejected like `Related`, counted
    /// separately, because these are what `relationship_inverse_pair` exists
    /// for and their numbers decide whether that guard is load-bearing.
    Inverse,
    /// The same string denoting two different things. Excluded from every
    /// statistic on purpose: the cosine is 1.0 by construction, so no threshold
    /// can separate them and including them would corrupt the negative
    /// population with a number that says nothing about the model.
    Homonym,
}

impl Label {
    /// Whether a threshold is supposed to admit this pair.
    const fn is_positive(self) -> bool {
        matches!(self, Self::Same)
    }

    /// Whether this pair belongs in the populations a threshold is fitted to.
    const fn is_measured(self) -> bool {
        !matches!(self, Self::Homonym)
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Same => "same",
            Self::Related => "related",
            Self::Unrelated => "unrelated",
            Self::Inverse => "inverse",
            Self::Homonym => "homonym",
        }
    }
}

/// One labelled pair of texts.
#[derive(Debug, Clone, Copy)]
struct Pair {
    left: &'static str,
    right: &'static str,
    label: Label,
    /// Why this pair is labelled the way it is, when that is not obvious.
    note: &'static str,
}

const fn pair(left: &'static str, right: &'static str, label: Label, note: &'static str) -> Pair {
    Pair {
        left,
        right,
        label,
        note,
    }
}

use Label::{Homonym, Inverse, Related, Same, Unrelated};

/// Entity names where **both** sides are longer than [`SHORT_NAME_CHARS`], so
/// the resolver would apply the loose pair of bars to either direction.
///
/// Pairs whose two names canonicalise to the same `node_id` are avoided
/// deliberately: those never reach the vector tier at all, so measuring them
/// would flatter the threshold with cases it is never asked about.
const LONG_NAME_PAIRS: &[Pair] = &[
    pair("Project Phoenix", "Phoenix Project", Same, "word order"),
    pair("Acme Corporation", "The Acme Corporation", Same, "article"),
    pair(
        "Kubernetes Control Plane",
        "K8s Control Plane",
        Same,
        "abbreviation",
    ),
    pair("The Payments Team", "Payments Team", Same, "article"),
    pair("Elizabeth Bennet", "Elizabeth Bennett", Same, "spelling"),
    pair(
        "United States Department of Energy",
        "US Department of Energy",
        Same,
        "abbreviation",
    ),
    pair(
        "Amazon Web Services",
        "Amazon Web Services Inc.",
        Same,
        "legal suffix",
    ),
    pair(
        "Postgres Migration Runbook",
        "PostgreSQL Migration Runbook",
        Same,
        "product spelling",
    ),
    pair(
        "billing-service-v2",
        "the v2 billing service",
        Same,
        "slug against prose",
    ),
    pair("Dr. Katherine Chen", "Katherine Chen", Same, "honorific"),
    pair(
        "San Francisco Bay Area",
        "the SF Bay Area",
        Same,
        "abbreviation",
    ),
    pair(
        "Customer Success Platform",
        "Customer Success Platform (CSP)",
        Same,
        "parenthetical",
    ),
    pair(
        "internal-tools-team",
        "the internal tooling team",
        Same,
        "slug against prose",
    ),
    pair(
        "Quarterly Business Review",
        "Quarterly Business Reviews",
        Same,
        "plural",
    ),
    pair(
        "Project Phoenix",
        "Project Chimera",
        Related,
        "sibling codename",
    ),
    pair(
        "Acme Corporation",
        "Acme Holdings",
        Related,
        "sibling legal entity",
    ),
    pair(
        "Kubernetes Control Plane",
        "Kubernetes Data Plane",
        Related,
        "sibling component",
    ),
    pair(
        "The Payments Team",
        "The Platform Team",
        Related,
        "sibling team",
    ),
    pair(
        "Dr. Katherine Chen",
        "Dr. Michael Chen",
        Related,
        "shared surname",
    ),
    pair(
        "United States Department of Energy",
        "United States Department of Commerce",
        Related,
        "sibling department",
    ),
    pair(
        "billing-service-v2",
        "billing-service-v1",
        Related,
        "distinct deployed versions",
    ),
    pair(
        "San Francisco Bay Area",
        "Greater Boston Area",
        Related,
        "sibling region",
    ),
    pair(
        "Customer Success Platform",
        "Customer Support Platform",
        Related,
        "one word apart",
    ),
    pair(
        "Quarterly Business Review",
        "Annual Business Review",
        Related,
        "one word apart",
    ),
    pair(
        "Postgres Migration Runbook",
        "Postgres Backup Runbook",
        Related,
        "one word apart",
    ),
    pair(
        "Amazon Web Services",
        "Google Cloud Platform",
        Related,
        "competitor",
    ),
    pair("Elizabeth Bennet", "Charlotte Lucas", Related, "two people"),
    pair(
        "Internal Tools Team",
        "External Partners Team",
        Related,
        "sibling team",
    ),
    pair("Project Phoenix", "Emperor Penguin Colony", Unrelated, ""),
    pair(
        "Acme Corporation",
        "Sourdough Starter Recipe",
        Unrelated,
        "",
    ),
    pair(
        "Kubernetes Control Plane",
        "Baroque Chamber Orchestra",
        Unrelated,
        "",
    ),
    pair(
        "The Payments Team",
        "Mount Kilimanjaro Base Camp",
        Unrelated,
        "",
    ),
    pair(
        "San Francisco Bay Area",
        "Postgres Migration Runbook",
        Unrelated,
        "",
    ),
    pair("Dr. Katherine Chen", "Amazon Web Services", Unrelated, ""),
    pair(
        "billing-service-v2",
        "Antarctic Ice Shelf Survey",
        Unrelated,
        "",
    ),
    pair(
        "Customer Success Platform",
        "Elizabeth Bennet",
        Unrelated,
        "",
    ),
];

/// Entity names where **both** sides are at most [`SHORT_NAME_CHARS`], so the
/// resolver applies the stricter pair of bars.
///
/// This is the regime the stricter band was invented for. If the negatives here
/// do not crowd higher than the negatives above, the band has no justification.
const SHORT_NAME_PAIRS: &[Pair] = &[
    pair("K8s", "Kubernetes", Same, "abbreviation"),
    pair("Postgres", "PostgreSQL", Same, "product spelling"),
    pair("NYC", "New York", Same, "abbreviation"),
    pair("JS", "JavaScript", Same, "abbreviation"),
    pair("AWS", "Amazon AWS", Same, "expansion"),
    pair("Ada L.", "Ada Lovelace", Same, "initial"),
    pair("Acme", "ACME Inc", Same, "legal suffix"),
    pair("TLS", "SSL/TLS", Same, "conventional pairing"),
    pair("GPT-4", "GPT4", Same, "punctuation"),
    pair("LA", "Los Angeles", Same, "abbreviation"),
    pair("Jane D.", "Jane Doe", Same, "initial"),
    pair("iOS", "Apple iOS", Same, "vendor prefix"),
    pair("Cloudflare", "Cloud Flare", Same, "spacing"),
    pair("Salesforce", "Sales Force", Same, "spacing"),
    pair("AWS", "GCP", Related, "competing cloud"),
    pair("Alice", "Alicia", Related, "two people, one letter apart"),
    pair("Phoenix", "Chimera", Related, "sibling codename"),
    pair(
        "Phoenix",
        "Phoenix, AZ",
        Related,
        "the project against the city",
    ),
    pair("Redis", "Memcached", Related, "competing cache"),
    pair("Bob", "Rob", Related, "two people, one letter apart"),
    pair("Q3", "Q4", Related, "adjacent period"),
    pair("v1.2.0", "v1.3.0", Related, "adjacent version"),
    pair("Ada", "Ida", Related, "two people, one letter apart"),
    pair("Python", "Ruby", Related, "competing language"),
    pair("Docker", "Podman", Related, "competing runtime"),
    pair("Jane Doe", "John Doe", Related, "two people"),
    pair("NYC", "Newark", Related, "neighbouring city"),
    pair("iOS", "Android", Related, "competing platform"),
    pair("TLS", "SSH", Related, "sibling protocol"),
    pair("LA", "Las Vegas", Related, "shared initials"),
    pair(
        "Barcelona",
        "Barcelona FC",
        Related,
        "the city against the club",
    ),
    pair("Terraform", "Ansible", Related, "competing tool"),
    pair("Anthropic", "Microsoft", Related, "two companies"),
    pair("Melbourne", "Manchester", Related, "two cities"),
    pair("K8s", "Basalt", Unrelated, ""),
    pair("Alice", "Tungsten", Unrelated, ""),
    pair("AWS", "Oregano", Unrelated, ""),
    pair("Q3", "Penguin", Unrelated, ""),
    pair("Postgres", "Sonnet", Unrelated, ""),
    pair("Phoenix", "Trombone", Unrelated, ""),
    pair("Ada", "Quartz", Unrelated, ""),
    pair("iOS", "Meadow", Unrelated, ""),
    pair("Cloudflare", "Marmalade", Unrelated, ""),
    pair("Wikipedia", "Limestone", Unrelated, ""),
    pair(
        "Phoenix",
        "Phoenix",
        Homonym,
        "the project against the city, written identically",
    ),
    pair(
        "Mercury",
        "Mercury",
        Homonym,
        "the planet against the element, written identically",
    ),
];

/// The note on a pair the exact tiers already resolve, so the vector tier never
/// sees it. Held as a constant rather than repeated, because the report counts
/// these and a typo would silently undercount them.
const EARLIER_TIER: &str = "already a tier-2 alias";

/// Relationship type names, in the upper-snake form
/// `relationship::normalise` folds them to — which is the form the store
/// actually embeds, so measuring anything else would measure a string this
/// server never sees.
const RELATIONSHIP_TYPE_PAIRS: &[Pair] = &[
    pair("USES", "UTILIZES", Same, EARLIER_TIER),
    pair("USES", "LEVERAGES", Same, ""),
    pair("USES", "EMPLOYS", Same, ""),
    pair("DEPENDS_ON", "RELIES_ON", Same, ""),
    pair("DEPENDS_ON", "REQUIRES", Same, EARLIER_TIER),
    pair("OWNS", "IS_OWNER_OF", Same, ""),
    pair("OWNS", "POSSESSES", Same, ""),
    pair("AUTHORED_BY", "WRITTEN_BY", Same, EARLIER_TIER),
    pair("AUTHORED_BY", "PENNED_BY", Same, ""),
    pair("LOCATED_IN", "SITUATED_IN", Same, ""),
    pair("MEMBER_OF", "BELONGS_TO", Same, EARLIER_TIER),
    pair("PREFERS", "FAVOURS", Same, ""),
    pair("REPORTS_TO", "REPORTING_TO", Same, ""),
    pair("MANAGES", "OVERSEES", Same, ""),
    pair("USES", "DEPENDS_ON", Related, "distinct curated types"),
    pair("OWNS", "MEMBER_OF", Related, "distinct curated types"),
    pair("PART_OF", "MEMBER_OF", Related, "distinct curated types"),
    pair("REPORTS_TO", "MEMBER_OF", Related, "distinct curated types"),
    pair("AUTHORED_BY", "OWNS", Related, "distinct curated types"),
    pair("LOCATED_IN", "PART_OF", Related, "distinct curated types"),
    pair("PREFERS", "USES", Related, "distinct curated types"),
    pair(
        "RELATED_TO",
        "DEPENDS_ON",
        Related,
        "the catch-all against a real type",
    ),
    pair("MENTORS", "MANAGES", Related, "adjacent, not equal"),
    pair("FUNDS", "OWNS", Related, "adjacent, not equal"),
    pair("REPLACES", "DEPENDS_ON", Related, ""),
    pair("COLLABORATES_WITH", "REPORTS_TO", Related, ""),
    pair("USES", "LOCATED_IN", Related, "distinct curated types"),
    pair("PREFERS", "AUTHORED_BY", Related, "distinct curated types"),
    pair("REPORTS_TO", "MANAGES", Inverse, ""),
    pair("OWNS", "OWNED_BY", Inverse, ""),
    pair("DEPENDS_ON", "REQUIRED_BY", Inverse, ""),
    pair("PART_OF", "HAS_PART", Inverse, ""),
    pair("MEMBER_OF", "HAS_MEMBER", Inverse, ""),
    pair("AUTHORED_BY", "AUTHORED", Inverse, ""),
    pair("USES", "USED_BY", Inverse, ""),
    pair("LOCATED_IN", "CONTAINS", Inverse, ""),
    pair("MANAGES", "MANAGED_BY", Inverse, ""),
    pair("FUNDS", "FUNDED_BY", Inverse, ""),
    pair("MENTORS", "MENTORED_BY", Inverse, ""),
    pair("SUPERSEDES", "SUPERSEDED_BY", Inverse, ""),
];

/// Every row of `relationship_inverse_pair` as migration `0001_init.sql` seeds
/// it.
///
/// Duplicated here rather than imported because it is SQL, not Rust, and there
/// is nothing to import. The point of holding it is coverage: an inverse pair
/// the table does not know about is one the direction guard cannot refuse,
/// however close the two names sit.
const SEEDED_INVERSE_PAIRS: &[(&str, &str)] = &[
    ("REPORTS_TO", "MANAGES"),
    ("OWNS", "OWNED_BY"),
    ("DEPENDS_ON", "REQUIRED_BY"),
    ("PART_OF", "HAS_PART"),
    ("MEMBER_OF", "HAS_MEMBER"),
    ("AUTHORED_BY", "AUTHORED"),
    ("USES", "USED_BY"),
    ("LOCATED_IN", "CONTAINS"),
];

/// One natural-language question, the stored sentence that answers it, and
/// same-topic sentences that do not.
///
/// The near misses are the whole point of the group. A question is trivially
/// far from a sentence about penguins; what a retrieval threshold would have to
/// survive is a sentence about the same service that happens not to answer
/// anything.
#[derive(Debug, Clone, Copy)]
struct QueryCase {
    question: &'static str,
    proof: &'static str,
    near_misses: &'static [&'static str],
}

const QUERY_CASES: &[QueryCase] = &[
    QueryCase {
        question: "who maintains the billing service",
        proof: "The billing service is maintained by the payments team.",
        near_misses: &[
            "The billing service was migrated to Postgres 16 in March.",
            "The payments team runs a weekly on-call rotation.",
        ],
    },
    QueryCase {
        question: "which database does the search index use",
        proof: "The search index is stored in Postgres using the pgvector extension.",
        near_misses: &[
            "The search index is rebuilt nightly at 02:00 UTC.",
            "Postgres 16 is the supported database version for all services.",
        ],
    },
    QueryCase {
        question: "who does Katherine Chen report to",
        proof: "Katherine Chen reports to the VP of Engineering, Marcus Webb.",
        near_misses: &[
            "Katherine Chen joined the platform team in 2021.",
            "Marcus Webb owns the engineering organisation's hiring plan.",
        ],
    },
    QueryCase {
        question: "what is the retention period for audit logs",
        proof: "Audit logs are retained for 400 days before deletion.",
        near_misses: &[
            "Audit logs are written to an append-only ledger table.",
            "Application logs are retained for 30 days.",
        ],
    },
    QueryCase {
        question: "where is the Frankfurt data centre",
        proof: "The Frankfurt data centre is on the Hanauer Landstrasse industrial park.",
        near_misses: &[
            "The Frankfurt data centre serves all EU customers.",
            "The Dublin data centre is in the Grange Castle business park.",
        ],
    },
    QueryCase {
        question: "how do I rotate the signing key",
        proof: "To rotate the signing key, publish the new JWK, wait one cache TTL, then \
                retire the old key.",
        near_misses: &[
            "Signing keys are held in the hardware security module.",
            "The signing key is rotated automatically every 90 days.",
        ],
    },
    QueryCase {
        question: "what caused the March outage",
        proof: "The March outage was caused by connection pool exhaustion in the checkout \
                service.",
        near_misses: &[
            "The March outage lasted 47 minutes and affected checkout.",
            "The February outage was caused by a bad DNS change.",
        ],
    },
    QueryCase {
        question: "which team owns Project Phoenix",
        proof: "Project Phoenix is owned by the internal tools team.",
        near_misses: &[
            "Project Phoenix replaced the legacy reporting stack.",
            "The internal tools team also owns the deployment pipeline.",
        ],
    },
    QueryCase {
        question: "what licence is the SDK released under",
        proof: "The SDK is released under the Apache 2.0 licence.",
        near_misses: &[
            "The SDK is published to the public package registry every Friday.",
            "The command line client is released under the MIT licence.",
        ],
    },
    QueryCase {
        question: "how many regions does the platform run in",
        proof: "The platform runs in six regions across three continents.",
        near_misses: &[
            "The platform is deployed with a blue-green strategy in every region.",
            "Six engineers maintain the regional failover tooling.",
        ],
    },
];

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// The distribution of one population of cosines.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct Summary {
    count: usize,
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    /// Population standard deviation. Population rather than sample because
    /// these are all the pairs there are, not a draw from a larger set.
    stddev: f64,
}

impl Summary {
    /// `None` for an empty population: a mean of nothing is not zero, and a
    /// report that printed 0.0000 there would read as a measurement.
    fn of(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / count as f64;

        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = if count.is_multiple_of(2) {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        Some(Self {
            count,
            min: sorted[0],
            max: sorted[count - 1],
            mean,
            median,
            stddev: variance.sqrt(),
        })
    }
}

/// How a threshold performs against a labelled population.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct Errors {
    /// Negatives the threshold admits. For the node resolver these are silent,
    /// irreversible merges of distinct entities.
    false_merges: usize,
    /// Positives the threshold rejects. Fragmentation: recoverable, visible,
    /// and much the cheaper failure.
    missed_merges: usize,
}

impl Errors {
    const fn total(self) -> usize {
        self.false_merges + self.missed_merges
    }
}

/// Counts both kinds of mistake a threshold makes. Admission is `>=`, matching
/// every comparison in `src/store`.
fn errors_at(threshold: f64, positives: &[f64], negatives: &[f64]) -> Errors {
    Errors {
        false_merges: negatives
            .iter()
            .filter(|value| **value >= threshold)
            .count(),
        missed_merges: positives.iter().filter(|value| **value < threshold).count(),
    }
}

/// The lowest two-decimal threshold that admits no negative at all.
///
/// Rounded to two places because a threshold quoted to four is false precision
/// on a corpus this size, and always *up* past the worst negative, so the
/// returned value is strictly greater than every negative rather than equal to
/// one of them — `>=` would otherwise admit the very pair it was derived from.
fn no_false_merge_threshold(negatives: &[f64]) -> Option<f64> {
    let worst = negatives.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !worst.is_finite() {
        return None;
    }

    Some(((worst * 100.0).floor() + 1.0) / 100.0)
}

/// The threshold that makes the fewest mistakes overall.
///
/// Ties break toward the *higher* threshold. Both regimes this is used for have
/// asymmetric costs — an over-merged node cannot be told apart afterwards,
/// while a fragmented one is a duplicate somebody can merge — so when two cuts
/// are indistinguishable on this corpus the conservative one is the answer.
fn balanced_threshold(positives: &[f64], negatives: &[f64]) -> Option<(f64, Errors)> {
    let mut candidates: Vec<f64> = positives
        .iter()
        .chain(negatives)
        .map(|value| ((value * 100.0).ceil()) / 100.0)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.push(1.01);
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();

    candidates
        .into_iter()
        .map(|threshold| (threshold, errors_at(threshold, positives, negatives)))
        .min_by(|left, right| {
            left.1
                .total()
                .cmp(&right.1.total())
                .then_with(|| right.0.total_cmp(&left.0))
        })
}

/// 1-based position of `target` in a descending ordering of `scores`, which is
/// expected to contain `target` itself. Ties count against the target.
fn rank_of(target: f64, scores: &[f64]) -> usize {
    1 + scores.iter().filter(|score| **score > target).count()
}

/// Cosine similarity. The embedder returns unit vectors, so this is a dot
/// product in practice, but the division costs nothing and keeps the number
/// correct if that ever stops being true.
fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot: f64 = left
        .iter()
        .zip(right)
        .map(|(l, r)| f64::from(*l) * f64::from(*r))
        .sum();
    let norm = |vector: &[f32]| -> f64 {
        vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt()
    };

    let scale = norm(left) * norm(right);
    if scale == 0.0 { 0.0 } else { dot / scale }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// One pair, measured.
#[derive(Debug, Clone, Serialize)]
struct Measurement {
    left: String,
    right: String,
    label: Label,
    note: &'static str,
    cosine: f64,
    /// Characters in the shorter of the two names. The resolver picks its bars
    /// off the length of the incoming name, and the shorter side is the one
    /// that makes a pair unstable, so this is what the length-regime analysis
    /// buckets on.
    shorter_chars: usize,
}

impl Measurement {
    fn is_positive(&self) -> bool {
        self.label.is_positive()
    }
}

/// A named regime, its pairs, and their distributions.
#[derive(Debug, Clone, Serialize)]
struct GroupReport {
    name: &'static str,
    /// How the two sides were embedded — the asymmetric case is not an
    /// afterthought, it is the reason the trait has two methods.
    embedding: &'static str,
    measurements: Vec<Measurement>,
    same: Option<Summary>,
    related: Option<Summary>,
    unrelated: Option<Summary>,
    inverse: Option<Summary>,
    /// All of `related`, `unrelated` and `inverse` together: the population a
    /// threshold has to keep out.
    negatives: Option<Summary>,
    /// `min(same) - max(negatives)`. Negative means the populations overlap and
    /// no threshold separates them.
    margin: Option<f64>,
}

impl GroupReport {
    fn values(&self, label: Label) -> Vec<f64> {
        self.measurements
            .iter()
            .filter(|measurement| measurement.label == label)
            .map(|measurement| measurement.cosine)
            .collect()
    }

    /// Every cosine carrying one of `labels`.
    fn values_of(&self, labels: &[Label]) -> Vec<f64> {
        self.measurements
            .iter()
            .filter(|measurement| labels.contains(&measurement.label))
            .map(|measurement| measurement.cosine)
            .collect()
    }

    fn positives(&self) -> Vec<f64> {
        self.measurements
            .iter()
            .filter(|measurement| measurement.is_positive())
            .map(|measurement| measurement.cosine)
            .collect()
    }

    /// Everything a threshold must reject, minus the homonyms, which no
    /// threshold can reject.
    fn negative_values(&self) -> Vec<f64> {
        self.measurements
            .iter()
            .filter(|measurement| !measurement.is_positive() && measurement.label.is_measured())
            .map(|measurement| measurement.cosine)
            .collect()
    }

    fn build(name: &'static str, embedding: &'static str, measurements: Vec<Measurement>) -> Self {
        let mut report = Self {
            name,
            embedding,
            measurements,
            same: None,
            related: None,
            unrelated: None,
            inverse: None,
            negatives: None,
            margin: None,
        };

        report.same = Summary::of(&report.values(Same));
        report.related = Summary::of(&report.values(Related));
        report.unrelated = Summary::of(&report.values(Unrelated));
        report.inverse = Summary::of(&report.values(Inverse));
        report.negatives = Summary::of(&report.negative_values());
        report.margin = match (report.same, report.negatives) {
            (Some(same), Some(negatives)) => Some(same.min - negatives.max),
            _ => None,
        };

        report
    }
}

/// How each question fared against the whole document pool.
#[derive(Debug, Clone, Serialize)]
struct QueryReport {
    question: String,
    /// Cosine between the question (query side) and the sentence that answers
    /// it (document side).
    answer: f64,
    /// Its 1-based rank among every document in the pool, ordered by cosine.
    /// This, not the absolute value, is what the semantic candidate list is
    /// built from.
    rank: usize,
    /// The closest same-topic sentence that does not answer the question.
    best_near_miss: f64,
    /// The closest sentence from another topic entirely.
    best_unrelated: f64,
    /// `answer - max(everything else)`. Negative means the wrong sentence won.
    margin: f64,
}

/// Whether the number in the code survives the measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum Verdict {
    /// The measurement supports the current value.
    Defensible,
    /// It admits pairs it must reject.
    TooLow,
    /// It rejects pairs a lower value would admit while still rejecting every
    /// negative.
    TooHigh,
    /// The corpus cannot say.
    Inconclusive,
}

impl Verdict {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Defensible => "DEFENSIBLE",
            Self::TooLow => "TOO LOW",
            Self::TooHigh => "TOO HIGH",
            Self::Inconclusive => "INCONCLUSIVE",
        }
    }
}

/// One constant, judged.
#[derive(Debug, Clone, Serialize)]
struct ConstantVerdict {
    constant: &'static str,
    /// What the constant decides. Half of any verdict on a threshold is the
    /// contract it was judged under.
    purpose: &'static str,
    /// What the recommended number does not say.
    caveat: &'static str,
    /// Where the recommendation sits relative to the two populations.
    boundary: Option<Boundary>,
    /// Set when the recommendation should not be read as a number to adopt.
    warning: Option<String>,
    current: f64,
    /// What the measurement says it should be, or `None` when the corpus does
    /// not support a recommendation.
    recommended: Option<f64>,
    verdict: Verdict,
    rationale: String,
}

/// The narrowest gap between two populations that is worth quoting a
/// threshold from.
///
/// Recommendations are rounded to two decimals, so a gap under this is a
/// boundary the rounding alone could have invented: move one pair of the corpus
/// and it moves with it. The value is still reported — with the gap beside it,
/// because the gap is the finding.
const THIN_GAP: f64 = 0.02;

/// Where a recommendation sits relative to the two populations it divides.
///
/// This is the number that says whether a calibration happened at all. A
/// threshold quoted without it is a decimal point standing in for a decision.
#[derive(Debug, Clone, Copy, Serialize)]
struct Boundary {
    /// `min(admit) - max(reject)`. Negative when the populations overlap, and
    /// then there is no threshold that gets both sides right.
    gap: f64,
    /// How many of the pairs it must admit score below the highest pair it must
    /// reject. Zero exactly when the populations separate.
    overlap_depth: usize,
}

impl Boundary {
    fn of(positives: &[f64], negatives: &[f64]) -> Option<Self> {
        let lowest_admit = positives.iter().copied().fold(f64::INFINITY, f64::min);
        let highest_reject = negatives.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !lowest_admit.is_finite() || !highest_reject.is_finite() {
            return None;
        }

        Some(Self {
            gap: lowest_admit - highest_reject,
            overlap_depth: positives
                .iter()
                .filter(|value| **value < highest_reject)
                .count(),
        })
    }

    /// What is wrong with quoting this recommendation as a number, if anything.
    fn warning(&self, admitted: usize) -> Option<String> {
        if self.overlap_depth > 0 {
            return Some(format!(
                "the populations OVERLAP. {} of the {admitted} pairs this must admit score \
                 below the highest pair it must reject, so no threshold gets both sides \
                 right and the recommendation is the top of the overlap rather than a \
                 boundary in the model. Which value to pick is a choice of which failure \
                 to accept, and only the cost of each is measurable here.",
                self.overlap_depth
            ));
        }

        if self.gap < THIN_GAP {
            return Some(format!(
                "the populations separate by only {:.4}, which is inside the rounding of \
                 the recommendation itself. The direction is trustworthy; the second \
                 decimal is not.",
                self.gap
            ));
        }

        None
    }
}

/// The smallest population either side of a threshold that this binary will
/// draw a conclusion from.
///
/// Below it the honest answer is [`Verdict::Inconclusive`]. A recommendation
/// fitted to three pairs is a number with a decimal point and no evidence, and
/// that is worse than no calibration at all, because it looks like one.
const MINIMUM_POPULATION: usize = 6;

/// What a threshold is for, and therefore what counts as a mistake against it.
///
/// Getting this wrong is the easiest way to produce a confident, wrong report.
/// The confident bars and the ambiguous floors are not the same kind of number:
/// a candidate floor that admits a `Related` pair is doing its job, and scoring
/// it as an error would condemn the one behaviour it exists for.
#[derive(Debug, Clone, Copy)]
struct Contract {
    /// Labels the threshold must admit.
    admit: &'static [Label],
    /// Labels it must keep out.
    reject: &'static [Label],
    /// Whether admitting a rejected pair is the expensive failure. When it is,
    /// the recommendation clears every rejected pair even at the cost of
    /// admitted ones; when it is not, the fewest-errors cut is the answer.
    admission_is_costly: bool,
    /// What the threshold does, printed beside the verdict so a reader can
    /// argue with the contract rather than having to infer it.
    purpose: &'static str,
    /// Anything the number alone would mislead a reader about.
    caveat: &'static str,
}

/// Attaching one node to another cannot be undone by anything a caller can
/// see, and leaves no trace that a decision was even made. A confident bar
/// therefore has to clear every pair that is not the same entity, and losing
/// true matches to a duplicate node is the cheap side of the trade.
const MERGE_CONTRACT: Contract = Contract {
    admit: &[Same],
    reject: &[Related, Unrelated, Inverse],
    admission_is_costly: true,
    purpose: "attaches an incoming name to an existing node — irreversible",
    caveat: "",
};

/// Coercing a type is not as bad as merging a node — the vocabulary is a small
/// table an operator can read, and `edge.raw_relationship_type` keeps whatever
/// the caller actually wrote — with one exception. Coercing a name onto its own
/// inverse reverses the direction of every fact written with it, and nothing
/// downstream can tell. So the bar still has to clear every rejected pair.
const TYPE_COERCION_CONTRACT: Contract = Contract {
    admit: &[Same],
    reject: &[Related, Unrelated, Inverse],
    admission_is_costly: true,
    purpose: "folds an incoming type name onto an existing one — reverses facts if the \
              two are inverses",
    caveat: "",
};

/// A candidate floor only decides what an agent is *told about*. A related name
/// surfaced as a near miss is the feature, not a mistake; an unrelated one is
/// noise. So the population it must admit is `same` and `related` together, and
/// the only thing it must keep out is the floor of the space.
const CANDIDATE_CONTRACT: Contract = Contract {
    admit: &[Same, Related],
    reject: &[Unrelated],
    admission_is_costly: false,
    purpose: "reports a near miss without attaching to it — recoverable",
    caveat: "the recommendation is the boundary between related and unrelated, which is \
             all a corpus can locate. Where inside that range the floor belongs is a \
             decision about how chatty the candidate list should be, and this instrument \
             cannot make it.",
};

/// Judges one constant against one group under its contract.
fn judge(
    constant: &'static str,
    current: f64,
    group: &GroupReport,
    contract: Contract,
) -> ConstantVerdict {
    let positives = group.values_of(contract.admit);
    let negatives = group.values_of(contract.reject);

    if positives.len() < MINIMUM_POPULATION || negatives.len() < MINIMUM_POPULATION {
        return ConstantVerdict {
            constant,
            purpose: contract.purpose,
            caveat: contract.caveat,
            boundary: None,
            warning: None,
            current,
            recommended: None,
            verdict: Verdict::Inconclusive,
            rationale: format!(
                "only {} pairs it must admit and {} it must reject in `{}`; too few to fit \
                 a threshold to",
                positives.len(),
                negatives.len(),
                group.name
            ),
        };
    }

    let here = errors_at(current, &positives, &negatives);
    let recommended = if contract.admission_is_costly {
        no_false_merge_threshold(&negatives)
    } else {
        balanced_threshold(&positives, &negatives).map(|(threshold, _)| threshold)
    };

    // Named in every branch, not only the failing one: the pair sitting at the
    // top of the reject population is what any recommendation is really being
    // set by, and a reader has to be able to see whether they believe it.
    let worst = group
        .measurements
        .iter()
        .filter(|measurement| contract.reject.contains(&measurement.label))
        .max_by(|left, right| left.cosine.total_cmp(&right.cosine));
    let boundary = worst.map_or_else(String::new, |measurement| {
        format!(
            " The highest pair it must reject is `{}` / `{}` at {:.4}{}.",
            measurement.left,
            measurement.right,
            measurement.cosine,
            if measurement.label == Inverse {
                ", which is an INVERSE"
            } else {
                ""
            }
        )
    });
    let price = recommended.map_or_else(String::new, |threshold| {
        let cost = errors_at(threshold, &positives, &negatives);
        format!(
            " At {threshold:.2} it would reject {} of the {} it must admit.",
            cost.missed_merges,
            positives.len()
        )
    });

    let (verdict, rationale) = if here.false_merges > 0 {
        (
            Verdict::TooLow,
            format!(
                "admits {} of the {} pairs it must reject.{boundary}{price}",
                here.false_merges,
                negatives.len(),
            ),
        )
    } else if let Some(threshold) = recommended
        && threshold < current
        && errors_at(threshold, &positives, &negatives).missed_merges < here.missed_merges
    {
        let lower = errors_at(threshold, &positives, &negatives);
        (
            Verdict::TooHigh,
            format!(
                "rejects {} of the {} pairs it must admit; {threshold:.2} rejects only {} \
                 and still admits {} of {} it must reject.{boundary}",
                here.missed_merges,
                positives.len(),
                lower.missed_merges,
                lower.false_merges,
                negatives.len()
            ),
        )
    } else {
        (
            Verdict::Defensible,
            format!(
                "admits 0 of the {} pairs it must reject and rejects {} of the {} it must \
                 admit.{boundary}",
                negatives.len(),
                here.missed_merges,
                positives.len(),
            ),
        )
    };

    let boundary = Boundary::of(&positives, &negatives);
    let warning = boundary.and_then(|boundary| boundary.warning(positives.len()));

    ConstantVerdict {
        constant,
        purpose: contract.purpose,
        caveat: contract.caveat,
        boundary,
        warning,
        current,
        recommended,
        verdict,
        rationale,
    }
}

/// One band of name lengths and how the negatives behave inside it.
#[derive(Debug, Clone, Serialize)]
struct LengthBucket {
    label: &'static str,
    low: usize,
    high: usize,
    positives: Option<Summary>,
    negatives: Option<Summary>,
}

/// The bands the length-regime analysis reports on.
///
/// The boundary at 12 is [`SHORT_NAME_CHARS`], so the question "does anything
/// change at 12?" is answered by comparing the second band with the third
/// rather than by assuming it.
const LENGTH_BUCKETS: &[(&str, usize, usize)] = &[
    ("1-6 chars", 1, 6),
    ("7-12 chars", 7, SHORT_NAME_CHARS),
    ("13-20 chars", SHORT_NAME_CHARS + 1, 20),
    ("21+ chars", 21, usize::MAX),
];

fn length_regime(groups: &[&GroupReport]) -> Vec<LengthBucket> {
    LENGTH_BUCKETS
        .iter()
        .map(|(label, low, high)| {
            let in_band = |measurement: &&Measurement| {
                measurement.shorter_chars >= *low
                    && measurement.shorter_chars <= *high
                    && measurement.label.is_measured()
            };
            let collect = |positive: bool| -> Vec<f64> {
                groups
                    .iter()
                    .flat_map(|group| group.measurements.iter())
                    .filter(in_band)
                    .filter(|measurement| measurement.is_positive() == positive)
                    .map(|measurement| measurement.cosine)
                    .collect()
            };

            LengthBucket {
                label,
                low: *low,
                high: *high,
                positives: Summary::of(&collect(true)),
                negatives: Summary::of(&collect(false)),
            }
        })
        .collect()
}

/// The band `store::node` says unrelated short English strings "routinely" sit
/// in. Quoted from the doc comment on `CONFIDENT_SIMILARITY`, and the whole
/// justification for the thresholds being where they are.
const CLAIMED_FLOOR_LOW: f64 = 0.6;
const CLAIMED_FLOOR_HIGH: f64 = 0.8;

/// The compressed-range argument, put next to what the model does.
#[derive(Debug, Clone, Serialize)]
struct FloorClaim {
    claimed_low: f64,
    claimed_high: f64,
    unrelated_long: Option<Summary>,
    unrelated_short: Option<Summary>,
    unrelated_query: Option<Summary>,
    /// The hard negatives. If the floor turns out to be low but these are
    /// high, the argument was wrong about *why* and right about *what*.
    related_long: Option<Summary>,
    related_short: Option<Summary>,
    /// Whether any unrelated pair at all reaches the claimed band.
    floor_reaches_the_claimed_band: bool,
    /// Whether the hard negatives climb past the loose confident bar anyway.
    hard_negatives_clear_the_confident_bar: bool,
}

impl FloorClaim {
    fn of(long: &GroupReport, short: &GroupReport, query: &GroupReport) -> Self {
        let unrelated_long = long.unrelated;
        let unrelated_short = short.unrelated;
        let highest_unrelated = [unrelated_long, unrelated_short]
            .into_iter()
            .flatten()
            .map(|summary| summary.max)
            .fold(f64::NEG_INFINITY, f64::max);
        let highest_related = [long.related, short.related]
            .into_iter()
            .flatten()
            .map(|summary| summary.max)
            .fold(f64::NEG_INFINITY, f64::max);

        Self {
            claimed_low: CLAIMED_FLOOR_LOW,
            claimed_high: CLAIMED_FLOOR_HIGH,
            unrelated_long,
            unrelated_short,
            unrelated_query: query.unrelated,
            related_long: long.related,
            related_short: short.related,
            floor_reaches_the_claimed_band: highest_unrelated >= CLAIMED_FLOOR_LOW,
            hard_negatives_clear_the_confident_bar: highest_related >= CONFIDENT_SIMILARITY,
        }
    }
}

/// What the resolver would actually do to one group, given the constants that
/// are in the code today.
///
/// Pairwise, which is the resolver's `confident.len() == 1` branch: each pair
/// is treated as though the neighbour were the only one in the tenant. That is
/// the shape that merges, so it is the shape worth simulating.
#[derive(Debug, Clone, Serialize)]
struct ResolutionOutcome {
    regime: &'static str,
    confident: f64,
    ambiguous: f64,
    same_merged: usize,
    same_reported: usize,
    /// Same-entity pairs below both bars: a duplicate node, and no candidate
    /// list to tell anyone it happened.
    same_silent: usize,
    negatives_merged: usize,
    negatives_reported: usize,
    negatives_ignored: usize,
    /// Named rather than counted. An irreversible merge is not a statistic.
    over_merges: Vec<String>,
}

impl ResolutionOutcome {
    fn of(regime: &'static str, group: &GroupReport, confident: f64, ambiguous: f64) -> Self {
        let mut outcome = Self {
            regime,
            confident,
            ambiguous,
            same_merged: 0,
            same_reported: 0,
            same_silent: 0,
            negatives_merged: 0,
            negatives_reported: 0,
            negatives_ignored: 0,
            over_merges: Vec::new(),
        };

        for measurement in &group.measurements {
            if !measurement.label.is_measured() {
                continue;
            }

            let merged = measurement.cosine >= confident;
            let reported = !merged && measurement.cosine >= ambiguous;

            if measurement.label.is_positive() {
                if merged {
                    outcome.same_merged += 1;
                } else if reported {
                    outcome.same_reported += 1;
                } else {
                    outcome.same_silent += 1;
                }
            } else if merged {
                outcome.negatives_merged += 1;
                outcome.over_merges.push(format!(
                    "{:.4}  {} / {}",
                    measurement.cosine, measurement.left, measurement.right
                ));
            } else if reported {
                outcome.negatives_reported += 1;
            } else {
                outcome.negatives_ignored += 1;
            }
        }

        outcome
    }
}

/// What the lexical guard in `store::node` does to one regime of this corpus.
///
/// The guard is a veto sitting beside the confident bar: a neighbour the cosine
/// calls confident is still refused if the two names contradict each other.
/// This measures both halves of that bargain — the over-merges it blocks and
/// the true matches it costs — because a guard is only worth having if the
/// first number is real and the second one is small.
#[derive(Debug, Clone, Serialize)]
struct GuardReport {
    regime: &'static str,
    confident: f64,
    /// Distinct entities at or above the bar that the guard refuses to fuse.
    /// Named rather than counted: this is what the guard is for.
    blocked: Vec<String>,
    /// Distinct entities at or above the bar it lets through anyway.
    admitted: Vec<String>,
    /// Same-entity pairs at or above the bar that the guard demotes to the
    /// ambiguous band. The price, paid in duplicate nodes a later merge undoes.
    cost: Vec<String>,
    /// Same-entity pairs at or above the bar that still merge.
    kept: usize,
    /// The guard's opinion of the whole labelled population, bar or no bar.
    /// A guard that only separated the pairs near the threshold would be a
    /// patch on this corpus rather than a signal about names.
    negatives_vetoed: usize,
    negatives_total: usize,
    positives_total: usize,
    /// Every same-entity pair the guard vetoes anywhere in the corpus, at the
    /// bar or far below it. The ones below the bar cost nothing today because
    /// no cosine would have merged them anyway — they are printed because they
    /// are what the guard would start costing if the bar ever moved down, and
    /// that is not a thing to discover after the fact.
    vetoed_positives: Vec<String>,
}

impl GuardReport {
    fn of(regime: &'static str, group: &GroupReport, confident: f64) -> Self {
        let mut report = Self {
            regime,
            confident,
            blocked: Vec::new(),
            admitted: Vec::new(),
            cost: Vec::new(),
            kept: 0,
            negatives_vetoed: 0,
            negatives_total: 0,
            positives_total: 0,
            vetoed_positives: Vec::new(),
        };

        for measurement in &group.measurements {
            if !measurement.label.is_measured() {
                continue;
            }

            let vetoed = names_contradict(&measurement.left, &measurement.right);
            let named = format!(
                "{:.4}  {} / {}",
                measurement.cosine, measurement.left, measurement.right
            );

            if measurement.label.is_positive() {
                report.positives_total += 1;
                if vetoed {
                    report.vetoed_positives.push(named.clone());
                }
            } else {
                report.negatives_total += 1;
                report.negatives_vetoed += usize::from(vetoed);
            }

            if measurement.cosine < confident {
                continue;
            }

            match (measurement.label.is_positive(), vetoed) {
                (true, true) => report.cost.push(named),
                (true, false) => report.kept += 1,
                (false, true) => report.blocked.push(named),
                (false, false) => report.admitted.push(named),
            }
        }

        report
    }
}

/// The lowest and highest candidate confident bar the sweep below walks, in
/// hundredths. The interesting band for entity names: 0.88 sits under every
/// plausible choice and 0.99 above the point where the tier stops firing at
/// all, so the row where a regime goes dead is inside the table rather than
/// off the end of it.
const SWEEP_LOW_HUNDREDTHS: i32 = 88;
const SWEEP_HIGH_HUNDREDTHS: i32 = 99;

/// One candidate confident bar, and both halves of what it would buy.
///
/// `_after_guard` is what the resolver would really do, because
/// [`names_contradict`] sits at the merge site: a neighbour is attached to only
/// if the cosine clears the bar *and* the names do not contradict. The
/// cosine-only column is printed beside it on purpose — the gap between the two
/// is exactly how much protection the guard supplies at that bar, and for a
/// one-token name that gap is zero by construction.
#[derive(Debug, Clone, Copy, Serialize)]
struct SweepRow {
    threshold: f64,
    same_merged: usize,
    same_merged_after_guard: usize,
    negatives_merged: usize,
    negatives_merged_after_guard: usize,
    /// How far the bar sits above the highest negative in the regime. Negative
    /// when the bar is under it, which is a false merge and not a margin.
    margin_to_worst_negative: f64,
    /// How far it sits above the highest negative the guard cannot veto — the
    /// one the cosine has to reject unaided. Where the guard covers a regime
    /// well this is the number that matters; where it does not, this column and
    /// the one before it nearly coincide, and that coincidence is the finding.
    margin_to_worst_unguardable: f64,
}

/// A regime's confident bar, swept across every two-decimal value it could
/// plausibly take.
#[derive(Debug, Clone, Serialize)]
struct Sweep {
    regime: &'static str,
    current: f64,
    positives_total: usize,
    negatives_total: usize,
    worst_negative: f64,
    worst_negative_pair: String,
    /// Whether [`names_contradict`] refuses the single worst negative. When it
    /// does, the raw margin overstates the exposure; when it does not, the raw
    /// margin is the entire defence.
    worst_negative_vetoed: bool,
    /// The highest negative the guard cannot veto, and so the floor any bar in
    /// this regime has to clear on the strength of the cosine alone.
    worst_unguardable: f64,
    worst_unguardable_pair: String,
    rows: Vec<SweepRow>,
}

impl Sweep {
    fn of(regime: &'static str, group: &GroupReport, current: f64) -> Self {
        let measured: Vec<&Measurement> = group
            .measurements
            .iter()
            .filter(|measurement| measurement.label.is_measured())
            .collect();

        let vetoed =
            |measurement: &Measurement| names_contradict(&measurement.left, &measurement.right);
        let named =
            |measurement: &Measurement| format!("{} / {}", measurement.left, measurement.right);

        let highest = |guardable_only: bool| -> (f64, String) {
            measured
                .iter()
                .filter(|measurement| !measurement.is_positive())
                .filter(|measurement| !(guardable_only && vetoed(measurement)))
                .max_by(|left, right| left.cosine.total_cmp(&right.cosine))
                .map_or((f64::NAN, "none".to_string()), |measurement| {
                    (measurement.cosine, named(measurement))
                })
        };

        let (worst_negative, worst_negative_pair) = highest(false);
        let (worst_unguardable, worst_unguardable_pair) = highest(true);
        let worst_negative_vetoed = worst_unguardable < worst_negative;

        let rows = (SWEEP_LOW_HUNDREDTHS..=SWEEP_HIGH_HUNDREDTHS)
            .map(|hundredths| {
                let threshold = f64::from(hundredths) / 100.0;
                let count = |positive: bool, apply_guard: bool| {
                    measured
                        .iter()
                        .filter(|measurement| measurement.is_positive() == positive)
                        .filter(|measurement| measurement.cosine >= threshold)
                        .filter(|measurement| !(apply_guard && vetoed(measurement)))
                        .count()
                };

                SweepRow {
                    threshold,
                    same_merged: count(true, false),
                    same_merged_after_guard: count(true, true),
                    negatives_merged: count(false, false),
                    negatives_merged_after_guard: count(false, true),
                    margin_to_worst_negative: threshold - worst_negative,
                    margin_to_worst_unguardable: threshold - worst_unguardable,
                }
            })
            .collect();

        let count_label = |positive: bool| {
            measured
                .iter()
                .filter(|measurement| measurement.is_positive() == positive)
                .count()
        };

        Self {
            regime,
            current,
            positives_total: count_label(true),
            negatives_total: count_label(false),
            worst_negative,
            worst_negative_pair,
            worst_negative_vetoed,
            worst_unguardable,
            worst_unguardable_pair,
            rows,
        }
    }
}

/// One inverse pair, its cosine, and whether the vocabulary knows about it.
#[derive(Debug, Clone, Serialize)]
struct InverseReport {
    left: String,
    right: String,
    cosine: f64,
    /// Whether `relationship_inverse_pair` holds this pair. An unguarded pair
    /// above the coercion threshold is a live direction-reversal bug, not a
    /// hypothetical one.
    guarded: bool,
    /// Whether the morphology rule derives the pair without being told —
    /// `X`/`X_BY` or `HAS_X`/`X_OF`. The other half of the same guard, and
    /// the half that covers coinages nobody has curated yet.
    mechanical: bool,
    /// Whether it clears [`TYPE_CONFIDENT_SIMILARITY`], and so would be coerced
    /// if the guard did not stop it.
    would_coerce: bool,
}

fn is_seeded_inverse(left: &str, right: &str) -> bool {
    SEEDED_INVERSE_PAIRS
        .iter()
        .any(|(a, b)| (*a == left && *b == right) || (*a == right && *b == left))
}

// ---------------------------------------------------------------------------
// Structural arithmetic
//
// Not measurements. RRF reads a rank and nothing else, so no corpus of cosines
// can calibrate a cutoff fraction; what these functions do is state, in
// numbers, what the constants in `search.rs` already imply. That is worth
// printing because the claims in the doc comments there are checkable and
// nothing was checking them.
// ---------------------------------------------------------------------------

/// The RRF contribution of one list at `rank`, 1-based.
fn rrf_score(rank: usize) -> f64 {
    1.0 / (RRF_K + rank as f64)
}

/// The deepest single-list rank that still clears `fraction * best`, capped at
/// [`CANDIDATE_DEPTH`]. `None` when even rank 1 is cut.
fn deepest_surviving_rank(fraction: f64, best: f64) -> Option<usize> {
    let floor = fraction * best;
    let depth = usize::try_from(CANDIDATE_DEPTH).unwrap_or(usize::MAX);
    (1..=depth).rev().find(|rank| rrf_score(*rank) >= floor)
}

/// The salience multiplier `search::salience_weight` applies, restated so the
/// band's reach can be computed here without a database.
fn salience_weight(salience: f64) -> f64 {
    SALIENCE_BAND_FLOOR + SALIENCE_BAND_SPAN * salience.clamp(0.0, 1.0)
}

/// The deepest rank a fully salient hit can hold and still outrank a
/// completely faded hit at rank 1 of the same list. The reach of the band.
fn salience_reach() -> Option<usize> {
    let faded_first = rrf_score(1) * salience_weight(0.0);
    let depth = usize::try_from(CANDIDATE_DEPTH).unwrap_or(usize::MAX);
    (1..=depth)
        .rev()
        .find(|rank| rrf_score(*rank) * salience_weight(1.0) > faded_first)
}

#[derive(Debug, Clone, Serialize)]
struct CutoffRow {
    precision: &'static str,
    fraction: f64,
    floor: f64,
    deepest_surviving_rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct Structural {
    rrf_k: f64,
    candidate_depth: i64,
    default_limit: usize,
    max_limit: usize,
    hybrid_best: f64,
    single_list_best: f64,
    hybrid: Vec<CutoffRow>,
    single_list: Vec<CutoffRow>,
    salience_floor: f64,
    salience_span: f64,
    /// The deepest rank the salience band can lift above a faded rank-1 hit.
    salience_reach: Option<usize>,
    /// Whether the band can overturn agreement between the two lists.
    band_can_overturn_agreement: bool,
}

fn structural() -> Structural {
    let hybrid_best = 2.0 * rrf_score(1);
    let single_best = rrf_score(1);
    let rows = |best: f64| {
        [
            ("Relaxed", CUTOFF_FRACTION_RELAXED),
            ("Standard", CUTOFF_FRACTION_STANDARD),
            ("Strict", CUTOFF_FRACTION_STRICT),
        ]
        .into_iter()
        .map(|(precision, fraction)| CutoffRow {
            precision,
            fraction,
            floor: fraction * best,
            deepest_surviving_rank: deepest_surviving_rank(fraction, best),
        })
        .collect::<Vec<_>>()
    };

    Structural {
        rrf_k: RRF_K,
        candidate_depth: CANDIDATE_DEPTH,
        default_limit: DEFAULT_LIMIT,
        max_limit: MAX_LIMIT,
        hybrid_best,
        single_list_best: single_best,
        hybrid: rows(hybrid_best),
        single_list: rows(single_best),
        salience_floor: SALIENCE_BAND_FLOOR,
        salience_span: SALIENCE_BAND_SPAN,
        salience_reach: salience_reach(),
        // A hit both lists rank first, faded to nothing, against a hit one list
        // ranks first at full salience.
        band_can_overturn_agreement: rrf_score(1) * salience_weight(1.0)
            > 2.0 * rrf_score(1) * salience_weight(0.0),
    }
}

// ---------------------------------------------------------------------------
// Running it
// ---------------------------------------------------------------------------

/// Texts per forward pass. Small enough that a two-word name is not padded out
/// to the length of a sentence, large enough that the corpus is a handful of
/// passes rather than two hundred.
const EMBED_BATCH: usize = 16;

#[derive(Debug, Clone, Serialize)]
struct Report {
    model: String,
    device: String,
    dimensions: usize,
    groups: Vec<GroupReport>,
    length_regime: Vec<LengthBucket>,
    floor_claim: FloorClaim,
    resolution: Vec<ResolutionOutcome>,
    guard: Vec<GuardReport>,
    sweep: Vec<Sweep>,
    inverse_pairs: Vec<InverseReport>,
    queries: Vec<QueryReport>,
    verdicts: Vec<ConstantVerdict>,
    structural: Structural,
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut json = false;
    for argument in &arguments {
        match argument.as_str() {
            "--json" => json = true,
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown argument `{other}`\n\n{USAGE}"),
        }
    }

    if !json {
        eprintln!("loading BAAI/bge-base-en-v1.5 …");
    }
    let embedder = CandleEmbedder::load()
        .await
        .context("could not load the embedding model")?;

    let report = measure(&embedder).await?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("could not serialise the report")?
        );
    } else {
        print_report(&report);
    }

    Ok(())
}

const USAGE: &str = "\
calibrate — measure what BAAI/bge-base-en-v1.5 actually does to mundu's corpora

    cargo run --features candle-embedder --bin calibrate [-- --json]

    --json    machine-readable output instead of the report
    --help    this text

Needs no database. Reads the threshold constants out of the library, so the
numbers it judges cannot drift from the numbers the server uses.";

/// Embeds every distinct text once, on the document side.
async fn embed_documents_once(
    embedder: &CandleEmbedder,
    texts: &[String],
) -> Result<HashMap<String, Vec<f32>>> {
    let mut vectors = HashMap::with_capacity(texts.len());
    for chunk in texts.chunks(EMBED_BATCH) {
        let embedded = embedder
            .embed_documents(chunk)
            .await
            .context("the model failed on a batch of documents")?;
        for (text, vector) in chunk.iter().zip(embedded) {
            vectors.insert(text.clone(), vector);
        }
    }
    Ok(vectors)
}

/// Unique, order-preserving.
fn distinct(texts: impl IntoIterator<Item = &'static str>) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for text in texts {
        if !seen.iter().any(|existing| existing == text) {
            seen.push(text.to_string());
        }
    }
    seen
}

async fn measure(embedder: &CandleEmbedder) -> Result<Report> {
    let pair_groups: [(&'static str, &'static [Pair]); 3] = [
        ("Entity names, long (> 12 chars)", LONG_NAME_PAIRS),
        ("Entity names, short (<= 12 chars)", SHORT_NAME_PAIRS),
        ("Relationship type names", RELATIONSHIP_TYPE_PAIRS),
    ];

    let names = distinct(
        pair_groups
            .iter()
            .flat_map(|(_, pairs)| pairs.iter())
            .flat_map(|pair| [pair.left, pair.right]),
    );
    let name_vectors = embed_documents_once(embedder, &names).await?;

    let mut groups: Vec<GroupReport> = Vec::with_capacity(pair_groups.len());
    for (name, pairs) in pair_groups {
        let measurements = pairs
            .iter()
            .map(|pair| {
                let left = name_vectors
                    .get(pair.left)
                    .expect("every corpus text was embedded");
                let right = name_vectors
                    .get(pair.right)
                    .expect("every corpus text was embedded");
                Measurement {
                    left: pair.left.to_string(),
                    right: pair.right.to_string(),
                    label: pair.label,
                    note: pair.note,
                    cosine: cosine(left, right),
                    shorter_chars: pair.left.chars().count().min(pair.right.chars().count()),
                }
            })
            .collect();
        groups.push(GroupReport::build(
            name,
            "document / document",
            measurements,
        ));
    }

    let inverse_pairs = groups
        .iter()
        .flat_map(|group| group.measurements.iter())
        .filter(|measurement| measurement.label == Inverse)
        .map(|measurement| InverseReport {
            left: measurement.left.clone(),
            right: measurement.right.clone(),
            cosine: measurement.cosine,
            guarded: is_seeded_inverse(&measurement.left, &measurement.right),
            mechanical: are_mechanical_inverses(&measurement.left, &measurement.right),
            would_coerce: measurement.cosine >= TYPE_CONFIDENT_SIMILARITY,
        })
        .collect();

    let (query_group, queries) = measure_queries(embedder).await?;

    let length_regime = length_regime(&[&groups[0], &groups[1]]);
    let verdicts = vec![
        judge(
            "store::node::CONFIDENT_SIMILARITY (name > 12 chars)",
            CONFIDENT_SIMILARITY,
            &groups[0],
            MERGE_CONTRACT,
        ),
        judge(
            "store::node::AMBIGUOUS_SIMILARITY (name > 12 chars)",
            AMBIGUOUS_SIMILARITY,
            &groups[0],
            CANDIDATE_CONTRACT,
        ),
        judge(
            "store::node::CONFIDENT_SIMILARITY_SHORT (name <= 12 chars)",
            CONFIDENT_SIMILARITY_SHORT,
            &groups[1],
            MERGE_CONTRACT,
        ),
        judge(
            "store::node::AMBIGUOUS_SIMILARITY (name <= 12 chars)",
            AMBIGUOUS_SIMILARITY,
            &groups[1],
            CANDIDATE_CONTRACT,
        ),
        judge(
            "store::relationship::CONFIDENT_SIMILARITY",
            TYPE_CONFIDENT_SIMILARITY,
            &groups[2],
            TYPE_COERCION_CONTRACT,
        ),
    ];

    let floor_claim = FloorClaim::of(&groups[0], &groups[1], &query_group);
    groups.push(query_group);

    let resolution = vec![
        ResolutionOutcome::of(
            "long  (> 12 chars)",
            &groups[0],
            CONFIDENT_SIMILARITY,
            AMBIGUOUS_SIMILARITY,
        ),
        ResolutionOutcome::of(
            "short (<= 12 chars)",
            &groups[1],
            CONFIDENT_SIMILARITY_SHORT,
            AMBIGUOUS_SIMILARITY,
        ),
    ];

    let guard = vec![
        GuardReport::of("long  (> 12 chars)", &groups[0], CONFIDENT_SIMILARITY),
        GuardReport::of(
            "short (<= 12 chars)",
            &groups[1],
            CONFIDENT_SIMILARITY_SHORT,
        ),
    ];

    let sweep = vec![
        Sweep::of("long  (> 12 chars)", &groups[0], CONFIDENT_SIMILARITY),
        Sweep::of(
            "short (<= 12 chars)",
            &groups[1],
            CONFIDENT_SIMILARITY_SHORT,
        ),
    ];

    Ok(Report {
        model: embedder.model_id().to_string(),
        device: format!("{:?}", embedder.device()),
        dimensions: embedder.dimensions(),
        groups,
        length_regime,
        floor_claim,
        resolution,
        guard,
        sweep,
        inverse_pairs,
        queries,
        verdicts,
        structural: structural(),
    })
}

/// The asymmetric group: questions through `embed_query`, sentences through
/// `embed_documents`.
///
/// Every question is scored against every sentence in the pool, not just its
/// own, because that is what the semantic candidate list does — it ranks the
/// whole tenant, and a threshold that only ever saw a question beside its own
/// answer would be fitted to a situation search never encounters.
async fn measure_queries(embedder: &CandleEmbedder) -> Result<(GroupReport, Vec<QueryReport>)> {
    let documents = distinct(
        QUERY_CASES
            .iter()
            .flat_map(|case| std::iter::once(case.proof).chain(case.near_misses.iter().copied())),
    );
    let document_vectors = embed_documents_once(embedder, &documents).await?;

    let mut measurements = Vec::with_capacity(QUERY_CASES.len() * documents.len());
    let mut queries = Vec::with_capacity(QUERY_CASES.len());

    for case in QUERY_CASES {
        let query = embedder
            .embed_query(case.question)
            .await
            .context("the model failed on a query")?;

        let mut scores = Vec::with_capacity(documents.len());
        let mut answer = f64::NAN;
        let mut best_near_miss = f64::NEG_INFINITY;
        let mut best_unrelated = f64::NEG_INFINITY;

        for document in &documents {
            let vector = document_vectors
                .get(document)
                .expect("every document was embedded");
            let similarity = cosine(&query, vector);
            scores.push(similarity);

            let label = if document == case.proof {
                answer = similarity;
                Same
            } else if case.near_misses.contains(&document.as_str()) {
                best_near_miss = best_near_miss.max(similarity);
                Related
            } else {
                best_unrelated = best_unrelated.max(similarity);
                Unrelated
            };

            measurements.push(Measurement {
                left: case.question.to_string(),
                right: document.clone(),
                label,
                note: "",
                cosine: similarity,
                shorter_chars: case.question.chars().count().min(document.chars().count()),
            });
        }

        queries.push(QueryReport {
            question: case.question.to_string(),
            answer,
            rank: rank_of(answer, &scores),
            best_near_miss,
            best_unrelated,
            margin: answer - best_near_miss.max(best_unrelated),
        });
    }

    Ok((
        GroupReport::build(
            "Query to proof (asymmetric)",
            "query / document",
            measurements,
        ),
        queries,
    ))
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

const RULE: &str =
    "===============================================================================";
const THIN: &str =
    "-------------------------------------------------------------------------------";

fn print_report(report: &Report) {
    println!("{RULE}");
    println!(" mundu threshold calibration");
    println!("{RULE}");
    println!("model       {}", report.model);
    println!(
        "device      {}   dimensions  {}",
        report.device, report.dimensions
    );
    println!(
        "pairs       {} labelled cosines across {} groups",
        report
            .groups
            .iter()
            .map(|group| group.measurements.len())
            .sum::<usize>(),
        report.groups.len()
    );
    println!(
        "\nEvery cosine below is between L2-normalised CLS-pooled vectors from the model\nthe server actually runs. Nothing here writes to a database or changes a\nconstant."
    );

    for group in &report.groups {
        print_group(group);
    }

    print_length_regime(&report.length_regime);
    print_floor_claim(&report.floor_claim);
    print_resolution(&report.resolution);
    print_guard(&report.guard);
    print_sweep(&report.sweep);
    print_inverse(&report.inverse_pairs);
    if let Some(group) = report.groups.iter().find(|group| group.inverse.is_some()) {
        print_inverse_interleaving(group);
    }
    print_queries(&report.queries);
    print_verdicts(&report.verdicts);
    print_structural(&report.structural);
}

fn print_summary_line(label: &str, summary: Option<Summary>) {
    match summary {
        Some(summary) => println!(
            "  {label:<12} n={:<4} min {:.4}  max {:.4}  mean {:.4}  median {:.4}  sd {:.4}",
            summary.count, summary.min, summary.max, summary.mean, summary.median, summary.stddev
        ),
        None => println!("  {label:<12} (none in this group)"),
    }
}

fn print_group(group: &GroupReport) {
    println!("\n{THIN}");
    println!(" {}   [{}]", group.name, group.embedding);
    println!("{THIN}");

    // The query group is 300 rows. Listing them all would bury the four numbers
    // that matter, so it is summarised per question further down instead.
    if group.measurements.len() <= 60 {
        let mut rows: Vec<&Measurement> = group.measurements.iter().collect();
        rows.sort_by(|left, right| right.cosine.total_cmp(&left.cosine));
        for measurement in rows {
            let note = if measurement.note.is_empty() {
                String::new()
            } else {
                format!("  ({})", measurement.note)
            };
            println!(
                "  {:.4}  {:<9}  {:<28} / {}{}",
                measurement.cosine,
                measurement.label.as_str(),
                measurement.left,
                measurement.right,
                note
            );
        }
        println!();
    }

    print_summary_line("same", group.same);
    print_summary_line("related", group.related);
    print_summary_line("unrelated", group.unrelated);
    if group.inverse.is_some() {
        print_summary_line("inverse", group.inverse);
    }
    print_summary_line("NEGATIVES", group.negatives);

    match group.margin {
        Some(margin) if margin > 0.0 => println!(
            "  separation   CLEAN: lowest same-pair {:.4} sits {:.4} above the highest \
             negative {:.4}",
            group.same.map_or(f64::NAN, |summary| summary.min),
            margin,
            group.negatives.map_or(f64::NAN, |summary| summary.max),
        ),
        Some(margin) => println!(
            "  separation   OVERLAP of {:.4}: lowest same-pair {:.4} is below the highest \
             negative {:.4}",
            -margin,
            group.same.map_or(f64::NAN, |summary| summary.min),
            group.negatives.map_or(f64::NAN, |summary| summary.max),
        ),
        None => println!("  separation   (not computable)"),
    }

    let positives = group.positives();
    let negatives = group.negative_values();
    if let Some(threshold) = no_false_merge_threshold(&negatives) {
        let errors = errors_at(threshold, &positives, &negatives);
        println!(
            "  thresholds   no-false-merge >= {threshold:.2} (loses {} of {} same-pairs)",
            errors.missed_merges,
            positives.len()
        );
    }
    if let Some((threshold, errors)) = balanced_threshold(&positives, &negatives) {
        println!(
            "               fewest-errors  >= {threshold:.2} ({} false merges, {} missed)",
            errors.false_merges, errors.missed_merges
        );
    }

    let homonyms: Vec<&Measurement> = group
        .measurements
        .iter()
        .filter(|measurement| measurement.label == Homonym)
        .collect();
    if !homonyms.is_empty() {
        println!(
            "\n  Excluded from the statistics above — identical strings, different referents:"
        );
        for measurement in homonyms {
            println!(
                "    {:.4}  {} / {}  ({})",
                measurement.cosine, measurement.left, measurement.right, measurement.note
            );
        }
        println!(
            "  No cosine threshold can separate these. In the store they never reach the\n  \
             vector tier at all: `canonical_node_id` maps them to one id, so tier 1 merges\n  \
             them outright. That is a resolution policy, not a calibration problem, but it\n  \
             is worth knowing it is not the threshold doing it."
        );
    }
}

fn print_length_regime(buckets: &[LengthBucket]) {
    println!("\n{THIN}");
    println!(" Length regime — is SHORT_NAME_CHARS = {SHORT_NAME_CHARS} a real boundary?");
    println!("{THIN}");
    println!(
        "  Entity-name pairs bucketed by the shorter of the two names. The claim under\n  \
         test is that shorter names crowd closer together, so the negatives — the pairs\n  \
         a threshold must reject — should climb as the band narrows.\n"
    );
    println!("  band          negatives                          same-pairs");
    for bucket in buckets {
        let negatives = bucket.negatives.map_or_else(
            || "n=0".to_string(),
            |summary| {
                format!(
                    "n={:<3} mean {:.4} max {:.4}",
                    summary.count, summary.mean, summary.max
                )
            },
        );
        let positives = bucket.positives.map_or_else(
            || "n=0".to_string(),
            |summary| {
                format!(
                    "n={:<3} mean {:.4} min {:.4}",
                    summary.count, summary.mean, summary.min
                )
            },
        );
        println!("  {:<13} {negatives:<34} {positives}", bucket.label);
    }

    let thin: Vec<&str> = buckets
        .iter()
        .filter(|bucket| {
            bucket
                .negatives
                .is_none_or(|summary| summary.count < MINIMUM_POPULATION)
        })
        .map(|bucket| bucket.label)
        .collect();
    if thin.is_empty() {
        println!(
            "\n  Every band carries enough pairs to read. Compare them: if the boundary at \
             {SHORT_NAME_CHARS}\n  were real, the negatives would step up as the bands narrow."
        );
    } else {
        println!(
            "\n  CAVEAT: {} is thin (under {MINIMUM_POPULATION} negatives), so read it as an \
             indication, not\n  a result. The bands either side of {SHORT_NAME_CHARS} are the \
             comparison that matters.",
            thin.join(", ")
        );
    }
}

fn print_floor_claim(claim: &FloorClaim) {
    println!("\n{THIN}");
    println!(" The compressed-range argument, tested");
    println!("{THIN}");
    println!(
        "  `store::node` justifies its thresholds with the claim that \"unrelated short\n  \
         English strings routinely sit at {:.1}-{:.1}\". That is a measurable claim and\n  \
         this is the measurement.\n",
        claim.claimed_low, claim.claimed_high
    );

    print_summary_line("unrel long", claim.unrelated_long);
    print_summary_line("unrel short", claim.unrelated_short);
    print_summary_line("unrel query", claim.unrelated_query);
    println!();
    print_summary_line("related long", claim.related_long);
    print_summary_line("related shrt", claim.related_short);

    println!();
    if claim.floor_reaches_the_claimed_band {
        println!(
            "  The floor is where the claim says it is: unrelated text does reach {:.1}.",
            claim.claimed_low
        );
    } else {
        println!(
            "  CONTRADICTED at the floor. Not one unrelated pair reaches {:.1}. Genuinely\n  \
             unrelated text lands in the 0.3-0.5 range, so the space is nothing like as\n  \
             compressed as the argument assumed.",
            claim.claimed_low
        );
    }

    if claim.hard_negatives_clear_the_confident_bar {
        println!(
            "  SUPPORTED where it counts. The pairs that crowd the top are not the\n  \
             unrelated ones, they are the *related* ones — sibling entities, one word\n  \
             apart — and those climb past the confident bar of {CONFIDENT_SIMILARITY:.2}. So \
             the conclusion\n  (thresholds must be high) survives, but the reason given for it \
             is the wrong\n  reason: it is near-synonymy that forces the bar up, not an \
             elevated floor."
        );
    } else {
        println!(
            "  And the hard negatives stay below the confident bar too, so nothing in this\n  \
             corpus forces the thresholds as high as they are."
        );
    }
}

fn print_resolution(outcomes: &[ResolutionOutcome]) {
    println!("\n{THIN}");
    println!(" What the constants in the code would do to this corpus");
    println!("{THIN}");
    println!(
        "  Pairwise, one neighbour at a time — the shape in which the resolver actually\n  \
         merges. `silent` is the bad-but-cheap outcome: a duplicate node with no\n  \
         candidate list to say so. `MERGED` on the negative row is the expensive one.\n"
    );

    for outcome in outcomes {
        println!(
            "  {} (confident {:.2}, ambiguous {:.2})",
            outcome.regime, outcome.confident, outcome.ambiguous
        );
        println!(
            "    same       {:>3} merged   {:>3} reported   {:>3} silently duplicated",
            outcome.same_merged, outcome.same_reported, outcome.same_silent
        );
        println!(
            "    negatives  {:>3} MERGED   {:>3} reported   {:>3} ignored",
            outcome.negatives_merged, outcome.negatives_reported, outcome.negatives_ignored
        );
        if outcome.over_merges.is_empty() {
            println!("    no over-merges on this corpus");
        } else {
            println!("    over-merges — distinct entities this would silently fuse:");
            for over_merge in &outcome.over_merges {
                println!("      {over_merge}");
            }
        }
        if !std::ptr::eq(outcome, &outcomes[outcomes.len() - 1]) {
            println!();
        }
    }
}

fn print_guard(reports: &[GuardReport]) {
    println!("\n{THIN}");
    println!(" What the lexical guard adds to those constants");
    println!("{THIN}");
    println!(
        "  `store::node::names_contradict` vetoes a confident neighbour whose name carries\n  \
         a word the incoming name contradicts. It cannot merge anything the cosine would\n  \
         not have merged on its own; it can only refuse. Every refusal on the same row\n  \
         is a duplicate node and a candidate list, which a later merge undoes.\n"
    );

    for report in reports {
        println!("  {} (confident {:.2})", report.regime, report.confident);
        println!(
            "    at the bar   {} of {} over-merges blocked, {} of {} true matches kept",
            report.blocked.len(),
            report.blocked.len() + report.admitted.len(),
            report.kept,
            report.kept + report.cost.len()
        );
        println!(
            "    whole corpus {} of {} negatives vetoed, {} of {} same-entity pairs vetoed",
            report.negatives_vetoed,
            report.negatives_total,
            report.vetoed_positives.len(),
            report.positives_total
        );

        if report.blocked.is_empty() {
            println!("    nothing to block on this corpus");
        } else {
            println!("    blocked — distinct entities the cosine alone would have fused:");
            for blocked in &report.blocked {
                println!("      {blocked}");
            }
        }
        if !report.admitted.is_empty() {
            println!("    STILL MERGED — distinct entities the guard cannot see:");
            for admitted in &report.admitted {
                println!("      {admitted}");
            }
        }
        if report.cost.is_empty() {
            println!("    costs no true match above the bar");
        } else {
            println!("    cost — same entity, now two nodes and a candidate list:");
            for cost in &report.cost {
                println!("      {cost}");
            }
        }

        let latent: Vec<&String> = report
            .vetoed_positives
            .iter()
            .filter(|vetoed| !report.cost.contains(vetoed))
            .collect();
        if !latent.is_empty() {
            println!("    free today, below the bar — what a lower bar would start costing:");
            for vetoed in latent {
                println!("      {vetoed}");
            }
        }
        if !std::ptr::eq(report, &reports[reports.len() - 1]) {
            println!();
        }
    }
}

/// The confident bar, swept, so that choosing one is a trade a reader can see
/// rather than a number somebody asserts.
fn print_sweep(sweeps: &[Sweep]) {
    println!("\n{THIN}");
    println!(" The confident bar, swept — what each candidate value would buy");
    println!("{THIN}");
    println!(
        "  `cosine` counts what the bar alone would do; `+guard` what survives\n  \
         `names_contradict`, which is what the resolver actually does. The gap between\n  \
         those two columns is the guard's whole contribution at that bar.\n\n  \
         `margin` is the bar minus the highest negative in the regime. `unguarded` is the\n  \
         bar minus the highest negative the guard CANNOT veto — the one the cosine has to\n  \
         reject on its own. Where the guard is inert the two columns converge, and that\n  \
         convergence is the reason the two regimes cannot share a number.\n"
    );

    for sweep in sweeps {
        println!("  {} (currently {:.2})", sweep.regime, sweep.current);
        println!(
            "    worst negative      {:.4}  {}{}",
            sweep.worst_negative,
            sweep.worst_negative_pair,
            if sweep.worst_negative_vetoed {
                "   [the guard vetoes this one]"
            } else {
                "   [the guard cannot see this one]"
            }
        );
        println!(
            "    worst unguardable   {:.4}  {}",
            sweep.worst_unguardable, sweep.worst_unguardable_pair
        );
        println!(
            "\n            same merged        negatives merged\n    \
             bar     of {:<2}             of {:<2}              margin   unguarded\n    \
             ---     cosine  +guard     cosine  +guard",
            sweep.positives_total, sweep.negatives_total
        );

        for row in &sweep.rows {
            let marker = if (row.threshold - sweep.current).abs() < 0.005 {
                "   <- current"
            } else {
                ""
            };
            println!(
                "    {:.2}       {:>2}      {:>2}         {:>2}      {:>2}       {:+.4}  {:+.4}{}",
                row.threshold,
                row.same_merged,
                row.same_merged_after_guard,
                row.negatives_merged,
                row.negatives_merged_after_guard,
                row.margin_to_worst_negative,
                row.margin_to_worst_unguardable,
                marker
            );
        }

        if !std::ptr::eq(sweep, &sweeps[sweeps.len() - 1]) {
            println!();
        }
    }
}

fn print_inverse_interleaving(group: &GroupReport) {
    let highest_inverse = group
        .measurements
        .iter()
        .filter(|measurement| measurement.label == Inverse)
        .map(|measurement| measurement.cosine)
        .fold(f64::NEG_INFINITY, f64::max);
    let synonyms = group.values_of(&[Same]);
    let coined = group
        .measurements
        .iter()
        .filter(|measurement| measurement.label == Same && measurement.note != EARLIER_TIER)
        .count();
    let below = synonyms
        .iter()
        .filter(|value| **value < highest_inverse)
        .count();

    println!(
        "\n  Interleaving: the highest inverse pair sits at {highest_inverse:.4}, above {below} \
         of the {} true\n  synonyms in this corpus. The two populations are not separated by \
         any cosine,\n  so no choice of threshold turns the direction guard into a redundancy \
         — it can\n  only be made unreachable by a threshold high enough to reject most real\n  \
         synonyms too, which is what {TYPE_CONFIDENT_SIMILARITY:.2} currently does.",
        synonyms.len()
    );
    println!(
        "\n  Reach of the tier: {coined} of the {} same-pairs are words the seeded \
         vocabulary does\n  not already hold as an alias, so those are the only ones tier 3 \
         is ever asked\n  about. The rest are resolved by an exact lookup before a vector is \
         computed.",
        synonyms.len()
    );
}

/// Which half of the direction guard, if either, refuses this pair.
fn caught_by(entry: &InverseReport) -> &'static str {
    match (entry.guarded, entry.mechanical) {
        (true, true) => "table + rule",
        (true, false) => "table only",
        (false, true) => "rule only",
        (false, false) => "NOTHING",
    }
}

fn print_inverse(pairs: &[InverseReport]) {
    println!("\n{THIN}");
    println!(" Inverse relationship pairs — is the direction guard load-bearing?");
    println!("{THIN}");
    println!(
        "  An inverse coerced onto its opposite reverses the direction of every fact\n  \
         written with it, and leaves a proof excerpt that still reads correctly. The\n  \
         guard refuses those, and it has two halves: the curated table\n  \
         `relationship_inverse_pair`, and the morphology rule `are_mechanical_inverses`\n  \
         (`X`/`X_BY` and `HAS_X`/`X_OF`). A pair neither half sees is a reversal waiting\n  \
         for the threshold to come down far enough to reach it.\n"
    );
    println!("  cosine  >= {TYPE_CONFIDENT_SIMILARITY:.2}?  caught by     pair");

    let mut sorted: Vec<&InverseReport> = pairs.iter().collect();
    sorted.sort_by(|left, right| right.cosine.total_cmp(&left.cosine));
    for entry in &sorted {
        println!(
            "  {:.4}  {:<8}  {:<12}  {} / {}",
            entry.cosine,
            if entry.would_coerce { "YES" } else { "no" },
            caught_by(entry),
            entry.left,
            entry.right
        );
    }

    print_inverse_coverage(&sorted);

    let would = sorted.iter().filter(|entry| entry.would_coerce).count();
    let unguarded_and_would = sorted
        .iter()
        .filter(|entry| entry.would_coerce && !entry.guarded && !entry.mechanical)
        .count();
    let highest = sorted.first().map_or(f64::NAN, |entry| entry.cosine);

    println!();
    if would == 0 {
        println!(
            "  VERDICT: belt and braces. No inverse pair reaches {TYPE_CONFIDENT_SIMILARITY:.2}; \
             the highest is {highest:.4}, so\n  \
             at the current threshold tier 3 would not have coerced any of them anyway.\n  \
             The guard is still worth keeping — it is what allows the threshold to be\n  \
             lowered later without reintroducing the failure — but on this corpus it is\n  \
             not what is preventing a reversal today."
        );
    } else {
        println!(
            "  VERDICT: LOAD-BEARING. {would} of {} inverse pairs reach \
             {TYPE_CONFIDENT_SIMILARITY:.2} and would be coerced\n  \
             onto their own opposite if nothing stopped them. The highest is {highest:.4}.",
            sorted.len()
        );
        if unguarded_and_would > 0 {
            println!(
                "\n  AND THERE IS A HOLE: {unguarded_and_would} of those are seen by neither\n  \
                 the table nor the morphology rule. Those coercions happen, and they\n  \
                 reverse facts:"
            );
            for entry in sorted
                .iter()
                .filter(|entry| entry.would_coerce && !entry.guarded && !entry.mechanical)
            {
                println!("    {:.4}  {} / {}", entry.cosine, entry.left, entry.right);
            }
        }
    }
}

/// The bar this instrument's own recommendation for
/// `store::relationship::CONFIDENT_SIMILARITY` produces on this corpus.
///
/// Quoted here so the coverage question can be asked at the number somebody
/// will actually propose rather than only at the number in the code. Lowering
/// the bar is a separate decision; what this section reports is whether the
/// guard would still hold if it were taken.
const PROPOSED_TYPE_BAR: f64 = 0.87;

/// Coverage, per half of the guard.
///
/// The point of counting it rather than asserting it: "the hole is closed" is a
/// claim about pairs nobody has listed, and the only honest form of that claim
/// is a number over the pairs somebody has. A pair in the `NOTHING` bucket is a
/// direction reversal that the coercion bar is the only thing still preventing.
fn print_inverse_coverage(sorted: &[&InverseReport]) {
    let total = sorted.len();
    let counts = |guarded: bool, mechanical: bool| {
        sorted
            .iter()
            .filter(|entry| entry.guarded == guarded && entry.mechanical == mechanical)
            .collect::<Vec<_>>()
    };

    let both = counts(true, true);
    let table_only = counts(true, false);
    let rule_only = counts(false, true);
    let neither = counts(false, false);
    let covered = total - neither.len();

    println!("\n  Coverage — every measured inverse pair, and what refuses it:\n");
    for (label, bucket) in [
        ("table + rule", &both),
        ("table only  ", &table_only),
        ("rule only   ", &rule_only),
        ("NOTHING     ", &neither),
    ] {
        println!("    {label}  {:>2}", bucket.len());
        for entry in bucket {
            println!("                   {} / {}", entry.left, entry.right);
        }
    }

    println!(
        "\n  {covered} of {total} refused by at least one half. The table alone reaches {}; \
         the rule\n  alone reaches {} — neither is a superset of the other, which is why both \
         run.",
        both.len() + table_only.len(),
        both.len() + rule_only.len()
    );

    // The question the next task will ask. A bar is only safe against reversal
    // down to the highest inverse pair nothing refuses; below that, tier 3 is
    // free to fold a relation onto its own opposite.
    let highest_uncovered = neither
        .iter()
        .map(|entry| entry.cosine)
        .fold(f64::NEG_INFINITY, f64::max);

    if neither.is_empty() {
        println!(
            "\n  Nothing measured here is unrefused, so no value of the coercion bar in\n  \
             [0, 1] can reverse a fact on this corpus — including the {PROPOSED_TYPE_BAR:.2} \
             the\n  recommendation block proposes. That is the property the bar has to have\n  \
             before it can come down, and it is a statement about this corpus rather\n  \
             than about English: an irregular participle (PAYS / PAID_BY) or a\n  \
             suppletive pair nobody has curated would still get through, and the\n  \
             answer is a row in the table."
        );
    } else {
        println!(
            "\n  The highest unrefused inverse pair sits at {highest_uncovered:.4}. A coercion \
             bar at or\n  below that is a bar at which tier 3 can reverse a fact, so \
             {PROPOSED_TYPE_BAR:.2} is safe only\n  if it stays above it."
        );
    }
}

fn print_queries(queries: &[QueryReport]) {
    println!("\n{THIN}");
    println!(" Query to proof, per question");
    println!("{THIN}");
    println!(
        "  `answer` is the verbatim sentence that answers the question; `near` is the\n  \
         closest same-topic sentence that does not; `other` is the closest sentence from\n  \
         a different topic. `rank` is where the answer landed in the whole pool, which is\n  \
         what the semantic candidate list is actually built out of.\n"
    );
    println!("  answer  near    other   margin   rank  question");
    for query in queries {
        println!(
            "  {:.4}  {:.4}  {:.4}  {:+.4}   {:>3}   {}",
            query.answer,
            query.best_near_miss,
            query.best_unrelated,
            query.margin,
            query.rank,
            query.question
        );
    }

    let first = queries.iter().filter(|query| query.rank == 1).count();
    println!(
        "\n  The answer ranked first for {first} of {} questions.",
        queries.len()
    );
}

fn print_verdicts(verdicts: &[ConstantVerdict]) {
    println!("\n{RULE}");
    println!(" Recommendations");
    println!("{RULE}");
    println!(
        "  Nothing below has been applied. `recommended` is what this corpus supports;\n  \
         moving a constant is a separate decision.\n"
    );

    for verdict in verdicts {
        let recommended = verdict
            .recommended
            .map_or_else(|| "  --  ".to_string(), |value| format!("{value:.2}"));
        println!("  {}", verdict.constant);
        println!("      purpose: {}", verdict.purpose);
        println!(
            "      current {:.2}   recommended {recommended}   {}",
            verdict.current,
            verdict.verdict.as_str()
        );
        println!("      {}", verdict.rationale);
        if let Some(boundary) = verdict.boundary {
            println!(
                "      separation between the two populations: {:+.4}",
                boundary.gap
            );
        }
        if let Some(warning) = &verdict.warning {
            println!("      WARNING: {warning}");
        }
        if !verdict.caveat.is_empty() {
            println!("      CAVEAT: {}", verdict.caveat);
        }
        println!();
    }

    let overlapping = verdicts
        .iter()
        .filter(|verdict| {
            verdict
                .boundary
                .is_some_and(|boundary| boundary.overlap_depth > 0)
        })
        .count();
    if overlapping > 0 {
        println!(
            "  {overlapping} of the {} constants above divide populations that OVERLAP. That \
             is the\n  headline result, and it is more important than any single recommended \
             value:\n  on this model there is no cosine that both merges what should merge \
             and\n  refuses what should not. Every number in the `recommended` column is a\n  \
             choice about which of the two failures to take, and the only thing this\n  \
             instrument can honestly supply is the price of each.",
            verdicts.len()
        );
        println!();
    }
}

fn print_structural(structural: &Structural) {
    println!("{RULE}");
    println!(" Search tuning — structural, NOT measured");
    println!("{RULE}");
    println!(
        "  RRF reads a rank and nothing else, so a cutoff fraction is not on the same\n  \
         scale as a cosine and no corpus of cosines can calibrate one. What follows is\n  \
         arithmetic on RRF_K = {:.0} and CANDIDATE_DEPTH = {}, printed because the claims\n  \
         in `search.rs` are checkable and nothing was checking them.\n",
        structural.rrf_k, structural.candidate_depth
    );

    println!(
        "  Hybrid, best hit ranked first in both lists (best = {:.6}):",
        structural.hybrid_best
    );
    print_cutoff_rows(&structural.hybrid);
    println!(
        "\n  One list only — Semantic or Lexical (best = {:.6}):",
        structural.single_list_best
    );
    print_cutoff_rows(&structural.single_list);

    let deepest_strict = structural
        .single_list
        .iter()
        .find(|row| row.precision == "Strict")
        .and_then(|row| row.deepest_surviving_rank);
    if let Some(rank) = deepest_strict
        && rank < usize::try_from(structural.candidate_depth).unwrap_or(usize::MAX)
    {
        let bites = if rank < structural.default_limit {
            format!(
                "shallower than the default limit of {}, so it would dock a page a caller \
                 gets\n  for asking nothing at all",
                structural.default_limit
            )
        } else {
            format!(
                "deeper than the default limit of {} but inside the hard cap of {}, so it \
                 would\n  dock a tail nothing measured",
                structural.default_limit, structural.max_limit
            )
        };
        println!(
            "\n  NOTE: in single-list mode the cutoff is arithmetic on RRF_K rather than a \
             reading\n  of the results. Every candidate scores 1/(RRF_K + rank), the same \
             sequence for\n  every query and every corpus, so a fraction of its first term \
             solves to a fixed\n  rank: Strict keeps the top {rank}, {bites}. Inert or \
             arbitrary, neither is a\n  meaning, so search does not apply the cutoff at all \
             when only one list came back;\n  the rows above are what it would otherwise have \
             done. The knob only bites in\n  Hybrid mode."
        );
    }

    println!(
        "\n  Salience band: weight = {:.1} + {:.1} * salience, so [{:.1}, 1.0].",
        structural.salience_floor, structural.salience_span, structural.salience_floor
    );
    match structural.salience_reach {
        Some(rank) if rank >= structural.default_limit => println!(
            "  Reach: a fully salient hit at rank {rank} of one list outranks a completely\n  \
             faded hit at rank 1 of that list — the whole default page of {}, so salience\n  \
             can overturn relevance rather than break ties in it. That follows from\n  \
             RRF_K = {:.0} flattening the head of the list, not from the band being generous.",
            structural.default_limit, structural.rrf_k
        ),
        Some(rank) if rank > 1 => println!(
            "  Reach: a fully salient hit at rank {rank} of one list outranks a completely\n  \
             faded hit at rank 1 of that list. That is inside the default page of {} and\n  \
             not across it, which is the property the band is supposed to have: it reorders\n  \
             the head of a page and cannot promote its tail over its head.",
            structural.default_limit
        ),
        Some(_) | None => {
            println!("  Reach: the band cannot reorder anything — ratings have stopped ranking.")
        }
    }
    println!(
        "  Against agreement: a hit both lists rank first, faded to salience 0, scores\n  \
         {:.6}; a hit one list ranks first at salience 1 scores {:.6}. The band {}\n  \
         overturn agreement between the lists.",
        2.0 * rrf_score(1) * salience_weight(0.0),
        rrf_score(1) * salience_weight(1.0),
        if structural.band_can_overturn_agreement {
            "CAN"
        } else {
            "cannot"
        }
    );

    println!(
        "\n  CANDIDATE_DEPTH = {}: not measurable here. The corpus is a few dozen documents\n  \
         and the depth is a recall bound over a whole tenant. What the query group does\n  \
         show is that the answer ranks near the top of its pool, so the depth is not the\n  \
         binding constraint on this data — which is not the same as saying 200 is right.",
        structural.candidate_depth
    );
}

fn print_cutoff_rows(rows: &[CutoffRow]) {
    for row in rows {
        let deepest = row.deepest_surviving_rank.map_or_else(
            || "none — the lists must agree".to_string(),
            |rank| {
                if i64::try_from(rank).unwrap_or(i64::MAX) >= CANDIDATE_DEPTH {
                    format!("{rank} (the whole list)")
                } else {
                    rank.to_string()
                }
            },
        );
        println!(
            "    {:<9} fraction {:.2}  floor {:.6}  deepest single-list rank kept: {deepest}",
            row.precision, row.fraction, row.floor
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// The corpus is data and the model is the model; neither is testable here. What
// is testable is the arithmetic between them, and a statistics helper that is
// quietly wrong turns a measurement into a confident fiction.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() < 1e-9
    }

    #[test]
    fn a_summary_reports_the_distribution_of_its_sample() {
        let summary = Summary::of(&[0.2, 0.4, 0.6, 0.8]).expect("a non-empty sample summarises");

        assert_eq!(summary.count, 4);
        assert!(close(summary.min, 0.2), "{summary:?}");
        assert!(close(summary.max, 0.8), "{summary:?}");
        assert!(close(summary.mean, 0.5), "{summary:?}");
        assert!(close(summary.median, 0.5), "{summary:?}");
    }

    /// An odd sample takes the middle value rather than averaging across it.
    #[test]
    fn the_median_of_an_odd_sample_is_its_middle_value() {
        let summary = Summary::of(&[0.9, 0.1, 0.5]).expect("a non-empty sample summarises");

        assert!(close(summary.median, 0.5), "{summary:?}");
    }

    /// A mean of nothing is not zero, and a report that printed one would be
    /// stating a measurement it never made.
    #[test]
    fn an_empty_sample_has_no_summary_rather_than_a_zeroed_one() {
        assert!(Summary::of(&[]).is_none());
    }

    #[test]
    fn errors_are_counted_in_both_directions_with_an_inclusive_admission() {
        let positives = [0.90, 0.80];
        let negatives = [0.85, 0.50];

        let errors = errors_at(0.85, &positives, &negatives);

        assert_eq!(errors.false_merges, 1, "0.85 is admitted by `>= 0.85`");
        assert_eq!(errors.missed_merges, 1, "0.80 falls below the bar");
        assert_eq!(errors.total(), 2);
    }

    /// The whole point of the value: `>=` must not readmit the negative it was
    /// derived from.
    #[test]
    fn the_no_false_merge_threshold_is_strictly_above_every_negative() {
        for negatives in [
            vec![0.8543, 0.1],
            vec![0.8500, 0.1],
            vec![0.9999],
            vec![0.0],
        ] {
            let threshold =
                no_false_merge_threshold(&negatives).expect("a non-empty population has one");
            let worst = negatives.iter().copied().fold(f64::MIN, f64::max);

            assert!(
                threshold > worst,
                "threshold {threshold} does not clear the worst negative {worst}"
            );
            assert_eq!(
                errors_at(threshold, &[], &negatives).false_merges,
                0,
                "threshold {threshold} still admits a negative"
            );
        }
    }

    #[test]
    fn an_empty_negative_population_has_no_no_false_merge_threshold() {
        assert!(no_false_merge_threshold(&[]).is_none());
    }

    #[test]
    fn the_balanced_threshold_makes_no_mistakes_when_the_populations_separate() {
        let positives = [0.93, 0.95, 0.99];
        let negatives = [0.40, 0.55, 0.61];

        let (threshold, errors) =
            balanced_threshold(&positives, &negatives).expect("both populations are non-empty");

        assert_eq!(errors.total(), 0, "threshold {threshold} should be perfect");
        assert!(
            threshold > 0.61 && threshold <= 0.93,
            "threshold {threshold} is not inside the gap"
        );
    }

    /// With no perfect cut available it has to find the cheapest one rather
    /// than the first one, and prefer the higher of two equals.
    #[test]
    fn the_balanced_threshold_minimises_total_errors_and_breaks_ties_upward() {
        let positives = [0.70, 0.90, 0.92];
        let negatives = [0.60, 0.71, 0.80];

        let (threshold, errors) =
            balanced_threshold(&positives, &negatives).expect("both populations are non-empty");

        assert_eq!(errors.total(), 1, "got {errors:?} at {threshold}");
        assert!(
            threshold > 0.80 && threshold <= 0.90,
            "the cheapest cut loses only the 0.70 positive, but this one is {threshold}"
        );
    }

    #[test]
    fn a_rank_is_one_based_and_counts_only_strictly_better_scores() {
        let scores = [0.9, 0.5, 0.7, 0.5];

        assert_eq!(rank_of(0.9, &scores), 1);
        assert_eq!(rank_of(0.7, &scores), 2);
        assert_eq!(rank_of(0.5, &scores), 3, "a tie does not push it to 4");
    }

    #[test]
    fn cosine_is_one_for_identical_vectors_and_zero_for_orthogonal_ones() {
        assert!(close(cosine(&[3.0, 4.0], &[3.0, 4.0]), 1.0));
        assert!(close(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0));
        assert!(close(cosine(&[1.0, 0.0], &[-1.0, 0.0]), -1.0));
    }

    /// A zero vector has no direction, so there is no angle to report. Return
    /// 0.0 rather than a NaN that would silently poison every summary it
    /// reached.
    #[test]
    fn cosine_against_a_zero_vector_is_zero_rather_than_not_a_number() {
        assert!(close(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0));
    }

    /// The doc comment on `CUTOFF_FRACTION_STANDARD` claims a single-list hit
    /// survives to rank 31. That is an arithmetic claim about `RRF_K`, and this
    /// is the check it never had.
    ///
    /// It was 27 at `RRF_K = 60` and `CUTOFF_FRACTION_STANDARD = 0.35`; the
    /// fraction was re-solved to 0.15 when `RRF_K` moved to 12 precisely so the
    /// depth would stay in the same place, and 31 is that depth. What the
    /// number has to be is *between* `DEFAULT_LIMIT` and `MAX_LIMIT`: shallower
    /// and the default precision docks a page nobody asked it to, deeper and it
    /// cannot bite at all.
    #[test]
    fn the_documented_standard_cutoff_really_does_fall_at_rank_31() {
        let best = 2.0 * rrf_score(1);
        let deepest = deepest_surviving_rank(CUTOFF_FRACTION_STANDARD, best);

        assert_eq!(deepest, Some(31));

        let deepest = deepest.expect("just asserted");
        assert!(
            DEFAULT_LIMIT < deepest && deepest < MAX_LIMIT,
            "the default precision keeps a single-list hit to rank {deepest}, \
             which is outside ({DEFAULT_LIMIT}, {MAX_LIMIT}) — it either docks \
             the default page or does nothing at the cap"
        );
    }

    /// And that `Strict` means "the two lists must agree".
    #[test]
    fn the_strict_cutoff_admits_no_single_list_hit_at_all() {
        let best = 2.0 * rrf_score(1);

        assert_eq!(deepest_surviving_rank(CUTOFF_FRACTION_STRICT, best), None);
    }

    #[test]
    fn the_relaxed_cutoff_keeps_the_whole_candidate_list() {
        let best = 2.0 * rrf_score(1);

        assert_eq!(
            deepest_surviving_rank(CUTOFF_FRACTION_RELAXED, best),
            usize::try_from(CANDIDATE_DEPTH).ok()
        );
    }

    #[test]
    fn the_salience_band_is_bounded_by_its_floor_and_span() {
        assert!(close(salience_weight(0.0), SALIENCE_BAND_FLOOR));
        assert!(close(
            salience_weight(1.0),
            SALIENCE_BAND_FLOOR + SALIENCE_BAND_SPAN
        ));
        assert!(
            close(salience_weight(-5.0), SALIENCE_BAND_FLOOR),
            "an out-of-range salience must not widen the band"
        );
        assert!(close(
            salience_weight(5.0),
            SALIENCE_BAND_FLOOR + SALIENCE_BAND_SPAN
        ));
    }

    /// Four of each is more than one and still under [`MINIMUM_POPULATION`].
    /// The refusal has to fire on the size of the population, not merely on it
    /// being degenerate.
    #[test]
    fn a_group_just_under_the_minimum_population_is_still_inconclusive() {
        let group = GroupReport::build(
            "test",
            "document / document",
            (0..8)
                .map(|index| Measurement {
                    left: format!("l{index}"),
                    right: format!("r{index}"),
                    label: if index < 4 { Same } else { Related },
                    note: "",
                    // Positives at 0.95..0.98, negatives at 0.90..0.93, with
                    // one negative pushed above the bar under test.
                    cosine: if index < 4 {
                        0.95 + f64::from(index) * 0.01
                    } else if index == 7 {
                        0.99
                    } else {
                        0.90 + f64::from(index - 4) * 0.01
                    },
                    shorter_chars: 8,
                })
                .collect(),
        );

        // Only four of each; the guard against fitting to a tiny corpus fires
        // first, which is itself the behaviour worth pinning.
        assert_eq!(
            judge("test", 0.95, &group, MERGE_CONTRACT).verdict,
            Verdict::Inconclusive
        );
    }

    #[test]
    fn a_group_too_small_to_fit_a_threshold_to_is_called_inconclusive() {
        let group = GroupReport::build(
            "tiny",
            "document / document",
            vec![Measurement {
                left: "a".to_string(),
                right: "b".to_string(),
                label: Same,
                note: "",
                cosine: 0.99,
                shorter_chars: 1,
            }],
        );

        let verdict = judge("tiny", 0.9, &group, MERGE_CONTRACT);

        assert_eq!(verdict.verdict, Verdict::Inconclusive);
        assert!(verdict.recommended.is_none());
    }

    fn synthetic_group(positives: &[f64], negatives: &[f64]) -> GroupReport {
        let measurements = positives
            .iter()
            .map(|value| (Same, *value))
            .chain(negatives.iter().map(|value| (Related, *value)))
            .enumerate()
            .map(|(index, (label, cosine))| Measurement {
                left: format!("l{index}"),
                right: format!("r{index}"),
                label,
                note: "",
                cosine,
                shorter_chars: 8,
            })
            .collect();

        GroupReport::build("synthetic", "document / document", measurements)
    }

    #[test]
    fn a_threshold_below_the_worst_negative_is_reported_as_too_low() {
        let group = synthetic_group(
            &[0.95, 0.96, 0.97, 0.98, 0.99, 0.99],
            &[0.80, 0.82, 0.84, 0.86, 0.88, 0.94],
        );

        let verdict = judge("c", 0.90, &group, MERGE_CONTRACT);

        assert_eq!(verdict.verdict, Verdict::TooLow, "{verdict:?}");
        assert_eq!(verdict.recommended, Some(0.95));
    }

    #[test]
    fn a_threshold_that_rejects_positives_a_safe_one_would_keep_is_too_high() {
        let group = synthetic_group(
            &[0.80, 0.82, 0.85, 0.90, 0.95, 0.99],
            &[0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        );

        let verdict = judge("c", 0.92, &group, MERGE_CONTRACT);

        assert_eq!(verdict.verdict, Verdict::TooHigh, "{verdict:?}");
        assert_eq!(verdict.recommended, Some(0.66));
    }

    #[test]
    fn a_threshold_in_the_gap_is_reported_as_defensible() {
        let group = synthetic_group(
            &[0.93, 0.94, 0.95, 0.96, 0.97, 0.99],
            &[0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        );

        let verdict = judge("c", 0.92, &group, MERGE_CONTRACT);

        assert_eq!(verdict.verdict, Verdict::Defensible, "{verdict:?}");
    }

    /// Two populations that separate cleanly can be divided by a number. Two
    /// that overlap cannot, and reporting a threshold for them without saying
    /// so is the false precision this whole report exists to avoid.
    #[test]
    fn a_recommendation_dividing_overlapping_populations_is_flagged_as_such() {
        let clean = judge(
            "c",
            0.92,
            &synthetic_group(
                &[0.95, 0.96, 0.97, 0.98, 0.99, 0.99],
                &[0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
            ),
            MERGE_CONTRACT,
        );
        let thin = judge(
            "c",
            0.92,
            &synthetic_group(
                &[0.95, 0.96, 0.97, 0.98, 0.99, 0.99],
                &[0.40, 0.45, 0.50, 0.55, 0.60, 0.9451],
            ),
            MERGE_CONTRACT,
        );
        let overlapping = judge(
            "c",
            0.92,
            &synthetic_group(
                &[0.72, 0.82, 0.90, 0.95, 0.97, 0.99],
                &[0.40, 0.45, 0.50, 0.55, 0.60, 0.9451],
            ),
            MERGE_CONTRACT,
        );

        assert!(
            clean.warning.is_none(),
            "0.30 of clear air is a boundary, not a coincidence: {clean:?}"
        );
        assert!((clean.boundary.expect("both populations are non-empty").gap - 0.30).abs() < 1e-9);

        assert!(
            thin.warning.is_some(),
            "a gap of 0.0049 is inside the rounding of the value quoted: {thin:?}"
        );
        assert_eq!(
            thin.boundary
                .expect("both populations are non-empty")
                .overlap_depth,
            0,
            "a thin gap is still a gap"
        );

        let boundary = overlapping
            .boundary
            .expect("both populations are non-empty");
        assert_eq!(
            boundary.overlap_depth, 3,
            "0.72, 0.82 and 0.90 all sit below the 0.9451 negative"
        );
        assert!(boundary.gap < 0.0);
        assert!(
            overlapping
                .warning
                .as_deref()
                .is_some_and(|warning| warning.contains("OVERLAP")),
            "{overlapping:?}"
        );
    }

    /// The bug this contract exists to prevent: judged as a merge bar, a
    /// candidate floor is condemned for admitting the near misses it is there
    /// to surface.
    #[test]
    fn a_candidate_floor_is_not_penalised_for_admitting_a_related_pair() {
        let group = GroupReport::build(
            "contract",
            "document / document",
            (0..18)
                .map(|index| Measurement {
                    left: format!("l{index}"),
                    right: format!("r{index}"),
                    label: match index {
                        0..=5 => Same,
                        6..=11 => Related,
                        _ => Unrelated,
                    },
                    cosine: match index {
                        0..=5 => 0.90 + f64::from(index) * 0.01,
                        6..=11 => 0.70 + f64::from(index - 6) * 0.01,
                        _ => 0.30 + f64::from(index - 12) * 0.01,
                    },
                    note: "",
                    shorter_chars: 8,
                })
                .collect(),
        );

        let merge = judge("bar", 0.80, &group, MERGE_CONTRACT);
        let candidate = judge("floor", 0.65, &group, CANDIDATE_CONTRACT);

        assert_eq!(
            merge.verdict,
            Verdict::Defensible,
            "0.80 clears every related pair, so as a merge bar it is fine: {merge:?}"
        );
        assert_eq!(
            candidate.verdict,
            Verdict::Defensible,
            "0.65 admits every same and related pair and no unrelated one, which is \
             exactly what a candidate floor is for: {candidate:?}"
        );
    }

    /// Every measured pair lands in exactly one bucket, so the counts add up to
    /// the corpus and a reader can check the arithmetic by eye.
    #[test]
    fn the_resolution_simulation_accounts_for_every_measured_pair() {
        let group = GroupReport::build(
            "outcome",
            "document / document",
            vec![
                Measurement {
                    left: "a".to_string(),
                    right: "b".to_string(),
                    label: Same,
                    note: "",
                    cosine: 0.99,
                    shorter_chars: 1,
                },
                Measurement {
                    left: "c".to_string(),
                    right: "d".to_string(),
                    label: Same,
                    note: "",
                    cosine: 0.85,
                    shorter_chars: 1,
                },
                Measurement {
                    left: "e".to_string(),
                    right: "f".to_string(),
                    label: Same,
                    note: "",
                    cosine: 0.50,
                    shorter_chars: 1,
                },
                Measurement {
                    left: "g".to_string(),
                    right: "h".to_string(),
                    label: Related,
                    note: "",
                    cosine: 0.95,
                    shorter_chars: 1,
                },
                Measurement {
                    left: "i".to_string(),
                    right: "j".to_string(),
                    label: Homonym,
                    note: "",
                    cosine: 1.0,
                    shorter_chars: 1,
                },
            ],
        );

        let outcome = ResolutionOutcome::of("test", &group, 0.92, 0.82);

        assert_eq!(outcome.same_merged, 1);
        assert_eq!(outcome.same_reported, 1);
        assert_eq!(outcome.same_silent, 1);
        assert_eq!(outcome.negatives_merged, 1, "0.95 clears the confident bar");
        assert_eq!(outcome.negatives_reported, 0);
        assert_eq!(outcome.negatives_ignored, 0);
        assert_eq!(
            outcome.over_merges.len(),
            1,
            "an irreversible merge is named, not counted"
        );
        assert_eq!(
            outcome.same_merged
                + outcome.same_reported
                + outcome.same_silent
                + outcome.negatives_merged
                + outcome.negatives_reported
                + outcome.negatives_ignored,
            4,
            "the homonym must not be counted; no threshold decides it"
        );
    }

    // ------------------------------------------------------------------
    // The corpus is the instrument. These check it is calibrated.
    // ------------------------------------------------------------------

    /// A "long name" pair with a short side would be measured under bars the
    /// resolver would never apply to it, and the group would quietly stop
    /// meaning what its name says.
    #[test]
    fn every_long_name_pair_has_two_long_names() {
        for pair in LONG_NAME_PAIRS {
            for name in [pair.left, pair.right] {
                assert!(
                    name.chars().count() > SHORT_NAME_CHARS,
                    "`{name}` is {} chars, so it is in the short regime",
                    name.chars().count()
                );
            }
        }
    }

    #[test]
    fn every_short_name_pair_has_two_short_names() {
        for pair in SHORT_NAME_PAIRS {
            for name in [pair.left, pair.right] {
                assert!(
                    name.chars().count() <= SHORT_NAME_CHARS,
                    "`{name}` is {} chars, so it is in the long regime",
                    name.chars().count()
                );
            }
        }
    }

    /// A duplicated pair is a silent thumb on the scale: the same cosine
    /// counted twice moves a mean without adding evidence.
    #[test]
    fn no_pair_appears_twice_in_any_group() {
        for pairs in [LONG_NAME_PAIRS, SHORT_NAME_PAIRS, RELATIONSHIP_TYPE_PAIRS] {
            for (index, pair) in pairs.iter().enumerate() {
                for other in &pairs[index + 1..] {
                    let same_way = pair.left == other.left && pair.right == other.right;
                    let other_way = pair.left == other.right && pair.right == other.left;
                    assert!(
                        !(same_way || other_way),
                        "`{}` / `{}` is in the corpus twice",
                        pair.left,
                        pair.right
                    );
                }
            }
        }
    }

    /// Cosine is symmetric, so a pair of the same text on both sides is a 1.0
    /// by construction. Only the homonyms are allowed to be that.
    #[test]
    fn only_homonyms_pair_a_text_with_itself() {
        for pairs in [LONG_NAME_PAIRS, SHORT_NAME_PAIRS, RELATIONSHIP_TYPE_PAIRS] {
            for pair in pairs {
                assert!(
                    pair.left != pair.right || pair.label == Homonym,
                    "`{}` is paired with itself but not labelled a homonym",
                    pair.left
                );
            }
        }
    }

    /// Every group has to be big enough on both sides to clear
    /// [`MINIMUM_POPULATION`], or the binary will refuse to conclude anything
    /// and the run is wasted.
    #[test]
    fn every_group_can_support_a_recommendation() {
        for (name, pairs) in [
            ("long", LONG_NAME_PAIRS),
            ("short", SHORT_NAME_PAIRS),
            ("relationship", RELATIONSHIP_TYPE_PAIRS),
        ] {
            let positives = pairs.iter().filter(|pair| pair.label.is_positive()).count();
            let negatives = pairs
                .iter()
                .filter(|pair| !pair.label.is_positive() && pair.label.is_measured())
                .count();

            assert!(
                positives >= MINIMUM_POPULATION,
                "`{name}` has only {positives} same-pairs"
            );
            assert!(
                negatives >= MINIMUM_POPULATION,
                "`{name}` has only {negatives} negatives"
            );
        }
    }

    /// The inverse group is the one that has to mirror the migration. A pair
    /// listed as seeded here but absent from `0001_init.sql` would report a
    /// guard that does not exist.
    #[test]
    fn every_seeded_inverse_pair_is_in_the_corpus() {
        for (left, right) in SEEDED_INVERSE_PAIRS {
            assert!(
                RELATIONSHIP_TYPE_PAIRS.iter().any(|pair| {
                    pair.label == Inverse
                        && ((pair.left == *left && pair.right == *right)
                            || (pair.left == *right && pair.right == *left))
                }),
                "the seeded inverse pair `{left}` / `{right}` is not measured"
            );
        }
    }

    /// And the other half: the corpus must carry inverse pairs the table does
    /// *not* hold, because that is the case the guard cannot cover and the one
    /// worth finding out about.
    #[test]
    fn the_corpus_carries_inverse_pairs_the_vocabulary_does_not_guard() {
        let unguarded = RELATIONSHIP_TYPE_PAIRS
            .iter()
            .filter(|pair| pair.label == Inverse)
            .filter(|pair| !is_seeded_inverse(pair.left, pair.right))
            .count();

        assert!(
            unguarded >= 2,
            "only {unguarded} unguarded inverse pairs; the guard's coverage is untested"
        );
    }

    /// **The other half of the guard, counted rather than described.** Every
    /// inverse pair this corpus measures has to be refused by the curated
    /// table, by the morphology rule, or by both. A pair neither half sees is a
    /// direction reversal that only the height of the coercion bar is still
    /// preventing — and the height of that bar is exactly what the
    /// recommendation block above is arguing about.
    ///
    /// Needs no model, so coverage is a check that runs in CI rather than a
    /// number a reader has to notice in the printed report.
    #[test]
    fn every_measured_inverse_pair_is_refused_by_one_half_of_the_guard_or_the_other() {
        let uncovered: Vec<String> = RELATIONSHIP_TYPE_PAIRS
            .iter()
            .filter(|pair| pair.label == Inverse)
            .filter(|pair| {
                !is_seeded_inverse(pair.left, pair.right)
                    && !are_mechanical_inverses(pair.left, pair.right)
            })
            .map(|pair| format!("{} / {}", pair.left, pair.right))
            .collect();

        assert!(
            uncovered.is_empty(),
            "refused by neither the table nor the morphology rule, so a coercion bar \
             below their cosine reverses facts: {uncovered:?}"
        );
    }

    /// And that it is the *rule* closing the gap rather than the table having
    /// quietly grown. The calibration found four unseeded inverse pairs —
    /// MENTORS/MENTORED_BY, MANAGES/MANAGED_BY, FUNDS/FUNDED_BY,
    /// SUPERSEDES/SUPERSEDED_BY — sharing one shape between them.
    #[test]
    fn the_morphology_rule_covers_the_inverse_pairs_the_table_does_not() {
        let rule_only = RELATIONSHIP_TYPE_PAIRS
            .iter()
            .filter(|pair| pair.label == Inverse)
            .filter(|pair| !is_seeded_inverse(pair.left, pair.right))
            .filter(|pair| are_mechanical_inverses(pair.left, pair.right))
            .count();

        assert!(
            rule_only >= 4,
            "only {rule_only} inverse pairs are the rule's alone; the four the calibration \
             found unseeded should all be"
        );
    }

    /// A near miss that is also somebody else's answer would be labelled twice
    /// and counted as both a hard negative and a positive.
    #[test]
    fn no_query_case_near_miss_is_another_cases_answer() {
        for case in QUERY_CASES {
            for near_miss in case.near_misses {
                assert!(
                    !QUERY_CASES.iter().any(|other| other.proof == *near_miss),
                    "`{near_miss}` is both a near miss and an answer"
                );
            }
        }
    }

    #[test]
    fn distinct_keeps_the_first_occurrence_and_drops_the_rest() {
        assert_eq!(
            distinct(["b", "a", "b", "c", "a"]),
            vec!["b".to_string(), "a".to_string(), "c".to_string()]
        );
    }
}
