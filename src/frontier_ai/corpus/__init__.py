"""FrontierCorpus v1 data pipeline (MASTER_CONTEXT §12–13, §37 step 3).

Stages that turn a frozen, verified source registry into a training-ready,
versioned dataset. This package *extends* the P004A/B corpus infrastructure
(``frontier_ai.data.corpora``, ``frontier_ai.data.mediawiki``,
``frontier_ai.tokenization.research_corpus``) — it does not fork it.

Stages are pure functions sharing one document model and one outcome type,
so each composes, tests, and reports independently. See ``pipeline.py`` for
the model and each stage module for its contract.

    from frontier_ai.corpus import PipelineDocument, normalize_documents, ...
"""

from .dedup import exact_dedup  # noqa: F401
from .frontier_docs import (  # noqa: F401
    FrontierBuild,
    FrontierDerivation,
    build_frontier_dataset,
    derive_documents,
    derive_frontier_documents,
    run_stages,
    verify_frozen_inputs,
)
from .langid import (  # noqa: F401
    DEFAULT_MIN_DECLARED_SHARE,
    langid_documents,
    letter_total,
    script_profile,
    top_script,
)
from .manifest import build_manifest, canonical_json, write_manifest  # noqa: F401
from .normalize import POLICY_VERSION, SUPPORTED_POLICIES, normalize_documents, normalize_text  # noqa: F401
from .pipeline import PipelineDocument, Removal, StageOutcome, sha256_text  # noqa: F401
from .quality import FLAG, REJECT, RULE_ORDER, QualityPolicy, quality_filter  # noqa: F401
from .registry import (  # noqa: F401
    FRONTIER_CORPUS_ID,
    FRONTIER_CORPUS_VERSION,
    FrontierSource,
    RegistryError,
    language_scripts,
    load_frontier_registry,
)
from .shards import (  # noqa: F401
    Shard,
    check_shards,
    pack,
    seeded_shuffle,
    shard_hashes,
    write_shards,
)
from .split import SPLIT_METHOD, train_holdout  # noqa: F401

__all__ = [
    "DEFAULT_MIN_DECLARED_SHARE",
    "FLAG",
    "FRONTIER_CORPUS_ID",
    "FRONTIER_CORPUS_VERSION",
    "POLICY_VERSION",
    "PipelineDocument",
    "QualityPolicy",
    "REJECT",
    "Removal",
    "RULE_ORDER",
    "RegistryError",
    "SPLIT_METHOD",
    "SUPPORTED_POLICIES",
    "Shard",
    "StageOutcome",
    "check_shards",
    "exact_dedup",
    "langid_documents",
    "language_scripts",
    "letter_total",
    "load_frontier_registry",
    "normalize_documents",
    "normalize_text",
    "pack",
    "quality_filter",
    "script_profile",
    "seeded_shuffle",
    "shard_hashes",
    "sha256_text",
    "top_script",
    "train_holdout",
    "write_manifest",
    "write_shards",
]
