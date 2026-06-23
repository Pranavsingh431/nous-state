"""
Centralized configuration for Nous.
All magic numbers live here. Override via environment variables.
"""

import os


class NousConfig:
    # BM25 retrieval
    BM25_K1: float = float(os.getenv("NOUS_BM25_K1", "1.5"))
    BM25_B: float = float(os.getenv("NOUS_BM25_B", "0.75"))
    BM25_TOP_K: int = int(os.getenv("NOUS_BM25_TOP_K", "30"))

    # Graph traversal
    GRAPH_MAX_HOPS: int = int(os.getenv("NOUS_GRAPH_MAX_HOPS", "2"))
    GRAPH_MIN_CONFIDENCE: float = float(os.getenv("NOUS_GRAPH_MIN_CONFIDENCE", "0.25"))
    GRAPH_MAX_ENTITIES: int = int(os.getenv("NOUS_GRAPH_MAX_ENTITIES", "15"))

    # Reranking
    RERANK_TOP_K: int = int(os.getenv("NOUS_RERANK_TOP_K", "10"))
    DENSE_WEIGHT: float = float(os.getenv("NOUS_DENSE_WEIGHT", "0.35"))
    BM25_WEIGHT: float = float(os.getenv("NOUS_BM25_WEIGHT", "0.65"))

    # Context assembly
    EVIDENCE_LIMIT_DEFAULT: int = int(os.getenv("NOUS_EVIDENCE_LIMIT", "10"))
    EVIDENCE_LIMIT_TEMPORAL: int = int(os.getenv("NOUS_EVIDENCE_LIMIT_TEMPORAL", "15"))
    EVIDENCE_LIMIT_MULTIHOP: int = int(os.getenv("NOUS_EVIDENCE_LIMIT_MULTIHOP", "12"))

    # Entity matching
    FUZZY_MATCH_MIN_WORD_OVERLAP: int = int(os.getenv("NOUS_FUZZY_MIN_OVERLAP", "1"))
    COUPLING_MIN_OVERLAP: int = int(os.getenv("NOUS_COUPLING_MIN_OVERLAP", "3"))
    COUPLING_MERGE_THRESHOLD: float = float(os.getenv("NOUS_COUPLING_MERGE_THRESHOLD", "0.97"))

    # Decay
    DECAY_HALF_LIFE_DAYS: float = float(os.getenv("NOUS_DECAY_HALF_LIFE_DAYS", "30.0"))

    # Embedding
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("NOUS_EMBEDDING_BATCH_SIZE", "32"))
    EMBEDDING_MODEL: str = os.getenv("NOUS_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

    # Novelty
    NOVELTY_PRIOR: float = float(os.getenv("NOUS_NOVELTY_PRIOR", "0.05"))


config = NousConfig()
