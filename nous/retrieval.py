"""
Hybrid Retrieval Layer for Nous.

Implements:
1. BM25Retriever      — BM25 full-text ranking over evidence events (pure Python)
2. TFIDFRetriever     — TF-IDF cosine similarity over evidence events (pure Python)
3. GraphTraverser     — BFS multi-hop traversal over relationship graph
4. HybridRetriever    — Combines BM25 + graph traversal, re-ranks with TF-IDF

No external dependencies. All pure Python (math module only).

Design:
  Query
    → BM25: fast term-based relevance over all evidence events
    → Graph: BFS from entity seeds, discover related entities up to N hops
    → Re-rank: TF-IDF cosine boosts semantically-dense results
    → Return top-K snippets + entity set to the Bayesian belief layer
"""
import math
import re
import collections
import urllib.request
import json
import hashlib
from typing import List, Dict, Tuple, Set, Optional


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "did", "does", "do", "what", "when",
    "where", "who", "why", "how", "which", "would", "could", "should",
    "be", "been", "being", "has", "have", "had", "with", "from", "about",
    "into", "than", "then", "their", "his", "her", "its", "they", "them",
    "he", "she", "i", "my", "me", "you", "your", "we", "us", "our",
    "this", "that", "it", "so", "if", "not", "no", "by", "as", "just",
    "can", "will", "also", "very", "really", "like", "more", "some",
    "know", "think", "feel", "want", "need", "going", "get", "got",
}

def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords and short tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_'-]*\b", text.lower())
    result = []
    for t in tokens:
        if t in _STOPWORDS or len(t) <= 2:
            continue
        result.append(t)
    return result

def _stem(token: str) -> str:
    """Very lightweight suffix stripping for English."""
    if len(token) <= 4:
        return token
    for suffix in ("ing", "tion", "ness", "ment", "ed", "er", "es", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token

def _tokenize_stemmed(text: str) -> List[str]:
    return [_stem(t) for t in _tokenize(text)]


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 (Okapi BM25) over a corpus of text documents.
    
    BM25 is the gold standard for full-text retrieval without neural models.
    Outperforms TF-IDF significantly on long documents (like conversation turns).
    
    Parameters k1, b are standard BM25 defaults (Robertson & Zaragoza 2009).
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Dict] = []           # original event dicts
        self._tokenized: List[List[str]] = [] # stemmed tokens per doc
        self._df: Dict[str, int] = {}         # document frequency per term
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}      # precomputed IDF per term

    def index(self, events: List[Dict]) -> None:
        """Build index from a list of evidence event dicts (with 'text' field)."""
        self._docs = list(events)
        self._tokenized = []
        self._df = collections.Counter()
        
        for event in self._docs:
            text = event.get("text", "")
            # Also include speaker as a pseudo-term for entity matching
            speaker = event.get("speaker", "")
            tokens = _tokenize_stemmed(f"{speaker} {text}")
            self._tokenized.append(tokens)
            for t in set(tokens):
                self._df[t] += 1
        
        N = len(self._docs)
        if N == 0:
            self._avgdl = 0.0
            return
        
        total_len = sum(len(toks) for toks in self._tokenized)
        self._avgdl = total_len / N
        
        # Precompute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {}
        for term, df in self._df.items():
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def add_event(self, event: Dict) -> None:
        """Incrementally add one event and rebuild IDF. Call after index()."""
        self._docs.append(event)
        text = event.get("text", "")
        speaker = event.get("speaker", "")
        tokens = _tokenize_stemmed(f"{speaker} {text}")
        self._tokenized.append(tokens)
        
        for t in set(tokens):
            self._df[t] = self._df.get(t, 0) + 1
        
        N = len(self._docs)
        total_len = sum(len(toks) for toks in self._tokenized)
        self._avgdl = total_len / N
        
        # Recompute IDF for changed terms only
        for t in set(tokens):
            df = self._df[t]
            self._idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def query(self, question: str, top_k: int = 10,
              entity_filter: Optional[Set[str]] = None) -> List[Tuple[float, Dict]]:
        """
        Retrieve top-K most relevant events for a question.
        
        Args:
            question: Natural language question.
            top_k: Number of results to return.
            entity_filter: If given, boost events mentioning these entities.
        
        Returns:
            List of (score, event_dict) sorted descending by score.
        """
        if not self._docs:
            return []
        
        query_tokens = _tokenize_stemmed(question)
        if not query_tokens:
            return []
        
        entity_lowers: Set[str] = set()
        if entity_filter:
            entity_lowers = {e.lower() for e in entity_filter}
        
        scores: List[float] = []
        
        for i, doc_tokens in enumerate(self._tokenized):
            dl = len(doc_tokens)
            if dl == 0:
                scores.append(0.0)
                continue
            
            tf_map = collections.Counter(doc_tokens)
            score = 0.0
            
            for qt in query_tokens:
                if qt not in self._idf:
                    continue
                idf = self._idf[qt]
                tf = tf_map.get(qt, 0)
                # BM25 term score
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                score += idf * numerator / denominator
            
            # Entity boost: reward events that mention the same entities as the query
            if entity_lowers:
                event_entities = {str(e).lower() for e in self._docs[i].get("entities", [])}
                speaker = (self._docs[i].get("speaker") or "").lower()
                overlap = entity_lowers & (event_entities | {speaker})
                score += len(overlap) * 1.5  # additive entity bonus
            
            scores.append(score)
        
        # Sort by score descending, break ties by timestamp (newer first)
        indexed = sorted(
            enumerate(scores),
            key=lambda x: (-x[1], -(self._docs[x[0]].get("timestamp") or 0))
        )
        
        seen_texts: Set[str] = set()
        results = []
        for idx, score in indexed:
            if score <= 0.0:
                break
            event = self._docs[idx]
            text = event.get("text", "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            results.append((score, event))
            if len(results) >= top_k:
                break
        
        return results


# ---------------------------------------------------------------------------
# TF-IDF Retriever (for re-ranking)
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """
    TF-IDF cosine similarity for re-ranking BM25 results.
    Pure Python — no numpy.
    """
    
    def __init__(self):
        self._docs: List[Dict] = []
        self._tfidf: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}

    def index(self, events: List[Dict]) -> None:
        self._docs = list(events)
        tokenized = [_tokenize_stemmed(e.get("text", "")) for e in self._docs]
        N = len(tokenized)
        df: Dict[str, int] = collections.Counter()
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1
        self._idf = {
            t: math.log((1 + N) / (1 + d)) + 1.0
            for t, d in df.items()
        }
        self._tfidf = []
        for toks in tokenized:
            tf = collections.Counter(toks)
            total = len(toks) or 1
            vec = {t: (tf[t] / total) * self._idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._tfidf.append({t: v / norm for t, v in vec.items()})

    def score(self, question: str, doc_idx: int) -> float:
        """Cosine similarity between query and document at doc_idx."""
        q_tokens = _tokenize_stemmed(question)
        if not q_tokens or doc_idx >= len(self._tfidf):
            return 0.0
        tf = collections.Counter(q_tokens)
        total = len(q_tokens)
        q_vec = {t: (tf[t] / total) * self._idf.get(t, 0.0) for t in tf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {t: v / q_norm for t, v in q_vec.items()}
        doc_vec = self._tfidf[doc_idx]
        return sum(q_vec.get(t, 0.0) * dv for t, dv in doc_vec.items())


# ---------------------------------------------------------------------------
# Graph Traverser
# ---------------------------------------------------------------------------

class GraphTraverser:
    """
    Multi-hop BFS traversal over the relationship graph.
    
    Given seed entities, expands outward N hops through the relationship edges
    to discover transitively related entities. This enables answering questions
    like "Who is the friend of the person who works at Google?" which require
    graph traversal rather than per-entity lookup.
    """
    
    def __init__(self):
        # Adjacency: source_entity -> list of {relation, target_entity, confidence}
        self._graph: Dict[str, List[Dict]] = collections.defaultdict(list)
        self._current_question: str = ""
    
    def build(self, relationship_edges: List[Dict]) -> None:
        """Build adjacency from a list of relationship edge dicts."""
        self._graph = collections.defaultdict(list)
        for edge in relationship_edges:
            src = edge.get("source_entity", "")
            tgt = edge.get("target_entity", "")
            if src and tgt:
                self._graph[src].append({
                    "relation": edge.get("relation", "related"),
                    "target_entity": tgt,
                    "confidence": edge.get("confidence", 0.5),
                    "evidence": edge.get("evidence", ""),
                })
    
    def add_edge(self, edge: Dict) -> None:
        """Incrementally add one edge."""
        src = edge.get("source_entity", "")
        if src:
            self._graph[src].append({
                "relation": edge.get("relation", "related"),
                "target_entity": edge.get("target_entity", ""),
                "confidence": edge.get("confidence", 0.5),
                "evidence": edge.get("evidence", ""),
            })
    
    def get_neighbors(self, entity: str, min_confidence: float = 0.3) -> List[Dict]:
        """Get all direct neighbors of entity."""
        return [
            e for e in self._graph.get(entity, [])
            if e["confidence"] >= min_confidence
        ]
    
    def bfs(self, seeds: List[str], max_hops: int = 2,
            min_confidence: float = 0.3,
            max_entities: int = 15) -> Dict[str, List[str]]:
        """
        BFS from seed entities up to max_hops.
        
        Returns:
            Dict mapping each discovered entity to the path from a seed,
            e.g. {"Melanie": ["Caroline", "friend", "Melanie"]}
        """
        visited: Dict[str, List[str]] = {}  # entity -> path
        queue = collections.deque()
        
        for seed in seeds:
            if seed not in visited:
                visited[seed] = [seed]
                queue.append((seed, 0))
        
        while queue and len(visited) < max_entities:
            entity, depth = queue.popleft()
            if depth >= max_hops:
                continue
            
            question_terms = set(self._current_question.lower().split()) if self._current_question else set()

            for edge in self._graph.get(entity, []):
                edge_conf = edge.get("confidence", 0.5)
                tgt = edge["target_entity"]
                if not tgt or tgt in visited:
                    continue
                tgt_words = set(tgt.lower().split())
                overlap = len(tgt_words & question_terms) if question_terms else 0
                relevance_boost = 1.0 + (0.3 * overlap)
                scored_conf = edge_conf * relevance_boost
                if scored_conf < min_confidence:
                    continue
                path = visited[entity] + [edge["relation"], tgt]
                visited[tgt] = path
                queue.append((tgt, depth + 1))
        
        # Return only non-seed entities (the discovered ones)
        return {k: v for k, v in visited.items() if k not in seeds}
    
    def find_paths(self, from_entity: str, to_entity: str,
                   max_hops: int = 3) -> List[List[str]]:
        """Find all paths between two entities (for explanation)."""
        paths = []
        stack = [([from_entity], set([from_entity]))]
        
        while stack:
            path, visited = stack.pop()
            current = path[-1]
            if current == to_entity:
                paths.append(path)
                continue
            if len(path) > max_hops * 2 + 1:
                continue
            for edge in self.get_neighbors(current):
                tgt = edge["target_entity"]
                if tgt not in visited:
                    new_path = path + [edge["relation"], tgt]
                    stack.append((new_path, visited | {tgt}))
        
        return sorted(paths, key=len)


# ---------------------------------------------------------------------------
# Dense Retriever (for re-ranking)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    OpenRouter API-based Dense Retriever.
    Pure Python (no numpy).
    """
    def __init__(self, api_key: str, model: str = "qwen/qwen3-embedding-8b"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/embeddings"
        
    def embed(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not self.api_key or not texts:
            return []
            
        # Batch max 32 at a time
        embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = json.dumps({
                "model": self.model,
                "input": batch
            }).encode("utf-8")
            
            req = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST"
            )
            
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    batch_embeddings = [None] * len(batch)
                    for item in resp_json.get("data", []):
                        batch_embeddings[item["index"]] = item["embedding"]
                    embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Embedding API error: {e}")
                # Fallback on error by returning Nones
                embeddings.extend([None] * len(batch))
                
        return embeddings

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Hybrid Retriever (main interface)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines BM25 full-text retrieval + graph-based entity traversal.
    
    This is the main retrieval layer that sits in front of the Bayesian
    belief engine. For any natural language question, it:
    
    1. Finds relevant evidence snippets via BM25 (much better than keyword overlap)
    2. Expands entity set via multi-hop graph traversal
    3. Re-ranks with TF-IDF cosine similarity
    4. Returns top-K snippets + full set of relevant entities
    
    Key improvement over the old search_evidence():
    - BM25 is the gold standard for full-text retrieval (used by Elasticsearch)
    - Multi-hop traversal enables multi-hop QA answers
    - Incremental indexing (add_event) keeps the index up to date during ingestion
    """
    
    def __init__(self, dense_retriever: Optional[DenseRetriever] = None):
        self.bm25 = BM25Retriever()
        self.tfidf = TFIDFRetriever()
        self.graph = GraphTraverser()
        self.dense_retriever = dense_retriever
        self._indexed = False
        self._needs_tfidf_rebuild = False
        self._event_count = 0
        self._debug_count = 0

    def build(self, events: List[Dict], relationship_edges: List[Dict]) -> None:
        """Full index build from scratch."""
        self.bm25.index(events)
        self.tfidf.index(events)
        self.graph.build(relationship_edges)
        self._indexed = True
        self._event_count = len(events)
        self._needs_tfidf_rebuild = False

    def add_event(self, event: Dict) -> None:
        """Incrementally add one evidence event."""
        self.bm25.add_event(event)
        self._event_count += 1
        # TF-IDF rebuild is expensive; batch it every 20 events
        self._needs_tfidf_rebuild = True
        if self._event_count % 20 == 0:
            self.tfidf.index(self.bm25._docs)
            self._needs_tfidf_rebuild = False

    def add_edge(self, edge: Dict) -> None:
        """Incrementally add one relationship edge."""
        self.graph.add_edge(edge)

    def retrieve(self,
                 question: str,
                 seed_entities: List[str],
                 top_k: int = 10,
                 max_hops: int = 2,
                 min_edge_confidence: float = 0.3) -> Dict:
        """
        Full hybrid retrieval pipeline.
        
        Args:
            question: The natural language question.
            seed_entities: Entity names already found in the question.
            top_k: Number of evidence snippets to return.
            max_hops: How many hops to traverse in the relationship graph.
            min_edge_confidence: Minimum edge confidence for graph traversal.
        
        Returns:
            {
              "snippets": List[str],           # formatted evidence snippets
              "entities": List[str],            # all relevant entity names
              "traversal_paths": Dict[str, List[str]],  # multi-hop paths
              "raw_events": List[Dict],         # raw event dicts for downstream use
            }
        """

        # --- Step 1: Graph traversal to expand entity set ---
        traversal = {}
        if seed_entities:
            self.graph._current_question = question
            traversal = self.graph.bfs(
                seeds=seed_entities,
                max_hops=max_hops,
                min_confidence=min_edge_confidence,
            )
        all_entities = list(seed_entities) + list(traversal.keys())
        
        # --- Step 2: BM25 retrieval ---
        bm25_results = self.bm25.query(
            question,
            top_k=30,  # over-fetch for re-ranking
            entity_filter=set(all_entities),
        )
        
        if not bm25_results:
            return {
                "snippets": [],
                "entities": all_entities,
                "traversal_paths": traversal,
                "raw_events": [],
            }
        
        # --- Step 3: Re-ranking ---
        reranked = []
        
        if self.dense_retriever is not None:
            max_bm25 = max(s for s, _ in bm25_results) or 1.0
            query_embeds = self.dense_retriever.embed([question])
            q_vec = query_embeds[0] if query_embeds else None
            
            if q_vec is not None:
                for bm25_score, event in bm25_results:
                    bm25_norm = bm25_score / max_bm25
                    c_vec = event.get("embedding")
                    if c_vec:
                        cosine = self.dense_retriever.cosine_similarity(q_vec, c_vec)
                        combined = 0.65 * bm25_norm + 0.35 * cosine
                    else:
                        combined = 0.65 * bm25_norm
                    reranked.append((combined, event))
            else:
                self._tfidf_rerank(question, bm25_results, reranked)
        else:
            self._tfidf_rerank(question, bm25_results, reranked)
        
        reranked.sort(key=lambda x: -x[0])
        top_events = [event for _, event in reranked[:top_k]]
        
        return {
            "snippets": [],          # caller formats with _format_evidence_event
            "entities": all_entities,
            "traversal_paths": traversal,
            "raw_events": top_events,
        }
    
    def get_neighbors(self, entity: str, min_confidence: float = 0.3) -> List[Dict]:
        """Get direct neighbors of an entity in the graph."""
        return self.graph.get_neighbors(entity, min_confidence)

    def _tfidf_rerank(self, question: str, bm25_results: List[Tuple[float, Dict]], reranked: List[Tuple[float, Dict]]):
        if self._needs_tfidf_rebuild and len(self.bm25._docs) < 2000:
            self.tfidf.index(self.bm25._docs)
            self._needs_tfidf_rebuild = False
        
        text_to_idx = {
            e.get("text", ""): i
            for i, e in enumerate(self.bm25._docs)
        }
        
        for bm25_score, event in bm25_results:
            idx = text_to_idx.get(event.get("text", ""), -1)
            tfidf_score = self.tfidf.score(question, idx) if idx >= 0 else 0.0
            combined = 0.7 * bm25_score + 0.3 * tfidf_score * 10
            reranked.append((combined, event))
