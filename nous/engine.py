from typing import Dict, List, Optional
import time
import re
import uuid
import collections
import logging
import os
import hashlib

from .persistence import PersistenceLayer
from .world_model import WorldModel
from .delta_log import DeltaLog
from .updater import BayesianUpdater
from .compressor import Compressor
from .coupling import IdentityCoupler
from .extractor import Extractor
from .dimension import Dimension
from .delta import Delta
from .surprise import compute_surprise
from .retrieval import HybridRetriever, DenseRetriever
from .query_rewriter import QueryRewriter
from .config import config

logger = logging.getLogger(__name__)


class Nous:
    """
    The main public API for the Nous Knowledge Evolution Engine.
    "Knowledge is prediction, not storage."
    
    v3: Rich context retrieval, fuzzy entity matching, evidence-augmented QA,
        relationship traversal for multi-hop, and coupling safety guards.
    """
    # Relationship attributes that link entities to each other
    RELATIONSHIP_ATTRS = {
        "friend", "partner", "spouse", "family", "sibling", "parent", "child",
        "colleague", "mentor", "pet", "owner", "roommate", "neighbor",
        "boss", "employee", "classmate"
    }

    SYMMETRIC_RELATIONS = {
        "friend", "partner", "spouse", "sibling", "colleague", "roommate",
        "neighbor", "classmate"
    }

    INVERSE_RELATIONS = {
        "parent": "child",
        "child": "parent",
        "pet": "owner",
        "owner": "pet",
        "boss": "employee",
        "employee": "boss",
        "mentor": "mentee",
        "mentee": "mentor",
    }

    VALID_GENERIC_RELATION_TARGETS = {
        "kid", "kids", "child", "children", "son", "daughter", "family",
        "fam", "friend", "friends", "best friend", "husband", "wife",
        "partner", "spouse", "mother", "father", "mom", "dad", "parent",
        "parents", "grandma", "grandmother", "grandpa", "grandfather",
        "mentor", "mentors", "teacher", "teachers", "colleague",
        "colleagues", "roommate", "roommates", "neighbor", "neighbors",
        "classmate", "classmates", "puppy", "pup", "dog", "cat", "kitty",
        "pet", "pets", "rescue dog", "church friends", "friends from church"
    }

    MULTI_VALUE_ATTRS = {
        "activity", "event", "achievement", "research", "travel", "purchase",
        "hobby", "interest", "plan", "goal", "education", "location", "pet",
        "friend", "family", "child", "topic", "preference"
    }

    _QUERY_STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
        "to", "for", "and", "or", "but", "did", "does", "do", "what", "when",
        "where", "who", "why", "how", "which", "would", "could", "should",
        "be", "been", "being", "has", "have", "had", "with", "from", "about",
        "into", "than", "then", "their", "his", "her", "its", "they", "them",
        "he", "she", "i", "my", "me", "you", "your"
    }
    
    def __init__(self, db_path: str = "memory.db", extractor=None, 
                 user_context: Optional[Dict[str, str]] = None):
        self.persistence = PersistenceLayer(db_path)
        self.world_model = WorldModel(self.persistence)
        self.delta_log = DeltaLog(self.persistence)
        self.updater = BayesianUpdater()
        self.compressor = Compressor()
        self.coupler = IdentityCoupler(min_overlap=3)
        self.extractor = extractor if extractor is not None else Extractor()
        self.user_context = user_context or {}
        
        # Track entities seen for auto-coupling
        self._entity_registry: Dict[str, set] = {}  # entity -> set of attributes
        self._load_entity_registry()
        self._evidence_events = self.persistence.load_all_evidence_events()
        self._relationship_edges = self.persistence.load_all_relationship_edges()
        self._load_relationship_registry()

        # --- Performance: O(1) relationship lookup dict ---
        # Key: (source_entity_lower, relation_lower) -> List[edge_dict]
        self._rel_dict: Dict[tuple, List[Dict]] = collections.defaultdict(list)
        for edge in self._relationship_edges:
            key = (edge["source_entity"].lower(), edge["relation"].lower())
            self._rel_dict[key].append(edge)

        # --- Performance: Per-entity delta cache ---
        # Populated lazily; invalidated on new deltas for that entity.
        self._delta_cache: Dict[str, List] = {}   # entity_prefix -> sorted deltas

        # --- Hybrid Retrieval Layer ---
        api_key = os.environ.get("OPENROUTER_API_KEY")
        dense_retriever = DenseRetriever(api_key) if api_key else None
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not found. Falling back to BM25-only retrieval.")

        self._retriever = HybridRetriever(dense_retriever=dense_retriever)
        self._retriever.build(self._evidence_events, self._relationship_edges)

        self.query_rewriter = QueryRewriter(api_key=api_key) if api_key else None

    def _load_entity_registry(self):
        """Rebuild entity registry from existing dimensions."""
        for dim in self.world_model.all_dimensions():
            parts = dim.id.rsplit(".", 1)
            if len(parts) == 2:
                entity, attr = parts
                if entity not in self._entity_registry:
                    self._entity_registry[entity] = set()
                self._entity_registry[entity].add(attr)

    def _load_relationship_registry(self):
        """Ensure graph-only entities are discoverable after loading from persistence."""
        for edge in self._relationship_edges:
            source = edge.get("source_entity")
            target = edge.get("target_entity")
            relation = edge.get("relation", "relationship")
            if source:
                self._entity_registry.setdefault(source, set()).add(relation)
            if target:
                self._entity_registry.setdefault(target, set()).add("related_entity")
        
    def observe(self, text: str, source: str = "user", reliability: float = 0.9, 
                timestamp: float = None) -> List[Delta]:
        """
        Ingest new information and update beliefs.
        Returns the list of deltas generated.
        """
        claims = self.extractor.extract(text)
        self._store_evidence_event(text, source=source, timestamp=timestamp, claims=claims)
        deltas = []
        
        for entity, attribute, value in claims:
            # Normalize the value for consistent storage
            value = self._normalize_value(value)
            
            dim_id = f"{entity}.{attribute}"
            dim = self.world_model.get_dimension(dim_id)
            
            # Perform Bayesian Update
            delta = self.updater.update(
                dimension=dim,
                observed_value=value,
                evidence=text,
                source=source,
                reliability=reliability,
                timestamp=timestamp
            )
            
            self.world_model.save_dimension(dim_id)
            self.delta_log.append(delta)
            deltas.append(delta)

            # Invalidate per-entity delta cache on new delta
            self._delta_cache.pop(entity, None)
            
            # Track entity in registry for auto-coupling
            if entity not in self._entity_registry:
                self._entity_registry[entity] = set()
            self._entity_registry[entity].add(attribute)

            if attribute in self.RELATIONSHIP_ATTRS and value != "unknown":
                self._store_relationship_edge(
                    source_entity=entity,
                    relation=attribute,
                    target_entity=value,
                    evidence=text,
                    confidence=reliability,
                    timestamp=timestamp,
                )
            
        # After ingesting, check for potential identity merges
        if claims:
            self._auto_coupling_check(claims)

        # Incremental embedding — fire and forget
        if self._retriever.dense_retriever and self._evidence_events:
            latest_event = self._evidence_events[-1]
            text_to_embed = latest_event.get("text", "")
            if text_to_embed:
                text_hash = hashlib.md5(text_to_embed.encode()).hexdigest()
                cached = self.persistence.get_embedding(text_hash)
                if not cached:
                    try:
                        vecs = self._retriever.dense_retriever.embed([text_to_embed])
                        if vecs:
                            self.persistence.store_embedding(text_hash, vecs[0])
                            latest_event["embedding"] = vecs[0]
                    except Exception:
                        pass
            
        return deltas

    def _store_evidence_event(self, text: str, source: str, timestamp: float, claims: List):
        """Persist the raw turn as searchable evidence for QA-style retrieval."""
        speaker = None
        body = text
        if ":" in text:
            possible_speaker, possible_body = text.split(":", 1)
            if possible_speaker.strip() and len(possible_speaker.strip()) <= 80:
                speaker = possible_speaker.strip()
                body = possible_body.strip()

        entities = set()
        if speaker:
            entities.add(speaker)
        for entity, _, value in claims:
            entities.add(entity)
            # Relationship values are often entity names; indexing them helps multi-hop-ish lookup.
            if value and any(part[:1].isupper() for part in str(value).split()):
                entities.add(str(value))

        terms = sorted(self._tokenize(f"{text} {' '.join(entities)}"))
        event = {
            "id": str(uuid.uuid4()),
            "text": text,
            "source": source,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "speaker": speaker,
            "entities": sorted(entities),
            "terms": terms,
        }
        self.persistence.save_evidence_event(event)
        self._evidence_events.append(event)
        self._retriever.add_event(event)  # keep BM25 index in sync

    def _store_relationship_edge(self, source_entity: str, relation: str, target_entity: str,
                                 evidence: str, confidence: float, timestamp: float):
        """Persist relationship facts as graph edges in addition to belief dimensions."""
        source_entity = source_entity.strip()
        target_entity = target_entity.strip()
        relation = relation.strip().lower()
        if not source_entity or not target_entity or source_entity.lower() == target_entity.lower():
            return
        if not self._valid_relationship_target(target_entity):
            return

        edges = [{
            "source_entity": source_entity,
            "relation": relation,
            "target_entity": target_entity,
            "confidence": confidence,
            "evidence": evidence,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }]

        inverse = relation if relation in self.SYMMETRIC_RELATIONS else self.INVERSE_RELATIONS.get(relation)
        if inverse:
            edges.append({
                "source_entity": target_entity,
                "relation": inverse,
                "target_entity": source_entity,
                "confidence": confidence,
                "evidence": evidence,
                "timestamp": timestamp if timestamp is not None else time.time(),
            })

        for edge in edges:
            self.persistence.save_relationship_edge(edge)
            self._upsert_relationship_cache(edge)
            self._retriever.add_edge(edge)  # keep hybrid retriever in sync
            self._entity_registry.setdefault(edge["source_entity"], set()).add(edge["relation"])
            self._entity_registry.setdefault(edge["target_entity"], set()).add("related_entity")

    def _upsert_relationship_cache(self, edge: Dict):
        """O(1) upsert into the relationship dict cache."""
        key = (edge["source_entity"].lower(), edge["relation"].lower())
        existing_list = self._rel_dict[key]
        for i, ex in enumerate(existing_list):
            if ex["target_entity"] == edge["target_entity"]:
                if edge.get("confidence", 0.0) >= ex.get("confidence", 0.0):
                    existing_list[i] = edge
                return
        existing_list.append(edge)
        # Also keep the flat list for backward-compat with persistence
        self._relationship_edges.append(edge)

    def _valid_relationship_target(self, target_entity: str) -> bool:
        """Reject abstract phrases accidentally extracted as relationship targets."""
        target = " ".join(str(target_entity).strip().split())
        lower = target.lower()
        if not target or len(target) > 60:
            return False
        if lower in self.VALID_GENERIC_RELATION_TARGETS:
            return True
        if any(ch.isdigit() for ch in target):
            return False
        words = lower.split()
        if len(words) > 4:
            return False
        invalid_markers = {
            "awesome", "great", "supportive", "important", "everything",
            "love", "reason", "thing", "things", "worth", "helpful",
            "amazing", "special", "cool", "nice", "fun", "work", "activity",
            "activities", "making", "enjoying"
        }
        if any(word in invalid_markers for word in words):
            return False
        # Proper names and title-cased groups are plausible entity targets.
        if any(part[:1].isupper() for part in target.split()):
            return True
        # Descriptive groups ending in known relationship nouns are plausible.
        return words[-1] in self.VALID_GENERIC_RELATION_TARGETS if words else False
            
    def predict(self, entity: str, attribute: str) -> Dict[str, float]:
        """
        What does the model expect for this entity's attribute?
        Returns a probability distribution.
        """
        return self.world_model.predict(entity, attribute)
        
    def query(self, entity: str, attribute: str) -> Dict[str, float]:
        """Retrieve current beliefs (alias for predict)."""
        return self.world_model.query_belief(entity, attribute)

    def answer_from_beliefs(self, question: str, confidence_threshold: float = 0.80) -> Optional[str]:
        """
        Attempt to answer a factual question directly from Bayesian beliefs,
        bypassing BM25 and LLM answering entirely.

        Returns the top-belief answer string if confidence >= threshold, else None.

        This is the core Nous advantage: for simple profile/fact questions
        where P(entity.attribute = value) is high, skip the noisy retrieval+LLM
        pipeline and return the structured belief directly.

        Only activated for profile/single-hop question types.
        """
        query_type = self._classify_question(question)
        if query_type not in ("profile",):
            return None

        # Try LLM-parsed (entity, attribute) first
        parsed_pairs: List[tuple] = []
        if hasattr(self.extractor, "parse_question"):
            try:
                parsed_pairs = self.extractor.parse_question(question)
            except Exception:
                pass

        # Fallback: heuristic extraction from known question patterns
        if not parsed_pairs:
            known_entities = self.get_all_entities()
            mentioned = self._fuzzy_entity_match(question, known_entities)
            q_lower = question.lower()
            attr_map = {
                "work": "employer", "job": "employer", "employ": "employer",
                "company": "employer", "compan": "employer",
                "live": "location", "lives": "location", "location": "location",
                "from": "location", "city": "location", "town": "location",
                "hobby": "hobby", "hobbies": "hobby", "interest": "interest",
                "study": "education", "studying": "education", "school": "education",
                "major": "education", "degree": "education",
                "research": "research", "researching": "research",
                "age": "age", "old": "age",
                "name": "identity", "call": "identity",
                "pet": "pet", "dog": "pet", "cat": "pet",
                "partner": "partner", "spouse": "partner", "marry": "partner",
                "role": "role", "title": "role",
            }
            for entity in mentioned:
                for kw, attr in attr_map.items():
                    if kw in q_lower:
                        parsed_pairs.append((entity, attr))

        # Try each (entity, attribute) pair
        known_entities = self.get_all_entities()
        for entity, attribute in parsed_pairs:
            # Fuzzy match entity name to a known entity
            candidates = self._fuzzy_entity_match(entity, known_entities) if entity not in known_entities else [entity]
            for matched_entity in (candidates or [entity]):
                dist = self.world_model.predict(matched_entity, attribute)
                if not dist:
                    continue
                top_items = sorted(
                    [(v, p) for v, p in dist.items() if v != "unknown"],
                    key=lambda x: -x[1]
                )
                if top_items and top_items[0][1] >= confidence_threshold:
                    return top_items[0][0]

        return None
        
    def query_relevant(self, question: str, top_k: int = 5, category: str = None) -> dict:
        """
        Given a natural language question, build rich context for the answering LLM.

        v5 Hybrid Strategy:
        1. Query decomposition — break complex multi-hop questions into sub-queries
        2. Fuzzy entity matching — find seed entities from the question text
        3. BM25 evidence retrieval — full BM25 ranking over all evidence events
        4. Multi-hop graph traversal — BFS up to 2 hops via HybridRetriever
        5. Rich profile building — Bayesian distributions → readable facts + evidence
        6. Fallback — LLM question parsing if fuzzy match finds nothing
        """
        sub_queries = [question]
        query_type = self._classify_question(question)
        if self.query_rewriter and len(question.split()) > 12 and query_type in ("multi-hop", "relationship"):
            sub_queries = self.query_rewriter.decompose(question)

        if len(sub_queries) > 1:
            merged_results = {}
            for sq in sub_queries:
                sq_results = self._query_relevant_single(sq, top_k=top_k, category=category)
                for key, val in sq_results.items():
                    if key not in merged_results:
                        merged_results[key] = val
                    else:
                        if isinstance(val, dict) and "evidence" in val:
                            existing = merged_results[key].get("evidence", [])
                            new_ev = val.get("evidence", [])
                            merged_results[key]["evidence"] = existing + new_ev
            return merged_results

        return self._query_relevant_single(question, top_k=top_k, category=category)

    def _query_relevant_single(self, question: str, top_k: int = 5, category: str = None) -> dict:
        """Single-query retrieval pipeline (no decomposition)."""
        query_type = self._classify_question(question)
        known_entities = self.get_all_entities()

        # --- Stage 1: Seed entity discovery (fuzzy matching) ---
        mentioned = self._fuzzy_entity_match(question, known_entities)

        # Fallback: LLM-based (entity, attr) parsing
        if not mentioned and hasattr(self.extractor, "parse_question"):
            parsed = self.extractor.parse_question(question)
            for entity, _ in parsed:
                matched = self._fuzzy_entity_match(entity, known_entities)
                for m in matched:
                    if m not in mentioned:
                        mentioned.append(m)
                if not matched and entity not in mentioned:
                    mentioned.append(entity)

        # Fallback: if still nothing, use top entities by attribute count (most-known)
        if not mentioned:
            ranked = sorted(
                self._entity_registry.items(),
                key=lambda x: -len(x[1])
            )
            mentioned = [e for e, _ in ranked[:top_k]]

        # --- Stage 2: Multi-hop graph traversal via HybridRetriever ---
        max_hops = 2 if query_type in ("relationship", "multi-hop") else 1
        traversal = self._retriever.graph.bfs(
            seeds=mentioned,
            max_hops=max_hops,
            min_confidence=0.3,
            max_entities=12,
        )
        all_entities = list(mentioned) + list(traversal.keys())

        # --- Stage 3: BM25 evidence retrieval ---
        evidence_limit = {
            "temporal": config.EVIDENCE_LIMIT_TEMPORAL,
            "relationship": config.EVIDENCE_LIMIT_MULTIHOP,
            "multi-hop": config.EVIDENCE_LIMIT_MULTIHOP,
            "profile": config.EVIDENCE_LIMIT_DEFAULT,
            "open": config.EVIDENCE_LIMIT_DEFAULT,
        }.get(query_type, config.EVIDENCE_LIMIT_DEFAULT)
        evidence_snippets = self.search_evidence(question, entities=all_entities, top_k=evidence_limit)

        # --- Stage 4: Build rich context ---
        entity_profiles = {}
        relationship_facts = []
        relationship_keys: set = set()

        for entity in mentioned[:top_k]:
            profile = self._build_rich_profile(entity)
            if profile:
                entity_profiles[entity] = profile

            for edge in self.get_relationships(entity):
                if query_type in ("relationship", "multi-hop") or edge.get("confidence", 0.0) >= 0.5:
                    key = (
                        edge["source_entity"].lower(),
                        edge["relation"].lower(),
                        edge["target_entity"].lower(),
                    )
                    if key not in relationship_keys:
                        relationship_keys.add(key)
                        relationship_facts.append(self._format_relationship_edge(edge))

            # Distribution-derived relation fallback
            entity_attrs = self.world_model.get_entity_attributes(entity)
            for attr, dist in entity_attrs.items():
                if attr in self.RELATIONSHIP_ATTRS:
                    sorted_related = sorted(dist.items(), key=lambda x: -x[1])
                    related_threshold = 0.10 if query_type == "relationship" else 0.30
                    related_limit = 5 if query_type == "relationship" else 1
                    for related_name, probability in sorted_related[:related_limit]:
                        if related_name != "unknown" and probability > related_threshold:
                            key = (entity.lower(), attr.lower(), related_name.lower())
                            if key not in relationship_keys:
                                relationship_keys.add(key)
                                relationship_facts.append(
                                    f"{entity} --{attr.replace('_', ' ')}--> {related_name} ({probability:.0%} belief)"
                                )

        # Add traversal-discovered entities (multi-hop)
        for related, path in traversal.items():
            if related not in entity_profiles and len(entity_profiles) < top_k + 4:
                profile = self._build_rich_profile(related)
                if profile:
                    entity_profiles[related] = profile
                # Annotate the relationship path in context
                if len(path) >= 3:
                    path_str = " → ".join(path)
                    key = tuple(path)
                    if key not in relationship_keys:
                        relationship_keys.add(key)
                        relationship_facts.append(f"[Multi-hop path] {path_str}")

        results = self._assemble_context(
            query_type=query_type,
            entity_profiles=entity_profiles,
            relationship_facts=relationship_facts,

            evidence_snippets=evidence_snippets,
        )

        return results

    def _assemble_context(self, query_type: str, entity_profiles: Dict[str, dict],
                          relationship_facts: List[str], evidence_snippets: List[str]) -> dict:
        """Create a stable context layout for the answering model."""
        results = {}
        seen_relations = set()
        relation_lines = []
        for fact in relationship_facts:
            if fact not in seen_relations:
                seen_relations.add(fact)
                relation_lines.append(fact)

        evidence_block = None
        if evidence_snippets:
            label = "Raw conversation snippets retrieved for this temporal question" if query_type == "temporal" else "Raw conversation snippets retrieved for this question"
            evidence_block = {"facts": [label], "evidence": evidence_snippets}

        if query_type == "temporal":
            if evidence_block:
                results["Relevant evidence"] = evidence_block
            if relation_lines:
                results["Relationships"] = {"facts": relation_lines[:10]}
            results.update(entity_profiles)
            return results

        if query_type == "relationship":
            if relation_lines:
                results["Relationships"] = {"facts": relation_lines[:12]}
            results.update(entity_profiles)
            if evidence_block:
                results["Relevant evidence"] = evidence_block
            return results

        # Profile/state questions should prefer compact state over raw evidence.
        results.update(entity_profiles)
        if relation_lines and query_type == "open":
            results["Relationships"] = {"facts": relation_lines[:8]}
        if evidence_block:
            results["Relevant evidence"] = evidence_block
        return results

    def _format_relationship_edge(self, edge: Dict) -> str:
        return (
            f"{edge['source_entity']} --{edge['relation'].replace('_', ' ')}--> "
            f"{edge['target_entity']} ({float(edge.get('confidence', 0.0)):.0%} confidence)"
        )

    def get_relationships(self, entity: str, relation: Optional[str] = None) -> List[Dict]:
        """Return graph relationship edges for an entity. O(1) via dict cache."""
        entity_lower = entity.lower()
        if relation:
            key = (entity_lower, relation.lower())
            matches = list(self._rel_dict.get(key, []))
        else:
            # Gather all relations for this entity
            matches = []
            for (src, _rel), edges in self._rel_dict.items():
                if src == entity_lower:
                    matches.extend(edges)
        matches.sort(key=lambda e: -float(e.get("confidence", 0.0)))
        return matches

    def _classify_question(self, question: str) -> str:
        """Route questions to the context shape most likely to answer them."""
        q = question.lower().strip()
        tokens = self._tokenize(question)

        temporal_markers = (
            "when", "what year", "what month", "what date", "how long",
            "how many years", "how many months", "ago", "before", "after"
        )
        if any(marker in q for marker in temporal_markers):
            return "temporal"

        # Multi-hop: possessive chain like "X's friend's Y" or
        # "the person who works at X" type bridging questions.
        multi_hop_patterns = (
            "'s friend", "'s sister", "'s brother", "'s partner", "'s spouse",
            "'s colleague", "'s mentor", "'s roommate", "'s neighbor",
            "the person who", "the one who", "the individual who",
            "friend of", "sister of", "brother of", "partner of",
            "colleague of", "coworker of", "boss of", "employee of",
        )
        if any(pat in q for pat in multi_hop_patterns):
            return "multi-hop"

        relationship_terms = self.RELATIONSHIP_ATTRS | {
            "friends", "kids", "children", "mother", "father", "parents",
            "siblings", "wife", "husband", "boyfriend", "girlfriend"
        }
        if tokens & relationship_terms:
            return "relationship"

        profile_terms = {
            "identity", "relationship", "status", "employer", "occupation",
            "role", "job", "career", "location", "live", "lives", "from",
            "model", "framework", "language", "preference"
        }
        if tokens & profile_terms:
            return "profile"

        if q.startswith(("what is ", "who is ", "where is ", "where does ")):
            return "profile"

        return "open"

    def search_evidence(self, question: str, entities: Optional[List[str]] = None,
                        top_k: int = 8) -> List[str]:
        """
        Retrieve relevant evidence using BM25 full-text ranking.
        Replaces the old keyword-overlap scoring with proper BM25 (Okapi BM25).
        BM25 handles term frequency saturation and document length normalization —
        both critical for conversation data where turns vary widely in length.
        """
        bm25_results = self._retriever.bm25.query(
            question,
            top_k=top_k * 2,  # Over-fetch to allow dedup without losing results
            entity_filter=set(entities or []),
        )
        seen_core: set = set()
        snippets = []
        for _score, event in bm25_results:
            formatted = self._format_evidence_event(event)
            # Dedup by core content: strip timestamps, speaker prefix, normalize
            raw_text = event.get("text", "").strip()
            # Strip "Speaker: " prefix
            core = re.sub(r'^[^:]{1,60}:\s*', '', raw_text)
            core = core.lower().strip()
            core = re.sub(r'\s+', ' ', core)[:120]
            if core and core not in seen_core:
                seen_core.add(core)
                snippets.append(formatted)
                if len(snippets) >= top_k:
                    break
        return snippets

    def _format_evidence_event(self, event: Dict) -> str:
        ts_str = ""
        timestamp = event.get("timestamp")
        event_dt = None
        if timestamp and timestamp > 1000000:
            try:
                import datetime
                event_dt = datetime.datetime.fromtimestamp(timestamp)
                ts_str = f"[{event_dt.strftime('%Y-%m-%d')}] "
            except Exception:
                pass
        text = event.get("text", "").strip()
        annotations = self._temporal_annotations(text, event_dt)
        if annotations:
            return f"{ts_str}{text} (Temporal normalization: {'; '.join(annotations)})"
        return f"{ts_str}{text}"

    def _temporal_annotations(self, text: str, event_dt) -> List[str]:
        """Resolve common relative time expressions against the turn timestamp."""
        if event_dt is None:
            return []

        import datetime
        import calendar

        text_lower = text.lower()
        base_date = event_dt.date()
        annotations = []
        seen = set()

        number_words = {
            "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12
        }

        def add(label: str, value: str):
            item = f"{label} = {value}"
            if item not in seen:
                seen.add(item)
                annotations.append(item)

        def month_shift(date_obj, offset: int):
            month_index = date_obj.month - 1 + offset
            year = date_obj.year + month_index // 12
            month = month_index % 12 + 1
            day = min(date_obj.day, calendar.monthrange(year, month)[1])
            return datetime.date(year, month, day)

        if "today" in text_lower:
            add("today", base_date.isoformat())
        if "yesterday" in text_lower:
            add("yesterday", (base_date - datetime.timedelta(days=1)).isoformat())
        if "tomorrow" in text_lower:
            add("tomorrow", (base_date + datetime.timedelta(days=1)).isoformat())
        if "last week" in text_lower:
            start = base_date - datetime.timedelta(days=7)
            end = base_date - datetime.timedelta(days=1)
            add("last week", f"{start.isoformat()} to {end.isoformat()}")
        if "next week" in text_lower:
            start = base_date + datetime.timedelta(days=1)
            end = base_date + datetime.timedelta(days=7)
            add("next week", f"{start.isoformat()} to {end.isoformat()}")
        if "this month" in text_lower:
            add("this month", base_date.strftime("%B %Y"))
        if "last month" in text_lower:
            add("last month", month_shift(base_date, -1).strftime("%B %Y"))
        if "next month" in text_lower:
            add("next month", month_shift(base_date, 1).strftime("%B %Y"))
        if "last year" in text_lower:
            add("last year", str(base_date.year - 1))
        if "this year" in text_lower:
            add("this year", str(base_date.year))
        if "next year" in text_lower:
            add("next year", str(base_date.year + 1))

        weekday_aliases = {
            "mon": 0, "monday": 0,
            "tue": 1, "tues": 1, "tuesday": 1,
            "wed": 2, "wednesday": 2,
            "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
            "fri": 4, "friday": 4,
            "sat": 5, "saturday": 5,
            "sun": 6, "sunday": 6,
        }
        for name, weekday in weekday_aliases.items():
            if re.search(rf"\blast\s+{name}\b", text_lower):
                days_back = (base_date.weekday() - weekday) % 7
                days_back = 7 if days_back == 0 else days_back
                add(f"last {name}", (base_date - datetime.timedelta(days=days_back)).isoformat())
            if re.search(rf"\bnext\s+{name}\b", text_lower):
                days_forward = (weekday - base_date.weekday()) % 7
                days_forward = 7 if days_forward == 0 else days_forward
                add(f"next {name}", (base_date + datetime.timedelta(days=days_forward)).isoformat())

        number_pattern = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        for raw_num, unit in re.findall(rf"\b{number_pattern}\s+(day|days|week|weeks|month|months|year|years)\s+ago\b", text_lower):
            n = int(raw_num) if raw_num.isdigit() else number_words.get(raw_num, 0)
            if not n:
                continue
            if unit.startswith("day"):
                value = (base_date - datetime.timedelta(days=n)).isoformat()
            elif unit.startswith("week"):
                value = (base_date - datetime.timedelta(weeks=n)).isoformat()
            elif unit.startswith("month"):
                value = month_shift(base_date, -n).strftime("%B %Y")
            else:
                value = str(base_date.year - n)
            add(f"{raw_num} {unit} ago", value)

        for raw_num in re.findall(rf"\b{number_pattern}\s+weekends?\s+ago\b", text_lower):
            n = int(raw_num) if raw_num.isdigit() else number_words.get(raw_num, 0)
            if n:
                add(f"{raw_num} weekends ago", (base_date - datetime.timedelta(weeks=n)).isoformat())

        return annotations

    def _tokenize(self, text: str) -> set:
        tokens = set(re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", str(text).lower()))
        normalized = set()
        for token in tokens:
            if token in self._QUERY_STOPWORDS or len(token) <= 2:
                continue
            normalized.add(token)
            # Very small stemming for LoCoMo-style wording variations.
            for suffix in ("ing", "ed", "es", "s"):
                if token.endswith(suffix) and len(token) > len(suffix) + 3:
                    normalized.add(token[:-len(suffix)])
                    break
        return normalized
    
    def _fuzzy_entity_match(self, text: str, known_entities: List[str]) -> List[str]:
        """
        Find entities mentioned in text using multiple matching strategies.
        Returns list of matched entity names.
        """
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        mentioned = []
        
        for entity in known_entities:
            entity_lower = entity.lower()
            
            # Skip very short entity names that would match too broadly
            if len(entity) <= 2:
                continue
            
            # Strategy 1: Exact substring match
            if entity_lower in text_lower:
                mentioned.append(entity)
                continue
            
            # Strategy 2: Word-level match (entity name words appear in text)
            entity_words = set(re.findall(r'\b\w+\b', entity_lower))
            # Filter out common words
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", 
                        "at", "to", "for", "and", "or", "but", "speaker", "unknown"}
            entity_significant = entity_words - stopwords
            
            if entity_significant and entity_significant.issubset(text_words):
                mentioned.append(entity)
                continue
            
            # Strategy 3: Partial word match (at least 1 significant word)
            if entity_significant and len(entity_significant & text_words) > 0:
                # Only if the matching word is ≥4 chars (avoid matching "I", "a", etc.)
                matching = entity_significant & text_words
                if any(len(w) >= 4 for w in matching):
                    mentioned.append(entity)
        
        return mentioned

    def _build_rich_profile(self, entity: str) -> dict:
        """
        Build a rich, readable profile for an entity.
        Returns dict with 'facts' (human-readable) and 'evidence' (raw delta snippets).
        """
        attrs = self.world_model.get_entity_attributes(entity)
            
        facts = []
        for attr, dist in attrs.items():
            # Filter out near-uniform distributions
            meaningful = {v: p for v, p in dist.items() 
                         if p > 0.10 and v != "unknown"}
            if not meaningful:
                continue
            
            # Sort by probability descending
            sorted_vals = sorted(meaningful.items(), key=lambda x: -x[1])
            
            # Build human-readable fact string
            top_val, top_prob = sorted_vals[0]
            if top_prob > 0.8:
                fact = f"{entity}'s {attr.replace('_', ' ')} is {top_val}"
            elif top_prob > 0.5:
                fact = f"{entity}'s {attr.replace('_', ' ')} is likely {top_val} ({top_prob:.0%})"
            else:
                alts = ", ".join(f"{v} ({p:.0%})" for v, p in sorted_vals[:3])
                fact = f"{entity}'s {attr.replace('_', ' ')} could be: {alts}"
            
            facts.append(fact)

            observed_values = self._get_observed_values(entity, attr, max_values=8)
            if attr in self.MULTI_VALUE_ATTRS and len(observed_values) > 1:
                facts.append(
                    f"Observed {entity} {attr.replace('_', ' ')} values include: "
                    + ", ".join(observed_values)
                )
        
        if not facts:
            return None
        
        # Get top evidence snippets from delta log
        evidence_snippets = self._get_top_evidence(entity, max_snippets=5)
        
        result = {"facts": facts}
        if evidence_snippets:
            result["evidence"] = evidence_snippets
            
        return result

    def _build_focused_profile(self, entity: str, question: str) -> dict:
        """Build a question-focused profile — only includes attributes relevant to the question."""
        attrs = self.world_model.get_entity_attributes(entity)
        if not attrs:
            return None

        # Extract question keywords for relevance scoring
        q_lower = question.lower()
        q_words = set(re.sub(r'[^a-z\s]', '', q_lower).split())
        stop_words = {'what', 'who', 'where', 'when', 'how', 'why', 'did', 'does',
                      'is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on',
                      'at', 'to', 'for', 'of', 'and', 'or', 'has', 'have', 'do',
                      'about', 'with', 'from', 'this', 'that', 'it', 'be', 'been',
                      'can', 'could', 'would', 'should', 'will', 'her', 'his',
                      'their', 'she', 'he', 'they', 'its'}
        q_keywords = q_words - stop_words

        # Keyword-to-attribute mapping for common question patterns
        keyword_attr_boost = {
            'work': {'employer', 'occupation', 'job', 'career', 'role'},
            'job': {'employer', 'occupation', 'job', 'career', 'role'},
            'career': {'employer', 'occupation', 'job', 'career', 'role'},
            'employ': {'employer', 'occupation'},
            'live': {'location', 'city', 'country', 'residence'},
            'location': {'location', 'city', 'country'},
            'city': {'location', 'city'},
            'hobby': {'hobby', 'hobbies', 'interest', 'activity'},
            'hobbies': {'hobby', 'hobbies', 'interest', 'activity'},
            'interest': {'hobby', 'interest', 'activity'},
            'activity': {'hobby', 'interest', 'activity', 'event'},
            'activities': {'hobby', 'interest', 'activity', 'event'},
            'paint': {'hobby', 'art', 'project', 'activity'},
            'painting': {'hobby', 'art', 'project', 'activity'},
            'study': {'education', 'school', 'major', 'degree'},
            'school': {'education', 'school'},
            'pet': {'pet', 'animal', 'dog', 'cat'},
            'age': {'age', 'birthday'},
            'family': {'family', 'child', 'parent', 'spouse', 'partner'},
            'friend': {'friend', 'relationship'},
            'plan': {'plan', 'goal', 'future'},
            'camp': {'travel', 'hobby', 'activity', 'event'},
            'camping': {'travel', 'hobby', 'activity', 'event'},
            'feel': {'feeling', 'emotion', 'opinion', 'thought'},
            'think': {'opinion', 'thought', 'belief'},
            'book': {'recommendation', 'interest', 'hobby'},
            'music': {'interest', 'hobby', 'preference'},
            'counsel': {'occupation', 'career', 'goal', 'plan'},
            'adopt': {'plan', 'goal', 'event'},
            'identity': {'identity', 'transition', 'gender'},
            'trans': {'identity', 'transition', 'gender'},
        }

        # Score each attribute by relevance
        scored_attrs = []
        for attr, dist in attrs.items():
            meaningful = {v: p for v, p in dist.items() if p > 0.10 and v != 'unknown'}
            if not meaningful:
                continue

            # Score: keyword overlap + semantic boost
            attr_words = set(attr.lower().replace('_', ' ').split())
            score = len(q_keywords & attr_words) * 3  # Direct match = 3 points

            # Check keyword-to-attribute boost
            for qw in q_keywords:
                boosted_attrs = keyword_attr_boost.get(qw, set())
                if attr.lower() in boosted_attrs or attr_words & boosted_attrs:
                    score += 2

            # Check if any value matches question keywords
            for v in meaningful:
                v_words = set(v.lower().split())
                if q_keywords & v_words:
                    score += 1

            scored_attrs.append((score, attr, dist, meaningful))

        # Sort by relevance, take top attributes
        scored_attrs.sort(key=lambda x: -x[0])

        # Always include high-relevance attrs (score > 0), plus top-5 by default
        relevant = [x for x in scored_attrs if x[0] > 0]
        if not relevant:
            # Fallback: take top-5 by probability strength
            relevant = scored_attrs[:5]
        else:
            # Cap at 8 most relevant
            relevant = relevant[:8]

        facts = []
        for score, attr, dist, meaningful in relevant:
            sorted_vals = sorted(meaningful.items(), key=lambda x: -x[1])
            top_val, top_prob = sorted_vals[0]
            if top_prob > 0.8:
                fact = f"{entity}'s {attr.replace('_', ' ')} is {top_val}"
            elif top_prob > 0.5:
                fact = f"{entity}'s {attr.replace('_', ' ')} is likely {top_val} ({top_prob:.0%})"
            else:
                alts = ', '.join(f'{v} ({p:.0%})' for v, p in sorted_vals[:3])
                fact = f"{entity}'s {attr.replace('_', ' ')} could be: {alts}"
            facts.append(fact)

            # For multi-value attributes, include observed values
            observed_values = self._get_observed_values(entity, attr, max_values=8)
            if attr in self.MULTI_VALUE_ATTRS and len(observed_values) > 1:
                facts.append(
                    f'Observed {entity} {attr.replace("_", " ")} values include: '
                    + ', '.join(observed_values)
                )

        if not facts:
            return None

        evidence_snippets = self._get_top_evidence(entity, max_snippets=3)
        result = {'facts': facts}
        if evidence_snippets:
            result['evidence'] = evidence_snippets
        return result

    def _get_observed_values(self, entity: str, attr: str, max_values: int = 8) -> List[str]:
        """Return ALL distinct values that have appeared for an entity attribute.

        Two sources:
        1. Delta posteriors — values that survived Bayesian updates with p > 0.05
        2. Delta evidence text — raw extracted values from the stored evidence string.
           Catches values that were probability-pruned (<0.01 threshold in updater)
           because newer observations pushed them out, but they genuinely existed.

        This is the key fix for multi-hop aggregation questions like
        "what activities does Melanie do?" where 5 separate single-activity
        sessions each push a different value into the distribution.
        """
        dim_id = f"{entity}.{attr}"
        values = []
        seen = set()

        for delta in self.delta_log.get_history(dim_id):
            # Source 1: Posterior distribution values
            candidates = sorted(delta.posterior.items(), key=lambda item: -item[1])
            for value, prob in candidates:
                if value == "unknown" or prob < 0.05:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                values.append(value)
                if len(values) >= max_values:
                    return values

            # Source 2: Evidence text — extract the value that caused this delta
            # The evidence string typically is the raw conversation text or
            # a summary of the observation. The dimension_id encodes the value
            # implicitly, but we can recover more from the prior→posterior diff:
            # the new key that appeared in posterior but NOT in prior is the new value.
            new_keys = set(delta.posterior.keys()) - set(delta.prior.keys())
            for value in new_keys:
                if value == "unknown":
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                values.append(value)
                if len(values) >= max_values:
                    return values

        return values
    
    def _get_top_evidence(self, entity: str, max_snippets: int = 5) -> List[str]:
        """
        Retrieve the most informative evidence snippets for an entity,
        using a per-entity delta cache to avoid loading the full log.
        """
        import datetime
        prefix = f"{entity}."

        # --- Use per-entity cache ---
        if entity not in self._delta_cache:
            all_deltas = self.delta_log.get_all()
            entity_deltas = [d for d in all_deltas if d.dimension_id.startswith(prefix)]
            entity_deltas.sort(key=lambda d: d.surprise, reverse=True)
            self._delta_cache[entity] = entity_deltas
        else:
            entity_deltas = self._delta_cache[entity]

        seen_evidence: set = set()
        snippets = []
        for delta in entity_deltas:
            evidence = (delta.evidence or "").strip()
            if evidence and evidence not in seen_evidence:
                seen_evidence.add(evidence)
                ts_str = ""
                if delta.timestamp and delta.timestamp > 1000000:
                    try:
                        dt = datetime.datetime.fromtimestamp(delta.timestamp)
                        ts_str = f" [{dt.strftime('%Y-%m-%d')}]"
                    except Exception:
                        pass
                snippets.append(f"{evidence}{ts_str}")
                if len(snippets) >= max_snippets:
                    break
        return snippets
    
    def query_at(self, entity: str, attribute: str, at_time: float) -> Dict[str, float]:
        """Time-travel query. Reconstructs beliefs exactly as they were at at_time."""
        dim_id = f"{entity}.{attribute}"
        return self.delta_log.query_at(dim_id, at_time)
        
    def history(self, entity: str, attribute: str) -> List[Delta]:
        """How has understanding evolved? Returns the delta log for this attribute."""
        dim_id = f"{entity}.{attribute}"
        return self.delta_log.get_history(dim_id)
        
    def explain(self, entity: str, attribute: str, value: str) -> List[Delta]:
        """
        Why does the model believe X? 
        Returns deltas that increased P(value).
        Also checks normalized variants of the value.
        """
        history = self.history(entity, attribute)
        value_normalized = self._normalize_value(value)
        explanations = []
        for delta in history:
            for check_val in [value, value_normalized]:
                prior_p = delta.prior.get(check_val, 0.0)
                posterior_p = delta.posterior.get(check_val, 0.0)
                if posterior_p > prior_p:
                    explanations.append(delta)
                    break  # Don't double-add
        return explanations
        
    def surprise(self, text: str) -> float:
        """
        How surprising would this observation be?
        Returns the information content in bits. High bits = very surprising.
        """
        claims = self.extractor.extract(text)
        total_surprise = 0.0
        
        for entity, attribute, value in claims:
            value = self._normalize_value(value)
            dim_id = f"{entity}.{attribute}"
            dim = self.world_model.get_dimension(dim_id)
            prior_prob = dim.get_probability(value, record_access=False)
            total_surprise += compute_surprise(prior_prob)
            
        return total_surprise
        
    def get_coupling(self, entity_a: str, entity_b: str) -> float:
        """
        Get the identity coupling score between two entities.
        A high score means they are likely the same entity.
        """
        dims_a = self.world_model.get_entity_attributes(entity_a)
        dims_b = self.world_model.get_entity_attributes(entity_b)
        return self.coupler.compute_coupling(dims_a, dims_b)
    
    def get_all_entities(self) -> List[str]:
        """Returns a list of all known entity names."""
        return list(self._entity_registry.keys())
    
    def get_entity_profile(self, entity: str) -> Dict[str, Dict[str, float]]:
        """Returns a full profile of an entity: all attributes with distributions."""
        attrs = self.world_model.get_entity_attributes(entity)
        return {attr: dist for attr, dist in attrs.items()}
        
    def apply_decay(self, current_time: float = None):
        """Apply entropy decay (forgetting) to all unused dimensions."""
        dims = self.world_model.all_dimensions()
        self.compressor.apply_decay(dims, current_time)
        for dim in dims:
            self.world_model.save_dimension(dim.id)
    
    def _normalize_value(self, value: str) -> str:
        """Normalize a value for consistent storage."""
        # Replace underscores with spaces
        value = value.replace("_", " ")
        # Collapse multiple spaces
        value = " ".join(value.split())
        return value.strip()
    
    def _auto_coupling_check(self, new_claims: List):
        """
        After new claims are ingested, check if any existing entities
        might be the same entity as the newly updated ones.
        
        v3: Uses name similarity as an additional gate and requires
        minimum attribute overlap before checking.
        """
        new_entities = set(c[0] for c in new_claims)
        
        for new_entity in new_entities:
            new_attrs = self._entity_registry.get(new_entity, set())
            if not new_attrs:
                continue
                
            for existing_entity, existing_attrs in self._entity_registry.items():
                if existing_entity == new_entity:
                    continue
                
                # Gate: require minimum overlapping attributes
                overlap = new_attrs.intersection(existing_attrs)
                if len(overlap) < 3:
                    continue
                
                coupling = self.get_coupling(new_entity, existing_entity)
                if self.coupler.should_merge(coupling, new_entity, existing_entity):
                    # Log the discovery (in production, you might auto-merge or prompt)
                    logger.info("Auto-coupling detected: '%s' ≈ '%s' (score: %.2f)",
                                new_entity, existing_entity, coupling)
            
    def embed_all_evidence(self):
        """Pre-warm the embedding cache for all stored evidence."""
        if self._retriever.dense_retriever is None:
            return
            
        texts_to_embed = []
        hash_to_text = {}
        
        # 1. Identify what needs embedding
        for event in self._evidence_events:
            text = event.get("text", "")
            if not text:
                continue
            text_hash = hashlib.md5(text.encode()).hexdigest()
            emb = self.persistence.get_embedding(text_hash)
            if emb is None:
                if text_hash not in hash_to_text:
                    hash_to_text[text_hash] = text
                    texts_to_embed.append((text_hash, text))
            else:
                event["embedding"] = emb
                    
        # 2. Batch embed missing texts
        if texts_to_embed:
            logger.info("Pre-warming embeddings for %d unique events...", len(texts_to_embed))
            batch_size = 32
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                hashes = [b[0] for b in batch]
                texts = [b[1] for b in batch]
                
                embeddings = self._retriever.dense_retriever.embed(texts)
                for h, emb in zip(hashes, embeddings):
                    if emb is not None:
                        self.persistence.store_embedding(h, emb)
            logger.info("Pre-warming complete.")
            
        # 3. Attach embeddings to all events in memory
        for event in self._evidence_events:
            if "embedding" not in event:
                text = event.get("text", "")
                if text:
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    emb = self.persistence.get_embedding(text_hash)
                    if emb is not None:
                        event["embedding"] = emb

    def close(self):
        """Closes the underlying database connection."""
        self.persistence.conn.close()
