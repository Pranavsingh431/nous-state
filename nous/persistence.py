import sqlite3
import json
import time
from collections import OrderedDict
from typing import List, Dict, Optional
from .dimension import Dimension
from .delta import Delta

class LRUCache:
    def __init__(self, maxsize=1000):
        self._cache = OrderedDict()
        self._maxsize = maxsize
    
    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]
    
    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

class PersistenceLayer:
    """
    SQLite backend for storing dimensions, deltas, and couplings.
    Zero external dependencies.
    """
    def __init__(self, db_path: str = "memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._embedding_cache = LRUCache(maxsize=1000)
        self._init_db()
        
    def _init_db(self):
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-32768")
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS dimensions (
                    id TEXT PRIMARY KEY,
                    distribution TEXT NOT NULL,
                    last_accessed REAL NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS deltas (
                    id TEXT PRIMARY KEY,
                    dimension_id TEXT NOT NULL,
                    prior TEXT NOT NULL,
                    posterior TEXT NOT NULL,
                    surprise REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_reliability REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    valid_time REAL,
                    cascaded_from TEXT
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deltas_dim ON deltas(dimension_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deltas_time ON deltas(timestamp)
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_events (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp REAL,
                    speaker TEXT,
                    entities TEXT NOT NULL,
                    terms TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_evidence_time ON evidence_events(timestamp)
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS relationship_edges (
                    source_entity TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    timestamp REAL,
                    PRIMARY KEY (source_entity, relation, target_entity)
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_source
                ON relationship_edges(source_entity)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_target
                ON relationship_edges(target_entity)
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

    def save_dimension(self, dim: Dimension):
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO dimensions (id, distribution, last_accessed)
                VALUES (?, ?, ?)
            """, (dim.id, json.dumps(dim.distribution), dim.last_accessed))

    def load_dimension(self, dim_id: str) -> Optional[Dimension]:
        cursor = self.conn.execute("SELECT * FROM dimensions WHERE id = ?", (dim_id,))
        row = cursor.fetchone()
        if row:
            return Dimension(row['id'], json.loads(row['distribution']), row['last_accessed'])
        return None
        
    def load_all_dimensions(self) -> List[Dimension]:
        cursor = self.conn.execute("SELECT * FROM dimensions")
        return [Dimension(row['id'], json.loads(row['distribution']), row['last_accessed']) 
                for row in cursor.fetchall()]

    def save_delta(self, delta: Delta):
        with self.conn:
            self.conn.execute("""
                INSERT INTO deltas (id, dimension_id, prior, posterior, surprise, 
                                    evidence, source, source_reliability, timestamp, 
                                    valid_time, cascaded_from)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (delta.id, delta.dimension_id, json.dumps(delta.prior), 
                  json.dumps(delta.posterior), delta.surprise, delta.evidence, 
                  delta.source, delta.source_reliability, delta.timestamp, 
                  delta.valid_time, delta.cascaded_from))

    def get_deltas(self, dimension_id: str = None) -> List[Delta]:
        if dimension_id:
            cursor = self.conn.execute("SELECT * FROM deltas WHERE dimension_id = ? ORDER BY timestamp ASC", (dimension_id,))
        else:
            cursor = self.conn.execute("SELECT * FROM deltas ORDER BY timestamp ASC")
            
        deltas = []
        for row in cursor.fetchall():
            d = dict(row)
            # Ensure JSON strings are properly parsed
            d['prior'] = json.loads(d['prior'])
            d['posterior'] = json.loads(d['posterior'])
            deltas.append(Delta.from_dict(d))
        return deltas

    def save_evidence_event(self, event: Dict):
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO evidence_events
                    (id, text, source, timestamp, speaker, entities, terms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event["id"],
                event["text"],
                event.get("source", ""),
                event.get("timestamp"),
                event.get("speaker"),
                json.dumps(event.get("entities", [])),
                json.dumps(event.get("terms", [])),
            ))

    def load_all_evidence_events(self) -> List[Dict]:
        cursor = self.conn.execute("SELECT * FROM evidence_events ORDER BY timestamp ASC")
        events = []
        for row in cursor.fetchall():
            event = dict(row)
            event["entities"] = json.loads(event["entities"])
            event["terms"] = json.loads(event["terms"])
            events.append(event)
        return events

    def save_relationship_edge(self, edge: Dict):
        with self.conn:
            self.conn.execute("""
                INSERT INTO relationship_edges
                    (source_entity, relation, target_entity, confidence, evidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_entity, relation, target_entity)
                DO UPDATE SET
                    confidence = MAX(confidence, excluded.confidence),
                    evidence = excluded.evidence,
                    timestamp = excluded.timestamp
            """, (
                edge["source_entity"],
                edge["relation"],
                edge["target_entity"],
                edge.get("confidence", 1.0),
                edge.get("evidence", ""),
                edge.get("timestamp"),
            ))

    def load_all_relationship_edges(self) -> List[Dict]:
        cursor = self.conn.execute("""
            SELECT * FROM relationship_edges
            ORDER BY source_entity ASC, relation ASC, confidence DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def store_embedding(self, text_hash: str, embedding: List[float]):
        self._embedding_cache.set(text_hash, embedding)
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO embeddings (text_hash, embedding, created_at)
                VALUES (?, ?, ?)
            """, (text_hash, json.dumps(embedding), time.time()))

    def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        # Check LRU cache first
        cached = self._embedding_cache.get(text_hash)
        if cached is not None:
            return cached
            
        # Fallback to SQLite
        cursor = self.conn.execute("SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,))
        row = cursor.fetchone()
        if row:
            embedding = json.loads(row['embedding'])
            self._embedding_cache.set(text_hash, embedding)
            return embedding
        return None
