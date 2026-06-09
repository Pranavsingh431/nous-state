import sqlite3
import json
from typing import List, Dict, Optional
from .dimension import Dimension
from .delta import Delta

class PersistenceLayer:
    """
    SQLite backend for storing dimensions, deltas, and couplings.
    Zero external dependencies.
    """
    def __init__(self, db_path: str = "memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
    def _init_db(self):
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
