from typing import Dict, List, Optional
import time
import re

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


class Nous:
    """
    The main public API for the Nous Knowledge Evolution Engine.
    "Knowledge is prediction, not storage."
    
    v2: Supports user context, auto-coupling checks, and value normalization.
    """
    def __init__(self, db_path: str = "memory.db", extractor=None, 
                 user_context: Optional[Dict[str, str]] = None):
        self.persistence = PersistenceLayer(db_path)
        self.world_model = WorldModel(self.persistence)
        self.delta_log = DeltaLog(self.persistence)
        self.updater = BayesianUpdater()
        self.compressor = Compressor()
        self.coupler = IdentityCoupler()
        self.extractor = extractor if extractor is not None else Extractor()
        self.user_context = user_context or {}
        
        # Track entities seen for auto-coupling
        self._entity_registry: Dict[str, set] = {}  # entity -> set of attributes
        self._load_entity_registry()
        
    def _load_entity_registry(self):
        """Rebuild entity registry from existing dimensions."""
        for dim in self.world_model.all_dimensions():
            parts = dim.id.rsplit(".", 1)
            if len(parts) == 2:
                entity, attr = parts
                if entity not in self._entity_registry:
                    self._entity_registry[entity] = set()
                self._entity_registry[entity].add(attr)
        
    def observe(self, text: str, source: str = "user", reliability: float = 0.9, 
                timestamp: float = None) -> List[Delta]:
        """
        Ingest new information and update beliefs.
        Returns the list of deltas generated.
        """
        claims = self.extractor.extract(text)
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
            
            # Track entity in registry for auto-coupling
            if entity not in self._entity_registry:
                self._entity_registry[entity] = set()
            self._entity_registry[entity].add(attribute)
            
        # After ingesting, check for potential identity merges
        if claims:
            self._auto_coupling_check(claims)
            
        return deltas
            
    def predict(self, entity: str, attribute: str) -> Dict[str, float]:
        """
        What does the model expect for this entity's attribute?
        Returns a probability distribution.
        """
        return self.world_model.predict(entity, attribute)
        
    def query(self, entity: str, attribute: str) -> Dict[str, float]:
        """Retrieve current beliefs (alias for predict)."""
        return self.world_model.query_belief(entity, attribute)
        
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
        Only checks entities with overlapping attributes (efficient).
        """
        new_entities = set(c[0] for c in new_claims)
        
        for new_entity in new_entities:
            new_attrs = self._entity_registry.get(new_entity, set())
            if not new_attrs:
                continue
                
            for existing_entity, existing_attrs in self._entity_registry.items():
                if existing_entity == new_entity:
                    continue
                # Only check entities with at least 1 overlapping attribute
                if new_attrs.intersection(existing_attrs):
                    coupling = self.get_coupling(new_entity, existing_entity)
                    if self.coupler.should_merge(coupling):
                        # Log the discovery (in production, you might auto-merge or prompt)
                        print(f"  ⚡ Auto-coupling detected: '{new_entity}' ≈ '{existing_entity}' "
                              f"(score: {coupling:.2f})")
            
    def close(self):
        """Closes the underlying database connection."""
        self.persistence.conn.close()
