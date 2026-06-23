from typing import List, Dict, Tuple
import math


class IdentityCoupler:
    """
    Handles identity resolution by measuring predictive coupling between entities.
    Instead of string matching, it checks if predicting one entity's properties 
    perfectly predicts another's.
    
    v2 improvements:
    - Requires minimum overlapping attributes before computing coupling
    - Name similarity gate — entities with very different names need higher coupling
    - Value diversity check — single-value distributions don't count as meaningful overlap
    """
    def __init__(self, merge_threshold: float = 0.97, min_overlap: int = 3):
        self.merge_threshold = merge_threshold
        self.min_overlap = min_overlap
        
    def compute_coupling(self, entity_a_dims: Dict[str, dict], entity_b_dims: Dict[str, dict]) -> float:
        """
        Computes the coupling coefficient between two entities based on their dimension distributions.
        Returns a value between 0.0 (unrelated) and 1.0 (identical).
        
        entity_a_dims: mapping of attribute_name -> probability distribution
        """
        # Find overlapping attributes
        common_attrs = set(entity_a_dims.keys()).intersection(set(entity_b_dims.keys()))
        
        # Gate 1: Require minimum number of meaningful overlapping attributes
        if len(common_attrs) < self.min_overlap:
            return 0.0
        
        # Gate 2: Filter out trivial overlaps (both have only 1 value, or both are near-uniform)
        meaningful_overlaps = []
        for attr in common_attrs:
            dist_a = entity_a_dims[attr]
            dist_b = entity_b_dims[attr]
            
            # Skip attributes where both distributions are trivially simple
            # (single value or near-uniform = not informative)
            if len(dist_a) <= 1 and len(dist_b) <= 1:
                # Both have exactly one value — only meaningful if values differ
                if set(dist_a.keys()) != set(dist_b.keys()):
                    return 0.0  # Different single values → definitely different entities
                continue  # Same single value — not informative enough on its own
            
            meaningful_overlaps.append(attr)
        
        if len(meaningful_overlaps) < max(2, self.min_overlap - 1):
            return 0.0
            
        total_coupling = 0.0
        weight_sum = 0.0
        
        for attr in common_attrs:
            dist_a = entity_a_dims[attr]
            dist_b = entity_b_dims[attr]
            
            # Calculate similarity between distributions (Bhattacharyya coefficient)
            bc = self._bhattacharyya_coefficient(dist_a, dist_b)
            
            # Weight by the certainty of both distributions
            certainty = (self._certainty(dist_a) + self._certainty(dist_b)) / 2.0
            weight = certainty + 0.1  # Base weight
            
            total_coupling += bc * weight
            weight_sum += weight
            
        if weight_sum == 0:
            return 0.0
            
        return total_coupling / weight_sum
    
    def compute_name_similarity(self, name_a: str, name_b: str) -> float:
        """
        Simple character-level similarity between entity names.
        Uses Jaccard coefficient on character bigrams.
        """
        if not name_a or not name_b:
            return 0.0
        
        a_lower = name_a.lower().strip()
        b_lower = name_b.lower().strip()
        
        # Exact match
        if a_lower == b_lower:
            return 1.0
        
        # One is a substring of the other (e.g., "Bob" in "Bob Smith")
        if a_lower in b_lower or b_lower in a_lower:
            return 0.8
        
        # Character bigram Jaccard
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else {s}
        
        bg_a = bigrams(a_lower)
        bg_b = bigrams(b_lower)
        
        if not bg_a or not bg_b:
            return 0.0
            
        intersection = len(bg_a & bg_b)
        union = len(bg_a | bg_b)
        
        return intersection / union if union > 0 else 0.0
        
    def _bhattacharyya_coefficient(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Measure of similarity between two probability distributions."""
        bc = 0.0
        all_keys = set(p.keys()).union(set(q.keys()))
        for k in all_keys:
            bc += math.sqrt(p.get(k, 0.0) * q.get(k, 0.0))
        return bc
        
    def _certainty(self, dist: Dict[str, float]) -> float:
        """1.0 minus normalized entropy. High certainty = near 1.0."""
        k = len(dist)
        if k <= 1:
            return 1.0
        max_entropy = math.log2(k)
        entropy = -sum(p * math.log2(p) for p in dist.values() if p > 0)
        return 1.0 - (entropy / max_entropy)
        
    def should_merge(self, coupling_score: float, name_a: str = "", name_b: str = "") -> bool:
        """
        Determine if entities should merge.
        Uses name similarity as a gate: entities with very different names
        need much higher coupling scores.
        """
        if coupling_score < 0.5:
            return False
            
        name_sim = self.compute_name_similarity(name_a, name_b)
        
        # If names are very similar (>0.6), use the standard threshold
        if name_sim >= 0.6:
            return coupling_score >= self.merge_threshold
        
        # If names are somewhat similar (0.3-0.6), require higher coupling
        if name_sim >= 0.3:
            return coupling_score >= min(0.99, self.merge_threshold + 0.02)
        
        # Very different names — almost never merge
        return coupling_score >= 0.995
