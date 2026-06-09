from typing import List, Dict, Tuple
import math

class IdentityCoupler:
    """
    Handles identity resolution by measuring predictive coupling between entities.
    Instead of string matching, it checks if predicting one entity's properties 
    perfectly predicts another's.
    """
    def __init__(self, merge_threshold: float = 0.85):
        self.merge_threshold = merge_threshold
        
    def compute_coupling(self, entity_a_dims: Dict[str, dict], entity_b_dims: Dict[str, dict]) -> float:
        """
        Computes the coupling coefficient between two entities based on their dimension distributions.
        Returns a value between 0.0 (unrelated) and 1.0 (identical).
        
        entity_a_dims: mapping of attribute_name -> probability distribution
        """
        # Find overlapping attributes
        common_attrs = set(entity_a_dims.keys()).intersection(set(entity_b_dims.keys()))
        
        if not common_attrs:
            return 0.0
            
        total_coupling = 0.0
        weight_sum = 0.0
        
        for attr in common_attrs:
            dist_a = entity_a_dims[attr]
            dist_b = entity_b_dims[attr]
            
            # Calculate similarity between distributions (e.g., using Bhattacharyya coefficient)
            bc = self._bhattacharyya_coefficient(dist_a, dist_b)
            
            # Weight by the certainty (inverse of entropy) of both distributions
            # We care more if they agree on very specific, certain facts than if they
            # agree on "unknown".
            certainty = (self._certainty(dist_a) + self._certainty(dist_b)) / 2.0
            weight = certainty + 0.1 # Base weight
            
            total_coupling += bc * weight
            weight_sum += weight
            
        if weight_sum == 0:
            return 0.0
            
        return total_coupling / weight_sum
        
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
        
    def should_merge(self, coupling_score: float) -> bool:
        """Determine if score exceeds merge threshold."""
        return coupling_score >= self.merge_threshold
