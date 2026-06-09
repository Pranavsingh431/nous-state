import time
from typing import List, Dict, Set
from .dimension import Dimension
from .surprise import mutual_information, kl_divergence

class Compressor:
    """
    Engine for maintaining the world model.
    Applies entropy decay (forgetting) and compresses redundant information.
    """
    def __init__(self, decay_half_life_days: float = 30.0):
        # Calculate lambda for decay: 0.5 = lambda ^ (half_life_seconds)
        half_life_seconds = decay_half_life_days * 24 * 60 * 60
        if half_life_seconds > 0:
            self.decay_rate = 0.5 ** (1.0 / half_life_seconds)
        else:
            self.decay_rate = 1.0 # No decay
            
    def apply_decay(self, dimensions: List[Dimension], current_time: float = None):
        """Applies entropy decay to all dimensions."""
        if current_time is None:
            current_time = time.time()
            
        for dim in dimensions:
            dim.apply_decay(self.decay_rate, current_time)
            
    def find_redundant_dimensions(self, dimensions: List[Dimension], 
                                  joint_distributions: Dict[tuple, Dict[tuple, float]],
                                  threshold: float = 0.9) -> List[str]:
        """
        Finds dimensions that can be compressed (removed) because their information 
        is completely derivable from other dimensions (high mutual information).
        
        joint_distributions maps (dim_id_1, dim_id_2) to a joint probability dict.
        In a full implementation, these joint dists are learned over time from co-occurrence.
        """
        redundant_ids = []
        dim_map = {d.id: d for d in dimensions}
        
        for (id_a, id_b), joint_dist in joint_distributions.items():
            if id_a not in dim_map or id_b not in dim_map:
                continue
                
            dim_a = dim_map[id_a]
            dim_b = dim_map[id_b]
            
            mi = mutual_information(joint_dist, dim_a.distribution, dim_b.distribution)
            
            entropy_a = dim_a.entropy()
            entropy_b = dim_b.entropy()
            
            # If A tells us almost everything about B
            if entropy_b > 0 and (mi / entropy_b) >= threshold:
                redundant_ids.append(id_b)
            # If B tells us almost everything about A
            elif entropy_a > 0 and (mi / entropy_a) >= threshold:
                redundant_ids.append(id_a)
                
        return list(set(redundant_ids))
