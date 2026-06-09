import math
import time
from typing import Dict, List, Optional

class Dimension:
    """
    A Dimension represents the agent's belief about one aspect of the world.
    It is modeled as a categorical probability distribution over possible values.
    
    novelty_prior: Small probability mass reserved for unseen values.
    This prevents surprise from always being maximal (20 bits) for new observations
    and provides a more calibrated information-theoretic signal.
    """
    NOVELTY_PRIOR = 0.05  # 5% mass for "anything else we haven't seen"
    
    def __init__(self, id: str, distribution: Dict[str, float] = None, last_accessed: float = None):
        self.id = id
        # Ensure we have at least 'unknown' to avoid empty distributions
        self.distribution = dict(distribution) if distribution else {"unknown": 1.0}
        self.last_accessed = last_accessed if last_accessed is not None else time.time()
        self._normalize()

    def _normalize(self):
        total = sum(self.distribution.values())
        if total <= 0:
            k = len(self.distribution)
            if k == 0:
                self.distribution = {"unknown": 1.0}
            else:
                self.distribution = {k: 1.0/len(self.distribution) for k in self.distribution}
        else:
            self.distribution = {k: v/total for k, v in self.distribution.items()}

    def get_probability(self, value: str, record_access: bool = True, current_time: float = None) -> float:
        """
        Get the probability of a value. Returns NOVELTY_PRIOR for unseen values
        instead of 0.0, providing calibrated surprise scoring.
        """
        if record_access:
            self.last_accessed = current_time if current_time is not None else time.time()
        return self.distribution.get(value, self.NOVELTY_PRIOR)

    def set_distribution(self, new_distribution: Dict[str, float], timestamp: float = None):
        """Directly set the distribution (e.g., after an update)."""
        self.distribution = dict(new_distribution)
        self._normalize()
        self.last_accessed = timestamp if timestamp is not None else time.time()

    def entropy(self) -> float:
        """Calculate Shannon entropy in bits."""
        return -sum(p * math.log2(p) for p in self.distribution.values() if p > 0)

    def apply_decay(self, decay_rate: float, current_time: float = None):
        """
        Decay toward maximum entropy (uniform distribution).
        decay_rate is lambda (0 < lambda < 1) per second.
        A lambda of 0.99999 means very slow decay.
        """
        if current_time is None:
            current_time = time.time()
        
        dt = max(0, current_time - self.last_accessed)
        if dt == 0:
            return
            
        k = len(self.distribution)
        if k == 0:
            return
            
        factor = math.pow(decay_rate, dt)
        uniform_p = 1.0 / k
        
        new_dist = {}
        for val, p in self.distribution.items():
            new_dist[val] = factor * p + (1 - factor) * uniform_p
            
        self.distribution = new_dist
        self._normalize()
        self.last_accessed = current_time

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "distribution": self.distribution,
            "last_accessed": self.last_accessed
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Dimension':
        return cls(
            id=data["id"],
            distribution=data["distribution"],
            last_accessed=data["last_accessed"]
        )
