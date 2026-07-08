import time
from typing import Dict, List, Optional
from .dimension import Dimension
from .delta import Delta
from .surprise import compute_surprise
from .config import config

class BayesianUpdater:
    """
    Engine for performing Bayesian updates on dimensions when new evidence arrives.
    """
    def __init__(self):
        pass
        
    def update(self, dimension: Dimension, observed_value: str, evidence: str, 
               source: str, reliability: float = 0.9, cascaded_from: Optional[str] = None,
               timestamp: float = None) -> Delta:
        """
        Performs a Bayesian update on the dimension given an observed value.
        Uses Bayes' rule: P(value|obs) = P(obs|value) * P(value) / P(obs)
        
        Args:
            dimension: The dimension to update
            observed_value: The value observed in the evidence
            evidence: Textual description of the evidence
            source: Source of the evidence
            reliability: Confidence in the source (0.0 to 1.0)
            cascaded_from: ID of a parent delta if this update was triggered by another
            
        Returns:
            A Delta object recording the change.
        """
        prior_dist = dict(dimension.distribution)
        
        # Calculate surprise based on prior belief
        prior_prob = dimension.get_probability(observed_value, record_access=False)
        surprise_bits = compute_surprise(prior_prob)

        # --- ABLATION: naive last-write-wins overwrite (no Bayesian belief tracking) ---
        # Collapses all mass onto the newest observed value, discarding the prior
        # distribution entirely. This is the "flat storage" baseline the paper's
        # Limitations section names as the key missing ablation.
        if config.ABLATE_UPDATE == "flat":
            posterior_dist = {observed_value: 1.0}
            dimension.set_distribution(posterior_dist, timestamp=timestamp)
            return Delta(
                dimension_id=dimension.id,
                prior=prior_dist,
                posterior=posterior_dist,
                surprise=surprise_bits,
                evidence=evidence,
                source=source,
                source_reliability=reliability,
                timestamp=timestamp if timestamp is not None else time.time(),
                cascaded_from=cascaded_from,
            )

        k = len(prior_dist)
        is_new_value = observed_value not in prior_dist
        if is_new_value:
            k += 1
            
        # Likelihoods:
        # P(obs | value_is_true) = reliability
        # P(obs | value_is_false) = (1 - reliability) / (k - 1)
        
        unnormalized = {}
        for val, p in prior_dist.items():
            if val == observed_value:
                unnormalized[val] = p * reliability
            else:
                unnormalized[val] = p * ((1.0 - reliability) / max(1, k - 1))
                
        if is_new_value:
            # If value was previously unknown, give it a small uniform prior for the update
            # We assume it was part of the 'unknown' mass, or just newly discovered
            prior_for_new = 1.0 / k if k > 0 else 1.0
            unnormalized[observed_value] = prior_for_new * reliability
            
        # Normalize to get posterior
        total = sum(unnormalized.values())
        if total > 0:
            posterior_dist = {val: p / total for val, p in unnormalized.items()}
        else:
            posterior_dist = {observed_value: 1.0}
            
        # If the highest probability is very close to 1.0, we might want to trim tiny probabilities
        # to prevent the dictionary from growing indefinitely with negligible values.
        posterior_dist = self._prune_distribution(posterior_dist)
            
        # Apply the update
        dimension.set_distribution(posterior_dist, timestamp=timestamp)
        
        # Create delta
        delta = Delta(
            dimension_id=dimension.id,
            prior=prior_dist,
            posterior=posterior_dist,
            surprise=surprise_bits,
            evidence=evidence,
            source=source,
            source_reliability=reliability,
            timestamp=timestamp if timestamp is not None else time.time(),
            cascaded_from=cascaded_from
        )
        return delta
        
    def _prune_distribution(self, dist: Dict[str, float], threshold: float = 0.01) -> Dict[str, float]:
        """Removes values with very low probability and renormalizes."""
        pruned = {k: v for k, v in dist.items() if v >= threshold}
        # Always keep at least the highest value
        if not pruned and dist:
            highest = max(dist.items(), key=lambda item: item[1])
            pruned = {highest[0]: highest[1]}
            
        # If 'unknown' was pruned but we want to keep it as a fallback, we could, 
        # but for clean dimensions we let it disappear.
        
        total = sum(pruned.values())
        if total > 0:
            return {k: v / total for k, v in pruned.items()}
        return dist
