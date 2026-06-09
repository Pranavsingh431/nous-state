import math
from typing import Dict

def compute_surprise(probability: float) -> float:
    """
    Computes information-theoretic surprise (surprisal) in bits.
    S = -log2(P)
    
    Args:
        probability: The prior probability of the observed event (0.0 to 1.0)
        
    Returns:
        Surprise in bits. 0 means completely expected, higher values mean more surprising.
    """
    if probability <= 0:
        # Cap at a large finite value for impossible events to avoid infinity in practical systems
        return 20.0 
    if probability >= 1:
        return 0.0
    return -math.log2(probability)

def kl_divergence(posterior: Dict[str, float], prior: Dict[str, float]) -> float:
    """
    Computes Kullback-Leibler divergence D_KL(posterior || prior) in bits.
    Measures the information gained when revising beliefs from prior to posterior.
    
    Args:
        posterior: The new probability distribution P
        prior: The old probability distribution Q
        
    Returns:
        Divergence in bits.
    """
    divergence = 0.0
    
    # Ensure prior has all keys from posterior for computation, using a small epsilon
    # to avoid division by zero.
    epsilon = 1e-10
    
    for val, p_val in posterior.items():
        if p_val > 0:
            q_val = prior.get(val, 0.0)
            if q_val <= 0:
                q_val = epsilon
            divergence += p_val * math.log2(p_val / q_val)
            
    return divergence

def mutual_information(joint_dist: Dict[tuple, float], 
                       marginal_a: Dict[str, float], 
                       marginal_b: Dict[str, float]) -> float:
    """
    Computes mutual information I(A; B) in bits.
    
    Args:
        joint_dist: Dict mapping (val_a, val_b) -> joint probability
        marginal_a: Marginal distribution of variable A
        marginal_b: Marginal distribution of variable B
    """
    mi = 0.0
    for (a_val, b_val), p_ab in joint_dist.items():
        if p_ab > 0:
            p_a = marginal_a.get(a_val, 1e-10)
            p_b = marginal_b.get(b_val, 1e-10)
            if p_a > 0 and p_b > 0:
                mi += p_ab * math.log2(p_ab / (p_a * p_b))
    return mi
