from dataclasses import dataclass, field
from typing import Dict, Optional
import uuid
import time
import json

@dataclass
class Delta:
    """
    A Delta represents a change in understanding.
    It records how a probabilistic dimension changed from a prior to a posterior
    distribution, the surprise of that change, and the evidence that caused it.
    """
    dimension_id: str
    prior: Dict[str, float]
    posterior: Dict[str, float]
    surprise: float  # Information content in bits (-log2 P)
    evidence: str
    source: str
    source_reliability: float
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    valid_time: Optional[float] = None
    cascaded_from: Optional[str] = None  # ID of the delta that triggered this one
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dimension_id": self.dimension_id,
            "prior": json.dumps(self.prior),
            "posterior": json.dumps(self.posterior),
            "surprise": self.surprise,
            "evidence": self.evidence,
            "source": self.source,
            "source_reliability": self.source_reliability,
            "timestamp": self.timestamp,
            "valid_time": self.valid_time,
            "cascaded_from": self.cascaded_from
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Delta':
        return cls(
            id=data["id"],
            dimension_id=data["dimension_id"],
            prior=json.loads(data["prior"]) if isinstance(data["prior"], str) else data["prior"],
            posterior=json.loads(data["posterior"]) if isinstance(data["posterior"], str) else data["posterior"],
            surprise=float(data["surprise"]),
            evidence=data["evidence"],
            source=data["source"],
            source_reliability=float(data["source_reliability"]),
            timestamp=float(data["timestamp"]),
            valid_time=data.get("valid_time"),
            cascaded_from=data.get("cascaded_from")
        )
