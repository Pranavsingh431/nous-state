from typing import List, Optional
from .delta import Delta
from .persistence import PersistenceLayer

class DeltaLog:
    """
    Immutable log of all changes in understanding.
    The primary source of truth for the evolution of the agent's memory.
    """
    def __init__(self, persistence: PersistenceLayer):
        self.db = persistence
        
    def append(self, delta: Delta):
        """Append a new delta to the log and persist it."""
        self.db.save_delta(delta)
        
    def get_history(self, dimension_id: str) -> List[Delta]:
        """Retrieve the evolution history of a specific dimension."""
        return self.db.get_deltas(dimension_id)
        
    def get_all(self) -> List[Delta]:
        """Retrieve the entire delta log."""
        return self.db.get_deltas()
        
    def query_at(self, dimension_id: str, timestamp: float) -> Optional[dict]:
        """
        Time-travel query. Reconstructs the belief distribution of a dimension 
        exactly as it was at the given timestamp.
        """
        history = self.get_history(dimension_id)
        
        # If no history, or timestamp is before the first delta
        if not history or timestamp < history[0].timestamp:
            return {"unknown": 1.0}
            
        # Find the most recent delta before or exactly at the timestamp
        latest_valid = None
        for delta in history:
            if delta.timestamp <= timestamp:
                latest_valid = delta
            else:
                break
                
        if latest_valid:
            return latest_valid.posterior
        return {"unknown": 1.0}
