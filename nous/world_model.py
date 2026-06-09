from typing import Dict, List, Optional
from .dimension import Dimension
from .persistence import PersistenceLayer

class WorldModel:
    """
    The agent's predictive world model. 
    Manages the collection of all dimensions (current beliefs).
    """
    def __init__(self, persistence: PersistenceLayer):
        self.db = persistence
        # In-memory cache of dimensions
        self.dimensions: Dict[str, Dimension] = {dim.id: dim for dim in self.db.load_all_dimensions()}
        
    def get_dimension(self, dim_id: str) -> Dimension:
        """Retrieves a dimension. If it doesn't exist, creates a uniform one."""
        if dim_id not in self.dimensions:
            # Create a new, uniform dimension if it doesn't exist
            dim = Dimension(id=dim_id)
            self.dimensions[dim_id] = dim
            self.db.save_dimension(dim)
        return self.dimensions[dim_id]
        
    def save_dimension(self, dim_id: str):
        """Persists the dimension to storage."""
        if dim_id in self.dimensions:
            self.db.save_dimension(self.dimensions[dim_id])
            
    def query_belief(self, entity: str, attribute: str) -> Dict[str, float]:
        """Returns the current probability distribution for an entity attribute."""
        dim_id = f"{entity}.{attribute}"
        dim = self.get_dimension(dim_id)
        return dict(dim.distribution)
        
    def predict(self, entity: str, attribute: str) -> Dict[str, float]:
        """
        Alias for query_belief, reflecting the predictive nature of the model.
        In future iterations, this can include cascaded inferences.
        """
        return self.query_belief(entity, attribute)
        
    def all_dimensions(self) -> List[Dimension]:
        """Returns a list of all current dimensions in the model."""
        return list(self.dimensions.values())
        
    def get_entity_attributes(self, entity: str) -> Dict[str, dict]:
        """Retrieve all attribute distributions for a specific entity."""
        prefix = f"{entity}."
        attrs = {}
        for dim_id, dim in self.dimensions.items():
            if dim_id.startswith(prefix):
                attr_name = dim_id[len(prefix):]
                attrs[attr_name] = dict(dim.distribution)
        return attrs
