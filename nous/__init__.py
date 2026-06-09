"""
nous-state — A probabilistic agent state layer.
Knowledge is prediction, not storage.

Usage:
    from nous import Nous
    from nous.llm_extractor import LLMExtractor

    memory = Nous("agent.db")
    memory.observe("Pranav works at Google DeepMind.")
    print(memory.predict("Pranav", "employer"))
"""

__version__ = "0.1.0"
__author__ = "Pranav Singh"
__license__ = "MIT"

from .engine import Nous
from .dimension import Dimension
from .delta import Delta

__all__ = [
    "Nous",
    "Dimension",
    "Delta",
    "__version__",
]
