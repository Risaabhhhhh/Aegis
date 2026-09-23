from engines.base import BaseEngine
from engines.groundedness import GroundednessEngine

# Register concrete engine instances here later (e.g. GroundednessEngine())
REGISTERED_ENGINES: list[BaseEngine] = [
    GroundednessEngine()
]
