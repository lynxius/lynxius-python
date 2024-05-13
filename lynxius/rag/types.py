from dataclasses import dataclass


@dataclass
class ContextChunk:
    document: str
    relevance: float
