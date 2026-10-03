from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
import json


@dataclass
class Document:
    """
    Standard Document model (Week 1 & Week 3).
    Represents raw or loaded document before chunking.
    """
    content: str
    doc_id: str = ""
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary (W1 concept)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize document to JSON string (W1 concept)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Reconstruct Document from dictionary."""
        return cls(**data)


@dataclass
class DocumentChunk:
    """
    Document Chunk model (Week 1 & Week 3).
    Represents a chunk after applying chunking strategy.
    """
    content: str
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        return cls(**data)
