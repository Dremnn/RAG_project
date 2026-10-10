from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Literal
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


# ==========================================
# DOMAIN MODELS & TOOL SCHEMAS (Week 1 & Week 6)
# ==========================================

@dataclass
class StudentRequest:
    """
    Data model representing a student procedure request in the database.
    """
    request_id: str
    mssv: str
    student_name: str
    request_type: str  # HOAN_THI, BAO_LUU, RUT_MON, CAP_BANG_DIEM, CAP_GIAY_XAC_NHAN
    course_name: str = ""
    exam_date: str = ""
    submission_date: str = ""
    reason: str = ""
    attachment: str = ""
    status: Literal["PENDING", "APPROVED", "REJECTED"] = "PENDING"
    note: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchRegulationsInput:
    """
    Input schema for searching academic regulations.
    """
    query: str
    top_k: int = 4
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"


@dataclass
class SubmitRequestInput:
    """
    Input schema for submitting an academic procedure request (Write Tool / Side Effect).
    """
    mssv: str
    student_name: str
    request_type: str
    course_name: str = ""
    exam_date: str = ""
    submission_date: str = ""
    reason: str = ""
    attachment: str = ""


@dataclass
class GetRequestStatusInput:
    """
    Input schema for verifying an existing request status (Verify Tool).
    """
    request_id: str
