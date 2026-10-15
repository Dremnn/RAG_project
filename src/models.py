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
# DOMAIN MODELS & TOOL SCHEMAS FOR CTXH (T5)
# ==========================================

@dataclass
class CTXHRegistration:
    """
    Data model representing a CTXH activity registration.
    """
    reg_id: str
    activity_code: str
    activity_name: str
    points: int
    converted_days: float
    status: Literal["CONFIRMED", "PENDING", "REJECTED"] = "CONFIRMED"
    registration_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StudentCTXHRecord:
    """
    Data model representing student CTXH profile and accumulated days.
    """
    mssv: str
    student_name: str
    faculty: str = ""
    course_year: str = "2022"
    required_days: float = 4.0
    accumulated_days: float = 0.0
    registered_activities: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchCTXHRulesInput:
    """Input schema for searching CTXH regulations and activity codes."""
    query: str
    top_k: int = 4
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"


@dataclass
class RegisterCTXHInput:
    """Input schema for registering a student to a CTXH activity (Write Tool)."""
    mssv: str
    activity_code: str
    points: Optional[int] = None
    converted_days: Optional[float] = None
    note: str = ""


@dataclass
class GetCTXHStatusInput:
    """Input schema for verifying student CTXH profile and graduation progress (Verify Tool)."""
    mssv: str


@dataclass
class CheckStudentExistsInput:
    """Input schema for checking if a student exists in the university database."""
    mssv: str


@dataclass
class CheckGraduationEligibilityInput:
    """Input schema for checking whether a student meets the minimum CTXH days for graduation."""
    mssv: str


@dataclass
class StudentProfile:
    """
    Data model representing master student information in students.json.
    """
    mssv: str
    student_name: str
    faculty: str
    course_year: str
    program_type: str = "Đại học chính quy (>= 4 năm)"
    training_years: float = 4.0
    required_ctxh_days: float = 4.0
    email: str = ""
    status: str = "Đang học"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
