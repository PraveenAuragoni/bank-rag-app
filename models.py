"""
Pydantic request/response models
"""

from pydantic import BaseModel, field_validator
from typing import Optional, List


# ── Request Models ────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    doc_filter: Optional[str] = None  # filter to specific doc type

    @field_validator("question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        if len(v) > 2000:
            raise ValueError("Question too long. Max 2000 characters.")
        # Basic prompt injection guard
        banned = [
            "ignore previous instructions",
            "ignore all instructions",
            "jailbreak",
            "you are now",
            "forget everything"
        ]
        lower = v.lower()
        for phrase in banned:
            if phrase in lower:
                raise ValueError("Invalid input detected.")
        return v.strip()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


# ── Response Models ───────────────────────────────────────

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    routed_to: str
    confidence: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    message_count: int


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int


class DocumentListResponse(BaseModel):
    documents: List[str]
    total: int


class HealthResponse(BaseModel):
    status: str
    docs_indexed: int
    version: str
