"""
Pydantic request/response schemas for the Research Agent API.

Wire contract between the React frontend (ported from shanalishah/researchagent)
and this FastAPI backend. Field names mirror that reference repo's models.py so
the frontend can be reused as-is; DateRange/Provider values are adapted to this
repo's actual pipeline_core.py (get_date_range, LLMConfig.provider).
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Provider = Literal["openai", "gemini", "groq", "free"]
DateRange = Literal["Last 3 Days", "Last Week", "Last Month", "All Time"]


class SearchRequest(BaseModel):
    """A single search submitted from the home screen."""

    query: str = Field(..., description="The user's research brief / what they're looking for")
    exclude: str = Field("", description="Topics to exclude (the 'NOT looking for' box)")
    date_range: DateRange = Field("Last Month", description="Recency window")
    provider: Provider = Field("free", description="Reasoning provider")
    top_n: int = Field(5, ge=1, le=10, description="How many top papers to highlight with a plain-English summary")
    api_key: Optional[str] = Field(None, description="Provider API key (in-memory only, never persisted)")
    model: Optional[str] = Field(None, description="Chat model id for the chosen provider")


class PaperOut(BaseModel):
    """One ranked paper as rendered by a result card."""

    rank: int
    arxiv_id: str
    title: str
    authors: List[str]
    venue: Optional[str] = None
    abstract: str
    arxiv_url: str
    pdf_url: Optional[str] = None
    score: float  # -1 means "too new to rate" (mirrors app.py's sentinel)
    too_new: bool
    focus: Literal["primary", "secondary", "off-topic"]
    relevance: float  # 0..1
    why: List[str] = Field(default_factory=list)
    summary: Optional[str] = None  # plain-English summary (LLM providers only, top_n papers only)


class Stage(BaseModel):
    """A pipeline stage, used both for the loader and the 'how it works' panel."""

    n: str
    name: str
    detail: str = ""
    seconds: Optional[float] = None


# ---- Server-Sent Event payloads (serialized to JSON in the `data:` line) ----

class StageEvent(BaseModel):
    type: Literal["stage"] = "stage"
    index: int  # 0-based stage index
    status: Literal["start", "done"]
    name: str
    detail: str = ""


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    papers: List[PaperOut]
    primary_count: int
    secondary_count: int
    total_seconds: float
    provider: Provider


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
