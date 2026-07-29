"""
Pydantic request/response schemas for the Research Agent API.

Wire contract between the React frontend (ported from shanalishah/researchagent)
and this FastAPI backend. Field names mirror that reference repo's models.py so
the frontend can be reused as-is; DateRange/Provider values are adapted to this
repo's actual pipeline_core.py (get_date_range, LLMConfig.provider).
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Provider = Literal[
    "openai", "gemini", "groq", "free",
    "anthropic", "openrouter", "ollama_local", "ollama_cloud",
]
DateRange = Literal["Last 3 Days", "Last Week", "Last Month", "Last 3 Months", "All Time"]


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
    # "cross_encoder" = focus/relevance came from an absolute threshold on a real
    # reranker score (scibert_classify_papers) — safe to show the number. "embedding"
    # = the CrossEncoder was unavailable and focus came from heuristic_classify_papers_free's
    # rank-percentile fallback instead (top 30% = primary) — same field, different
    # meaning, not calibrated against the same thresholds. See docs/asta-ui-comparison-design.md §4.
    relevance_basis: Literal["cross_encoder", "embedding"] = "embedding"
    evidence: List[str] = Field(default_factory=list)  # top matched abstract sentences (docs/asta-ui-comparison-design.md §5)
    why: List[str] = Field(default_factory=list)
    summary: Optional[str] = None  # plain-English summary (LLM providers only, top_n papers only)


class CriterionOut(BaseModel):
    """One relevance criterion decomposed from the brief (QIL v3 Stage 2)."""

    name: str
    definition: str
    strength: str = "should"  # "must" | "should"


class QueryUnderstanding(BaseModel):
    """What QIL (query_intelligence.py) actually produced for this query, shown to
    the user as "How your query was read". `criteria` is display-only for now
    (QIL v3 Stage 2) — papers are not yet scored per-criterion the way Asta's are
    (that's Stage 3, not built). See docs/asta-ui-comparison-design.md §3."""

    intent: str = "general"
    search_terms: List[str] = Field(default_factory=list)
    excluded_terms: List[str] = Field(default_factory=list)
    quality_modifier: str = "any"
    source: str = "rules"  # "llm_groq" | "llm_openrouter" | "rules"
    # Brief-sourced hard filters (R6/R7). These narrow the LanceDB search, so they
    # must be visible — a filter the user can't see is one they can't correct.
    date_window: Optional[str] = None  # display form, e.g. "2022–2024" / "from 2020"
    authors: List[str] = Field(default_factory=list)
    venues: List[str] = Field(default_factory=list)
    criteria: List[CriterionOut] = Field(default_factory=list)


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
    query_understanding: Optional[QueryUnderstanding] = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
