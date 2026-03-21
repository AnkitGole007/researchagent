"""
data_pipeline/fetch_corpus.py
S2 Bulk Search ingestion pipeline.

Usage:
    python data_pipeline/fetch_corpus.py          # incremental (default)
    python data_pipeline/fetch_corpus.py --full   # full refresh
"""
import logging
import os
import sys
from typing import List, Optional

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from data_pipeline.schema import PaperRecord, create_db, upsert_paper
except ImportError:
    # Script mode: `python data_pipeline/fetch_corpus.py` — project root not on sys.path.
    # Insert it and retry. This is a no-op when imported as a module.
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from data_pipeline.schema import PaperRecord, create_db, upsert_paper

# Load .env so S2_API_KEY is available even when run as a standalone script
load_dotenv()

logger = logging.getLogger(__name__)

# Semantic Scholar fields to request via bulk search.
# NOTE: authors.citationCount is NOT supported by the bulk endpoint (causes 400).
# Author-level citation counts can be fetched via the per-paper detail endpoint
# in a follow-up enrichment pass (P2 backlog, G-01).
_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "authors",          # returns authorId + name by default
    "year",
    "publicationDate",
    "venue",
    "citationCount",
    "externalIds",
    "openAccessPdf",
    "s2FieldsOfStudy",
]


def parse_s2_paper(raw: dict) -> Optional[PaperRecord]:
    """
    Convert a raw S2 API dict to a PaperRecord.
    Returns None if the paper has no arXiv ID (non-arXiv papers skipped).

    Note: max_author_citations is set to 0 because authors.citationCount is
    not available from the bulk search endpoint. Can be enriched later.
    """
    try:
        ids = raw.get("externalIds") or {}
        arxiv_id = ids.get("ArXiv", "")
        if not arxiv_id:
            return None

        authors = raw.get("authors") or []
        pdf_url = (raw.get("openAccessPdf") or {}).get(
            "url", f"https://arxiv.org/pdf/{arxiv_id}"
        )
        pub_date = (
            raw.get("publicationDate")
            or str(raw.get("year", "2024")) + "-01-01"
        )

        return PaperRecord(
            arxiv_id=arxiv_id,
            s2_id=raw.get("paperId", ""),
            title=(raw.get("title") or "").replace("\n", " ").strip(),
            abstract=(raw.get("abstract") or "").replace("\n", " ").strip(),
            authors=[a.get("name", "") for a in authors],
            submitted_date=pub_date,
            venue=raw.get("venue") or None,
            citation_count=raw.get("citationCount", 0) or 0,
            max_author_citations=0,  # not available from bulk endpoint
            pdf_url=pdf_url,
            arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
            fields_of_study=[
                f.get("category", "")
                for f in (raw.get("s2FieldsOfStudy") or [])
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("parse error: %s", exc)
        return None


_S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=2, max=30),
    reraise=True,
)
def _fetch_page(url: str, params: dict, headers: dict) -> dict:
    """Single paginated GET with retry."""
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _bulk_search_rest(
    query: str,
    fields: List[str],
    limit: int,
    api_key: Optional[str],
) -> List[dict]:
    """
    Cursor-paginated bulk search via direct REST calls.

    The semanticscholar Python client (v0.11.0) hangs on Windows/Anaconda
    due to an async event loop conflict (nest_asyncio + httpx).  Direct
    requests calls are used instead and are confirmed working.
    """
    headers = {"x-api-key": api_key} if api_key else {}
    params: dict = {
        "query": query,
        "fields": ",".join(fields),
        "limit": 1000,  # max per page for bulk
    }

    all_papers: List[dict] = []
    while True:
        data = _fetch_page(_S2_BULK_URL, params, headers)
        batch = data.get("data") or []
        all_papers.extend(batch)
        logger.debug("Page: %d papers (total so far: %d)", len(batch), len(all_papers))

        token = data.get("token")
        if not token or len(all_papers) >= limit:
            break
        params["token"] = token

    return all_papers[:limit]


def fetch_papers_bulk(
    api_key: Optional[str] = None,
    max_papers: int = 20_000,
    **_,
) -> List[PaperRecord]:
    """
    Pull papers from S2 via Bulk Search REST API.
    Returns deduplicated list of PaperRecord (arXiv papers only).
    """
    resolved_key = api_key or os.getenv("S2_API_KEY")
    raw_list = _bulk_search_rest(
        query="machine learning artificial intelligence",
        fields=_FIELDS,
        limit=max_papers,
        api_key=resolved_key,
    )

    seen: set = set()
    results: List[PaperRecord] = []
    for raw in raw_list:
        paper = parse_s2_paper(raw)
        if paper and paper.arxiv_id not in seen:
            seen.add(paper.arxiv_id)
            results.append(paper)

    logger.info("Fetched %d arXiv papers", len(results))
    return results


def run_ingestion(
    db_path: str = "data_pipeline/corpus.db",
    max_papers: int = 20_000,
    incremental: bool = True,
) -> int:
    """
    End-to-end ingestion: fetch papers from S2, upsert into SQLite.
    Returns the number of papers upserted.
    """
    conn = create_db(db_path)
    papers = fetch_papers_bulk(max_papers=max_papers)
    for i, paper in enumerate(papers):
        upsert_paper(conn, paper)
        if i % 1_000 == 0:
            conn.commit()
    conn.commit()
    logger.info(
        "Ingestion complete: %d papers upserted to %s", len(papers), db_path
    )
    return len(papers)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_ingestion(incremental="--full" not in sys.argv)
