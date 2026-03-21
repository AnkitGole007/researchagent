"""
data_pipeline/schema.py
Shared schema: PaperRecord dataclass + SQLite DDL helpers.
"""
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PaperRecord:
    arxiv_id: str
    s2_id: str
    title: str
    abstract: str
    authors: List[str]
    submitted_date: str
    venue: Optional[str]
    citation_count: int
    max_author_citations: int
    pdf_url: str
    arxiv_url: str
    fields_of_study: List[str]


_DDL = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id             TEXT PRIMARY KEY,
    s2_id                TEXT,
    title                TEXT NOT NULL,
    abstract             TEXT,
    authors              TEXT NOT NULL,
    submitted_date       TEXT NOT NULL,
    venue                TEXT,
    citation_count       INTEGER DEFAULT 0,
    max_author_citations INTEGER DEFAULT 0,
    pdf_url              TEXT,
    arxiv_url            TEXT,
    fields_of_study      TEXT,
    ingested_at          TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_date ON papers(submitted_date);
CREATE INDEX IF NOT EXISTS idx_cites ON papers(citation_count DESC);
"""


def create_db(db_path: str) -> sqlite3.Connection:
    """Create (or open) the corpus SQLite database and ensure schema exists."""
    conn = sqlite3.connect(db_path)
    conn.executescript(_DDL)
    conn.commit()
    return conn


def upsert_paper(conn: sqlite3.Connection, p: PaperRecord) -> None:
    """Insert or update a paper, refreshing mutable fields on conflict."""
    conn.execute(
        """
        INSERT INTO papers
            (arxiv_id, s2_id, title, abstract, authors, submitted_date,
             venue, citation_count, max_author_citations, pdf_url, arxiv_url,
             fields_of_study)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            citation_count       = excluded.citation_count,
            max_author_citations = excluded.max_author_citations,
            venue                = excluded.venue,
            ingested_at          = datetime('now')
        """,
        (
            p.arxiv_id,
            p.s2_id,
            p.title,
            p.abstract,
            json.dumps(p.authors),
            p.submitted_date,
            p.venue,
            p.citation_count,
            p.max_author_citations,
            p.pdf_url,
            p.arxiv_url,
            json.dumps(p.fields_of_study),
        ),
    )
