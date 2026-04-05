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
    source: str = "Semantic Scholar"
    source_id: str = ""
    doi: Optional[str] = None


_TABLE_DDL = """
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
    source               TEXT,
    source_id            TEXT,
    doi                  TEXT,
    is_indexed           INTEGER DEFAULT 0,
    ingested_at          TEXT DEFAULT (datetime('now'))
);
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_date ON papers(submitted_date);
CREATE INDEX IF NOT EXISTS idx_cites ON papers(citation_count DESC);
CREATE INDEX IF NOT EXISTS idx_source_id ON papers(source_id);
CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi);
"""


def create_db(db_path: str) -> sqlite3.Connection:
    """Create (or open) the corpus SQLite database and ensure schema exists."""
    conn = sqlite3.connect(db_path)
    # 1. Ensure table exists
    conn.execute(_TABLE_DDL)
    
    # 2. Safely inject missing columns for migrations
    cols_to_add = [
        ("source", "TEXT DEFAULT 'Semantic Scholar'"),
        ("source_id", "TEXT"),
        ("doi", "TEXT"),
        ("is_indexed", "INTEGER DEFAULT 0")
    ]
    for col_name, col_def in cols_to_add:
        try:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
            pass # Column already exists
    
    # 3. Ensure indexes exist (safe after columns are injected)
    conn.executescript(_INDEX_DDL)
        
    conn.commit()
    return conn


def upsert_paper(conn: sqlite3.Connection, p: PaperRecord) -> None:
    # Insert or update a paper record.
    # Uses arxiv_id as the surrogate primary key (ADR-012).
    conn.execute(
        """
        INSERT INTO papers
            (arxiv_id, s2_id, title, abstract, authors, submitted_date,
             venue, citation_count, max_author_citations, pdf_url, arxiv_url,
             fields_of_study, source, source_id, doi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            citation_count       = MAX(papers.citation_count, excluded.citation_count),
            max_author_citations = MAX(papers.max_author_citations, excluded.max_author_citations),
            venue                = CASE 
                WHEN excluded.venue IS NOT NULL AND excluded.venue != 'arXiv.org' AND excluded.venue != 'ArXiv' THEN excluded.venue 
                ELSE COALESCE(papers.venue, excluded.venue) 
            END,
            is_indexed           = CASE
                WHEN (excluded.title IS NOT NULL AND excluded.title != papers.title) OR
                     (excluded.abstract IS NOT NULL AND excluded.abstract != papers.abstract) THEN 0
                ELSE papers.is_indexed
            END,
            title                = COALESCE(excluded.title, papers.title),
            s2_id                = COALESCE(excluded.s2_id, papers.s2_id),
            doi                  = COALESCE(excluded.doi, papers.doi),
            source               = CASE 
                WHEN papers.source = 'Semantic Scholar' OR excluded.source = 'Semantic Scholar' THEN 'Semantic Scholar'
                ELSE excluded.source
            END,
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
            p.source,
            p.source_id,
            p.doi,
        ),
    )
