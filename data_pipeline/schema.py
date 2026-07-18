"""
data_pipeline/schema.py
Shared schema: PaperRecord dataclass + LanceDB schema and connection helpers.
SQLite removed — LanceDB on R2 is the sole store.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import pyarrow as pa

VECTOR_DIM = 768
TABLE_NAME = "papers"


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


# Canonical LanceDB schema for the papers table.
# vector column is nullable — papers without embeddings have vector=None, has_embedding=False.
LANCEDB_SCHEMA = pa.schema([
    pa.field("arxiv_id",             pa.string()),
    pa.field("s2_id",                pa.string()),
    pa.field("title",                pa.string()),
    pa.field("abstract",             pa.string()),
    pa.field("authors",              pa.string()),        # JSON-encoded list
    pa.field("year",                 pa.int32()),
    pa.field("submitted_date",       pa.string()),
    pa.field("venue",                pa.string()),
    pa.field("citation_count",       pa.int32()),
    pa.field("max_author_citations", pa.int32()),
    pa.field("fields_of_study",      pa.string()),        # JSON-encoded list
    pa.field("pdf_url",              pa.string()),
    pa.field("arxiv_url",            pa.string()),
    pa.field("source",               pa.string()),
    pa.field("doi",                  pa.string()),
    pa.field("has_embedding",        pa.bool_()),
    pa.field("vector",               pa.list_(pa.float32(), VECTOR_DIM)),
])


def _extract_year(submitted_date: str) -> int:
    try:
        return int((submitted_date or "")[:4])
    except ValueError:
        return 0


def connect_lancedb(local_path: Optional[str] = None):
    """
    Connect to LanceDB. Uses local_path for testing; otherwise R2 via env vars.
    Returns a LanceDBConnection.
    """
    import lancedb

    if local_path:
        return lancedb.connect(local_path)

    endpoint  = os.getenv("R2_ENDPOINT")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket    = os.getenv("R2_BUCKET")

    missing = [k for k, v in {
        "R2_ENDPOINT": endpoint,
        "R2_ACCESS_KEY_ID": access_key,
        "R2_SECRET_ACCESS_KEY": secret_key,
        "R2_BUCKET": bucket,
    }.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing R2 env vars: {missing}. Set them or pass local_path= for local mode."
        )

    uri = f"s3://{bucket}/lancedb"
    return lancedb.connect(
        uri,
        storage_options={
            "endpoint_url": endpoint,
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
        },
    )


def get_or_create_papers_table(db):
    """
    Open the 'papers' table if it exists, otherwise create it with LANCEDB_SCHEMA.
    Returns a LanceTable.
    """
    existing = db.list_tables().tables
    if TABLE_NAME in existing:
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=LANCEDB_SCHEMA, mode="create")


def query_table(table, filter: Optional[str] = None, columns: Optional[List[str]] = None):
    """Read rows from a LanceDB table's underlying lance dataset. Returns a pandas DataFrame."""
    ds = table.to_lance()
    return ds.to_table(filter=filter, columns=columns).to_pandas()


def paper_to_row(p: PaperRecord, vector=None, has_embedding: bool = False) -> dict:
    """Convert PaperRecord to a LanceDB row dict."""
    return {
        "arxiv_id":             p.arxiv_id,
        "s2_id":                p.s2_id or "",
        "title":                p.title or "",
        "abstract":             p.abstract or "",
        "authors":              json.dumps(p.authors) if isinstance(p.authors, list) else (p.authors or "[]"),
        "year":                 _extract_year(p.submitted_date),
        "submitted_date":       p.submitted_date or "",
        "venue":                p.venue or "",
        "citation_count":       int(p.citation_count or 0),
        "max_author_citations": int(p.max_author_citations or 0),
        "fields_of_study":      json.dumps(p.fields_of_study) if isinstance(p.fields_of_study, list) else (p.fields_of_study or "[]"),
        "pdf_url":              p.pdf_url or "",
        "arxiv_url":            p.arxiv_url or "",
        "source":               p.source or "Semantic Scholar",
        "doi":                  p.doi or "",
        "has_embedding":        has_embedding,
        "vector":               (vector.tolist() if hasattr(vector, "tolist") else vector) if vector is not None else [0.0] * VECTOR_DIM,
    }


def _load_existing_ids(table) -> set:
    """Return set of all arxiv_ids currently in the table. Used by upsert to detect new vs existing."""
    return set(query_table(table, columns=["arxiv_id"])["arxiv_id"].tolist())


def _escape_sql(s: str) -> str:
    return s.replace("'", "''")


# Fields where a blank/None re-fetch should never clobber a good existing value
# (e.g. arXiv-only re-scan overwriting an S2-enriched venue/title with "").
_COALESCE_FIELDS = (
    "title", "abstract", "authors", "submitted_date", "venue",
    "fields_of_study", "pdf_url", "arxiv_url", "source", "doi", "s2_id",
)
UPDATE_BATCH = 500  # rows per merge_insert flush, matches build_index.py's WRITE_BATCH


def upsert_papers_batch(table, papers: List[PaperRecord], existing_ids: Optional[set] = None) -> tuple:
    """
    Batch upsert papers into LanceDB.

    New papers (arxiv_id not in table): inserted with vector=[0]*768, has_embedding=False.
    Existing papers: metadata merged (new value wins unless blank, citation counts take the
    max, existing vector/has_embedding preserved unless title or abstract actually changed).

    Args:
        table: LanceTable (papers table)
        papers: list of PaperRecord to upsert
        existing_ids: pre-fetched set of arxiv_ids (pass to avoid repeated DB calls in tight loops)

    Returns:
        (inserted_count, updated_count)
    """
    if not papers:
        return 0, 0

    if existing_ids is None:
        existing_ids = _load_existing_ids(table)

    new_rows = []
    existing_papers = []

    for p in papers:
        if p.arxiv_id not in existing_ids:
            new_rows.append(paper_to_row(p, vector=None, has_embedding=False))
            existing_ids.add(p.arxiv_id)
        else:
            existing_papers.append(p)

    inserted = 0
    updated = 0

    if new_rows:
        table.add(new_rows)
        inserted = len(new_rows)

    if existing_papers:
        id_list = ", ".join(f"'{_escape_sql(p.arxiv_id)}'" for p in existing_papers)
        existing_df = query_table(table, filter=f"arxiv_id IN ({id_list})")
        existing_map = {row["arxiv_id"]: row.to_dict() for _, row in existing_df.iterrows()}

        rows_to_write = []
        for p in existing_papers:
            old = existing_map.get(p.arxiv_id)
            if old is None:
                continue
            new = paper_to_row(p, vector=None, has_embedding=False)
            merged = dict(old)
            for key in _COALESCE_FIELDS:
                merged[key] = new[key] if new[key] not in (None, "", "[]") else old.get(key)
            merged["citation_count"] = max(int(old.get("citation_count") or 0), new["citation_count"])
            merged["max_author_citations"] = max(int(old.get("max_author_citations") or 0), new["max_author_citations"])
            merged["year"] = _extract_year(merged["submitted_date"])
            content_changed = merged["title"] != old.get("title") or merged["abstract"] != old.get("abstract")
            if content_changed:
                merged["has_embedding"] = False
                merged["vector"] = [0.0] * VECTOR_DIM
            rows_to_write.append(merged)

        for start in range(0, len(rows_to_write), UPDATE_BATCH):
            batch = rows_to_write[start : start + UPDATE_BATCH]
            table.merge_insert("arxiv_id").when_matched_update_all().execute(batch)
        updated = len(existing_papers)

    return inserted, updated
