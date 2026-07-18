"""
tests/test_integration.py
End-to-end integration test: fetch 100 papers from S2 → upsert to LanceDB
→ build embeddings + FTS/ANN indexes.

Skipped unless S2_API_KEY is set in environment (or .env).
Run with:
    pytest -m integration -v tests/test_integration.py
"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("S2_API_KEY"),
    reason="S2_API_KEY not set — skipping live integration test",
)
def test_pipeline_100_papers(tmp_path):
    """
    Smoke test: ingest 100 papers from S2 bulk API, build the LanceDB index,
    assert the table exists and has ≥ 1 embedded arXiv paper.
    """
    from data_pipeline.fetch_corpus import run_ingestion
    from data_pipeline.build_index import run_index_build
    from data_pipeline.schema import connect_lancedb, get_or_create_papers_table

    lancedb_path = str(tmp_path / "lancedb")

    # Step 1 — ingestion
    # incremental=True + days=1 keeps the arXiv scout (Stage 1) to a ~1-day window across
    # all 17 sub-categories. incremental=False sets a 365-day scout window with no cap on
    # per-category results (up to 10k each) — that's intentional for production's --full
    # resync, but turns this "100-paper" smoke test into a multi-hour run. max_papers=100
    # only bounds the S2 bulk stage (Stage 2), never the arXiv scout.
    run_ingestion(lancedb_local_path=lancedb_path, max_papers=100, incremental=True, days=1)
    table = get_or_create_papers_table(connect_lancedb(lancedb_path))
    n = table.count_rows()
    assert n >= 1, (
        f"Expected at least 1 arXiv paper, got {n}. "
        "S2 bulk search returns ~30% arXiv hit rate so 100 raw → ≥1 expected."
    )

    # Step 2 — index build
    run_index_build(lancedb_local_path=lancedb_path, meta_dir=str(tmp_path))

    # Re-open: run_index_build writes through its own connection/table handle, and a
    # LanceDB table object doesn't see writes made through a separate handle on the
    # same path until reopened.
    table = get_or_create_papers_table(connect_lancedb(lancedb_path))
    embedded = table.count_rows("has_embedding = true")
    assert embedded >= 1, "No papers were embedded"
    assert (tmp_path / "build_meta.json").exists(), "build_meta.json not created"
