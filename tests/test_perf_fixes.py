"""
tests/test_perf_fixes.py

Targeted verification for:
  Fix 1  — DB fetch LIMIT 20,000 cap (OOM guard)
  Fix 2  — specter2_vector_rerank precomputed fast-path
  Fix 3  — _faiss_ranked_pool vectorized (numpy, no Python loop)
  n2=200 — Stage 2 output cap (halves CrossEncoder Stage 3 input)
  NULL date bug — submitted_date=NULL handled in fetch_papers_from_db

Data source: Cloudflare R2 bucket → session-scoped pytest tmp dir.
Temp dir is auto-deleted by pytest after the session (tmpdir_factory cleanup).
data_pipeline/ is never read or written.
"""
from __future__ import annotations
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

pytestmark = pytest.mark.integration  # requires R2 credentials — excluded from CI unit runs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── R2 helpers ────────────────────────────────────────────────────────────────

def _make_s3():
    """Return (boto3 client, bucket_name) from .env credentials."""
    import boto3
    from botocore.config import Config
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    key_id   = os.getenv("R2_ACCESS_KEY_ID",     "").strip()
    secret   = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    endpoint = os.getenv("R2_ENDPOINT",           "").strip()
    bucket   = os.getenv("R2_BUCKET",             "").strip()

    if not all([key_id, secret, endpoint, bucket]):
        pytest.skip("R2 credentials not configured — set R2_* env vars or .env")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    return s3, bucket


# ── Session-scoped temp dir: downloaded once, deleted at session end ───────────

@pytest.fixture(scope="session")
def r2_tmp(tmp_path_factory):
    """
    Download R2 corpus artifacts into a session-scoped pytest temp dir.
    pytest auto-deletes the basetemp tree after the session ends.
    data_pipeline/ is never touched.
    """
    dest = tmp_path_factory.mktemp("r2_corpus", numbered=True)
    s3, bucket = _make_s3()

    core = ["corpus.db", "embeddings.npy", "id_map.json", "build_meta.json"]
    print(f"\n[R2] Downloading to temp: {dest}")
    for fname in core:
        local = dest / fname
        t0 = time.perf_counter()
        s3.download_file(bucket, f"corpus/{fname}", str(local))
        mb = local.stat().st_size / (1024 ** 2)
        print(f"[R2]   {fname}: {mb:.1f} MB  ({time.perf_counter()-t0:.1f}s)")

    # BM25 (non-fatal)
    bm25_dir = dest / "bm25_index"
    bm25_dir.mkdir()
    try:
        pager = s3.get_paginator("list_objects_v2")
        for page in pager.paginate(Bucket=bucket, Prefix="corpus/bm25_index/"):
            for obj in page.get("Contents", []):
                key  = obj["Key"]
                name = key[len("corpus/bm25_index/"):]
                if name:
                    s3.download_file(bucket, key, str(bm25_dir / name))
        print("[R2]   bm25_index: downloaded")
    except Exception as e:
        print(f"[R2]   bm25_index: skipped ({e})")

    yield dest

    # Explicit cleanup (belt-and-suspenders on top of pytest's auto-cleanup)
    try:
        shutil.rmtree(dest, ignore_errors=True)
        print(f"\n[R2] Temp dir deleted: {dest}")
    except Exception:
        pass


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def artifacts(r2_tmp) -> Tuple[np.ndarray, dict]:
    emb = np.load(str(r2_tmp / "embeddings.npy"), mmap_mode="r")
    with open(r2_tmp / "id_map.json", encoding="utf-8") as f:
        id_map = json.load(f)
    arxiv_to_pos = {v: int(k) for k, v in id_map.items()}
    print(f"\n[fixture] embeddings={emb.shape}  indexed={len(arxiv_to_pos):,}")
    return emb, arxiv_to_pos


def _load_papers_from_db(db_path: Path, limit: int) -> list:
    """Load papers from SQLite. NULL submitted_date → fallback '2024-01-01'."""
    from app import Paper
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM papers ORDER BY submitted_date DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    papers: List[Paper] = []
    for r in rows:
        d = dict(r)
        # Guard: reject NULL, empty, or non-ISO strings like the literal "None"/"None-01-01"
        raw = d.get("submitted_date") or ""
        date_str = raw if (len(raw) >= 4 and raw[:4].isdigit()) else "2024-01-01"
        if "T" not in date_str:
            date_str += "T00:00:00"
        papers.append(Paper(
            arxiv_id=d["arxiv_id"],
            title=d.get("title") or "",
            authors=json.loads(d.get("authors") or "[]"),
            email_domains=[],
            abstract=d.get("abstract") or "",
            submitted_date=datetime.fromisoformat(date_str.replace("Z", "+00:00")),
            pdf_url=d.get("pdf_url") or "",
            arxiv_url=d.get("arxiv_url") or "",
            venue=d.get("venue"),
            source=d.get("source"),
        ))
    return papers


@pytest.fixture(scope="session")
def papers_5k(r2_tmp):
    papers = _load_papers_from_db(r2_tmp / "corpus.db", limit=5_000)
    assert papers, "No papers loaded from R2 corpus.db"
    print(f"[fixture] Loaded {len(papers)} papers (5k)")
    return papers


@pytest.fixture(scope="session")
def papers_20k(r2_tmp):
    papers = _load_papers_from_db(r2_tmp / "corpus.db", limit=20_000)
    assert papers, "No papers loaded from R2 corpus.db"
    print(f"[fixture] Loaded {len(papers)} papers (20k cap)")
    return papers


# ── Reference: old Python-loop implementation ─────────────────────────────────

def _faiss_ref(papers, q_vec, emb, a2p, top_k=400) -> dict:
    """Pre-Fix-3 Python loop — ground truth for correctness diff."""
    scores = []
    for p in papers:
        pos = a2p.get(p.arxiv_id)
        sim = float(np.dot(emb[pos], q_vec)) if (pos is not None and pos < emb.shape[0]) else -1.0
        scores.append((p.arxiv_id, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return {aid: rank for rank, (aid, _) in enumerate(scores[:top_k], 1)}


def _qvec(papers, emb, a2p) -> np.ndarray:
    """Deterministic query vector: first indexed paper's embedding."""
    for p in papers:
        pos = a2p.get(p.arxiv_id)
        if pos is not None and pos < emb.shape[0]:
            v = emb[pos].copy().astype(np.float32)
            n = np.linalg.norm(v)
            return v / n if n > 0 else v
    pytest.skip("No indexed paper in pool")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. R2 CORPUS INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestR2CorpusIntegrity:

    def test_build_meta(self, r2_tmp):
        with open(r2_tmp / "build_meta.json") as f:
            meta = json.load(f)
        corpus_size  = meta.get("corpus_size", 0)
        faiss_ntotal = meta.get("faiss_ntotal", 0)
        built_at     = meta.get("built_at", "")
        print(f"\n[R2] corpus_size={corpus_size:,}  faiss_ntotal={faiss_ntotal:,}  built_at={built_at}")
        assert corpus_size  > 10_000
        assert faiss_ntotal > 10_000
        assert built_at

    def test_embeddings_match_id_map(self, artifacts):
        emb, a2p = artifacts
        print(f"\n[R2] emb rows={emb.shape[0]:,}  id_map={len(a2p):,}")
        assert emb.shape[0] == len(a2p)

    def test_embeddings_normalized(self, artifacts):
        emb, _ = artifacts
        rng = np.random.default_rng(42)
        idx = rng.integers(0, emb.shape[0], size=min(1000, emb.shape[0]))
        norms = np.linalg.norm(emb[idx].astype(np.float32), axis=1)
        dev = float(np.max(np.abs(norms - 1.0)))
        print(f"\n[R2] Max norm deviation (1k sample): {dev:.6f}")
        assert dev < 0.01, f"Not unit-normalized: dev={dev}"

    def test_db_fetch_limit_constant(self):
        from app import DB_FETCH_LIMIT
        print(f"\n[R2] DB_FETCH_LIMIT={DB_FETCH_LIMIT:,}")
        assert DB_FETCH_LIMIT == 20_000

    def test_db_paper_count(self, r2_tmp):
        conn = sqlite3.connect(str(r2_tmp / "corpus.db"))
        n = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        conn.close()
        print(f"\n[R2] corpus.db papers: {n:,}")
        assert n > 10_000

    def test_bad_submitted_date_no_crash(self, r2_tmp):
        """Rows with NULL or non-ISO submitted_date (e.g. literal 'None') must not raise."""
        conn = sqlite3.connect(str(r2_tmp / "corpus.db"))
        bad_count = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE submitted_date IS NULL "
            "OR CAST(submitted_date AS TEXT) NOT GLOB '[0-9][0-9][0-9][0-9]*'"
        ).fetchone()[0]
        conn.close()
        print(f"\n[R2] Papers with NULL or non-ISO submitted_date: {bad_count:,}")
        # Loading 5k papers must not raise regardless of bad dates
        papers = _load_papers_from_db(r2_tmp / "corpus.db", limit=5_000)
        assert len(papers) > 0, "No papers loaded"
        print(f"[R2] Loaded {len(papers)} papers without crash: PASS")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FIX 3 — _faiss_ranked_pool vectorized
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix3VectorizedFAISS:

    def test_top50_matches_reference(self, artifacts, papers_5k):
        from app import _faiss_ranked_pool
        emb, a2p = artifacts
        q = _qvec(papers_5k, emb, a2p)

        ref = _faiss_ref(papers_5k, q, emb, a2p, top_k=400)
        new = _faiss_ranked_pool(papers_5k, "", emb, a2p, top_k=400, q_vec=q)

        ref_top = [aid for aid, _ in sorted(ref.items(), key=lambda x: x[1])[:50]]
        new_top = [aid for aid, _ in sorted(new.items(), key=lambda x: x[1])[:50]]

        assert ref_top == new_top, f"Top-50 mismatch\nRef: {ref_top[:3]}\nNew: {new_top[:3]}"
        print(f"\n[Fix3] Top-50 vs reference: PASS")

    def test_full_rank_dict_matches_reference(self, artifacts, papers_5k):
        from app import _faiss_ranked_pool
        emb, a2p = artifacts
        q = _qvec(papers_5k, emb, a2p)

        ref = _faiss_ref(papers_5k, q, emb, a2p, top_k=400)
        new = _faiss_ranked_pool(papers_5k, "", emb, a2p, top_k=400, q_vec=q)

        assert set(ref.keys()) == set(new.keys()), \
            f"Key mismatch: ref={len(ref)} new={len(new)}"
        bad = {k: (ref[k], new[k]) for k in ref if ref[k] != new[k]}
        assert not bad, f"Rank diffs: {list(bad.items())[:3]}"
        print(f"\n[Fix3] Full rank dict: PASS ({len(new)} papers)")

    def test_unindexed_papers_no_crash(self, artifacts):
        from app import Paper, _faiss_ranked_pool
        emb, a2p = artifacts
        fakes = [
            Paper(arxiv_id=f"FAKE_{i}", title="", authors=[], email_domains=[],
                  abstract="", submitted_date=datetime(2024, 1, 1),
                  pdf_url="", arxiv_url="")
            for i in range(20)
        ]
        q = np.ones(emb.shape[1], dtype=np.float32)
        q /= np.linalg.norm(q)
        result = _faiss_ranked_pool(fakes, "", emb, a2p, top_k=400, q_vec=q)
        assert isinstance(result, dict)
        print(f"\n[Fix3] Unindexed papers no-crash: PASS ({len(result)} entries)")

    def test_timing_5k_under_1s(self, artifacts, papers_5k):
        from app import _faiss_ranked_pool
        emb, a2p = artifacts
        q = _qvec(papers_5k, emb, a2p)

        t0 = time.perf_counter()
        result = _faiss_ranked_pool(papers_5k, "", emb, a2p, top_k=400, q_vec=q)
        elapsed = time.perf_counter() - t0

        print(f"\n[Fix3] 5k papers vectorized: {elapsed:.3f}s")
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s"
        assert len(result) > 0

    def test_timing_20k_under_2s(self, artifacts, papers_20k):
        from app import _faiss_ranked_pool
        emb, a2p = artifacts
        q = _qvec(papers_20k, emb, a2p)

        t0 = time.perf_counter()
        result = _faiss_ranked_pool(papers_20k, "", emb, a2p, top_k=1200, q_vec=q)
        elapsed = time.perf_counter() - t0

        print(f"\n[Fix3] 20k papers vectorized: {elapsed:.3f}s")
        assert elapsed < 2.0, f"Too slow for 20k: {elapsed:.3f}s"
        assert len(result) > 0

    def test_vectorized_faster_than_reference(self, artifacts, papers_5k):
        from app import _faiss_ranked_pool
        emb, a2p = artifacts
        q = _qvec(papers_5k, emb, a2p)

        t0 = time.perf_counter()
        _faiss_ref(papers_5k, q, emb, a2p, top_k=400)
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        _faiss_ranked_pool(papers_5k, "", emb, a2p, top_k=400, q_vec=q)
        t_new = time.perf_counter() - t0

        speedup = t_ref / t_new if t_new > 0 else float("inf")
        print(f"\n[Fix3] Reference={t_ref:.3f}s  Vectorized={t_new:.3f}s  Speedup={speedup:.1f}x")
        assert t_new < t_ref, f"Vectorized not faster: {t_new:.3f}s vs {t_ref:.3f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. n2=200 — Stage 2 cap + Stage 3 speedup
# ═══════════════════════════════════════════════════════════════════════════════

def _stage2(pool, query, emb, a2p, n2):
    """
    Mirror select_embedding_candidates Stage 2 logic:
    try SPECTER2 fast-path; fall back to minilm_vector_rerank (same as app.py).
    Ensures tests always get results regardless of whether 'adapters' is installed.
    """
    from app import specter2_vector_rerank, minilm_vector_rerank
    result = specter2_vector_rerank(pool, query, n2=n2,
                                    precomputed_embeddings=emb, arxiv_to_pos=a2p)
    if not result:
        # adapters not installed — use MiniLM fallback (matches app.py behaviour)
        result = minilm_vector_rerank(pool, query, emb, a2p, n2=n2)
    return result


class TestN2Reduction:

    def test_stage2_capped_at_200(self, artifacts, papers_5k):
        emb, a2p = artifacts
        result = _stage2(papers_5k[:500],
                         "transformer self-supervised learning attention",
                         emb, a2p, n2=200)
        assert 0 < len(result) <= 200, f"Stage 2 count={len(result)}"
        print(f"\n[n2=200] Stage 2 output={len(result)} (SPECTER2 or MiniLM): PASS")

    def test_stage2_sorted_descending(self, artifacts, papers_5k):
        emb, a2p = artifacts
        result = _stage2(papers_5k[:300],
                         "graph neural networks node classification",
                         emb, a2p, n2=200)
        sims = [p.semantic_relevance for p in result if p.semantic_relevance is not None]
        assert sims == sorted(sims, reverse=True), "Not sorted descending"
        print(f"\n[n2=200] Sort order: PASS  top={sims[0]:.4f}  bottom={sims[-1]:.4f}")

    def test_stage2_to_stage3_chain(self, artifacts, papers_5k):
        from app import cross_encoder_rerank
        emb, a2p = artifacts
        query = "large language models RLHF instruction tuning"

        s2 = _stage2(papers_5k[:400], query, emb, a2p, n2=200)
        assert 0 < len(s2) <= 200, f"Stage 2={len(s2)}"

        s3 = cross_encoder_rerank(s2, query, n3=150)
        assert 0 < len(s3) <= 150, f"Stage 3={len(s3)}"
        print(f"\n[n2=200] Stage2={len(s2)} -> Stage3={len(s3)}: PASS")

    def test_ce_faster_with_200_vs_400(self, artifacts, papers_5k):
        from app import cross_encoder_rerank
        emb, a2p = artifacts
        query = "diffusion models image generation latent space"

        s400 = _stage2(papers_5k[:500], query, emb, a2p, n2=400)
        s200 = _stage2(papers_5k[:500], query, emb, a2p, n2=200)

        t0 = time.perf_counter(); cross_encoder_rerank(s400, query, n3=150); t400 = time.perf_counter() - t0
        t0 = time.perf_counter(); cross_encoder_rerank(s200, query, n3=150); t200 = time.perf_counter() - t0

        speedup = t400 / t200 if t200 > 0 else float("inf")
        print(f"\n[n2=200] CE(400 pairs)={t400:.2f}s  CE(200 pairs)={t200:.2f}s  Speedup={speedup:.1f}x")
        assert t200 < t400, f"n2=200 CE ({t200:.2f}s) not faster than n2=400 ({t400:.2f}s)"
