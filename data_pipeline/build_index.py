"""
data_pipeline/build_index.py
Incremental embedding pipeline: embed unindexed papers and write vectors into LanceDB.

FAISS, embeddings.npy, id_map.json, and bm25_index are no longer used.
LanceDB on R2 is the sole vector store (has_embedding column tracks which papers are embedded).

Usage:
    python data_pipeline/build_index.py               # incremental (papers with has_embedding=False)
    python data_pipeline/build_index.py --full        # reset all vectors, re-embed everything
    python data_pipeline/build_index.py --local PATH  # use local LanceDB dir (testing)
"""
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np

from data_pipeline.schema import (
    LANCEDB_SCHEMA,
    TABLE_NAME,
    VECTOR_DIM,
    connect_lancedb,
    get_or_create_papers_table,
    query_table,
    _escape_sql,
)

logger = logging.getLogger(__name__)

EMBED_BATCH = 32   # papers per SPECTER2 inference batch
WRITE_BATCH = 500  # rows per LanceDB merge_insert flush
CHECKPOINT_CHUNK = 2000  # papers embedded+written per checkpoint (crash loses at most one chunk, not the whole backlog)
INDEX_REBUILD_EVERY_N_CHUNKS = 10  # ~20k papers between mid-run index rebuilds on a long backfill


def load_unindexed_papers(table, force_full: bool = False) -> list:
    """
    Return paper dicts (arxiv_id, title, abstract) for papers that need embedding.
    force_full=True resets all has_embedding flags first.
    """
    if force_full:
        logger.info("force-full: resetting has_embedding=False for all rows…")
        table.update(where="has_embedding = true", values={"has_embedding": False})

    try:
        df = query_table(
            table, filter="has_embedding = false",
            columns=["arxiv_id", "title", "abstract", "submitted_date"],
        )
    except Exception as exc:
        logger.error("Failed to query unindexed papers: %s", exc)
        return []

    # ponytail: newest-first — same total work, but retrieval on recent-date
    # queries (the ones actually breaking today) gets fixed soonest instead of
    # waiting on the full historical backlog.
    df = df.sort_values("submitted_date", ascending=False)
    papers = df.to_dict("records")
    logger.info("Papers requiring embedding: %d (newest first)", len(papers))
    return papers


def load_embedding_model(model_name: str = "allenai/specter2_base"):
    """
    Load the embedding model once. Returns (model, tokenizer_or_None, is_specter2).
    Split out from embed_papers_with_model so a long backfill can reuse one loaded
    model across many checkpointed chunks instead of reloading it every call.
    """
    is_specter2 = "specter2" in model_name

    if is_specter2:
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        logger.info("Loading SPECTER2 base model + proximity adapter…")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoAdapterModel.from_pretrained(model_name)
        model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
        model.set_active_adapters("proximity")
        try:
            active_adapters = model.active_adapters
            if "proximity" not in active_adapters:
                logger.warning("[SPECTER2] adapter not active before encode: %s", active_adapters)
                model.set_active_adapters("proximity")
        except Exception as e:
            logger.warning("[SPECTER2] could not verify adapter activation: %s", e)
        model.eval()
        return model, tokenizer, True
    else:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name), None, False


def embed_papers_with_model(papers: list, model, tokenizer, is_specter2: bool) -> np.ndarray:
    """Encode one chunk of papers with an already-loaded model. Returns float32 (N, 768), L2-normalised."""
    if is_specter2:
        import torch

        sep = tokenizer.sep_token
        texts = [(p.get("title") or "") + sep + (p.get("abstract") or "") for p in papers]
        all_vecs = []

        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            vecs = out.last_hidden_state[:, 0, :]
            vecs = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)
            all_vecs.append(vecs.cpu().float().numpy())
            if i % (EMBED_BATCH * 20) == 0:
                logger.info("  Embedded %d / %d papers in this chunk…", i + len(batch), len(texts))

        return np.vstack(all_vecs).astype("float32")
    else:
        texts = [(p.get("title") or "") + "\n\n" + (p.get("abstract") or "") for p in papers]
        vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")


def embed_papers(papers: list, model_name: str = "allenai/specter2_base") -> np.ndarray:
    """Backward-compatible one-shot wrapper: load model once, encode all papers, discard model."""
    model, tokenizer, is_specter2 = load_embedding_model(model_name)
    try:
        return embed_papers_with_model(papers, model, tokenizer, is_specter2)
    finally:
        del model
        gc.collect()


def write_vectors_to_lancedb(table, papers: list, embeddings: np.ndarray) -> int:
    """
    Update LanceDB rows with computed embeddings.
    Uses merge_insert (full-row replace) so vector + has_embedding are written atomically.
    Returns number of rows updated.
    """
    if len(papers) != len(embeddings):
        raise ValueError(f"papers/embeddings length mismatch: {len(papers)} vs {len(embeddings)}")

    # We need full rows for merge_insert. Fetch existing metadata from LanceDB.
    arxiv_ids = [p["arxiv_id"] for p in papers]
    id_list = ", ".join(f"'{_escape_sql(aid)}'" for aid in arxiv_ids)
    filter_str = f"arxiv_id IN ({id_list})"

    try:
        existing_df = query_table(table, filter=filter_str)
    except Exception as exc:
        logger.error("Failed to fetch existing rows for vector write: %s", exc)
        return 0

    # Build lookup: arxiv_id → existing metadata row (as dict)
    existing_map = {row["arxiv_id"]: row for _, row in existing_df.iterrows()}
    pos_map = {p["arxiv_id"]: i for i, p in enumerate(papers)}

    updated = 0
    rows_to_write = []

    for arxiv_id, idx in pos_map.items():
        meta = existing_map.get(arxiv_id)
        if meta is None:
            logger.warning("arxiv_id %s not found in LanceDB during vector write — skipping.", arxiv_id)
            continue

        vec = embeddings[idx].tolist()
        row = meta.to_dict()
        row["vector"] = vec
        row["has_embedding"] = True
        rows_to_write.append(row)

    for start in range(0, len(rows_to_write), WRITE_BATCH):
        batch = rows_to_write[start : start + WRITE_BATCH]
        (
            table.merge_insert("arxiv_id")
            .when_matched_update_all()
            .execute(batch)
        )
        updated += len(batch)
        logger.info("  Vectors written %d / %d", min(start + WRITE_BATCH, len(rows_to_write)), len(rows_to_write))

    return updated


def rebuild_fts_index(table) -> None:
    """Rebuild FTS indexes (one per field — lancedb 0.25+ requires separate index per field)."""
    from lancedb.index import FTS
    logger.info("Rebuilding FTS indexes on title and abstract…")
    table.create_index("title",    config=FTS(with_position=True), replace=True)
    table.create_index("abstract", config=FTS(with_position=True), replace=True)


def rebuild_vector_index(table) -> None:
    """Rebuild ANN vector index."""
    from lancedb.index import IvfHnswSq
    logger.info("Rebuilding vector index (IVF_HNSW_SQ, cosine)…")
    table.create_index("vector", config=IvfHnswSq(distance_type="cosine"), replace=True)


def write_build_meta(output_dir: str, update_arxiv_ts: bool, update_s2_ts: bool, row_count: int) -> None:
    meta_path = os.path.join(output_dir, "build_meta.json")
    existing = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass

    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    meta = {
        "built_at": now_iso,
        "last_arxiv_at": existing.get("last_arxiv_at") or now_iso,
        "last_s2_at": existing.get("last_s2_at") or now_iso,
        "corpus_size": row_count,
        "storage": "lancedb",
        "schema_version": 2,
    }
    if update_arxiv_ts:
        meta["last_arxiv_at"] = now_iso
    if update_s2_ts:
        meta["last_s2_at"] = now_iso

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Build metadata saved: %s", meta_path)


def run_index_build(
    lancedb_local_path: str = None,
    model_name: str = "allenai/specter2_base",
    force_full: bool = False,
    update_arxiv_ts: bool = False,
    update_s2_ts: bool = False,
    meta_dir: str = "data_pipeline",
) -> None:
    """
    Incremental index build:
    1. Query LanceDB for papers with has_embedding=False
    2. Embed + write in checkpointed chunks (CHECKPOINT_CHUNK papers at a time),
       so a crash mid-run loses at most one chunk, not the whole backlog —
       everything already written stays committed and load_unindexed_papers()
       naturally skips it on the next run.
    3. Rebuild FTS + ANN indexes every INDEX_REBUILD_EVERY_N_CHUNKS chunks (not
       just once at the very end) — a crash during index rebuild then only
       costs re-running the rebuild on top of already-embedded rows, not lost
       embedding work, and newly-embedded papers become searchable well before
       the whole backlog finishes on a long run.
    4. Update build_meta.json
    """
    db = connect_lancedb(lancedb_local_path)
    table = get_or_create_papers_table(db)

    papers = load_unindexed_papers(table, force_full=force_full)
    if not papers:
        logger.info("No papers to embed. Index is up to date.")
        row_count = table.count_rows()
        write_build_meta(meta_dir, update_arxiv_ts, update_s2_ts, row_count)
        return

    logger.info(
        "Embedding %d papers with %s (checkpointed every %d, index rebuild every %d chunks)…",
        len(papers), model_name, CHECKPOINT_CHUNK, INDEX_REBUILD_EVERY_N_CHUNKS,
    )
    model, tokenizer, is_specter2 = load_embedding_model(model_name)

    total_written = 0
    try:
        for chunk_i, start in enumerate(range(0, len(papers), CHECKPOINT_CHUNK), start=1):
            chunk = papers[start : start + CHECKPOINT_CHUNK]
            embeddings = embed_papers_with_model(chunk, model, tokenizer, is_specter2)
            total_written += write_vectors_to_lancedb(table, chunk, embeddings)
            del embeddings
            gc.collect()
            logger.info(
                "Checkpoint: %d / %d papers embedded and written to LanceDB",
                min(start + CHECKPOINT_CHUNK, len(papers)), len(papers),
            )
            if chunk_i % INDEX_REBUILD_EVERY_N_CHUNKS == 0:
                logger.info("Mid-run index rebuild (chunk %d)…", chunk_i)
                rebuild_fts_index(table)
                rebuild_vector_index(table)
    finally:
        del model
        gc.collect()

    logger.info("Vectors written: %d", total_written)

    rebuild_fts_index(table)
    rebuild_vector_index(table)

    row_count = table.count_rows()
    write_build_meta(meta_dir, update_arxiv_ts, update_s2_ts, row_count)
    logger.info("Index build complete. Total rows in LanceDB: %d", row_count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed papers and write vectors to LanceDB")
    parser.add_argument("--full",         action="store_true", help="Reset all embeddings and rebuild from scratch")
    parser.add_argument("--update-arxiv", action="store_true", help="Update last_arxiv_at timestamp in build_meta.json")
    parser.add_argument("--update-s2",    action="store_true", help="Update last_s2_at timestamp in build_meta.json")
    parser.add_argument("--local",        default=None, metavar="PATH", help="Local LanceDB dir (for testing, e.g. data_pipeline/lancedb)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_index_build(
        lancedb_local_path=args.local,
        force_full=args.full,
        update_arxiv_ts=args.update_arxiv,
        update_s2_ts=args.update_s2,
    )
