"""
data_pipeline/build_index.py
Build FAISS (inner-product / cosine) and optional BM25 indexes from corpus.db.

Usage:
    python data_pipeline/build_index.py
"""
import json
import logging
import os
import sqlite3

import faiss
import numpy as np

logger = logging.getLogger(__name__)

try:
    import bm25s

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def load_papers_from_db(db_path: str) -> list:
    """Return all papers from corpus.db as plain dicts (authors decoded from JSON)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM papers ORDER BY submitted_date DESC"
    ).fetchall()
    conn.close()

    papers = []
    for row in rows:
        d = dict(row)
        d["authors"] = json.loads(d.get("authors") or "[]")
        papers.append(d)
    return papers


def embed_papers(
    papers: list,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Encode title + abstract with a SentenceTransformer model.
    Returns float32 ndarray of shape (N, dim), L2-normalised.
    """
    from sentence_transformers import SentenceTransformer  # lazy import

    model = SentenceTransformer(model_name)
    texts = [
        (p.get("title") or "") + "\n\n" + (p.get("abstract") or "")
        for p in papers
    ]
    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an exact inner-product (cosine equivalent for normalised vecs) index.
    """
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    return idx


def save_index(index: faiss.Index, path: str) -> None:
    """Write FAISS index to disk."""
    faiss.write_index(index, path)
    logger.info("FAISS saved: %s (%d vectors)", path, index.ntotal)


def build_bm25_index(papers: list, save_dir: str) -> None:
    """Build and save a BM25 index over title+abstract text."""
    if not HAS_BM25:
        logger.warning("bm25s not installed — skipping BM25 index")
        return

    corpus = [
        (p.get("title") or "") + " " + (p.get("abstract") or "")
        for p in papers
    ]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    retriever.save(save_dir)
    logger.info("BM25 saved: %s/", save_dir)


def run_index_build(
    db_path: str = "data_pipeline/corpus.db",
    output_dir: str = "data_pipeline",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """
    Full index build pipeline:
    1. Load papers from SQLite
    2. Embed with SentenceTransformer
    3. Build + save FAISS index
    4. Build + save BM25 index
    5. Write id_map.json (int index → arxiv_id)
    """
    os.makedirs(output_dir, exist_ok=True)

    papers = load_papers_from_db(db_path)
    if not papers:
        logger.error("No papers found — run fetch_corpus.py first")
        return

    logger.info("Building index for %d papers…", len(papers))

    # Embeddings
    embeddings = embed_papers(papers, model_name)
    emb_path = os.path.join(output_dir, "embeddings_minilm.npy")
    np.save(emb_path, embeddings)
    logger.info("Embeddings: %.1f MB", embeddings.nbytes / 1e6)

    # FAISS index
    faiss_path = os.path.join(output_dir, "index_minilm.faiss")
    save_index(build_faiss_index(embeddings), faiss_path)

    # BM25 index
    bm25_dir = os.path.join(output_dir, "bm25_index")
    build_bm25_index(papers, bm25_dir)

    # ID map: positional index → arxiv_id
    id_map = {str(i): p["arxiv_id"] for i, p in enumerate(papers)}
    id_map_path = os.path.join(output_dir, "id_map.json")
    with open(id_map_path, "w", encoding="utf-8") as fh:
        json.dump(id_map, fh)

    logger.info("Index build complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_index_build()
