"""Research Agent — FastAPI backend.

Serves the built React frontend (frontend/dist, once built) and the streaming
search API, backed directly by backend/pipeline_core.py's real LanceDB
pipeline — no mock fallback (unlike the reference repo's prototype-phase
main.py): F-02 already proved this pipeline works end-to-end against the real
corpus, so there's nothing to demo a mock in place of.

Run (dev):  uvicorn backend.main:app --reload --port 8000
"""
import asyncio
import json
import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline_core as pc
from .models import SearchRequest, Stage
from .runner import STAGE_NAMES, pipeline_events

app = FastAPI(title="Research Agent API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SENTINEL = object()


def _sse(payload) -> str:
    if hasattr(payload, "model_dump_json"):
        return f"data: {payload.model_dump_json()}\n\n"
    return f"data: {json.dumps(payload)}\n\n"


@app.on_event("startup")
def _warm_corpus() -> None:
    """Sync build_meta.json (freshness tracking) and warm the model caches in the background."""
    import threading
    threading.Thread(target=pc.download_corpus_artifacts, daemon=True).start()


@app.get("/api/health")
def health():
    table = pc.get_lancedb_table()
    return {"status": "ok", "corpus_ready": table is not None}


@app.get("/api/stages", response_model=list[Stage])
def stages():
    return [Stage(n=str(i + 1), name=name) for i, name in enumerate(STAGE_NAMES)]


@app.post("/api/search")
async def search(req: SearchRequest):
    async def event_stream():
        gen = pipeline_events(req)
        try:
            while True:
                # Advance the synchronous pipeline one event at a time in a worker
                # thread so heavy model inference never blocks the event loop.
                ev = await asyncio.to_thread(next, gen, _SENTINEL)
                if ev is _SENTINEL:
                    break
                yield _sse(ev)
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- Static frontend (production) ----
_DIST = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="frontend")
else:
    @app.get("/")
    def _root_placeholder():
        return JSONResponse({"status": "backend up", "frontend": "not built — run the Vite dev server"})
