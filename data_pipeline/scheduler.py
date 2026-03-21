"""
data_pipeline/scheduler.py
Daily pipeline wrapper: fetch → build index.

Cron usage (Unix/macOS):
    0 2 * * * cd /repo && python data_pipeline/scheduler.py >> logs/pipeline.log 2>&1

Windows Task Scheduler: point to this script with the project root as working directory.
"""
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def run() -> None:
    """Run fetch → build sequentially; exit non-zero on any failure."""
    logger.info("=== Pipeline run: %s ===", datetime.now().isoformat())

    steps = [
        ([sys.executable, "data_pipeline/fetch_corpus.py"], "Fetch"),
        ([sys.executable, "data_pipeline/build_index.py"], "Build index"),
    ]

    for cmd, label in steps:
        logger.info("--- %s ---", label)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("%s failed (exit code %d)", label, result.returncode)
            sys.exit(result.returncode)

    logger.info("=== Done ===")


if __name__ == "__main__":
    run()
