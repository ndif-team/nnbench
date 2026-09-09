"""Name-based Docker benchmark CLI. See backends/README.md."""
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.jobs.cli import interrupted, main  # noqa: E402

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
