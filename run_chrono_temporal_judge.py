from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "chronosynth"))

from chronoagent_harness.temporal_faithfulness_eval import main


if __name__ == "__main__":
    main()
