#!/usr/bin/env python3
"""Launch the shared-core desktop application in development or a bundle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop.app import main

if __name__ == "__main__":
    raise SystemExit(main())
