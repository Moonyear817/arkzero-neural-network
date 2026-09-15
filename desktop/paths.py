"""Resource paths and writable workspace; never write inside an application bundle."""

import os
import sys
from pathlib import Path

RESOURCE_ROOT = Path(__file__).resolve().parents[1]
IN_BUNDLE = ".app/Contents/" in str(RESOURCE_ROOT) or ".app/Contents/" in sys.executable
WORKSPACE_ROOT = Path(os.environ.get("ARKNIGHTS_ZERO_WORKSPACE", str(
    Path.home() / "Library/Application Support/Arknights Zero" if IN_BUNDLE else RESOURCE_ROOT
))).expanduser().resolve()
DATA_DIR = WORKSPACE_ROOT / "data/real"
if not DATA_DIR.exists():
    DATA_DIR = RESOURCE_ROOT / "data/real"


def ensure_workspace():
    for folder in ("configs", "checkpoints", "logs"):
        (WORKSPACE_ROOT / folder).mkdir(parents=True, exist_ok=True)
    default = WORKSPACE_ROOT / "configs/desktop_default.yaml"
    source = RESOURCE_ROOT / "configs/desktop_default.yaml"
    if not default.exists() and source.exists():
        default.write_bytes(source.read_bytes())
    model = RESOURCE_ROOT / "bundled_checkpoints/best.pt"
    destination = WORKSPACE_ROOT / "checkpoints/best.pt"
    if IN_BUNDLE and model.exists() and not destination.exists():
        destination.write_bytes(model.read_bytes())
