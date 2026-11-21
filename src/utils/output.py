"""Create independent artifact directories within a single experiment run."""
from pathlib import Path

CATEGORIES = ("configs", "models", "metrics", "predictions", "figures", "logs", "splits", "reports")

def output_dirs(base, create=True):
    base = Path(base)
    paths = {category: base / category for category in CATEGORIES}
    if create:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    return paths
