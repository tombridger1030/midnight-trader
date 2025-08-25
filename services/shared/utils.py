"""Shared helper utilities."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List


def load_universe(path: str | Path) -> List[str]:
    """Load list of tickers from a CSV file."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        return [row[0].strip() for row in reader if row]
