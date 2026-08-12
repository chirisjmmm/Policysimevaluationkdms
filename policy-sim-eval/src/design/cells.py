"""Cell configuration loading + run assembly (sec 6.1)."""
from __future__ import annotations

import yaml

from src.io.loaders import LoadReport, load_cell


def load_cells_config(path: str = "configs/cells.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_cells(cells_config: dict) -> dict:
    """Returns {cell_id: (runs, LoadReport)}."""
    out = {}
    for cell in cells_config["cells"]:
        runs, report = load_cell(cell["runs_dir"], cell["id"])
        out[cell["id"]] = (runs, report)
    return out
