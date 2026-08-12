"""Driver: generates the A1 manual-review artifacts (v4 sec 6) for every
single-pathway cell and every branch cell (at leaf-81 level). Only runs A1
(fast, ~1-2s/cell) -- does not need the slow A2 embedding pass, so this can
be run independently of the full `python -m src.cli` pipeline.

Writes, per cell, into outputs/audit/:
  {cell_id}_a1_diagnostic.md   -- Step 1 (sec 6.1): automatic, read-only
  {cell_id}_a1_sample.csv      -- Step 3 (sec 6.3): empty `label` column,
                                   a human fills in A/B/C for each row
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.audit.a1_diagnostic import run_diagnostic
from src.audit.a1_sampler import write_audit_csv
from src.branch.paths import extract_runs
from src.branch.tree_loader import load_branch_tree
from src.design.cells import load_cells_config
from src.io.loaders import load_cell
from src.metrics.acc_micro import run_a1


def main():
    eval_cfg = yaml.safe_load(open(ROOT / "configs/eval_default.yaml", encoding="utf-8"))
    cells_cfg = load_cells_config(str(ROOT / "configs/cells.yaml"))
    out_dir = ROOT / "outputs" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = {}

    def get_config(policy_id):
        if policy_id not in configs:
            configs[policy_id] = yaml.safe_load(open(ROOT / f"configs/policy_{policy_id}.yaml", encoding="utf-8"))
        return configs[policy_id]

    summary = []

    for cell in cells_cfg["cells"]:
        cid, pid = cell["id"], cell["policy"]
        config = get_config(pid)
        runs, _ = load_cell(str(ROOT / cell["runs_dir"]), cid)
        a1 = run_a1(runs, config["fact_table"])
        _write_artifacts(cid, a1, out_dir)
        summary.append((cid, a1["supported"], a1["contradicted"], a1["unscored"]))

    for bcell in cells_cfg.get("branch_cells", []):
        cid, pid = bcell["id"], bcell["policy"]
        config = get_config(pid)
        tree = load_branch_tree(str(ROOT / bcell["tree_path"]), cid)
        leaf_runs = extract_runs(tree, "leaf")
        a1 = run_a1(leaf_runs, config["fact_table"])
        _write_artifacts(cid, a1, out_dir)
        summary.append((cid, a1["supported"], a1["contradicted"], a1["unscored"]))

    print("\n=== A1 audit artifacts written to", out_dir, "===")
    for cid, sup, con, uns in summary:
        print(f"  {cid}: supported={sup} contradicted={con} unscored={uns}")


def _write_artifacts(cid: str, a1: dict, out_dir: Path):
    diag_path = out_dir / f"{cid}_a1_diagnostic.md"
    run_diagnostic(a1, cid, out_path=str(diag_path))
    csv_path = out_dir / f"{cid}_a1_sample.csv"
    n_sampled = write_audit_csv(a1["evidence"], cid, str(csv_path))
    print(f"{cid}: wrote {diag_path.name} and {csv_path.name} ({n_sampled} contradiction rows sampled)")


if __name__ == "__main__":
    main()
