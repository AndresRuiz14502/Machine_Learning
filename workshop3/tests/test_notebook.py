"""
Smoke test for the Workshop 3 notebook: every code cell runs top to bottom without Jupyter.

The notebook is read-only (it loads JSON results and replays one session), so executing its
code cells with ``exec`` from the notebooks folder takes a few seconds and needs no kernel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "w3_reinforcement_learning.ipynb"


def test_notebook_code_cells_execute(capsys) -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert cells, "notebook has no code cells"
    assert all(c.get("outputs", []) == [] for c in notebook["cells"] if c["cell_type"] == "code"), (
        "commit the notebook without stored outputs"
    )

    namespace: dict = {"__name__": "__main__"}
    previous = Path.cwd()
    os.chdir(NOTEBOOK.parent)  # Jupyter starts kernels in the notebook's folder
    try:
        for index, source in enumerate(cells):
            try:
                exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
            except Exception as exc:  # pragma: no cover - the assertion message carries the cell
                raise AssertionError(f"notebook code cell {index} failed: {exc!r}") from exc
    finally:
        os.chdir(previous)

    printed = capsys.readouterr().out
    assert "MovieLensRecEnv Trajectory" in printed
    assert namespace["metrics"]["num_test_users"] > 0
