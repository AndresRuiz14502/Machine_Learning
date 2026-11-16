"""
Render the Workshop 3 result tables from the JSON sources of truth.

Writes, from `workshop3/report/metrics_summary.json`:
- `workshop3/report/tables/test_results.tex`: main test table (K=10 and K=20), keeps
  `\\label{tab:test_comparison}` for the documentation consistency tests.
- `workshop3/report/tables/hyperparameters.tex`: validation results of the architecture x
  hyperparameter grid plus the champion and the gamma = 0 ablation.
- the block between `<!-- results:start -->` and `<!-- results:end -->` in `workshop3/README.md`.

Run from the repository root after `evaluate.py`:

    python workshop3/src/render_tables.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "workshop3" / "report"
TABLE_DIR = REPORT_DIR / "tables"
README = PROJECT_ROOT / "workshop3" / "README.md"
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"
ROOT_README = PROJECT_ROOT / "README.md"
ROOT_START, ROOT_END = "<!-- w3-results:start -->", "<!-- w3-results:end -->"
ROOT_POLICIES = ("D3QN", "MyopicBandit", "GreedyFunkSVD", "GreedyHybridNeuMF")

POLICY_ORDER = ("Random", "Popularity", "BiasOnly", "GreedyFunkSVD", "GreedyHybridNeuMF", "MyopicBandit", "D3QN")
METRIC_KEYS = ("mean_return", "precision_at_k", "ndcg_at_k", "map_at_k", "catalog_coverage", "mean_novelty_bits")
BEST_IS_HIGHER = {key: True for key in METRIC_KEYS}


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_names(metrics: Dict[str, Any], latex: bool) -> Dict[str, str]:
    agents = metrics.get("agents", {})
    d3qn = agents.get("D3QN", {})
    gamma = d3qn.get("gamma", "?")
    architecture = str(d3qn.get("config", {}).get("architecture", "flat"))
    arch = {"slot": "slot-shared", "flat": "flat MLP"}.get(architecture, architecture)
    g = r"$\gamma$" if latex else "γ"
    mu = r"($\mu + b_u + b_i$)" if latex else "(μ + b_u + b_i)"
    return {
        "Random": "Random",
        "Popularity": "Popularity",
        "BiasOnly": f"Bias {mu}",
        "GreedyFunkSVD": "Greedy FunkSVD (W1)",
        "GreedyHybridNeuMF": "Greedy HybridNeuMF (W2)",
        "MyopicBandit": f"Myopic DQN ({g} = 0)",
        "D3QN": f"D3QN ({g} = {gamma:g}, {arch})" if isinstance(gamma, (int, float)) else "D3QN",
    }


def _fmt(key: str, value: float, std: float | None = None, latex: bool = True) -> str:
    if key == "catalog_coverage":
        return f"{100.0 * value:.2f}" + (r"\%" if latex else "%")
    if key == "mean_return":
        pm = r" $\pm$ " if latex else " ± "
        return f"{value:.4f}" + (f"{pm}{std:.2f}" if std is not None else "")
    if key == "mean_novelty_bits":
        return f"{value:.2f}"
    return f"{value:.4f}"


def _best(rows: Dict[str, Dict[str, float]], key: str) -> str:
    return max(rows, key=lambda p: rows[p][key])


def render_test_table(metrics: Dict[str, Any]) -> str:
    names = policy_names(metrics, latex=True)
    k10, k20 = metrics["test_k10"], metrics["test_k20"]
    best10 = {key: _best(k10, key) for key in METRIC_KEYS}
    best20 = {key: _best(k20, key) for key in METRIC_KEYS}

    def cells(block: Dict[str, Dict[str, float]], policy: str, best: Dict[str, str]) -> List[str]:
        out = []
        for key in METRIC_KEYS:
            text = _fmt(key, block[policy][key], block[policy].get("std_return") if key == "mean_return" else None)
            out.append(rf"\textbf{{{text}}}" if best[key] == policy else text)
        return out

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Held-out test sessions ("
            f"{metrics['num_test_users']}"
            r" users, one episode each). Return is the mean episode return ($\pm$ SD); P, NDCG and MAP "
            r"are computed at $K$; coverage counts distinct recommended movies over the movies of the "
            r"evaluated slates; novelty is mean self-information in bits. Best value per column in bold.}"
        ),
        r"\label{tab:test_comparison}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l cccccc cccccc@{}}",
        r"\toprule",
        r" & \multicolumn{6}{c}{$K = 10$} & \multicolumn{6}{c}{$K = 20$} \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
        r"Policy & Return & P & NDCG & MAP & Cov. & Nov. & Return & P & NDCG & MAP & Cov. & Nov. \\",
        r"\midrule",
    ]
    for policy in POLICY_ORDER:
        if policy not in k10:
            continue
        row = [names[policy], *cells(k10, policy, best10), *cells(k20, policy, best20)]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    return "\n".join(lines)


def render_hyperparameter_table(metrics: Dict[str, Any]) -> str:
    rows = metrics["hyperparameter_table"]
    by_id = {r["config_id"]: r for r in rows}
    grid = [r for r in rows if re.match(r"^(flat|slot)_cfg", r["config_id"])]
    base_ids = sorted({re.sub(r"^(flat|slot)_", "", r["config_id"]) for r in grid})
    # Selection rule of train.py: highest smoothed validation return among the grid runs
    selected = max(grid, key=lambda r: r["smoothed_val_return"])["config_id"] if grid else None
    labels = {
        "cfg1_baseline": r"Base (see text)",
        "cfg2_high_lr": r"$\eta = 10^{-3}$",
        "cfg3_low_lr": r"$\eta = 10^{-4}$",
        "cfg4_gamma_090": r"$\gamma = 0.90$",
        "cfg5_gamma_099": r"$\gamma = 0.99$",
        "cfg6_fast_eps": r"$\delta_\epsilon = 0.990$",
        "cfg7_slow_eps": r"$\delta_\epsilon = 0.998$",
        "cfg8_high_l2": r"$\lambda = 10^{-4}$",
    }

    def cell(config_id: str) -> List[str]:
        r = by_id.get(config_id)
        if r is None:
            return ["--", "--"]
        smooth = f"{r['smoothed_val_return']:.4f}"
        if config_id == selected:
            smooth = rf"\textbf{{{smooth}}}"
        return [smooth, f"{r['best_val_return']:.4f}"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Validation return of the grid (800 episodes per run; smoothed = mean of the last five "
            r"greedy evaluations on the validation sessions; best = maximum over training). Bold: selected run.}"
        ),
        r"\label{tab:hyperparameters}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l cc cc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Flat MLP} & \multicolumn{2}{c}{Slot-shared} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Configuration & Smoothed & Best & Smoothed & Best \\",
        r"\midrule",
    ]
    for base in base_ids:
        lines.append(" & ".join([labels.get(base, base), *cell(f"flat_{base}"), *cell(f"slot_{base}")]) + r" \\")
    lines.append(r"\midrule")
    for config_id, label in (("d3qn_tuned", "Champion (1{,}500 ep.)"), ("bandit_gamma0", r"$\gamma = 0$ ablation (1{,}500 ep.)")):
        r = by_id.get(config_id)
        if r is not None:
            arch = "Flat MLP" if r.get("architecture") == "flat" else "Slot-shared"
            lines.append(
                rf"{label}, {arch} & \multicolumn{{2}}{{c}}{{{r['smoothed_val_return']:.4f} / {r['best_val_return']:.4f}}} & & \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def render_readme_block(metrics: Dict[str, Any]) -> str:
    names = policy_names(metrics, latex=False)
    out = [README_START, ""]
    for horizon, title in (("test_k10", "K = 10"), ("test_k20", "K = 20")):
        block = metrics[horizon]
        k = 10 if horizon == "test_k10" else 20
        out += [
            f"**{title}** ({metrics['num_test_users']} held-out test sessions, one episode each)",
            "",
            f"| Policy | Return (mean ± SD) | P@{k} | R@{k} | NDCG@{k} | MAP@{k} | HR@{k} | Coverage | Novelty (bits) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for policy in POLICY_ORDER:
            if policy not in block:
                continue
            m = block[policy]
            out.append(
                f"| {names[policy]} | {_fmt('mean_return', m['mean_return'], m['std_return'], latex=False)} | "
                f"{m['precision_at_k']:.4f} | {m['recall_at_k']:.4f} | {m['ndcg_at_k']:.4f} | {m['map_at_k']:.4f} | "
                f"{m['hr_at_k']:.4f} | {_fmt('catalog_coverage', m['catalog_coverage'], latex=False)} | "
                f"{m['mean_novelty_bits']:.2f} |"
            )
        out.append("")
    w1 = metrics.get("workshop1_comparison", {}).get("w1_report_funksvd", {})
    w2 = metrics.get("workshop2_comparison", {}).get("w2_hybrid_neumf_static", {})
    if w1 and w2:
        out += [
            "**Static references (different protocol).** Workshop 1 and Workshop 2 report one-shot rankings in "
            "which each test user's liked items compete with sampled unrated movies, so these numbers give "
            "context and are not comparable one-to-one with the sessions above:",
            "",
            "| Model (own report) | Test RMSE | P@10 | NDCG@10 | HR@10 |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| FunkSVD (Workshop 1; RMSE for k = 50) | {w1['test_rmse_k50']:.4f} | {w1['precision_at_10']:.4f} | "
            f"{w1['ndcg_at_10']:.4f} | {w1['hr_at_10']:.4f} |",
            f"| HybridNeuMF (Workshop 2) | {w2['test_rmse']:.4f} | {w2['precision_at_10']:.4f} | "
            f"{w2['ndcg_at_10']:.4f} | {w2['hr_at_10']:.4f} |",
            "",
        ]
    paired = metrics.get("paired_comparisons", {})
    pair_rows = [(k, v) for k, v in paired.items() if isinstance(v, dict) and "policy_a" in v]
    if pair_rows:
        out += [
            "**Paired per-user differences at K = 10** (A − B; paired bootstrap 95% CI, Wilcoxon signed-rank p-value)",
            "",
            "| A − B | Return diff [95% CI] | p | NDCG@10 diff [95% CI] | p |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for _, v in pair_rows:
            r, n = v["return"], v["ndcg_at_10"]
            out.append(
                f"| {names[v['policy_a']]} − {names[v['policy_b']]} | {r['mean_diff']:.4f} "
                f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | {r['wilcoxon_p_value']:.4f} | "
                f"{n['mean_diff']:.4f} [{n['ci95_low']:.4f}, {n['ci95_high']:.4f}] | {n['wilcoxon_p_value']:.4f} |"
            )
        out.append("")
    out.append(README_END)
    return "\n".join(out)


def render_root_block(metrics: Dict[str, Any]) -> str:
    """Compact Workshop 3 headline table for the repository-level README."""
    names = policy_names(metrics, latex=False)
    k10, k20 = metrics["test_k10"], metrics["test_k20"]
    out = [
        ROOT_START,
        "",
        f"Held-out test sessions ({metrics['num_test_users']} users, one episode each, K = 10; "
        "coverage also at K = 20). Full tables and paired statistics: `workshop3/README.md`.",
        "",
        "| Policy | Return (mean ± SD) | P@10 | NDCG@10 | MAP@10 | Coverage@10 | Coverage@20 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for policy in ROOT_POLICIES:
        if policy not in k10:
            continue
        m = k10[policy]
        out.append(
            f"| {names[policy]} | {_fmt('mean_return', m['mean_return'], m['std_return'], latex=False)} | "
            f"{m['precision_at_k']:.4f} | {m['ndcg_at_k']:.4f} | {m['map_at_k']:.4f} | "
            f"{_fmt('catalog_coverage', m['catalog_coverage'], latex=False)} | "
            f"{_fmt('catalog_coverage', k20[policy]['catalog_coverage'], latex=False)} |"
        )
    out += ["", ROOT_END]
    return "\n".join(out)


def update_block(path: Path, start: str, end: str, block: str) -> bool:
    """Replace the text between ``start`` and ``end`` markers in ``path`` (False if absent)."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), flags=re.S)
    if not pattern.search(text):
        return False
    path.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    return True


def main() -> None:
    metrics = _load(REPORT_DIR / "metrics_summary.json")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "test_results.tex").write_text(render_test_table(metrics), encoding="utf-8")
    (TABLE_DIR / "hyperparameters.tex").write_text(render_hyperparameter_table(metrics), encoding="utf-8")
    w3_readme = update_block(README, README_START, README_END, render_readme_block(metrics))
    root_readme = update_block(ROOT_README, ROOT_START, ROOT_END, render_root_block(metrics))
    print("[render_tables] wrote report/tables/test_results.tex and report/tables/hyperparameters.tex")
    print(f"[render_tables] workshop3/README.md results block {'updated' if w3_readme else 'markers not found'}")
    print(f"[render_tables] README.md Workshop 3 block {'updated' if root_readme else 'markers not found'}")


if __name__ == "__main__":
    main()
