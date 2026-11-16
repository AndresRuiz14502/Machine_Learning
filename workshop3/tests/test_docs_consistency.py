"""
Documentation consistency tests for Workshop 3.

Every number, path, and artifact claimed by the Workshop 3 documentation (README files,
IEEE report, and notebook) must agree with the machine-generated sources of truth:

- ``workshop3/report/metrics_summary.json``   (test evaluation, W1/W2 references, tuning)
- ``workshop3/data/dataset_stats.json``       (cohort and split statistics)
- ``workshop3/report/random_diagnostics.json`` (100-episode random-policy validation)
- ``runs/evaluation_summary.json``            (official Workshop 2 results on ``main``)

Conventions enforced here (documentation writers must follow them):

- Ranking metrics are written with 4 decimals (``0.9818``); returns and novelty with
  4 or 2 decimals; catalog coverage as a percentage with 2 decimals (``36.64%``).
- Every number written with exactly 4 decimals must be traceable to one of the JSON
  sources above (or to ``DERIVED_CONSTANTS``). Derived quantities such as relative
  improvements or ratios must be written with at most 2 decimals.
- The main results table of the paper keeps ``\\label{tab:test_comparison}``.

Run from the repository root:

    python -m pytest workshop3/tests/test_docs_consistency.py -v
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
W3_DIR = REPO_ROOT / "workshop3"
REPORT_DIR = W3_DIR / "report"
W3_README = W3_DIR / "README.md"
ROOT_README = REPO_ROOT / "README.md"
NOTEBOOK = W3_DIR / "notebooks" / "w3_reinforcement_learning.ipynb"
PAPER_PDF = W3_DIR / "group7_workshop3.pdf"
W2_SUMMARY = REPO_ROOT / "runs" / "evaluation_summary.json"

GITHUB_URL = "github.com/AndresRuiz14502/Machine_Learning"
OFFICIAL_MOVIE_COUNT = "87,585"
STALE_MOVIE_COUNT = "84,432"
MAX_PAPER_PAGES = 6

# metrics_summary.json policy key -> regex identifying that policy's table row.
POLICY_ROW_PATTERNS = {
    "Random": r"\bRandom\b",
    "BiasOnly": r"\bBias\b",
    "Popularity": r"\bPopularity\b",
    "GreedyFunkSVD": r"Greedy\s+FunkSVD",
    "GreedyHybridNeuMF": r"Greedy\s+HybridNeuMF",
    "MyopicBandit": r"\bMyopic\b",
    "D3QN": r"\bD3QN\b",
}

# Metrics that every policy row of a results table must report.
RESULT_METRICS = (
    "mean_return",
    "precision_at_k",
    "ndcg_at_k",
    "map_at_k",
    "catalog_coverage",
    "mean_novelty_bits",
)

# Mathematical constants allowed with 4 decimals without being stored in a JSON source.
DERIVED_CONSTANTS = {
    "5.6439",  # log2(50): entropy of a uniform policy over the 50 candidate slots
}

# Artifacts the documentation declares as versioned (must exist and must not be ignored).
VERSIONED_ARTIFACTS = (
    "workshop3/README.md",
    "workshop3/requirements.txt",
    "workshop3/group7_workshop3.pdf",
    "workshop3/checkpoints/agent_best.pt",
    "workshop3/data/processed_cache.npz",
    "workshop3/data/dataset_stats.json",
    "workshop3/data/movie_lookup.json",
    "workshop3/notebooks/w3_reinforcement_learning.ipynb",
    "workshop3/report/main.tex",
    "workshop3/report/bibliography.bib",
    "workshop3/report/metrics_summary.json",
    "workshop3/report/training_logs.json",
    "workshop3/report/random_diagnostics.json",
)

# Local-only or regenerable artifacts that must stay out of version control.
IGNORED_ARTIFACTS = (
    "workshop3/tools/tectonic.exe",
    "workshop3/data/ml-32m/ml-32m/ratings.csv",
    "dataSet/ratings.csv",
    "workshop3/report/main.pdf",
    "workshop3/report/main.aux",
    "workshop3/report/main.log",
    "workshop3/src/__pycache__/env.cpython-312.pyc",
)

# Import name -> distribution name expected in workshop3/requirements.txt.
IMPORT_TO_DISTRIBUTION = {
    "gymnasium": "gymnasium",
    "IPython": "ipython",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pytest": "pytest",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "tqdm": "tqdm",
}
LOCAL_IMPORT_ROOTS = {"workshop3", "src", "env", "agent", "baselines", "train", "evaluate", "data_prep"}

FOUR_DECIMALS = re.compile(r"(?<![\w.])\d+\.\d{4}(?!\d)")


# --------------------------------------------------------------------------------------
# Source-of-truth loaders
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics() -> dict:
    return _load_json(REPORT_DIR / "metrics_summary.json")


def _dataset_stats() -> dict:
    return _load_json(W3_DIR / "data" / "dataset_stats.json")


def _random_diagnostics() -> dict:
    return _load_json(REPORT_DIR / "random_diagnostics.json")


def _walk_numbers(obj: object) -> Iterator[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_numbers(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_numbers(value)


@lru_cache(maxsize=None)
def _known_four_decimal_values() -> frozenset:
    known = set(DERIVED_CONSTANTS)
    sources = [_metrics(), _dataset_stats(), _random_diagnostics()]
    if W2_SUMMARY.exists():
        sources.append(_load_json(W2_SUMMARY))  # official Workshop 1/2 benchmarks on main
    for source in sources:
        known.update(f"{abs(v):.4f}" for v in _walk_numbers(source))
    return frozenset(known)


# --------------------------------------------------------------------------------------
# Document loaders
# --------------------------------------------------------------------------------------
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _normalize(text: str) -> str:
    """Strip Markdown/LaTeX formatting so numbers can be matched as plain text."""
    for old, new in (
        ("{,}", ","),
        ("\\_", "_"),
        ("\\,", ""),
        ("\\%", "%"),
        ("$", ""),
        ("**", ""),
        ("~", " "),
        ("\\times", "x"),
    ):
        text = text.replace(old, new)
    return text


def _strip_tex_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())


def _resolve_tex_inputs(path: Path, depth: int = 0) -> str:
    if depth > 5 or not path.exists():
        return ""
    tex = _strip_tex_comments(path.read_text(encoding="utf-8"))

    def _replace(match: re.Match) -> str:
        target = REPORT_DIR / match.group(1)
        if target.suffix != ".tex":
            target = target.with_suffix(".tex")
        return _resolve_tex_inputs(target, depth + 1)

    return re.sub(r"\\(?:input|include)\{([^}]+)\}", _replace, tex)


@lru_cache(maxsize=None)
def _paper_tex() -> str:
    return _resolve_tex_inputs(REPORT_DIR / "main.tex")


def _root_readme_w3_section() -> str:
    text = _read(ROOT_README)
    heading = re.search(r"^(#{2,3})\s+[^\n]*Workshop\s*3\b[^\n]*$", text, flags=re.M)
    if heading is None:
        return ""
    level = len(heading.group(1))
    rest = text[heading.end():]
    following = re.search(rf"^#{{1,{level}}}\s", rest, flags=re.M)
    end = heading.end() + (following.start() if following else len(rest))
    return text[heading.start():end]


def _notebook_sources(cell_type: str) -> str:
    if not NOTEBOOK.exists():
        return ""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == cell_type
    )


def _documents() -> dict:
    return {
        "workshop3/README.md": _read(W3_README),
        "README.md (Workshop 3 section)": _root_readme_w3_section(),
        "workshop3/report (main.tex + sections)": _paper_tex(),
        "workshop3 notebook (markdown cells)": _notebook_sources("markdown"),
    }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _format_options(metric: str, value: float) -> tuple:
    if metric == "catalog_coverage":
        return (f"{100.0 * value:.2f}%",)
    if metric in ("mean_return", "std_return", "mean_novelty_bits"):
        return (f"{value:.4f}", f"{value:.2f}")
    return (f"{value:.4f}",)


def _missing_policy_values(rows: Iterable[str], horizons: tuple) -> list:
    rows = list(rows)
    metrics = _metrics()
    problems = []
    for policy in metrics[horizons[0]]:
        pattern = POLICY_ROW_PATTERNS.get(policy)
        if pattern is None:
            problems.append(f"{policy}: register a row pattern in POLICY_ROW_PATTERNS")
            continue
        row_text = _normalize(" ".join(row for row in rows if re.search(pattern, row)))
        if not row_text.strip():
            problems.append(f"{policy}: no table row found")
            continue
        for horizon in horizons:
            for metric in RESULT_METRICS:
                options = _format_options(metric, metrics[horizon][policy][metric])
                if not any(option in row_text for option in options):
                    problems.append(f"{policy} [{horizon}] {metric}: expected one of {options}")
    return problems


def _markdown_table_rows(text: str) -> list:
    return [line for line in text.splitlines() if line.lstrip().startswith("|")]


def _paper_results_table_rows() -> list:
    for match in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", _paper_tex(), flags=re.S):
        if "\\label{tab:test_comparison}" in match.group(1):
            return [line for line in match.group(1).splitlines() if "&" in line]
    return []


def _is_ignored(rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", rel_path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        pytest.skip(f"git check-ignore unavailable: {result.stderr.decode(errors='replace')}")
    return result.returncode == 0


def _tracked_files(prefix: str) -> list:
    result = subprocess.run(["git", "ls-files", prefix], cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("git ls-files unavailable")
    return result.stdout.splitlines()


def _w3_readme_tree() -> list:
    """Parse the ``workshop3/`` tree block of the README into (path, is_dir, comment)."""
    block = re.search(r"```text\s*\n(workshop3/\s*\n.*?)```", _read(W3_README), flags=re.S)
    if block is None:
        return []
    line_re = re.compile(r"^(?P<indent>(?:[│|] {3}| {4})*)[├└]── (?P<rest>.+)$")
    entries, stack = [], []
    for line in block.group(1).splitlines()[1:]:
        match = line_re.match(line.rstrip())
        if match is None:
            continue
        depth = len(match.group("indent")) // 4
        name_part, _, comment = match.group("rest").partition("#")
        name = re.split(r"\s{2,}", name_part.strip())[0]
        stack = stack[:depth] + [name.rstrip("/")]
        entries.append(("/".join(["workshop3", *stack]), name.endswith("/"), comment.strip().lower()))
    return entries


def _pdf_page_count(path: Path) -> int:
    """Return /Count of the root /Pages node (also searched inside compressed object streams)."""
    data = path.read_bytes()
    blobs = [data]
    for match in re.finditer(rb"(?<!end)stream\r?\n", data):
        if b"/ObjStm" not in data[max(0, match.start() - 400):match.start()]:
            continue
        end = data.find(b"endstream", match.end())
        if end == -1:
            continue
        try:
            blobs.append(zlib.decompressobj().decompress(data[match.end():end]))
        except zlib.error:
            continue
    blob = b"\n".join(blobs)
    roots = [
        int(count)
        for body in re.findall(rb"<<([^<>]*?/Type\s*/Pages\b[^<>]*?)>>", blob, flags=re.S)
        if b"/Parent" not in body
        for count in re.findall(rb"/Count\s+(\d+)", body)[:1]
    ]
    if roots:
        return max(roots)
    return len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", blob))


# --------------------------------------------------------------------------------------
# Dataset statistics
# --------------------------------------------------------------------------------------
def test_movie_count_is_official_everywhere() -> None:
    problems = [name for name, text in _documents().items() if STALE_MOVIE_COUNT in _normalize(text)]
    assert not problems, f"Stale movie count {STALE_MOVIE_COUNT} found in: {problems}"
    for name in ("workshop3/README.md", "workshop3/report (main.tex + sections)"):
        assert OFFICIAL_MOVIE_COUNT in _normalize(_documents()[name]), f"{name} must cite {OFFICIAL_MOVIE_COUNT} movies"


def test_dataset_statistics_match_dataset_stats_json() -> None:
    stats = _dataset_stats()
    expected = {
        "workshop3/README.md": ("total_raw_ratings", "total_raw_users", "filtered_ratings", "filtered_users", "filtered_movies"),
        "workshop3/report (main.tex + sections)": ("filtered_ratings", "filtered_users", "filtered_movies"),
    }
    problems = []
    for name, keys in expected.items():
        text = _normalize(_documents()[name])
        problems += [f"{name}: missing {key}={stats[key]:,}" for key in keys if f"{stats[key]:,}" not in text]
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------------------
# Results tables and headline numbers
# --------------------------------------------------------------------------------------
def test_w3_readme_results_table_matches_metrics_summary() -> None:
    problems = _missing_policy_values(_markdown_table_rows(_read(W3_README)), ("test_k10",))
    assert not problems, "\n".join(problems)


def test_w3_readme_reports_k20_headline() -> None:
    d3qn = _metrics()["test_k20"]["D3QN"]
    text = _normalize(_read(W3_README))
    problems = [
        f"D3QN [test_k20] {metric}: expected one of {_format_options(metric, d3qn[metric])}"
        for metric in ("mean_return", "ndcg_at_k", "catalog_coverage")
        if not any(option in text for option in _format_options(metric, d3qn[metric]))
    ]
    assert not problems, "\n".join(problems)


def test_paper_results_table_matches_metrics_summary() -> None:
    rows = _paper_results_table_rows()
    assert rows, "Paper results table with \\label{tab:test_comparison} not found"
    problems = _missing_policy_values(rows, ("test_k10", "test_k20"))
    assert not problems, "\n".join(problems)


def test_root_readme_has_consistent_workshop3_section() -> None:
    section = _normalize(_root_readme_w3_section())
    assert section, "Root README.md needs a '## ... Workshop 3 ...' section"
    k10, k20 = _metrics()["test_k10"]["D3QN"], _metrics()["test_k20"]["D3QN"]
    checks = [(f"[test_k10] {m}", _format_options(m, k10[m])) for m in RESULT_METRICS if m != "mean_novelty_bits"]
    checks.append(("[test_k20] catalog_coverage", _format_options("catalog_coverage", k20["catalog_coverage"])))
    problems = [f"D3QN {label}: expected one of {opts}" for label, opts in checks if not any(o in section for o in opts)]
    for needle in ("agent_best.pt", "workshop3/README.md"):
        if needle not in section:
            problems.append(f"missing reference to {needle}")
    assert not problems, "\n".join(problems)


def test_every_four_decimal_number_is_traceable_to_a_json_source() -> None:
    known = _known_four_decimal_values()
    problems = []
    for name, text in _documents().items():
        untraceable = sorted({n for n in FOUR_DECIMALS.findall(_normalize(text)) if n not in known})
        if untraceable:
            problems.append(f"{name}: {untraceable}")
    assert not problems, "Numbers not found in any JSON source:\n" + "\n".join(problems)


# --------------------------------------------------------------------------------------
# Workshop 1 / Workshop 2 references
# --------------------------------------------------------------------------------------
def test_metrics_summary_has_workshop2_reference_matching_main() -> None:
    block = _metrics().get("workshop2_comparison", {}).get("w2_hybrid_neumf_full_catalog")
    assert block, "metrics_summary.json lacks workshop2_comparison.w2_hybrid_neumf_full_catalog"
    official = _load_json(W2_SUMMARY)
    ranking = official["ranking_comparison"]["HybridNeuMF (W2 Deep CF)"]
    expected = {
        "test_rmse": official["hybrid_test_metrics"]["RMSE"],
        "test_mae": official["hybrid_test_metrics"]["MAE"],
        "roc_auc": official["hybrid_test_metrics"]["roc_auc"],
        "precision_at_10": ranking["P@10"],
        "ndcg_at_10": ranking["NDCG@10"],
        "map_at_10": ranking["MAP@10"],
        "hr_at_10": ranking["HitRate@10"],
    }
    problems = [
        f"{key}: {block.get(key)} != {round(value, 4)}"
        for key, value in expected.items()
        if block.get(key) is None or abs(block[key] - round(value, 4)) > 1e-4
    ]
    assert not problems, "\n".join(problems)


def test_docs_cite_workshop2_reference_numbers() -> None:
    block = _metrics().get("workshop2_comparison", {}).get("w2_hybrid_neumf_full_catalog", {})
    assert block, "metrics_summary.json lacks the Workshop 2 reference block"
    problems = []
    for name in ("workshop3/README.md", "workshop3/report (main.tex + sections)"):
        text = _normalize(_documents()[name])
        for key in ("test_rmse", "ndcg_at_10"):
            if f"{block[key]:.4f}" not in text:
                problems.append(f"{name}: missing W2 {key}={block[key]:.4f}")
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------------------
# Paths, artifacts, and version control
# --------------------------------------------------------------------------------------
def test_w3_readme_tree_paths_exist() -> None:
    entries = _w3_readme_tree()
    assert entries, "workshop3/README.md must contain a ```text tree starting with 'workshop3/'"
    problems = []
    for path, is_dir, comment in entries:
        if any(token in path for token in ("..", "*", "<", "{")):
            continue
        if any(flag in comment for flag in ("git-ignored", "not versioned", "local only")):
            if not _is_ignored(path + ("/" if is_dir else "")):
                problems.append(f"{path}: documented as ignored but not ignored by .gitignore")
        elif not (REPO_ROOT / path).exists():
            problems.append(f"{path}: listed in README tree but does not exist")
    assert not problems, "\n".join(problems)


def test_w3_readme_tree_lists_every_tracked_directory() -> None:
    tracked_dirs = {p.split("/")[1] for p in _tracked_files("workshop3") if p.count("/") >= 2}
    listed = {path.split("/")[1] for path, _, _ in _w3_readme_tree() if path.count("/") >= 1}
    assert tracked_dirs <= listed, f"Tracked directories missing from README tree: {sorted(tracked_dirs - listed)}"


def test_referenced_workshop3_files_exist() -> None:
    problems = []
    pattern = re.compile(r"workshop3[\\/][\w./\\-]+?\.(?:py|ipynb|json|pt|npz|pdf|tex|png|txt|md)\b")
    for name, text in _documents().items():
        for ref in sorted({m.replace("\\", "/") for m in pattern.findall(text)}):
            if not (REPO_ROOT / ref).exists():
                problems.append(f"{name}: {ref}")
    assert not problems, "Referenced files that do not exist:\n" + "\n".join(problems)


def test_mentioned_checkpoints_exist() -> None:
    existing = {p.name for p in REPO_ROOT.rglob("*.pt") if ".git" not in p.parts}
    texts = {**_documents(), "README.md": _read(ROOT_README), "notebook code": _notebook_sources("code")}
    problems = [
        f"{name}: {ckpt}"
        for name, text in texts.items()
        for ckpt in sorted(set(re.findall(r"[\w-]+\.pt\b", _normalize(text))))
        if ckpt not in existing
    ]
    assert not problems, "Checkpoints mentioned but missing:\n" + "\n".join(problems)


def test_versioned_artifacts_exist_and_are_not_ignored() -> None:
    problems = []
    for rel in VERSIONED_ARTIFACTS:
        if not (REPO_ROOT / rel).exists():
            problems.append(f"{rel}: missing")
        elif _is_ignored(rel):
            problems.append(f"{rel}: ignored by .gitignore")
    events = sorted((W3_DIR / "runs" / "workshop3").glob("*/events.out.tfevents.*"))
    if not events:
        problems.append("workshop3/runs/workshop3/: no TensorBoard event files")
    problems += [f"{e.relative_to(REPO_ROOT).as_posix()}: ignored" for e in events if _is_ignored(e.relative_to(REPO_ROOT).as_posix())]
    assert not problems, "\n".join(problems)


def test_local_and_regenerable_artifacts_are_ignored() -> None:
    not_ignored = [rel for rel in IGNORED_ARTIFACTS if not _is_ignored(rel)]
    assert not not_ignored, f"Should be ignored by the root .gitignore: {not_ignored}"


def test_single_gitignore_at_repository_root() -> None:
    assert not (W3_DIR / ".gitignore").exists(), "Merge workshop3/.gitignore into the root .gitignore"


def test_single_compiled_report_pdf() -> None:
    pdfs = sorted(
        p.relative_to(REPO_ROOT).as_posix() for p in REPO_ROOT.rglob("group7_workshop3.pdf") if ".git" not in p.parts
    )
    assert pdfs == ["workshop3/group7_workshop3.pdf"], f"Expected a single compiled report, found {pdfs}"


def test_report_json_has_no_absolute_local_paths() -> None:
    offenders = [
        name
        for name in ("metrics_summary.json", "training_logs.json", "random_diagnostics.json")
        if re.search(r"[A-Za-z]:\\\\Users|/home/|/Users/", _read(REPORT_DIR / name))
    ]
    assert not offenders, f"Store repository-relative paths instead of absolute local paths in: {offenders}"


def test_requirements_cover_workshop3_imports() -> None:
    raw = (W3_DIR / "requirements.txt").read_bytes()
    assert not raw.startswith((b"\xff\xfe", b"\xfe\xff")), (
        "workshop3/requirements.txt must be UTF-8 (PowerShell `pip freeze >` writes UTF-16)"
    )
    req_text = raw.decode("utf-8")
    assert "+cu" not in req_text, "Pin torch without a local CUDA suffix (document CUDA wheels in the README)"
    declared = {
        re.split(r"[<>=!~\[; ]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")
        for line in req_text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    }
    sources = [p.read_text(encoding="utf-8") for p in sorted(W3_DIR.glob("src/*.py")) + sorted(W3_DIR.glob("tests/*.py"))]
    sources += [_read(REPO_ROOT / "src" / name) for name in ("baselines.py", "data_loader.py", "features.py")]
    sources.append(_notebook_sources("code"))
    imported = set()
    for source in sources:
        imported.update(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", source, flags=re.M))
    missing = sorted(
        IMPORT_TO_DISTRIBUTION[name].lower()
        for name in imported
        if name in IMPORT_TO_DISTRIBUTION and IMPORT_TO_DISTRIBUTION[name].lower() not in declared
    )
    if any("torch.utils.tensorboard" in s for s in sources) and "tensorboard" not in declared:
        missing.append("tensorboard")
    unknown = sorted(
        name
        for name in imported
        if name not in IMPORT_TO_DISTRIBUTION
        and name not in LOCAL_IMPORT_ROOTS
        and name not in sys.stdlib_module_names
        and name != "__future__"
    )
    assert not missing, f"workshop3/requirements.txt is missing: {missing}"
    assert not unknown, f"Unmapped third-party imports (add to IMPORT_TO_DISTRIBUTION): {unknown}"


# --------------------------------------------------------------------------------------
# IEEE report (course template and guidelines)
# --------------------------------------------------------------------------------------
def test_paper_follows_course_template_sections() -> None:
    tex = _paper_tex()
    missing = [
        title
        for title in ("Introduction", "Methods and Materials", "Results", "Conclusion")
        if not re.search(rf"\\section\*?\{{[^}}]*{title}", tex)
    ]
    assert not missing, f"Course template sections missing: {missing}"


def test_paper_abstract_has_three_sentences() -> None:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", _paper_tex(), flags=re.S)
    assert match, "abstract not found"
    text = re.sub(r"\$[^$]*\$", "X", match.group(1)).strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]
    assert len(sentences) == 3, f"Abstract must have 3 sentences (course guideline), found {len(sentences)}"


def test_paper_links_github_repository() -> None:
    for name in ("workshop3/README.md", "workshop3/report (main.tex + sections)"):
        assert GITHUB_URL in _documents()[name], f"{name} must link https://{GITHUB_URL}"


def test_paper_citations_and_figures_resolve() -> None:
    tex = _paper_tex()
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", _read(REPORT_DIR / "bibliography.bib")))
    cited = {
        key.strip()
        for group in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", tex)
        for key in group.split(",")
    }
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    missing_figures = [f for f in figures if not (REPORT_DIR / f).exists() and not (REPORT_DIR / f"{f}.png").exists()]
    assert not (cited - bib_keys), f"Citations without BibTeX entry: {sorted(cited - bib_keys)}"
    assert not missing_figures, f"Missing figure files: {missing_figures}"


def test_compiled_paper_has_at_most_six_pages() -> None:
    assert PAPER_PDF.exists(), f"{PAPER_PDF.relative_to(REPO_ROOT)} not found"
    pages = _pdf_page_count(PAPER_PDF)
    assert 0 < pages <= MAX_PAPER_PAGES, f"Paper has {pages} pages; course guideline allows {MAX_PAPER_PAGES}"


# --------------------------------------------------------------------------------------
# Notebook
# --------------------------------------------------------------------------------------
def test_notebook_reads_results_from_json_and_imports_resolve() -> None:
    code = _notebook_sources("code")
    assert "metrics_summary.json" in code, "Notebook must load results from metrics_summary.json"
    sys.path[:0] = [str(REPO_ROOT), str(W3_DIR / "src")]
    try:
        modules = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", code, flags=re.M))
        unresolved = sorted(m for m in modules if importlib.util.find_spec(m.split(".")[0]) is None)
        local = sorted(
            m for m in modules if m.startswith("workshop3.") and importlib.util.find_spec(m) is None
        )
    finally:
        del sys.path[:2]
    assert not unresolved and not local, f"Notebook imports that do not resolve: {unresolved + local}"
