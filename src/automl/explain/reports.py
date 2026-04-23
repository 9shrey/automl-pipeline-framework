"""Render markdown explainability reports.

No PNG dependency — we keep matplotlib optional and emit a markdown table
that renders nicely in GitHub / VS Code. The CSV next to it is the source of
truth for downstream tooling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["write_importance_report"]


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        # tabulate is missing — fall back to a CSV-style code block.
        return "```\n" + df.to_csv(index=False) + "```"


def write_importance_report(
    *,
    out_dir: Path,
    permutation: pd.DataFrame | None = None,
    shap: pd.DataFrame | None = None,
    top_n: int = 20,
) -> dict[str, Path]:
    """Write CSVs + a single ``explain.md`` summary into ``out_dir``.

    Returns a mapping of artifact name → path for any artifacts that were
    actually written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    sections: list[str] = ["# Explainability Report\n"]

    if permutation is not None and not permutation.empty:
        p = out_dir / "permutation_importance.csv"
        permutation.to_csv(p, index=False)
        written["permutation"] = p
        sections.append("## Permutation importance (top {n})\n\n{tbl}\n".format(
            n=top_n, tbl=_df_to_markdown(permutation.head(top_n))
        ))

    if shap is not None and not shap.empty:
        p = out_dir / "shap_importance.csv"
        shap.to_csv(p, index=False)
        written["shap"] = p
        sections.append("## SHAP importance (top {n})\n\n{tbl}\n".format(
            n=top_n, tbl=_df_to_markdown(shap.head(top_n))
        ))

    if len(sections) == 1:
        sections.append("_no importance signals were produced._\n")

    md_path = out_dir / "explain.md"
    md_path.write_text("\n".join(sections), encoding="utf-8")
    written["explain"] = md_path
    return written

