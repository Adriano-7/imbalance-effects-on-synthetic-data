"""Script to find balanced binary tabular datasets in OpenML 
Example:
    uv run python src/find_balanced_binary_datasets.py
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import openml
from dotenv import load_dotenv


DEFAULT_OUTPUT = Path("results/balanced_binary_datasets.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-instances", type=int, default=500)
    parser.add_argument("--max-instances", type=int, default=10_000)
    parser.add_argument(
        "--min-minority-pct",
        type=float,
        default=45.0,
        help="Minimum percentage represented by the smaller class (default: 35).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of candidates to include; omit to report every match.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown report path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def find_candidates(args: argparse.Namespace):
    datasets = openml.datasets.list_datasets(
        number_classes=2,
        number_instances=f"{args.min_instances}..{args.max_instances}",
        output_format="dataframe",
    )
    required = [
        "did",
        "name",
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures",
        "MajorityClassSize",
        "MinorityClassSize",
    ]
    candidates = datasets.dropna(subset=required).copy()
    candidates["minority_pct"] = (
        100 * candidates["MinorityClassSize"] / candidates["NumberOfInstances"]
    )
    candidates["natural_ir"] = (
        candidates["MajorityClassSize"] / candidates["MinorityClassSize"]
    )
    candidates = candidates[candidates["minority_pct"] >= args.min_minority_pct]
    candidates = candidates.sort_values(
        ["minority_pct", "NumberOfInstances"], ascending=[False, False]
    ).drop_duplicates(subset="name", keep="first")
    return candidates


def write_report(candidates, args: argparse.Namespace) -> None:
    columns = [
        "did",
        "name",
        "NumberOfInstances",
        "NumberOfFeatures",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures",
        "MinorityClassSize",
        "MajorityClassSize",
        "minority_pct",
        "natural_ir",
    ]
    report = candidates[columns].copy()
    if args.limit is not None:
        report = report.head(args.limit)
    report["minority_pct"] = report["minority_pct"].round(1)
    report["natural_ir"] = report["natural_ir"].round(2)
    report["OpenML"] = report["did"].map(
        lambda dataset_id: f"[dataset {dataset_id}](https://www.openml.org/d/{dataset_id})"
    )
    report = report.rename(
        columns={
            "name": "Dataset",
            "NumberOfInstances": "Rows",
            "NumberOfFeatures": "Features",
            "NumberOfNumericFeatures": "Numeric features",
            "NumberOfSymbolicFeatures": "Categorical features",
            "MinorityClassSize": "Minority rows",
            "MajorityClassSize": "Majority rows",
            "minority_pct": "Minority %",
            "natural_ir": "Natural IR",
        }
    )
    report = report.drop(columns="did")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = "\n".join(
        [
            "# Balanced Binary Dataset Candidates",
            "",
            f"Generated: {generated_at}",
            "",
            "Selection criteria:",
            f"- Binary classification; {args.min_instances:,}–{args.max_instances:,} rows",
            f"- Minority class at least {args.min_minority_pct:g}% of rows",
            "- One row per OpenML dataset name; sorted by class balance, then size",
            "",
            report.to_markdown(index=False),
            "",
            (
                f"{len(candidates)} candidates matched; showing the first {len(report)}."
                if args.limit is not None
                else f"{len(candidates)} candidates matched; showing all candidates."
            ),
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {len(report)} candidates to {args.output}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    openml.config.apikey = os.getenv("OPENML_API_KEY")
    if not openml.config.apikey:
        raise RuntimeError("OPENML_API_KEY is not set. Add it to .env before running this script.")

    write_report(find_candidates(args), args)


if __name__ == "__main__":
    main()
