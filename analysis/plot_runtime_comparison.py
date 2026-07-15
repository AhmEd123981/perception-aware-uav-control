"""Plot perception runtime per controller as a bar chart.

Reads the runtime CSV produced by the day7 thesis package and writes
results/figures/runtime_perception_comparison.png.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="results/perception_logs/day7_thesis_package/runtime_profile.csv",
    )
    parser.add_argument(
        "--out",
        default="results/figures/runtime_perception_comparison.png",
    )
    args = parser.parse_args()

    controllers, values = [], []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            controllers.append(row["controller"])
            values.append(float(row["mean_inference_time_ms"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    bars = ax.bar(controllers, values, color="#3373BF")
    ax.set_ylabel("Mean perception runtime per frame [ms]")
    ax.set_title("Perception Runtime by Controller")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
