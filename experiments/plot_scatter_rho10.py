"""
Scatter plots at rho=10 across all 14 datasets.

  Plot 1: runtime vs. number of off-diagonal measurements
  Plot 2: e_off (off-diagonal energy) vs. number of off-diagonal measurements

Run:
    python -m experiments.plot_scatter_rho10 --output-dir results/
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────

df = pd.DataFrame([
    ["adult",                 6,   0.059, 0.018, 4.6,   12.2,   0.16],
    ["SeoulBike",            10,   0.203, 0.340, 5.3,   37.1,   0.55],
    ["LifeExpectancy",       15,   0.398, 0.656, 5.1,   61.4,   1.24],
    ["Parkinsons",           16,   0.451, 0.392, 6.3,   55.9,   1.10],
    ["thyroid_ann",          21,   0.052, 0.026, 10.1,  42.5,   0.64],
    ["ibm",                  23,   0.081, 0.230, 17.6,  38.1,   0.86],
    ["pol_reg",              26,   0.103, 0.165, 9.9,  120.6,   2.76],
    ["BreastCancerWisconsin",32,   0.383, 0.739, 5.5,   49.0,   1.78],
    ["bank32nh",             32,   0.015, 0.071, 26.4,  49.7,   1.52],
    ["Ailerons",             33,   0.152, 0.746, 7.3,   95.8,   2.96],
    ["spambase",             57,   0.064, 0.069, 8.5,   16.9,   0.65],
    ["communities_crime",   102,   0.245, 0.895, 17.2, 255.3,  47.8],
    ["indian_pines",        187,   0.555, 0.859, 2.5,  518.0, 3117.0],
    ["madeline",            260,   0.016, 0.248, 201.2, 39.2,   4.16],
], columns=[
    "dataset", "d", "mean_abs_r", "e_off", "r_eff",
    "offdiag_measurements_rho10", "runtime_rho10",
])

LABEL_DATASETS = {"adult", "LifeExpectancy", "communities_crime", "indian_pines", "madeline"}

HIGHLIGHT_COLORS = {
    "adult":             "#aacc00",  # lime green
    "LifeExpectancy":    "#7b1fa2",  # violet
    "communities_crime": "#64b5f6",  # light blue
    "indian_pines":      "#00796b",  # dark green teal
    "madeline":          "#cc00aa",  # magenta
}

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


def _scatter(ax, df, x_col, y_col):
    """Plot gray unlabelled points first, then colored highlighted ones on top."""
    other = df[~df["dataset"].isin(LABEL_DATASETS)]
    highlighted = df[df["dataset"].isin(LABEL_DATASETS)]
    ax.scatter(other[x_col], other[y_col],
               s=110, color="0.75", edgecolors="0.55", linewidths=0.6, zorder=2)
    for _, r in highlighted.iterrows():
        c = HIGHLIGHT_COLORS[r["dataset"]]
        ax.scatter(r[x_col], r[y_col],
                   s=110, color=c, edgecolors=c, linewidths=0.7, zorder=3)


def _add_labels(ax, xs, ys, texts, colors, offsets=None):
    """Place bold labels near points. offsets: list of (dx, dy) in points per label."""
    if offsets is None:
        offsets = [(8, 5)] * len(xs)
    for x, y, t, c, (dx, dy) in zip(xs, ys, texts, colors, offsets):
        ax.annotate(
            t, xy=(x, y), fontsize=12,
            fontweight="bold", color=c,
            xytext=(dx, dy), textcoords="offset points",
            ha="left",
        )


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Manual offsets (dx, dy) in points for each of the 5 highlighted datasets,
# ordered as they appear in LABEL_DATASETS iteration (sorted by dataset name).
# Tuned to avoid overlap across both plots.
P1_OFFSETS = {          # plot1: x=measurements, y=runtime
    "adult":             (  8,   5),
    "LifeExpectancy":    (  8,   5),
    "communities_crime": (  8,  -14),
    "indian_pines":      (-10,   8),
    "madeline":          (  8,   5),
}
P2_OFFSETS = {          # plot2: x=e_off, y=measurements
    "adult":             (  8,   5),
    "LifeExpectancy":    (  8,  -14),
    "communities_crime": (  8,  -14),
    "indian_pines":      (  8,   5),
    "madeline":          (  8,  -16),
}


def _p1_labels(ax, hi):
    _add_labels(
        ax,
        xs=hi["offdiag_measurements_rho10"].tolist(),
        ys=hi["runtime_rho10"].tolist(),
        texts=[f'{r["dataset"]} ($d={int(r["d"])}$)' for _, r in hi.iterrows()],
        colors=[HIGHLIGHT_COLORS[r["dataset"]] for _, r in hi.iterrows()],
        offsets=[P1_OFFSETS[r["dataset"]] for _, r in hi.iterrows()],
    )


def _p2_labels(ax, hi):
    _add_labels(
        ax,
        xs=hi["e_off"].tolist(),
        ys=hi["offdiag_measurements_rho10"].tolist(),
        texts=[f'{r["dataset"]} ($d={int(r["d"])}$)' for _, r in hi.iterrows()],
        colors=[HIGHLIGHT_COLORS[r["dataset"]] for _, r in hi.iterrows()],
        offsets=[P2_OFFSETS[r["dataset"]] for _, r in hi.iterrows()],
    )


# ── Plot 1 variants: runtime vs measurements ──────────────────────────────────

def _p1_base(ax, xscale="linear", yscale="log"):
    _scatter(ax, df, "offdiag_measurements_rho10", "runtime_rho10")
    if xscale == "log": ax.set_xscale("log")
    if yscale == "log": ax.set_yscale("log")
    ax.set_xlabel(r"Off-diagonal measurements at $\rho=10$", fontsize=13)
    ax.set_ylabel(r"Runtime at $\rho=10$ (s)", fontsize=13)
    _style(ax)
    hi = df[df["dataset"].isin(LABEL_DATASETS)]
    _p1_labels(ax, hi)


def plot1_logy(output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _p1_base(ax, xscale="linear", yscale="log")
    ax.set_box_aspect(1)
    fig.tight_layout()
    _save(fig, output_dir, "scatter_p1_logy_compact")


def plot1_loglog(output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _p1_base(ax, xscale="log", yscale="log")
    ax.set_box_aspect(1)
    fig.tight_layout()
    _save(fig, output_dir, "scatter_p1_loglog_compact")


def plot1_linear(output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _p1_base(ax, xscale="linear", yscale="linear")
    ax.set_box_aspect(1)
    fig.tight_layout()
    _save(fig, output_dir, "scatter_p1_linear_compact")


# ── Plot 2 variants: e_off vs measurements ────────────────────────────────────

def _p2_base(ax, yscale="linear"):
    _scatter(ax, df, "e_off", "offdiag_measurements_rho10")
    if yscale == "log": ax.set_yscale("log")
    ax.set_xlabel(r"Off-diagonal energy $e_{\mathrm{off}}$", fontsize=13)
    ax.set_ylabel(r"Off-diagonal measurements at $\rho=10$", fontsize=13)
    _style(ax)
    hi = df[df["dataset"].isin(LABEL_DATASETS)]
    _p2_labels(ax, hi)


def plot2_linear(output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _p2_base(ax, yscale="linear")
    ax.set_box_aspect(1)
    fig.tight_layout()
    _save(fig, output_dir, "scatter_p2_linear_compact")


def plot2_logy(output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    _p2_base(ax, yscale="log")
    ax.set_box_aspect(1)
    _save(fig, output_dir, "scatter_p2_logy_compact")


def plot2_loglog(output_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _p2_base(ax, yscale="log")
    ax.set_xscale("log")
    ax.set_box_aspect(1)
    fig.tight_layout()
    _save(fig, output_dir, "scatter_p2_loglog_compact")


def _save(fig, output_dir, name):
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(p / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved: {p / name}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/",
                        help="Directory to save figures")
    args = parser.parse_args()

    plot1_logy(args.output_dir)
    plot1_loglog(args.output_dir)
    plot1_linear(args.output_dir)
    plot2_linear(args.output_dir)
    plot2_logy(args.output_dir)
    plot2_loglog(args.output_dir)
