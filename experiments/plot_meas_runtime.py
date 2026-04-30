"""
Three publication figures for measurements and runtime analysis.

  Figure 1: off-diagonal measurement count vs rho  (line plot + side table)
  Figure 2: runtime vs rho                          (line plot + side table)
  Figure 3: % of d(d+1)/2 entries measured          (grouped bar, greyscale)

Run:
    python -m experiments.plot_meas_runtime --output-dir results/
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Table data (hardcoded from paper tables) ──────────────────────────────────

RHOS       = [1e-4,      1e-2,   1.0,    10.0]
RHO_LABELS = [r'$10^{-4}$', r'$10^{-2}$', r'$1$', r'$10$']

DATASETS = [
    ('adult',             6),
    ('LifeExpectancy',    15),
    ('communities_crime', 102),
    ('indian_pines',      187),
    ('madeline',          260),
]
# Keys without LaTeX escaping, for dict lookup
KEYS = ['adult', 'LifeExpectancy', 'communities_crime', 'indian_pines', 'madeline']

# off-diagonal measurement means and SE (lower triangle only)
# means: from paper table; SE: computed from 10 trials in results/a30_b05_final_privcov
MEAS = {
    'adult':             np.array([2.6,   3.4,  11.6,  12.2]),
    'LifeExpectancy':    np.array([4.9,  13.5,  43.7,  61.4]),
    'communities_crime': np.array([7.7,  11.0,  97.5, 255.3]),
    'indian_pines':      np.array([10.1, 47.9, 372.5, 518.0]),
    'madeline':          np.array([9.1,   9.9,  15.0,  39.2]),
}
MEAS_SE = {
    'adult':             np.array([0.2,  0.2,   0.4,   0.3]),
    'LifeExpectancy':    np.array([0.3,  1.2,   1.2,   2.2]),
    'communities_crime': np.array([0.3,  0.8,   4.7,   8.9]),
    'indian_pines':      np.array([0.5,  5.9,  10.6,  14.0]),
    'madeline':          np.array([0.4,  0.7,   0.8,   1.4]),
}

# runtime mean and SE (seconds)
TIME_MEAN = {
    'adult':             np.array([0.05,  0.05,    0.13,    0.16]),
    'LifeExpectancy':    np.array([0.12,  0.28,    0.95,    1.24]),
    'communities_crime': np.array([0.39,  0.63,   10.4,    47.8]),
    'indian_pines':      np.array([0.70,  8.01, 2714.0,  3117.0]),
    'madeline':          np.array([0.83,  0.84,    1.28,    4.16]),
}
TIME_SE = {
    'adult':             np.array([0.01, 0.01,   0.01,   0.01]),
    'LifeExpectancy':    np.array([0.02, 0.04,   0.06,   0.03]),
    'communities_crime': np.array([0.04, 0.06,   1.6,    2.8]),
    'indian_pines':      np.array([0.07, 2.22, 369.0,  697.0]),
    'madeline':          np.array([0.11, 0.11,   0.10,   0.32]),
}

HIGHLIGHT_COLORS = {
    'adult':             '#aacc00',  # lime green
    'LifeExpectancy':    '#7b1fa2',  # violet
    'communities_crime': '#64b5f6',  # light blue
    'indian_pines':      '#00796b',  # dark green teal
    'madeline':          '#cc00aa',  # magenta
}
COLORS = [HIGHLIGHT_COLORS[k] for k in KEYS]


def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _fmt(v, se=None):
    """Format a value (with optional SE) for the side table."""
    if v >= 1000:
        s = f'{v:.0f}'
    elif v >= 10:
        s = f'{v:.1f}'
    else:
        s = f'{v:.2f}'
    if se is not None:
        if se >= 100:
            s += f' ± {se:.0f}'
        elif se >= 1:
            s += f' ± {se:.1f}'
        else:
            s += f' ± {se:.2f}'
    return s


def make_line_table_fig(ylabel, title, y_data, se_data=None,
                        marker='o', filename='fig'):
    """
    Generic helper: left = line plot, right = matplotlib table.
    y_data  : dict key -> np.array (mean values)
    se_data : dict key -> np.array (SE values), or None
    """
    fig = plt.figure(figsize=(10, 3.8))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.05)
    ax_plot  = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    for (disp, d), key, color in zip(DATASETS, KEYS, COLORS):
        label = f'{disp} ($d={d}$)'
        m = y_data[key]
        ax_plot.plot(RHOS, m, marker=marker, linewidth=1.5, markersize=4,
                     color=color, label=label)
        if se_data is not None:
            s  = se_data[key]
            lo = np.maximum(m - s, m * 1e-3)
            ax_plot.fill_between(RHOS, lo, m + s, alpha=0.2, color=color)

    ax_plot.set_xscale('log')
    ax_plot.set_yscale('log')
    ax_plot.set_xlabel(r'Privacy budget $\rho$', fontsize=10)
    ax_plot.set_ylabel(ylabel, fontsize=10)
    ax_plot.set_title(title, fontsize=10)
    ax_plot.set_xticks(RHOS)
    ax_plot.set_xticklabels(RHO_LABELS, fontsize=9)
    ax_plot.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    _clean_axes(ax_plot)

    # ── Side table ────────────────────────────────────────────────────────────
    row_labels = [f'{disp}\n($d={d}$)' for disp, d in DATASETS]
    col_labels = [r'$\rho=10^{-4}$', r'$\rho=10^{-2}$', r'$\rho=1$', r'$\rho=10$']
    cell_text  = []
    for key in KEYS:
        row = []
        for i in range(len(RHOS)):
            se_i = se_data[key][i] if se_data is not None else None
            row.append(_fmt(y_data[key][i], se_i))
        cell_text.append(row)

    ax_table.axis('off')
    tbl = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)

    # Colour row label cells to match line colours
    for i, color in enumerate(COLORS):
        tbl[(i + 1, -1)].set_facecolor(matplotlib.colors.to_rgba(color, alpha=0.15))

    return fig, filename


def fig1_measurements(output_dir):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    for (name, d), key, color in zip(DATASETS, KEYS, COLORS):
        label = f'{name} ($d={d}$)'
        m  = MEAS[key]
        se = MEAS_SE[key]
        ax.plot(RHOS, m, marker='o', linewidth=2.5, markersize=12,
                color=color, label=label)
        lo = np.maximum(m - se, m * 1e-3)
        ax.fill_between(RHOS, lo, m + se, alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Privacy budget $\rho$', fontsize=16)
    ax.set_ylabel('Off-diagonal measurements', fontsize=16)
    ax.set_xticks(RHOS)
    ax.set_xticklabels(RHO_LABELS, fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    _clean_axes(ax)
    fig.set_size_inches(5, 5)
    _save(fig, output_dir, 'fig1_measurements_compact')


def fig2_runtime(output_dir):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    for (name, d), key, color in zip(DATASETS, KEYS, COLORS):
        m  = TIME_MEAN[key]
        se = TIME_SE[key]
        ax.plot(RHOS, m, marker='o', linewidth=2.5, markersize=12,
                color=color, label=f'{name} ($d={d}$)')
        lo = np.maximum(m - se, m * 1e-3)
        ax.fill_between(RHOS, lo, m + se, alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Privacy budget $\rho$', fontsize=16)
    ax.set_ylabel('Runtime (s)', fontsize=16)
    ax.set_xticks(RHOS)
    ax.set_xticklabels(RHO_LABELS, fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    _clean_axes(ax)
    fig.set_size_inches(5, 5)
    _save(fig, output_dir, 'fig2_runtime_compact')



def fig3_pct_greyscale(output_dir):
    """Grouped bar chart: % of d(d+1)/2 entries measured, greyscale."""
    greys = ['0.80', '0.55', '0.30', '0.05']  # light → dark for rho low → high

    fig, ax = plt.subplots(figsize=(6, 3.8))

    n_ds      = len(DATASETS)
    n_rho     = len(RHOS)
    bar_width = 0.18
    x_pos     = np.arange(n_ds)

    for ri, (grey, rl) in enumerate(zip(greys, RHO_LABELS)):
        pcts = []
        for key, (_, d) in zip(KEYS, DATASETS):
            total = d * (d + 1) / 2
            pcts.append(MEAS[key][ri] / total * 100)
        offset = (ri - (n_rho - 1) / 2) * bar_width
        ax.bar(x_pos + offset, pcts, width=bar_width,
               color=grey, label=f'$\\rho={{{rl.strip("$")}}}$',
               edgecolor='white', linewidth=0.4)

    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{disp}\n($d={d}$)' for disp, d in DATASETS],
                       rotation=15, ha='right', fontsize=8.5)
    ax.set_ylabel(r'$\%$ of $d(d+1)/2$ entries measured', fontsize=10)
    ax.set_title('Fraction of possible entries measured', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9,
              title=r'$\rho$', title_fontsize=8)
    _clean_axes(ax)
    fig.tight_layout()
    _save(fig, output_dir, 'fig3_pct_greyscale_compact')


def fig4_meas_vs_rho_greyscale(output_dir):
    """Measurements vs rho for the 5 selected datasets, colored + linestyles."""
    linestyles = ['-',  '--',  '-.',  ':',  (0, (5, 1))]
    markers    = ['o',  's',   '^',   'D',  'v']

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # dy offsets (points) for the end-of-line labels to avoid overlap
    label_dy = {
        'adult':             -8,
        'LifeExpectancy':     6,
        'communities_crime':  6,
        'indian_pines':       6,
        'madeline':          -8,
    }

    for (name, d), key, ls, mk in zip(DATASETS, KEYS, linestyles, markers):
        m     = MEAS[key]
        se    = MEAS_SE[key]
        color = HIGHLIGHT_COLORS[key]
        ax.plot(RHOS, m, marker=mk, linewidth=2.5, markersize=10,
                color=color, linestyle=ls)
        lo = np.maximum(m - se, m * 1e-3)
        ax.fill_between(RHOS, lo, m + se, alpha=0.15, color=color)
        # label at last rho
        ax.annotate(f'{name} ($d={d}$)',
                    xy=(RHOS[-1], m[-1]),
                    xytext=(8, label_dy[key]),
                    textcoords='offset points',
                    fontsize=13, fontweight='bold', color=color,
                    va='center', ha='left')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Privacy budget $\rho$', fontsize=15)
    ax.set_ylabel('Off-diagonal measurements', fontsize=15)
    ax.set_xticks(RHOS)
    ax.set_xticklabels(RHO_LABELS, fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_box_aspect(1)
    _clean_axes(ax)
    _save(fig, output_dir, 'fig4_meas_vs_rho_compact')


def _save(fig, output_dir, name):
    try:
        fig.tight_layout()
    except Exception:
        pass
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(p / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    print(f'Saved: {p / name}.png')
    plt.close(fig)


def fig_legend(output_dir):
    """Standalone legend for the 5 highlighted datasets."""
    linestyles = ['-', '--', '-.', ':', (0, (5, 1))]
    markers    = ['o', 's',  '^',  'D', 'v']

    handles = []
    for (name, d), key, ls, mk in zip(DATASETS, KEYS, linestyles, markers):
        color = HIGHLIGHT_COLORS[key]
        h, = plt.plot([], [], marker=mk, linewidth=2, markersize=7,
                      color=color, linestyle=ls,
                      label=f'{name} ($d={d}$)')
        handles.append(h)
    plt.close()

    fig = plt.figure(figsize=(3.5, 2.2))
    fig.legend(handles=handles, loc='center', fontsize=11,
               frameon=True, framealpha=0.9,
               handlelength=2.5, handletextpad=0.6,
               borderpad=0.8)
    _save(fig, output_dir, 'legend_datasets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='results/',
                        help='Directory to save the three figures')
    args = parser.parse_args()

    fig1_measurements(args.output_dir)
    fig2_runtime(args.output_dir)
    fig3_pct_greyscale(args.output_dir)
    fig4_meas_vs_rho_greyscale(args.output_dir)
    fig_legend(args.output_dir)
