"""Figures for the README: correlation heatmap, box plot, dominance chart."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import FEATURES, LABELS, TARGET

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blues", BLUES)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

SUBTITLE = "S&P 500 companies (n = 495), 6-month median ratings, May to Oct 2026."


def _title(fig, title, subtitle, x=0.05):
    fig.suptitle(title, x=x, ha="left", fontsize=15, fontweight="bold", y=0.97)
    fig.text(x, 0.925, subtitle, fontsize=9.5, color=INK2, va="top")


def correlation_heatmap(corr, path):
    labels = list(corr.columns)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, 8.4), dpi=200)
    im = ax.imshow(corr.values, cmap=CMAP, vmin=0.4, vmax=1.0)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            txt = "1" if i == j else f"{v:.2f}".lstrip("0")
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="white" if v >= 0.75 else INK,
                    fontweight="bold" if (v >= 0.9 and i != j) else "normal")
    for i in range(n + 1):
        ax.axhline(i - 0.5, color=SURFACE, lw=2)
        ax.axvline(i - 0.5, color=SURFACE, lw=2)
    ax.set_xticks(range(n), labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(n), labels, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    cb.set_label("Pearson r", color=INK2)
    _title(fig, "Glassdoor Rating Intercorrelations", SUBTITLE + " All r significant at p < .001.")
    fig.subplots_adjust(left=0.2, right=0.95, top=0.89, bottom=0.17)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def box_plot(df, path):
    order = [TARGET] + sorted(FEATURES, key=lambda k: -df[k].median())
    data = [df[k].to_numpy() for k in order][::-1]
    names = [LABELS[k] for k in order][::-1]
    fig, ax = plt.subplots(figsize=(11, 8.6), dpi=200)
    bp = ax.boxplot(data, orientation="horizontal", widths=0.42, patch_artist=True, whis=1.5,
                    boxprops=dict(facecolor="#9ec5f4", edgecolor="#184f95", lw=1.2),
                    medianprops=dict(color="#0d366b", lw=2.4),
                    whiskerprops=dict(color="#184f95", lw=1.2), capprops=dict(color="#184f95", lw=1.2),
                    flierprops=dict(marker="o", markersize=4.5, markerfacecolor="#3987e5",
                                    markeredgecolor=SURFACE, markeredgewidth=0.8, alpha=0.85))
    bp["boxes"][-1].set_facecolor("#3987e5")
    for i, d in enumerate(data):
        y = i + 1
        q1, md, q3 = np.percentile(d, [25, 50, 75])
        lo = bp["whiskers"][2 * i].get_xdata()[1]
        hi = bp["whiskers"][2 * i + 1].get_xdata()[1]
        ax.text(md, y + 0.25, f"{md:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        for v in (q1, q3):
            ax.text(v, y - 0.25, f"{v:.2f}", ha="center", va="top", fontsize=8.5, color=INK2)
        for v in (lo, hi):
            ax.text(v, y + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yticks(range(1, len(order) + 1), names, fontsize=10.5)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("Rating (1 to 5 stars)", fontsize=10)
    ax.set_xlim(1.75, 4.85)
    ax.set_ylim(0.4, len(order) + 0.65)
    ax.grid(axis="x", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0)
    _title(fig, "Distribution of Glassdoor Ratings",
           SUBTITLE + " Bold = median; below box = Q1 and Q3;\nat whisker ends = most extreme "
           "value within 1.5 × IQR. Dots are outliers.", x=0.04)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.87, bottom=0.08)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def dominance_chart(gen, path):
    d = gen.sort_values("share_of_R2")
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    y = np.arange(len(d))
    colors = ["#256abf" if r <= 3 else "#9ec5f4" for r in d["rank"]]
    ax.barh(y, d["share_of_R2"] * 100, color=colors, height=0.6)
    err = np.vstack([(d["share_of_R2"] - d["share_ci_low"]) * 100, (d["share_ci_high"] - d["share_of_R2"]) * 100])
    ax.errorbar(d["share_of_R2"] * 100, y, xerr=err, fmt="none", ecolor=INK2, capsize=3, lw=1)
    for yi, (v, hi) in enumerate(zip(d["share_of_R2"], d["share_ci_high"])):
        ax.text(hi * 100 + 0.4, yi, f"{v * 100:.1f}%", va="center", fontsize=10)
    ax.set_yticks(y, d["variable"], fontsize=10.5)
    ax.set_xlabel("Share of explained variance (% of R² = .934)", fontsize=10)
    ax.set_xlim(0, 27)
    ax.grid(axis="x", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _title(fig, "Which Sub-Ratings Explain Overall Glassdoor Ratings?",
           "General dominance weights with 95% bootstrap intervals (1,000 resamples). "
           "Dark bars = top tier.", x=0.04)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.83, bottom=0.13)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
