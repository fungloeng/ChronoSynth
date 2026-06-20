#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "paper_results" / "supporting_tables.json"
OUT_DIR = ROOT / "figure_preview_rq_candidates"
OUT_DIR.mkdir(exist_ok=True)


COLORS = {
    "full": "#2F6B8F",
    "relation": "#78A9D1",
    "nomem": "#D98C4A",
    "kqg": "#7A7A7A",
    "green": "#6FAE75",
    "red": "#C85C5C",
    "light_blue": "#DCEAF5",
    "light_orange": "#F6DFC8",
    "dark": "#2F2F2F",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def finish(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def load_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_rq2_ablation_panel() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5), gridspec_kw={"hspace": 0.52, "wspace": 0.34})

    methods = ["Full", "Relation-only", "No memory"]
    cron_quality = {
        "BLEU-4": [0.2715, 0.2963, 0.1785],
        "ROUGE-L": [0.4947, 0.5087, 0.3502],
        "CIDEr": [2.5643, 2.6533, 1.6276],
    }
    multitq_quality = {
        "BLEU-4": [0.3087, 0.2635, 0.1479],
        "ROUGE-L": [0.5275, 0.4886, 0.3897],
        "CIDEr": [2.7035, 2.2389, 1.3809],
    }
    cron_cost = {"Latency (s)": [1.967, 1.820, 1.713], "Tokens": [636.14, 520.0, 343.91]}
    multitq_cost = {"Latency (s)": [1.790, 2.240, 4.302], "Tokens": [699.20, 560.0, 359.07]}

    def grouped_bar(ax: plt.Axes, data: dict, title: str) -> None:
        x = np.arange(len(methods))
        width = 0.24
        palette = [COLORS["full"], COLORS["relation"], COLORS["nomem"]]
        keys = list(data.keys())
        for i, key in enumerate(keys):
            values = data[key]
            ax.bar(x + (i - 1) * width, values, width, label=key, color=palette[i], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title)
        finish(ax)

    grouped_bar(axes[0, 0], cron_quality, "(a) CRONQUESTIONS quality")
    axes[0, 0].set_ylabel("Metric value")
    grouped_bar(axes[0, 1], multitq_quality, "(b) MultiTQ quality")
    grouped_bar(axes[1, 0], cron_cost, "(c) CRONQUESTIONS cost")
    axes[1, 0].set_ylabel("Value")
    grouped_bar(axes[1, 1], multitq_cost, "(d) MultiTQ cost")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    save(fig, "rq2_ablation_panel_candidate")


def _prepare_group_matrix(records: list[dict]) -> tuple[list[str], np.ndarray]:
    rows = []
    vals = []
    for item in records:
        label = item["group"].replace("operator=", "op: ").replace("edge_count=", "edge: ").replace("answer_type=", "ans: ")
        rows.append(label)
        vals.append([item["delta_bleu4"], item["delta_cider"], item["N"]])
    return rows, np.array(vals, dtype=float)


def plot_rq3_heatmaps() -> None:
    data = load_data()
    cron_rows, cron = _prepare_group_matrix(data["grouped_robustness_cronquestions"])
    multitq_rows, multitq = _prepare_group_matrix(data["grouped_robustness_multitq"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), gridspec_kw={"wspace": 0.42})

    for ax, mat, rows, title in [
        (axes[0], cron, cron_rows, "(a) CRONQUESTIONS grouped gains"),
        (axes[1], multitq, multitq_rows, "(b) MultiTQ grouped gains"),
    ]:
        norm = mat[:, :2]
        im = ax.imshow(norm, cmap="YlGnBu", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$\Delta$BLEU-4", r"$\Delta$CIDEr"])
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows)
        ax.set_title(title)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=COLORS["dark"])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Positive gain")
    save(fig, "rq3_heatmap_pair_candidate")


def plot_rq3_small_multiples() -> None:
    data = load_data()
    cron = data["grouped_robustness_cronquestions"]
    multitq = data["grouped_robustness_multitq"]

    def pick(records: list[dict], prefix: str) -> list[dict]:
        return [r for r in records if r["group"].startswith(prefix)]

    panels = [
        ("CRONQ operators", pick(cron, "operator=")),
        ("CRONQ edge count", pick(cron, "edge_count=")),
        ("CRONQ answer type", pick(cron, "answer_type=")),
        ("MultiTQ operators", pick(multitq, "operator=")),
        ("MultiTQ edge count", pick(multitq, "edge_count=")),
        ("MultiTQ answer type", pick(multitq, "answer_type=")),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.6), gridspec_kw={"hspace": 0.62, "wspace": 0.48})
    for ax, (title, items) in zip(axes.ravel(), panels):
        labels = [x["group"].split("=")[1] for x in items]
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(x - width / 2, [x["delta_bleu4"] for x in items], width, color=COLORS["full"], label=r"$\Delta$BLEU-4", zorder=3)
        ax.bar(x + width / 2, [x["delta_cider"] for x in items], width, color=COLORS["nomem"], label=r"$\Delta$CIDEr", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_title(title)
        finish(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    save(fig, "rq3_small_multiples_candidate")


def plot_rq3_radar() -> None:
    data = load_data()
    cron_ops = [x for x in data["grouped_robustness_cronquestions"] if x["group"].startswith("operator=")]
    multitq_ops = [x for x in data["grouped_robustness_multitq"] if x["group"].startswith("operator=")]
    labels = sorted(set([x["group"].split("=")[1] for x in cron_ops + multitq_ops]))

    def normalized(records: list[dict]) -> list[float]:
        mapping = {x["group"].split("=")[1]: x["delta_cider"] for x in records}
        vals = np.array([mapping.get(label, 0.0) for label in labels], dtype=float)
        vmax = vals.max() if vals.max() > 0 else 1.0
        return (vals / vmax).tolist()

    cron_vals = normalized(cron_ops)
    multitq_vals = normalized(multitq_ops)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    cron_vals += cron_vals[:1]
    multitq_vals += multitq_vals[:1]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), subplot_kw={"projection": "polar"})
    for ax, vals, title, color in [
        (axes[0], cron_vals, "(a) CRONQUESTIONS operators", COLORS["full"]),
        (axes[1], multitq_vals, "(b) MultiTQ operators", COLORS["nomem"]),
    ]:
        ax.plot(angles, vals, color=color, linewidth=1.6)
        ax.fill(angles, vals, color=color, alpha=0.18)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
        ax.set_title(title, y=1.08)
    save(fig, "rq3_radar_candidate")


def plot_rq4_quality_cost_panel() -> None:
    data = load_data()
    full_runtime = data["full_runtime"]
    variance = data["variance_multitq_demo100"]
    cache = data["cache_multitq_demo100"]

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.45), gridspec_kw={"wspace": 0.52})

    ax = axes[0]
    scatter_data = [x for x in full_runtime if "avg_total_tokens" in x]
    marker_map = {"ChronoSynth-full": "o", "ChronoSynth-no-memory": "s"}
    color_map = {"CRONQUESTIONS": COLORS["full"], "MultiTQ": COLORS["nomem"]}
    for item in scatter_data:
        ax.scatter(
            item["avg_total_tokens"],
            item["CIDEr"],
            s=50,
            marker=marker_map[item["method"]],
            color=color_map[item["dataset"]],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax.annotate(
            f"{item['dataset'].replace('CRONQUESTIONS', 'CRONQ')}\n{item['method'].replace('ChronoSynth-', '').replace('no-memory', 'no-mem')}",
            (item["avg_total_tokens"], item["CIDEr"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.2,
        )
    ax.set_xlabel("Average total tokens")
    ax.set_ylabel("CIDEr")
    ax.set_title("(a) Quality-cost trade-off")
    finish(ax)

    ax = axes[1]
    methods = ["Full", "No memory"]
    x = np.arange(2)
    cider = [variance[0]["CIDEr_mean"], variance[1]["CIDEr_mean"]]
    cider_err = [variance[0]["CIDEr_std"], variance[1]["CIDEr_std"]]
    latency = [variance[0]["avg_latency_s_mean"], variance[1]["avg_latency_s_mean"]]
    latency_err = [variance[0]["avg_latency_s_std"], variance[1]["avg_latency_s_std"]]
    ax.errorbar(x, cider, yerr=cider_err, fmt="o-", color=COLORS["full"], capsize=2, label="CIDEr")
    ax2 = ax.twinx()
    ax2.errorbar(x, latency, yerr=latency_err, fmt="s--", color=COLORS["nomem"], capsize=2, label="Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("CIDEr")
    ax2.set_ylabel("Avg latency (s)")
    ax.set_title("(b) Repeated-run stability")
    finish(ax)

    ax = axes[2]
    labels = ["Wall", "Build"]
    miss = [cache["miss_wall_s"], cache["miss_build_s"]]
    hit = [cache["hit_wall_s"], cache["hit_build_s"]]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, miss, width, color=COLORS["red"], label="Cache miss", zorder=3)
    ax.bar(x + width / 2, hit, width, color=COLORS["green"], label="Cache hit", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds (log)")
    ax.set_title("(c) Cache behavior")
    finish(ax)
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color=COLORS["full"], label="CIDEr"),
        mpl.lines.Line2D([0], [0], marker="s", color=COLORS["nomem"], linestyle="--", label="Latency"),
    ]
    axes[1].legend(handles=handles, frameon=False, loc="upper left")
    axes[2].legend(frameon=False, loc="upper right")
    save(fig, "rq4_quality_cost_panel_candidate")


def plot_rq5_scalability_matrix() -> None:
    data = load_data()
    rows = data["scalability_multitq"]
    scale = [x["scale_percent"] for x in rows]
    metrics = [
        ("BLEU-4", "BLEU-4"),
        ("ROUGE-L", "ROUGE-L"),
        ("CIDEr", "CIDEr"),
        ("avg_latency_s", "Avg latency (s)"),
        ("cache_build_s", "Cache build (s)"),
        ("cache_size_mb", "Cache size (MB)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.2), gridspec_kw={"hspace": 0.58, "wspace": 0.36})
    for ax, (key, title) in zip(axes.ravel(), metrics):
        values = [x[key] for x in rows]
        ax.plot(scale, values, "-o", color=COLORS["full"], linewidth=1.5, markersize=4)
        ax.set_title(title)
        ax.set_xlabel("Memory scale (%)")
        finish(ax)
    save(fig, "rq5_scalability_matrix_candidate")


def main() -> None:
    setup_style()
    load_data()
    plot_rq2_ablation_panel()
    plot_rq3_heatmaps()
    plot_rq3_small_multiples()
    plot_rq3_radar()
    plot_rq4_quality_cost_panel()
    plot_rq5_scalability_matrix()


if __name__ == "__main__":
    main()
