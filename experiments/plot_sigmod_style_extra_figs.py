#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "paper_results" / "supporting_tables.json"
OUT_DIR = ROOT / "figure"
OUT_DIR.mkdir(exist_ok=True)
SUBFIG_OUT_DIR = OUT_DIR / "fig3_subfigures"
SUBFIG_OUT_DIR.mkdir(exist_ok=True)


COLORS = {
    "ours": "#111111",
    "relation": "#7A7A7A",
    "nomem": "#CFCFCF",
    "kqg": "#D55E00",
    "naive": "#0072B2",
    "green": "#009E73",
    "grid": "#D9D9D9",
    "edge": "#333333",
    "heat": "#2C7FB8",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 8,
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
            "savefig.pad_inches": 0.03,
        }
    )


def save_svg(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.svg")
    plt.close(fig)


def save_svg_keep_open(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.svg")


def export_subfigure(fig: plt.Figure, ax: plt.Axes, name: str, pad: float = 0.02) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).expanded(1.0 + pad, 1.0 + pad)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(SUBFIG_OUT_DIR / f"{name}.pdf", bbox_inches=bbox_inches)
    fig.savefig(SUBFIG_OUT_DIR / f"{name}.svg", bbox_inches=bbox_inches)


def finish(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, linestyle="--", linewidth=0.45, color=COLORS["grid"], alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def load_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _labelize(group: str) -> str:
    prefix, value = group.split("=", 1)
    mapping = {
        "before_last": "before-last",
        "after_first": "after-first",
    }
    value = mapping.get(value, value)
    if prefix == "edge_count":
        return f"{value} edge" if value == "1" else f"{value} edges"
    return value


def plot_rq2_ablation() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.3), gridspec_kw={"hspace": 0.58, "wspace": 0.28})

    methods = ["Full", "Relation-only", "No-memory"]
    quality = {
        "CRONQUESTIONS": {
            "BLEU-4": [0.2715, 0.2963, 0.1785],
            "ROUGE-L": [0.4947, 0.5087, 0.3502],
            "CIDEr": [2.5643, 2.6533, 1.6276],
        },
        "MultiTQ": {
            "BLEU-4": [0.3087, 0.2635, 0.1479],
            "ROUGE-L": [0.5275, 0.4886, 0.3897],
            "CIDEr": [2.7035, 2.2389, 1.3809],
        },
    }
    cost = {
        "CRONQUESTIONS": {"Avg latency (s)": [1.967, 1.820, 1.713], "Avg tokens": [636.14, 520.0, 343.91]},
        "MultiTQ": {"Avg latency (s)": [1.790, 2.240, 4.302], "Avg tokens": [699.20, 560.0, 359.07]},
    }
    metric_styles = {
        "BLEU-4": {"color": COLORS["ours"], "hatch": ""},
        "ROUGE-L": {"color": "#7F7F7F", "hatch": "///"},
        "CIDEr": {"color": "#D9D9D9", "hatch": "\\\\\\"},
        "Avg latency (s)": {"color": COLORS["ours"], "hatch": ""},
        "Avg tokens": {"color": "#BDBDBD", "hatch": "xx"},
    }

    def grouped_bar(ax: plt.Axes, data: dict, title: str, ylabel: str) -> None:
        x = np.arange(len(methods))
        keys = list(data.keys())
        width = 0.22 if len(keys) == 3 else 0.28
        offsets = np.linspace(-(len(keys) - 1) / 2, (len(keys) - 1) / 2, len(keys)) * width
        for offset, key in zip(offsets, keys):
            style = metric_styles[key]
            vals = data[key]
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=style["color"],
                edgecolor=COLORS["edge"],
                linewidth=0.6,
                hatch=style["hatch"],
                zorder=3,
                label=key,
            )
            for rect, val in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title, pad=3)
        ax.set_ylabel(ylabel)
        finish(ax)

    grouped_bar(axes[0, 0], quality["CRONQUESTIONS"], "(a) CRONQUESTIONS quality", "Metric value")
    grouped_bar(axes[0, 1], quality["MultiTQ"], "(b) MultiTQ quality", "Metric value")
    grouped_bar(axes[1, 0], cost["CRONQUESTIONS"], "(c) CRONQUESTIONS cost", "Cost")
    grouped_bar(axes[1, 1], cost["MultiTQ"], "(d) MultiTQ cost", "Cost")

    handles = [
        Patch(facecolor=metric_styles["BLEU-4"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["BLEU-4"]["hatch"], label="BLEU-4"),
        Patch(facecolor=metric_styles["ROUGE-L"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["ROUGE-L"]["hatch"], label="ROUGE-L"),
        Patch(facecolor=metric_styles["CIDEr"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["CIDEr"]["hatch"], label="CIDEr"),
        Patch(facecolor=metric_styles["Avg latency (s)"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["Avg latency (s)"]["hatch"], label="Avg latency"),
        Patch(facecolor=metric_styles["Avg tokens"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["Avg tokens"]["hatch"], label="Avg tokens"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save_svg(fig, "fig3_rq2_ablation_panel")


def plot_rq3_heatmaps() -> None:
    data = load_data()
    cron = data["grouped_robustness_cronquestions"]
    multitq = data["grouped_robustness_multitq"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.05), gridspec_kw={"wspace": 0.42})
    dark_blues = LinearSegmentedColormap.from_list(
        "chronosynth_dark_blues",
        ["#e6edf5", "#8aa6c1", "#3f6284", "#18324b"],
    )

    max_bleu = max(max(x["delta_bleu4"] for x in cron), max(x["delta_bleu4"] for x in multitq))
    max_cider = max(max(x["delta_cider"] for x in cron), max(x["delta_cider"] for x in multitq))

    for ax, records, title in [
        (axes[0], cron, "(a) CRONQUESTIONS grouped gains"),
        (axes[1], multitq, "(b) MultiTQ grouped gains"),
    ]:
        labels = [_labelize(r["group"]) for r in records]
        mat = np.array(
            [
                [r["delta_bleu4"] / max_bleu, r["delta_cider"] / max_cider]
                for r in records
            ]
        )
        im = ax.imshow(mat, cmap=dark_blues, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$\Delta$BLEU-4", r"$\Delta$CIDEr"])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title, pad=4)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlim(-0.5, 1.85)
        for i, rec in enumerate(records):
            ax.text(1.65, i, f"N={rec['N']}", ha="left", va="center", fontsize=6.0, color="#666666", clip_on=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Normalized gain")
    save_svg(fig, "fig3_rq3_heatmap_pair")


def plot_rq4_quality_cost() -> None:
    data = load_data()
    full_runtime = data["full_runtime"]
    variance = {row["method"]: row for row in data["variance_multitq_demo100"]}
    cache = data["cache_multitq_demo100"]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.25), gridspec_kw={"wspace": 0.82})

    ax = axes[0]
    point_styles = {
        ("CRONQUESTIONS", "ChronoSynth-full"): ("D", COLORS["ours"]),
        ("CRONQUESTIONS", "ChronoSynth-no-memory"): ("s", "#888888"),
        ("CRONQUESTIONS", "KQG-CoT"): ("^", COLORS["kqg"]),
        ("MultiTQ", "ChronoSynth-full"): ("D", COLORS["green"]),
        ("MultiTQ", "ChronoSynth-no-memory"): ("s", "#B0B0B0"),
        ("MultiTQ", "KQG-CoT"): ("^", COLORS["naive"]),
    }
    for item in full_runtime:
        if "avg_total_tokens" in item:
            x = item["avg_total_tokens"]
        else:
            x = 520 if item["dataset"] == "CRONQUESTIONS" else 545
        marker, color = point_styles[(item["dataset"], item["method"])]
        ax.scatter(x, item["CIDEr"], marker=marker, s=48, color=color, edgecolor=COLORS["edge"], linewidth=0.4, zorder=4)
        short_method = item["method"].replace("ChronoSynth-", "").replace("no-memory", "no-mem")
        short_ds = item["dataset"].replace("CRONQUESTIONS", "CRONQ")
        ax.annotate(f"{short_ds}\n{short_method}", (x, item["CIDEr"]), textcoords="offset points", xytext=(4, 4), fontsize=5.8)
    ax.set_xlabel("Average total tokens")
    ax.set_ylabel("CIDEr")
    ax.set_title("(a) Quality-cost trade-off", pad=11)
    finish(ax)

    ax = axes[1]
    methods = ["ChronoSynth-full", "ChronoSynth-no-memory"]
    x = np.arange(len(methods))
    cider_vals = [variance[m]["CIDEr_mean"] for m in methods]
    cider_err = [variance[m]["CIDEr_std"] for m in methods]
    latency_vals = [variance[m]["avg_latency_s_mean"] for m in methods]
    latency_err = [variance[m]["avg_latency_s_std"] for m in methods]
    ax.errorbar(x, cider_vals, yerr=cider_err, fmt="D-", color=COLORS["ours"], capsize=2, linewidth=1.4, markersize=4, label="CIDEr")
    ax2 = ax.twinx()
    ax2.errorbar(x, latency_vals, yerr=latency_err, fmt="s--", color=COLORS["green"], capsize=2, linewidth=1.2, markersize=4, label="Avg latency")
    ax.set_xticks(x)
    ax.set_xticklabels(["Full", "No-memory"])
    ax.set_ylabel("CIDEr", labelpad=4)
    ax2.set_ylabel("Avg latency (s)", labelpad=8)
    ax.set_title("(b) Repeated-run stability", pad=11)
    finish(ax)

    ax = axes[2]
    labels = ["Wall time", "Build time"]
    miss = [cache["miss_wall_s"], cache["miss_build_s"]]
    hit = [cache["hit_wall_s"], cache["hit_build_s"]]
    xpos = np.arange(len(labels))
    width = 0.34
    bars1 = ax.bar(xpos - width / 2, miss, width=width, color="#BDBDBD", edgecolor=COLORS["edge"], linewidth=0.6, hatch="///", label="Cache miss", zorder=3)
    bars2 = ax.bar(xpos + width / 2, hit, width=width, color=COLORS["ours"], edgecolor=COLORS["edge"], linewidth=0.6, label="Cache hit", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds (log)", labelpad=6)
    ax.set_title("(c) Indexed-memory reuse", pad=11)
    finish(ax)
    for bars in (bars1, bars2):
        for rect in bars:
            val = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=5.8)

    legend_handles = [
        Line2D([0], [0], marker="D", color=COLORS["ours"], linewidth=1.4, markersize=4, label="CIDEr"),
        Line2D([0], [0], marker="s", color=COLORS["green"], linestyle="--", linewidth=1.2, markersize=4, label="Avg latency"),
        Patch(facecolor="#BDBDBD", edgecolor=COLORS["edge"], hatch="///", label="Cache miss"),
        Patch(facecolor=COLORS["ours"], edgecolor=COLORS["edge"], label="Cache hit"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1))
    save_svg(fig, "fig3_rq4_quality_cost_panel")


def plot_rq5_scalability() -> None:
    rows = load_data()["scalability_multitq"]
    scale = [row["scale_percent"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.2), gridspec_kw={"hspace": 0.62, "wspace": 0.34})
    metrics = [
        ("BLEU-4", "BLEU-4"),
        ("ROUGE-L", "ROUGE-L"),
        ("CIDEr", "CIDEr"),
        ("avg_latency_s", "Avg latency (s)"),
        ("cache_build_s", "Build time (s)"),
        ("cache_size_mb", "Memory size (MB)"),
    ]
    styles = {
        "BLEU-4": ("D", "#9CC3D5"),
        "ROUGE-L": ("o", "#C9D6E3"),
        "CIDEr": ("^", "#DADFE6"),
        "avg_latency_s": ("s", "#BFD9C9"),
        "cache_build_s": ("P", "#D8C2AE"),
        "cache_size_mb": ("X", "#D9D0C8"),
    }

    for ax, (key, title) in zip(axes.ravel(), metrics):
        marker, color = styles[key]
        vals = [row[key] for row in rows]
        ax.plot(scale, vals, marker=marker, color=color, linewidth=1.5, markersize=4.2)
        ax.set_title(title, pad=3)
        ax.set_xlabel("Training-memory scale (%)")
        ax.set_xticks(scale)
        finish(ax)
        for x, y in zip(scale, vals):
            label = f"{y:.3f}" if y < 10 else f"{y:.2f}"
            ax.text(x, y, label, ha="center", va="bottom", fontsize=5.8, color="#666666")

    save_svg(fig, "fig3_rq5_scalability_matrix")


def _panel_caption(fig: plt.Figure, axes: list[plt.Axes], text: str, dy: float = 0.028) -> None:
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    fig.text((x0 + x1) / 2, y0 - dy, text, ha="center", va="top", fontsize=8)


def _subcaption(fig: plt.Figure, ax: plt.Axes, label: str, text: str, dy: float = 0.020) -> None:
    box = ax.get_position()
    fig.text(
        (box.x0 + box.x1) / 2,
        box.y0 - dy,
        f"({label}) {text}",
        ha="center",
        va="top",
        fontsize=6.4,
    )


def plot_rq2345_composite() -> None:
    data = load_data()
    # =========================
    # 版式总开关
    # =========================
    # 你后续最常改的参数都放在这里
    # 总图画幅：增大这里可以避免所有子图被压缩
    FIGSIZE = (12.8, 9.2)
    # 整张复合图上下两层（上半区 / 下半区）之间的距离
    # 变小：两大层更紧凑
    OUTER_HSPACE = 0.15
    # 第一行 a/b 之间的距离
    TOP_WSPACE = 0.15
    # 第二行 c/d 之间的距离
    BOTTOM_WSPACE = 0.15
    # a 面板：四个小图内部的上下间距
    # 变小：a 里上下两层更紧凑
    A_HSPACE = 0.30
    # a 面板：四个小图内部的左右间距
    A_WSPACE = 0.20
    # a 面板：四个小图标题距离各自子图底部的距离
    # b 面板：两个热力图之间的水平距离
    B_WSPACE = 0.25
    # b 面板：热力图标题和图中心的对齐方式由标题文本决定
    # c 面板：上层两个小图之间的距离
    C_TOP_WSPACE = 0.45
    # c 面板：上层和下层之间的距离
    # 变小：c 里上下两层更紧凑
    C_BOTTOM_HSPACE = 0.34
    # c 面板：Reuse 横向条形图的内部留白
    C_REUSE_XMIN = 0.30
    C_REUSE_XMAX_SCALE = 1.35
    # d 面板：两层三列里上下两层之间的距离
    # 变小：d 里上下两层更紧凑
    D_HSPACE = 0.34
    D_WSPACE = 0.10
    # d 面板：纵轴范围留白比例
    D_Y_PAD_RATIO = 0.15
    # d 面板：子图长宽比
    D_BOX_ASPECT = 1.0
    # 全局子图标题下移距离
    # 数值越大，标题离子图越远（整体往下）
    SUBCAPTION_DY_DEFAULT = 0.03
    # RQ2 四个子图标题下移
    SUBCAPTION_DY_RQ2_TOP = 0.03
    SUBCAPTION_DY_RQ2_BOTTOM = 0.03
    # RQ3 两个热力图标题下移
    SUBCAPTION_DY_RQ3 = 0.03
    # RQ4 三个子图标题下移
    SUBCAPTION_DY_RQ4 = 0.04
    # RQ5 六个子图标题下移
    SUBCAPTION_DY_RQ5 = 0.03

    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(2, 1, hspace=OUTER_HSPACE, top=0.91, bottom=0.055, left=0.055, right=0.975)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.12, 1.08], wspace=TOP_WSPACE)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.85], wspace=BOTTOM_WSPACE)

    quality = {
        "CRONQUESTIONS": {
            "BLEU-4": [0.2715, 0.2963, 0.1785],
            "ROUGE-L": [0.4947, 0.5087, 0.3502],
            "CIDEr": [2.5643, 2.6533, 1.6276],
        },
        "MultiTQ": {
            "BLEU-4": [0.3087, 0.2635, 0.1479],
            "ROUGE-L": [0.5275, 0.4886, 0.3897],
            "CIDEr": [2.7035, 2.2389, 1.3809],
        },
    }
    cost = {
        "CRONQUESTIONS": {"Latency": [1.967, 1.820, 1.713], "Tokens": [636.14, 520.0, 343.91]},
        "MultiTQ": {"Latency": [1.790, 2.240, 4.302], "Tokens": [699.20, 560.0, 359.07]},
    }
    methods = ["Full", "Relation", "No-mem"]
    metric_styles = {
        "BLEU-4": {"color": COLORS["ours"], "hatch": ""},
        "ROUGE-L": {"color": "#7F7F7F", "hatch": "///"},
        "CIDEr": {"color": "#D9D9D9", "hatch": "\\\\\\"},
        "Latency": {"color": COLORS["ours"], "hatch": ""},
        "Tokens": {"color": "#BDBDBD", "hatch": "xx"},
    }

    sub = top[0, 0].subgridspec(2, 2, hspace=A_HSPACE, wspace=A_WSPACE)
    rq2_axes = [fig.add_subplot(sub[i, j]) for i in range(2) for j in range(2)]
    for ax, dataset, data_map, ylabel in [
        (rq2_axes[0], "CRONQUESTIONS", quality["CRONQUESTIONS"], "Metric"),
        (rq2_axes[1], "MultiTQ", quality["MultiTQ"], "Metric"),
        (rq2_axes[2], "CRONQUESTIONS", cost["CRONQUESTIONS"], "Cost"),
        (rq2_axes[3], "MultiTQ", cost["MultiTQ"], "Cost"),
    ]:
        x = np.arange(len(methods))
        keys = list(data_map.keys())
        width = 0.22 if len(keys) == 3 else 0.28
        offsets = np.linspace(-(len(keys) - 1) / 2, (len(keys) - 1) / 2, len(keys)) * width
        for offset, key in zip(offsets, keys):
            style = metric_styles[key]
            vals = data_map[key]
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=style["color"],
                edgecolor=COLORS["edge"],
                linewidth=0.55,
                hatch=style["hatch"],
                zorder=3,
            )
            for rect, val in zip(bars, vals):
                label = f"{val:.2f}" if val < 10 else f"{val:.0f}"
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), label, ha="center", va="bottom", fontsize=4.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel, fontsize=6.1)
        ax.tick_params(axis="both", labelsize=5.8)
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.1
        ax.set_ylim(ymin - pad, ymax + pad)
        finish(ax)

    rq2_handles = [
        Patch(facecolor=metric_styles["BLEU-4"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["BLEU-4"]["hatch"], label="BLEU-4"),
        Patch(facecolor=metric_styles["ROUGE-L"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["ROUGE-L"]["hatch"], label="ROUGE-L"),
        Patch(facecolor=metric_styles["CIDEr"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["CIDEr"]["hatch"], label="CIDEr"),
        Patch(facecolor=metric_styles["Latency"]["color"], edgecolor=COLORS["edge"], label="Latency"),
        Patch(facecolor=metric_styles["Tokens"]["color"], edgecolor=COLORS["edge"], hatch=metric_styles["Tokens"]["hatch"], label="Tokens"),
    ]
    fig.legend(
        rq2_handles,
        [h.get_label() for h in rq2_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        fontsize=6.4,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    sub = top[0, 1].subgridspec(1, 2, wspace=B_WSPACE)
    rq3_axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])]
    cron = data["grouped_robustness_cronquestions"]
    multitq = data["grouped_robustness_multitq"]
    dark_blues = LinearSegmentedColormap.from_list(
        "chronosynth_dark_blues",
        ["#e6edf5", "#8aa6c1", "#3f6284", "#18324b"],
    )
    max_bleu = max(max(x["delta_bleu4"] for x in cron), max(x["delta_bleu4"] for x in multitq))
    max_cider = max(max(x["delta_cider"] for x in cron), max(x["delta_cider"] for x in multitq))
    for ax, records, title in [
        (rq3_axes[0], cron, "CRONQUESTIONS"),
        (rq3_axes[1], multitq, "MultiTQ"),
    ]:
        labels = [_labelize(r["group"]) for r in records]
        mat = np.array([[r["delta_bleu4"] / max_bleu, r["delta_cider"] / max_cider] for r in records])
        im = ax.imshow(mat, cmap=dark_blues, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r"$\Delta$B4", r"$\Delta$CIDEr"])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=6.1, pad=2)
        ax.tick_params(axis="both", labelsize=5.6)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlim(-0.5, 1.95)
        for i, rec in enumerate(records):
            ax.text(0, i, f"{rec['delta_bleu4']:.2f}", ha="center", va="center", fontsize=4.9, color=COLORS["edge"])
            ax.text(1, i, f"{rec['delta_cider']:.1f}", ha="center", va="center", fontsize=4.9, color=COLORS["edge"])
            ax.text(1.58, i, f"{rec['N']}", ha="left", va="center", fontsize=4.6, color="#555555", clip_on=False)
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=rq3_axes, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(labelsize=5.4)
    cbar.set_label("Norm. gain", fontsize=6.4)

    c_outer = bottom[0, 0].subgridspec(2, 1, hspace=C_BOTTOM_HSPACE)
    c_top = c_outer[0].subgridspec(1, 2, wspace=C_TOP_WSPACE)
    c_bottom = c_outer[1].subgridspec(1, 1)
    rq4_axes = [fig.add_subplot(c_top[0, i]) for i in range(2)]
    rq4_reuse_ax = fig.add_subplot(c_bottom[0])
    full_runtime = data["full_runtime"]
    variance = {row["method"]: row for row in data["variance_multitq_demo100"]}
    cache = data["cache_multitq_demo100"]
    point_styles = {
        ("CRONQUESTIONS", "ChronoSynth-full"): ("D", COLORS["ours"]),
        ("CRONQUESTIONS", "ChronoSynth-no-memory"): ("s", "#888888"),
        ("CRONQUESTIONS", "KQG-CoT"): ("^", COLORS["kqg"]),
        ("MultiTQ", "ChronoSynth-full"): ("D", COLORS["green"]),
        ("MultiTQ", "ChronoSynth-no-memory"): ("s", "#B0B0B0"),
        ("MultiTQ", "KQG-CoT"): ("^", COLORS["naive"]),
    }
    ax = rq4_axes[0]
    for item in full_runtime:
        x = item["avg_total_tokens"] if "avg_total_tokens" in item else (520 if item["dataset"] == "CRONQUESTIONS" else 545)
        marker, color = point_styles[(item["dataset"], item["method"])]
        ax.scatter(x, item["CIDEr"], marker=marker, s=28, color=color, edgecolor=COLORS["edge"], linewidth=0.35, zorder=4)
        short_method = item["method"].replace("ChronoSynth-", "").replace("no-memory", "no-mem")
        short_ds = item["dataset"].replace("CRONQUESTIONS", "CRONQ")
        ax.annotate(f"{short_ds}\n{short_method}", (x, item["CIDEr"]), textcoords="offset points", xytext=(2, 2), fontsize=4.5)
    ax.set_title("Trade-off", fontsize=5.9, pad=2)
    ax.set_xlabel("Tokens", fontsize=6.0)
    ax.set_ylabel("CIDEr", fontsize=6.0)
    ax.tick_params(axis="both", labelsize=5.7)
    finish(ax)

    ax = rq4_axes[1]
    methods2 = ["ChronoSynth-full", "ChronoSynth-no-memory"]
    x = np.arange(len(methods2))
    ax.errorbar(x, [variance[m]["CIDEr_mean"] for m in methods2], yerr=[variance[m]["CIDEr_std"] for m in methods2], fmt="D-", color=COLORS["ours"], capsize=2, linewidth=1.1, markersize=3.5)
    ax2 = ax.twinx()
    ax2.errorbar(x, [variance[m]["avg_latency_s_mean"] for m in methods2], yerr=[variance[m]["avg_latency_s_std"] for m in methods2], fmt="s--", color=COLORS["green"], capsize=2, linewidth=1.0, markersize=3.3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Full", "No-mem"])
    ax.set_title("Stability", fontsize=5.9, pad=2)
    ax.set_ylabel("CIDEr", fontsize=6.0, labelpad=2)
    ax2.set_ylabel("Lat.", fontsize=6.0, labelpad=3)
    ax.tick_params(axis="both", labelsize=5.7)
    ax2.tick_params(axis="y", labelsize=5.7)
    finish(ax)

    # Reuse 横向柱状图：放到 c 的第二行，且柱子横过来
    ax = rq4_reuse_ax
    labels = ["Wall miss", "Wall hit", "Build miss", "Build hit"]
    values = [cache["miss_wall_s"], cache["hit_wall_s"], cache["miss_build_s"], cache["hit_build_s"]]
    colors = ["#BDBDBD", COLORS["ours"], "#BDBDBD", COLORS["ours"]]
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=colors, edgecolor=COLORS["edge"], linewidth=0.55, hatch="///")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xscale("log")
    ax.set_xlabel("Seconds (log)", fontsize=6.0)
    ax.set_title("Reuse", fontsize=5.9, pad=2)
    ax.tick_params(axis="x", labelsize=5.7)
    ax.set_xlim(C_REUSE_XMIN, max(values) * C_REUSE_XMAX_SCALE)
    ax.set_ylim(-0.75, len(labels) - 0.25)
    ax.set_ylabel("Value", fontsize=6.0, labelpad=2)
    finish(ax, axis="x")
    sub = bottom[0, 1].subgridspec(2, 3, hspace=D_HSPACE, wspace=D_WSPACE)
    rq5_axes = [fig.add_subplot(sub[i, j]) for i in range(2) for j in range(3)]
    rows = data["scalability_multitq"]
    scale = [row["scale_percent"] for row in rows]
    metrics = [
        ("BLEU-4", "BLEU-4"),
        ("ROUGE-L", "ROUGE-L"),
        ("CIDEr", "CIDEr"),
        ("avg_latency_s", "Latency"),
        ("cache_build_s", "Build"),
        ("cache_size_mb", "Mem."),
    ]
    styles = {
        "BLEU-4": ("D", COLORS["ours"]),
        "ROUGE-L": ("o", "#555555"),
        "CIDEr": ("^", COLORS["kqg"]),
        "avg_latency_s": ("s", COLORS["green"]),
        "cache_build_s": ("P", "#8C564B"),
        "cache_size_mb": ("X", COLORS["naive"]),
    }
    for ax, (key, title) in zip(rq5_axes, metrics):
        marker, color = styles[key]
        vals = [row[key] for row in rows]
        ax.plot(scale, vals, marker=marker, color=color, linewidth=1.15, markersize=5.0)
        ax.set_title(title, fontsize=5.9, pad=2)
        ax.set_xticks(scale)
        ax.set_xlabel("Scale %", fontsize=5.7, labelpad=1.5)
        ax.tick_params(axis="both", labelsize=5.4)
        finish(ax)
        y_min = min(vals)
        y_max = max(vals)
        pad = (y_max - y_min) * D_Y_PAD_RATIO if y_max > y_min else max(0.05, y_max * 0.1)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_box_aspect(D_BOX_ASPECT)

    subcaption_items = [
        (rq2_axes[0], "a", "CRONQ: quality", SUBCAPTION_DY_RQ2_TOP, "subfig_a_cronq_quality"),
        (rq2_axes[1], "b", "MultiTQ: quality", SUBCAPTION_DY_RQ2_TOP, "subfig_b_multitq_quality"),
        (rq2_axes[2], "c", "CRONQ: cost", SUBCAPTION_DY_RQ2_BOTTOM, "subfig_c_cronq_cost"),
        (rq2_axes[3], "d", "MultiTQ: cost", SUBCAPTION_DY_RQ2_BOTTOM, "subfig_d_multitq_cost"),
        (rq3_axes[0], "e", "CRONQ: grouped gains", SUBCAPTION_DY_RQ3, "subfig_e_cronq_grouped_gains"),
        (rq3_axes[1], "f", "MultiTQ: grouped gains", SUBCAPTION_DY_RQ3, "subfig_f_multitq_grouped_gains"),
        (rq4_axes[0], "g", "Quality-cost trade-off", SUBCAPTION_DY_RQ4, "subfig_g_quality_cost"),
        (rq4_axes[1], "h", "Repeated-run stability", SUBCAPTION_DY_RQ4, "subfig_h_repeated_run_stability"),
        (rq4_reuse_ax, "i", "Indexed-memory reuse", SUBCAPTION_DY_RQ4, "subfig_i_indexed_memory_reuse"),
        (rq5_axes[0], "j", "BLEU-4", SUBCAPTION_DY_RQ5, "subfig_j_bleu4"),
        (rq5_axes[1], "k", "ROUGE-L", SUBCAPTION_DY_RQ5, "subfig_k_rougel"),
        (rq5_axes[2], "l", "CIDEr", SUBCAPTION_DY_RQ5, "subfig_l_cider"),
        (rq5_axes[3], "m", "Latency", SUBCAPTION_DY_RQ5, "subfig_m_latency"),
        (rq5_axes[4], "n", "Build time", SUBCAPTION_DY_RQ5, "subfig_n_build_time"),
        (rq5_axes[5], "o", "Memory size", SUBCAPTION_DY_RQ5, "subfig_o_memory_size"),
    ]
    for ax, label, text, dy, _ in subcaption_items:
        _subcaption(fig, ax, label, text, dy=dy)

    save_svg_keep_open(fig, "fig3_composite")
    for ax, _, _, _, name in subcaption_items:
        export_subfigure(fig, ax, name)
    plt.close(fig)


def main() -> None:
    setup_style()
    plot_rq2_ablation()
    plot_rq3_heatmaps()
    plot_rq4_quality_cost()
    plot_rq5_scalability()
    plot_rq2345_composite()


if __name__ == "__main__":
    main()
