#!/usr/bin/env python
"""Create publication-grade evidence figures for the workload compiler paper."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "nature"
OUT.mkdir(parents=True, exist_ok=True)
MPLCONFIG = OUT / ".mplconfig"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 6.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    }
)

PALETTE = {
    "ours": "#F0C0CC",
    "ours_dark": "#B54F6A",
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8F8F8F",
    "neutral_dark": "#4D4D4D",
    "green": "#2E9E44",
    "red": "#B64342",
    "gold": "#D6A420",
    "axis": "#272727",
}

MAIN_DIR = ROOT / "results" / "sys_real_external_workload_compiler_k6_5seed_rerun"
R039 = ROOT / "results" / "workload_compiler_ablation_r039" / "summary.csv"
R040 = ROOT / "results" / "workload_compiler_k_feature_r040" / "summary.csv"


DISPLAY = {
    "ParetoProbe-Sys": "Workload compiler",
    "QueryOnly-RF": "Query-only RF",
    "CostGreedy-cal": "Cost greedy",
    "StaticBest-cal": "Static best",
    "ScalarTelemetry-RF": "Scalar telemetry RF",
    "Static-bm25_pre": "BM25 static",
    "Static-bm25_post": "BM25 post-filter",
    "Static-dense_e5_base_v2_pre": "E5-base static",
    "Static-rerank_depth_20": "Rerank depth 20",
    "Static-external_splade_cocondenser_ensembledistil_pre": "SPLADE static",
    "Static-external_qwen3_embedding_06b_pre": "Qwen static",
    "Static-external_colbertv2_pre": "ColBERTv2 static",
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.1,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["axis"],
    )


def lighten_spines(ax: plt.Axes) -> None:
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["axis"])
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=PALETTE["axis"], labelsize=6.2)


def family_color(method: str) -> str:
    if method == "ParetoProbe-Sys":
        return PALETTE["ours"]
    if method in {"QueryOnly-RF", "CostGreedy-cal", "StaticBest-cal", "ScalarTelemetry-RF"}:
        return PALETTE["baseline_mid"]
    return PALETTE["neutral_light"]


def read_gate() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with (MAIN_DIR / "rank_gate.json").open("r", encoding="utf-8") as f:
        gate = json.load(f)
    rank_input = pd.read_csv(MAIN_DIR / "rank_gate_input.csv")
    sys_summary = pd.read_csv(MAIN_DIR / "sys_summary.csv")
    return gate, rank_input, sys_summary


def compute_block_rank_stats(gate: dict, rank_input: pd.DataFrame) -> pd.DataFrame:
    df = rank_input.copy()
    df["rank"] = df.groupby("block")["score"].rank(ascending=False, method="average")
    stats = (
        df.groupby("method")["rank"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "block_rank_mean"})
        .reset_index()
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    stats["avg_rank"] = stats["method"].map(gate["average_ranks"])
    stats["avg_rank"] = stats["avg_rank"].fillna(stats["block_rank_mean"])
    stats["display"] = stats["method"].map(DISPLAY).fillna(stats["method"])
    return stats


def compute_metrics(gate: dict, sys_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        sys_summary.groupby("method", as_index=False)
        .agg(
            mean_score=("mean_score", "mean"),
            satisfaction_rate=("satisfaction_rate", "mean"),
            mean_latency_ms=("mean_latency_ms", "mean"),
            mean_regret=("mean_regret", "mean"),
        )
        .copy()
    )
    metrics["avg_rank"] = metrics["method"].map(gate["average_ranks"])
    metrics["display"] = metrics["method"].map(DISPLAY).fillna(metrics["method"])
    return metrics


def panel_main_rank(ax: plt.Axes, gate: dict, rank_stats: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "ParetoProbe-Sys",
        "QueryOnly-RF",
        "CostGreedy-cal",
        "StaticBest-cal",
        "ScalarTelemetry-RF",
        "Static-bm25_pre",
        "Static-dense_e5_base_v2_pre",
        "Static-rerank_depth_20",
        "Static-external_splade_cocondenser_ensembledistil_pre",
        "Static-external_qwen3_embedding_06b_pre",
        "Static-external_colbertv2_pre",
    ]
    plot_df = rank_stats[rank_stats["method"].isin(methods)].copy()
    plot_df["order"] = plot_df["method"].map({m: i for i, m in enumerate(methods)})
    plot_df = plot_df.sort_values("order")
    y = np.arange(len(plot_df))
    colors = [family_color(m) for m in plot_df["method"]]
    ax.barh(
        y,
        plot_df["avg_rank"],
        xerr=plot_df["sem"],
        height=0.62,
        color=colors,
        edgecolor=PALETTE["axis"],
        linewidth=0.35,
        error_kw={"elinewidth": 0.6, "capsize": 1.8, "capthick": 0.6},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display"])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title("Rank-1 gate across 30 blocks", loc="left", fontsize=7.5, pad=4)
    ax.set_xlim(0, 23.5)
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["avg_rank"] + row["sem"] + 0.55,
            yi,
            f"{row['avg_rank']:.1f}",
            va="center",
            ha="left",
            fontsize=5.8,
            color=PALETTE["axis"],
        )
    holm = gate["pairwise_holm"]
    callouts = {
        "QueryOnly-RF": "pH=0.013",
        "CostGreedy-cal": "pH=0.012",
        "ScalarTelemetry-RF": "pH=0.0016",
    }
    for method, label in callouts.items():
        row = plot_df[plot_df["method"] == method].iloc[0]
        idx = plot_df.index.get_loc(row.name)
        reject = holm[method]["reject"]
        color = PALETTE["green"] if reject else PALETTE["red"]
        ax.text(17.0, idx, label, va="center", fontsize=5.6, color=color)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.45)
    lighten_spines(ax)
    return plot_df[["method", "display", "avg_rank", "sem"]]


def panel_score(ax: plt.Axes, metrics: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "ParetoProbe-Sys",
        "QueryOnly-RF",
        "StaticBest-cal",
        "CostGreedy-cal",
        "ScalarTelemetry-RF",
        "Static-bm25_pre",
    ]
    plot_df = metrics[metrics["method"].isin(methods)].copy()
    plot_df["order"] = plot_df["method"].map({m: i for i, m in enumerate(methods)})
    plot_df = plot_df.sort_values("order")
    y = np.arange(len(plot_df))
    xmin = 0.315
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        color = family_color(row["method"])
        ax.hlines(yi, xmin, row["mean_score"], color=color, linewidth=2.4)
        ax.scatter(
            row["mean_score"],
            yi,
            s=30 if row["method"] == "ParetoProbe-Sys" else 22,
            color=color,
            edgecolor=PALETTE["axis"],
            linewidth=0.35,
            zorder=3,
        )
        ax.text(
            row["mean_score"] + 0.0009,
            yi,
            f"{row['mean_score']:.3f}",
            va="center",
            fontsize=5.7,
            color=PALETTE["axis"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display"])
    ax.invert_yaxis()
    ax.set_xlim(xmin, 0.358)
    ax.set_xlabel("Mean constrained score (higher is better)")
    ax.set_title("Utility at matched gate protocol", loc="left", fontsize=7.5, pad=4)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.45)
    lighten_spines(ax)
    return plot_df[
        [
            "method",
            "display",
            "mean_score",
            "satisfaction_rate",
            "mean_latency_ms",
            "mean_regret",
            "avg_rank",
        ]
    ]


def panel_k_sweep(ax: plt.Axes, r039: pd.DataFrame, r040: pd.DataFrame) -> pd.DataFrame:
    k_df = r039[r039["mode"] == "transductive"].sort_values("k").copy()
    ax.plot(
        k_df["k"],
        k_df["candidate_avg_rank"],
        color=PALETTE["baseline_dark"],
        linewidth=1.2,
        zorder=1,
    )
    for _, row in k_df.iterrows():
        face = PALETTE["ours"] if bool(row["pass_gate"]) else "white"
        edge = PALETTE["ours_dark"] if bool(row["pass_gate"]) else PALETTE["neutral_dark"]
        ax.scatter(
            row["k"],
            row["candidate_avg_rank"],
            s=34,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
            zorder=3,
        )
        if not bool(row["pass_gate"]):
            ax.text(
                row["k"],
                row["candidate_avg_rank"] + 0.2,
                f"F{int(row['failures'])}",
                ha="center",
                va="bottom",
                fontsize=5.2,
                color=edge,
            )
    elbow = r040[r040["variant"] == "auto_elbow_features_full"].iloc[0]
    sil = r040[r040["variant"] == "auto_silhouette_features_full"].iloc[0]
    ax.scatter(
        [5.38],
        [elbow["candidate_avg_rank"]],
        marker="*",
        s=76,
        color=PALETTE["gold"],
        edgecolor=PALETTE["axis"],
        linewidth=0.35,
        zorder=4,
    )
    ax.text(
        6.1,
        elbow["candidate_avg_rank"] - 0.36,
        "auto elbow",
        fontsize=5.5,
        color=PALETTE["axis"],
        ha="left",
    )
    ax.scatter(
        [4.28],
        [sil["candidate_avg_rank"]],
        marker="x",
        s=42,
        color=PALETTE["red"],
        linewidth=1.0,
        zorder=4,
    )
    ax.text(4.46, sil["candidate_avg_rank"] + 0.1, "silhouette fails", fontsize=5.5, color=PALETTE["red"])
    ax.text(
        2.05,
        3.45,
        "filled marker = Rank-1 pass",
        fontsize=5.4,
        color=PALETTE["neutral_dark"],
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Number of workload cells (k)")
    ax.set_ylabel("Average rank")
    ax.set_title("k sensitivity and unsupervised selection", loc="left", fontsize=7.5, pad=4)
    ax.set_xticks([2, 3, 4, 5, 6, 8, 10, 12])
    ax.set_xlim(1.5, 12.5)
    ax.set_ylim(3.2, 7.6)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.45)
    lighten_spines(ax)
    auto_rows = pd.DataFrame(
        [
            {
                "variant": "auto_elbow_features_full",
                "k": "auto",
                "candidate_avg_rank": elbow["candidate_avg_rank"],
                "pass_gate": elbow["pass_gate"],
                "failures": elbow["failures"],
                "selected_k_counts": elbow["selected_k_counts"],
            },
            {
                "variant": "auto_silhouette_features_full",
                "k": "auto",
                "candidate_avg_rank": sil["candidate_avg_rank"],
                "pass_gate": sil["pass_gate"],
                "failures": sil["failures"],
                "selected_k_counts": sil["selected_k_counts"],
            },
        ]
    )
    return pd.concat(
        [
            k_df[["variant", "k", "candidate_avg_rank", "pass_gate", "failures"]],
            auto_rows,
        ],
        ignore_index=True,
    )


def panel_feature_ablation(ax: plt.Axes, r040: pd.DataFrame) -> pd.DataFrame:
    variants = [
        ("fixed_k6_features_full", "Full telemetry"),
        ("fixed_k6_features_no_dense", "No dense"),
        ("fixed_k6_features_metadata_only", "Metadata only"),
        ("fixed_k6_features_no_bm25", "No BM25"),
        ("fixed_k6_features_no_probe", "No probe"),
        ("fixed_k6_features_no_overlap", "No overlap"),
        ("fixed_k6_features_query_only", "Query only"),
    ]
    mapping = dict(variants)
    plot_df = r040[r040["variant"].isin(mapping)].copy()
    plot_df["display"] = plot_df["variant"].map(mapping)
    plot_df["order"] = plot_df["variant"].map({v: i for i, (v, _) in enumerate(variants)})
    plot_df = plot_df.sort_values("order")
    y = np.arange(len(plot_df))
    xmin = 3.2
    for yi, (_, row) in enumerate(plot_df.iterrows()):
        passed = bool(row["pass_gate"])
        color = PALETTE["ours"] if passed else PALETTE["baseline_soft"]
        edge = PALETTE["ours_dark"] if passed else PALETTE["baseline_dark"]
        ax.hlines(yi, xmin, row["candidate_avg_rank"], color=color, linewidth=2.6)
        ax.scatter(
            row["candidate_avg_rank"],
            yi,
            s=30 if passed else 24,
            color=color,
            edgecolor=edge,
            linewidth=0.45,
            zorder=3,
        )
        ax.text(
            row["candidate_avg_rank"] + 0.08,
            yi,
            f"{row['candidate_avg_rank']:.1f} / F{int(row['failures'])}",
            va="center",
            fontsize=5.5,
            color=edge,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display"])
    ax.invert_yaxis()
    ax.set_xlim(xmin, 6.95)
    ax.set_xlabel("Average rank / failed comparisons")
    ax.set_title("Feature-family ablation", loc="left", fontsize=7.5, pad=4)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.45)
    lighten_spines(ax)
    return plot_df[
        [
            "variant",
            "display",
            "pass_gate",
            "failures",
            "candidate_avg_rank",
            "mean_score",
            "mean_regret",
            "selected_k_counts",
        ]
    ]


def write_markdown_notes(outputs: dict[str, Path], source_files: dict[str, Path], qa: dict[str, str]) -> None:
    legend = """# Figure Legend

**Figure. Workload-level compilation achieves rank-first constrained retrieval and identifies the useful telemetry.**

**a,** Average rank across 30 complete NFCorpus selectivity-budget-seed blocks. Lower is better. Error bars show the standard error of block-level ranks. Holm-adjusted pairwise tests compare the workload compiler against each baseline in the shared rank gate.

**b,** Mean constrained score under the same gate protocol. The score includes ranking quality with latency and metadata-filter penalties; higher is better.

**c,** Sensitivity to the number of workload cells. Filled markers indicate variants that pass the Rank-1 gate. The unsupervised elbow selector chooses k=5 and passes; the silhouette selector is shown as a negative control.

**d,** Feature-family ablation at k=6. Removing retrieval telemetry families causes gate failures, while the full telemetry variant remains first-ranked.

Statistics: n=30 complete blocks. Rank-gate comparisons use the existing Friedman/Holm protocol from `rank_gate.json`.
"""
    (OUT / "workload_compiler_evidence_legend.md").write_text(legend, encoding="utf-8")

    qa_lines = [
        "# Figure QA Notes",
        "",
        "Core conclusion: workload-level physical-plan compilation ranks first under constrained hybrid retrieval, and ablations indicate that retrieval telemetry is the useful signal.",
        "Figure archetype: quantitative grid.",
        "Backend: Python/matplotlib only.",
        "Final size: 183 mm x 142 mm double-column composite.",
        "Exports:",
    ]
    for key, path in outputs.items():
        qa_lines.append(f"- {key}: `{path.relative_to(ROOT)}`")
    qa_lines.extend(
        [
            "",
            "Source data:",
        ]
    )
    for key, path in source_files.items():
        qa_lines.append(f"- {key}: `{path.relative_to(ROOT)}`")
    qa_lines.extend(
        [
            "",
            "Checks:",
        ]
    )
    for key, value in qa.items():
        qa_lines.append(f"- {key}: {value}")
    qa_lines.extend(
        [
            "",
            "Review risks:",
            "- Current figure supports the one-workload claim; a second dataset should become a separate robustness panel if added.",
            "- NV-Embed and RankZephyr are not in the rendered evidence because their true outputs were not available in the current result files.",
            "- The k-selection panel intentionally shows silhouette failure to avoid cherry-pick framing.",
        ]
    )
    (OUT / "workload_compiler_evidence_qa.md").write_text("\n".join(qa_lines) + "\n", encoding="utf-8")


def main() -> None:
    gate, rank_input, sys_summary = read_gate()
    rank_stats = compute_block_rank_stats(gate, rank_input)
    metrics = compute_metrics(gate, sys_summary)
    r039 = pd.read_csv(R039)
    r040 = pd.read_csv(R040)

    fig = plt.figure(figsize=(7.2, 5.6), dpi=300)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        top=0.90,
        bottom=0.095,
        wspace=0.42,
        hspace=0.44,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    source_main = panel_main_rank(axes[0], gate, rank_stats)
    add_panel_label(axes[0], "a")
    source_score = panel_score(axes[1], metrics)
    add_panel_label(axes[1], "b")
    source_k = panel_k_sweep(axes[2], r039, r040)
    add_panel_label(axes[2], "c")
    source_feature = panel_feature_ablation(axes[3], r040)
    add_panel_label(axes[3], "d")

    fig.suptitle(
        "Workload-compiled retrieval is rank-first under constrained hybrid search",
        x=0.095,
        y=0.972,
        ha="left",
        fontsize=8.5,
        fontweight="bold",
    )

    source_files = {
        "main_rank": OUT / "source_main_rank.csv",
        "score": OUT / "source_score.csv",
        "k_selection": OUT / "source_k_selection.csv",
        "feature_ablation": OUT / "source_feature_ablation.csv",
    }
    source_main.to_csv(source_files["main_rank"], index=False)
    source_score.to_csv(source_files["score"], index=False)
    source_k.to_csv(source_files["k_selection"], index=False)
    source_feature.to_csv(source_files["feature_ablation"], index=False)

    base = OUT / "workload_compiler_evidence"
    outputs = {
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
        "tiff": base.with_suffix(".tiff"),
        "png_preview": base.with_suffix(".png"),
    }
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["png_preview"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    svg_text = outputs["svg"].read_text(encoding="utf-8", errors="ignore")
    text_nodes = svg_text.count("<text")
    with Image.open(outputs["tiff"]) as im:
        tiff_info = f"{im.size[0]}x{im.size[1]} px, mode={im.mode}"
    qa = {
        "editable_svg_text": f"{text_nodes} SVG text nodes",
        "tiff_resolution": tiff_info,
        "rank_gate_blocks": str(gate["blocks"]),
        "rank_gate_pass": str(gate["pass_gate"]),
        "backend_exclusivity": "All rendering and QA exports were produced with Python/matplotlib/Pillow.",
    }
    write_markdown_notes(outputs, source_files, qa)

    print("Generated:")
    for path in outputs.values():
        print(path.relative_to(ROOT))
    for path in source_files.values():
        print(path.relative_to(ROOT))
    print((OUT / "workload_compiler_evidence_legend.md").relative_to(ROOT))
    print((OUT / "workload_compiler_evidence_qa.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
