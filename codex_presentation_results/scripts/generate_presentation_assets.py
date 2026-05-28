from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

DISPLAY_NAME = {
    "sst2": "SST-2",
    "mrpc": "MRPC",
    "rte": "RTE",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "qqp": "QQP",
}

ROW_END = " \\\\"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def write_overlap_table(overlap: Dict[str, object], out_path: Path) -> None:
    ks: List[int] = overlap["k_values"]
    mean = overlap["mean"]
    std = overlap["std"]
    vmin = overlap["min"]
    vmax = overlap["max"]

    lines = [
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "k & Mean overlap $\\phi(k)$ & Std & Min & Max" + ROW_END,
        "\\midrule",
    ]
    for k in ks:
        lines.append(
            f"{k} & {_fmt(mean[str(k)])} & {_fmt(std[str(k)])} & {_fmt(vmin[str(k)])} & {_fmt(vmax[str(k)])}" + ROW_END
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svd_table(svd: Dict[str, object], out_path: Path) -> None:
    mean = svd["mean_energy_at_k"]
    k_for = svd["k_for_threshold"]

    lines = [
        "\\begin{tabular}{rr}",
        "\\toprule",
        "k & Mean cumulative energy" + ROW_END,
        "\\midrule",
    ]
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        lines.append(f"{k} & {_fmt(mean[str(k)])}" + ROW_END)
    lines.extend(
        [
            "\\midrule",
            f"$k$ for 80\\% energy & {k_for['0.8']}" + ROW_END,
            f"$k$ for 90\\% energy & {k_for['0.9']}" + ROW_END,
            f"$k$ for 95\\% energy & {k_for['0.95']}" + ROW_END,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_task_table(metrics: Dict[str, object], out_path: Path) -> None:
    tasks = metrics["config"]["tasks"]
    sample_counts = metrics["sample_counts"]
    oracle = metrics["tracks"]["oracle_lora"]["per_task"]
    lora = metrics["tracks"]["per_task_lora"]["per_task"]
    fft = metrics["tracks"]["per_task_fft"]["per_task"]

    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Task & N & Oracle & LoRA & Full-FT & Oracle-LoRA & Oracle-FT" + ROW_END,
        "\\midrule",
    ]

    for task in tasks:
        o = oracle[task]["primary_metric"]
        l = lora[task]["primary_metric"]
        f = fft[task]["primary_metric"]
        lines.append(
            f"{DISPLAY_NAME.get(task, task.upper())} & {sample_counts[task]} & {_fmt(o)} & {_fmt(l)} & {_fmt(f)} & {_fmt(o-l)} & {_fmt(o-f)}"
            + ROW_END
        )

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_table(metrics: Dict[str, object], out_path: Path) -> None:
    oracle = metrics["tracks"]["oracle_lora"]["aggregate"]
    lora = metrics["tracks"]["per_task_lora"]["aggregate"]
    fft = metrics["tracks"]["per_task_fft"]["aggregate"]

    keys = ["macro_primary", "weighted_primary", "macro_accuracy", "weighted_accuracy"]
    labels = {
        "macro_primary": "Macro primary",
        "weighted_primary": "Weighted primary",
        "macro_accuracy": "Macro accuracy",
        "weighted_accuracy": "Weighted accuracy",
    }

    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Metric & Oracle & LoRA & Full-FT & Oracle-LoRA & Oracle-FT" + ROW_END,
        "\\midrule",
    ]
    for key in keys:
        o = oracle[key]
        l = lora[key]
        f = fft[key]
        lines.append(f"{labels[key]} & {_fmt(o)} & {_fmt(l)} & {_fmt(f)} & {_fmt(o-l)} & {_fmt(o-f)}" + ROW_END)

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_overview_table(
    overlap: Dict[str, object],
    svd: Dict[str, object],
    metrics: Dict[str, object],
    out_path: Path,
) -> None:
    oracle_agg = metrics["tracks"]["oracle_lora"]["aggregate"]
    fft_agg = metrics["tracks"]["per_task_fft"]["aggregate"]
    runtime = metrics["runtime_sec"]

    lines = [
        "\\begin{tabular}{p{0.28\\linewidth}p{0.67\\linewidth}}",
        "\\toprule",
        "Experiment & Key measured result" + ROW_END,
        "\\midrule",
        (
            "Exp 1: Full-FT SVD (SST-2 q/v) & "
            f"Mean cumulative energy at k=64 is {_fmt(svd['mean_energy_at_k']['64'])}; "
            f"k={svd['k_for_threshold']['0.8']} reaches 80\\% energy."
            + ROW_END
        ),
        (
            "Exp 2: LoRA-FT subspace overlap (SST-2 q/v) & "
            f"Mean overlap rises from {_fmt(overlap['mean']['1'])} at k=1 to {_fmt(overlap['mean']['64'])} at k=64 "
            f"(computed over {overlap['num_matrices']} matrices)."
            + ROW_END
        ),
        (
            "Exp 3: Oracle-routed LoRA experts across 6 GLUE tasks & "
            f"Weighted primary = {_fmt(oracle_agg['weighted_primary'])}; "
            f"vs per-task Full-FT weighted primary delta = {_fmt(oracle_agg['weighted_primary'] - fft_agg['weighted_primary'])}; "
            f"end-to-end runtime = {_fmt(runtime['total'], 2)}s."
            + ROW_END
        ),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_overlap_plot(overlap: Dict[str, object], out_path: Path) -> None:
    ks = overlap["k_values"]
    mean = np.array([overlap["mean"][str(k)] for k in ks], dtype=np.float64)
    std = np.array([overlap["std"][str(k)] for k in ks], dtype=np.float64)

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(ks, mean, marker="o", linewidth=2.2, color="#1D39C4", label="Mean overlap")
    plt.fill_between(
        ks,
        np.clip(mean - std, 0.0, None),
        np.clip(mean + std, 0.0, 1.0),
        alpha=0.2,
        color="#1D39C4",
        label="+/- 1 std",
    )
    plt.xscale("log", base=2)
    plt.xticks(ks, [str(k) for k in ks])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Top-k subspace dimension")
    plt.ylabel("Overlap phi(k)")
    plt.title("LoRA vs Full-FT Subspace Overlap (SST-2 q/v matrices)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_svd_plot(svd: Dict[str, object], out_path: Path) -> None:
    ks = [1, 2, 4, 8, 16, 32, 64, 128]
    vals = [svd["mean_energy_at_k"][str(k)] for k in ks]

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(ks, vals, marker="o", linewidth=2.2, color="#389E0D")
    plt.xscale("log", base=2)
    plt.xticks(ks, [str(k) for k in ks])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Top-k singular directions")
    plt.ylabel("Mean cumulative spectral energy")
    plt.title("Full-FT Delta Spectral Energy Keypoints (SST-2 q/v)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_per_task_plot(metrics: Dict[str, object], out_path: Path) -> None:
    tasks = metrics["config"]["tasks"]
    labels = [DISPLAY_NAME.get(t, t.upper()) for t in tasks]

    oracle = [metrics["tracks"]["oracle_lora"]["per_task"][t]["primary_metric"] for t in tasks]
    lora = [metrics["tracks"]["per_task_lora"]["per_task"][t]["primary_metric"] for t in tasks]
    fft = [metrics["tracks"]["per_task_fft"]["per_task"][t]["primary_metric"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.25

    plt.figure(figsize=(10.8, 5.4))
    plt.bar(x - width, oracle, width=width, label="Oracle-routed LoRA", color="#1D39C4")
    plt.bar(x, lora, width=width, label="Per-task LoRA", color="#389E0D")
    plt.bar(x + width, fft, width=width, label="Per-task Full-FT", color="#D46B08")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Primary metric")
    plt.title("Oracle LoRA vs Per-task LoRA vs Per-task Full-FT")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_delta_plot(metrics: Dict[str, object], out_path: Path) -> None:
    tasks = metrics["config"]["tasks"]
    labels = [DISPLAY_NAME.get(t, t.upper()) for t in tasks]
    deltas = []
    for t in tasks:
        o = metrics["tracks"]["oracle_lora"]["per_task"][t]["primary_metric"]
        f = metrics["tracks"]["per_task_fft"]["per_task"][t]["primary_metric"]
        deltas.append(o - f)

    x = np.arange(len(tasks))
    colors = ["#389E0D" if d >= 0 else "#CF1322" for d in deltas]

    plt.figure(figsize=(10.8, 4.8))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.bar(x, deltas, color=colors)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Oracle - Full-FT (primary)")
    plt.title("Per-task delta: Oracle-routed LoRA minus Full-FT")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_all_in_one_tex(out_root: Path) -> None:
    text = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\title{Presentation Results Pack: LoRA Geometry + Oracle Routing}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Experiment Overview}
\input{tables/table_experiment_overview.tex}

\section*{Experiment 1: Full-FT Spectral Energy (SST-2)}
\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{figures/svd_energy_keypoints.png}
\caption{Mean cumulative spectral energy across SST-2 query/value update matrices.}
\end{figure}
\input{tables/table_svd_energy_summary.tex}

\section*{Experiment 2: LoRA vs Full-FT Subspace Overlap (SST-2)}
\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{figures/subspace_overlap_mean_std.png}
\caption{Subspace overlap $\phi(k)$ with mean and $\pm 1$ std across matrices.}
\end{figure}
\input{tables/table_subspace_overlap.tex}

\section*{Experiment 3: Oracle-Routed LoRA vs Baselines (6 GLUE tasks)}
\begin{figure}[H]
\centering
\includegraphics[width=0.90\linewidth]{figures/oracle_vs_baselines_per_task.png}
\caption{Primary metric comparison by task.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.90\linewidth]{figures/oracle_minus_fft_delta.png}
\caption{Per-task primary-metric delta between oracle-routed LoRA and per-task Full-FT.}
\end{figure}

\input{tables/table_oracle_vs_baselines_per_task.tex}

\vspace{0.5em}
\input{tables/table_oracle_vs_baselines_aggregate.tex}

\end{document}
"""
    (out_root / "presentation_results_all_in_one.tex").write_text(text, encoding="utf-8")


def write_figure_bullets(
    out_root: Path,
    overlap: Dict[str, object],
    svd: Dict[str, object],
    metrics: Dict[str, object],
) -> None:
    oracle_agg = metrics["tracks"]["oracle_lora"]["aggregate"]
    fft_agg = metrics["tracks"]["per_task_fft"]["aggregate"]
    text = "\n".join(
        [
            "- Figure: svd_energy_keypoints.png",
            f"  - Mean cumulative energy reaches {svd['mean_energy_at_k']['64']:.4f} at k=64.",
            f"  - 80% mean energy is reached at k={svd['k_for_threshold']['0.8']}.",
            "  - Use this to argue full-FT updates are directionally concentrated.",
            "- Figure: subspace_overlap_mean_std.png",
            f"  - Mean overlap phi(1)={overlap['mean']['1']:.4f}, phi(16)={overlap['mean']['16']:.4f}, phi(64)={overlap['mean']['64']:.4f}.",
            f"  - Computed across {overlap['num_matrices']} SST-2 query/value matrices.",
            "  - Use this to show LoRA captures part of the full-FT update subspace.",
            "- Figure: oracle_vs_baselines_per_task.png",
            "  - Oracle-routed LoRA and per-task LoRA are identical (oracle routing consistency = 0 diff).",
            "  - Oracle-routed LoRA beats per-task Full-FT on RTE, MNLI, and QNLI, but trails on SST-2, MRPC, and QQP.",
            "- Figure: oracle_minus_fft_delta.png",
            f"  - Weighted primary delta (oracle minus Full-FT): {oracle_agg['weighted_primary'] - fft_agg['weighted_primary']:.4f}.",
            "  - Task deltas make the tradeoff transparent instead of reporting only one average.",
        ]
    )
    (out_root / "figure_bullets.txt").write_text(text + "\n", encoding="utf-8")


def write_speaker_script(
    out_root: Path,
    overlap: Dict[str, object],
    svd: Dict[str, object],
    metrics: Dict[str, object],
) -> None:
    oracle_agg = metrics["tracks"]["oracle_lora"]["aggregate"]
    fft_agg = metrics["tracks"]["per_task_fft"]["aggregate"]
    text = "\n".join(
        [
            "Slide script: LoRA geometry and oracle-routed expert results",
            "",
            "1) Setup",
            "We evaluated RoBERTa-base checkpoints in three steps: full-FT geometry on SST-2, LoRA-vs-full-FT overlap on SST-2, and oracle-routed LoRA across six GLUE tasks.",
            "",
            "2) Experiment 1: Full-FT spectral energy",
            f"Across 24 SST-2 query/value matrices, mean cumulative spectral energy is {svd['mean_energy_at_k']['64']:.4f} at k=64.",
            f"Mean energy crosses 80% at k={svd['k_for_threshold']['0.8']}.",
            "Interpretation: full fine-tuning updates are not uniform; energy is concentrated in a subset of directions.",
            "",
            "3) Experiment 2: Subspace overlap",
            f"LoRA/full-FT overlap increases from phi(1)={overlap['mean']['1']:.4f} to phi(64)={overlap['mean']['64']:.4f}.",
            f"These overlap stats are averaged over {overlap['num_matrices']} query/value matrices.",
            "Interpretation: LoRA does not match all full-FT directions, but it tracks part of the task-relevant subspace.",
            "",
            "4) Experiment 3: Oracle-routed LoRA vs baselines",
            "Oracle-routed LoRA equals per-task LoRA exactly in this setup, because routing is task-oracle and each sample is sent to its matching expert.",
            f"Weighted primary metric is {oracle_agg['weighted_primary']:.4f} for oracle-routed LoRA versus {fft_agg['weighted_primary']:.4f} for per-task full-FT.",
            f"That is a weighted primary delta of {oracle_agg['weighted_primary'] - fft_agg['weighted_primary']:.4f}.",
            "Per-task deltas show gains on RTE, MNLI, and QNLI, and drops on SST-2, MRPC, and QQP.",
            "",
            "5) Caveat",
            "This is an oracle-routing upper bound, not a learned router. The value is in showing feasibility and transparent per-task behavior.",
        ]
    )
    (out_root / "speaker_script.txt").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    root = _repo_root() / "codex_presentation_results"
    data_dir = root / "data"
    fig_dir = root / "figures"
    table_dir = root / "tables"

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    overlap_svd = _read_json(data_dir / "overlap_svd_metrics.json")
    metrics = _read_json(data_dir / "oracle_routing_metrics_snapshot.json")

    overlap = overlap_svd["overlap"]
    svd = overlap_svd["svd"]

    write_overlap_table(overlap, table_dir / "table_subspace_overlap.tex")
    write_svd_table(svd, table_dir / "table_svd_energy_summary.tex")
    write_per_task_table(metrics, table_dir / "table_oracle_vs_baselines_per_task.tex")
    write_aggregate_table(metrics, table_dir / "table_oracle_vs_baselines_aggregate.tex")
    write_experiment_overview_table(overlap, svd, metrics, table_dir / "table_experiment_overview.tex")

    make_overlap_plot(overlap, fig_dir / "subspace_overlap_mean_std.png")
    make_svd_plot(svd, fig_dir / "svd_energy_keypoints.png")
    make_per_task_plot(metrics, fig_dir / "oracle_vs_baselines_per_task.png")
    make_delta_plot(metrics, fig_dir / "oracle_minus_fft_delta.png")

    write_all_in_one_tex(root)
    write_figure_bullets(root, overlap, svd, metrics)
    write_speaker_script(root, overlap, svd, metrics)

    print(f"Wrote assets under: {root}")


if __name__ == "__main__":
    main()
