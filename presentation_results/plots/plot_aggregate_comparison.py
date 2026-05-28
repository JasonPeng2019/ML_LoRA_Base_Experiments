"""
Aggregate metric comparison: macro and weighted primary across methods.
Saves to presentation_results/plots/aggregate_comparison.png
Data hardcoded from outputs/oracle_routing_metrics.json
"""
import os
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["Oracle LoRA MoE", "Per-task LoRA", "Per-task FFT"]
MACRO = [0.8690, 0.8690, 0.8539]
WEIGHTED = [0.8919, 0.8919, 0.8937]

# Parameter efficiency numbers (from model card / download_verify_summary.json)
TRAINABLE_M = [1.18, 1.18, 125.8]  # millions per task adapter/model

x = np.arange(len(METHODS))
width = 0.32

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Panel A: macro vs weighted ---
ax = axes[0]
b1 = ax.bar(x - width / 2, MACRO, width=width, label="Macro primary", color="#1D39C4", zorder=3)
b2 = ax.bar(x + width / 2, WEIGHTED, width=width, label="Weighted primary", color="#389E0D", zorder=3)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8)

ax.set_ylim(0.82, 0.92)
ax.set_xticks(x)
ax.set_xticklabels(METHODS, fontsize=9)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Aggregate Performance\n(Macro and Weighted Primary Metric)", fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.legend(fontsize=9)

# Highlight macro advantage
ax.annotate("+1.5pp macro\nover FFT",
            xy=(x[0] - width / 2, MACRO[0]),
            xytext=(x[0] - 0.55, 0.860),
            fontsize=8, color="#1D39C4",
            arrowprops=dict(arrowstyle="->", color="#1D39C4", lw=1.0))

# --- Panel B: trainable parameters ---
ax2 = axes[1]
colors = ["#1D39C4", "#389E0D", "#D46B08"]
bars = ax2.bar(x, TRAINABLE_M, color=colors, zorder=3, width=0.5)

for bar, val in zip(bars, TRAINABLE_M):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
             f"{val:.1f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(METHODS, fontsize=9)
ax2.set_ylabel("Trainable Parameters (M) per task", fontsize=10)
ax2.set_title("Parameter Efficiency\n(Trainable parameters per task)", fontsize=10)
ax2.grid(axis="y", alpha=0.3, zorder=0)

# Annotation for ratio
ax2.annotate("107× fewer params\nthan FFT",
             xy=(x[0], TRAINABLE_M[0]),
             xytext=(x[0] - 0.3, 60),
             fontsize=9, color="#1D39C4",
             arrowprops=dict(arrowstyle="->", color="#1D39C4", lw=1.2))

plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "aggregate_comparison.png")
plt.savefig(out_path, dpi=180)
print(f"Saved: {out_path}")
plt.close()
