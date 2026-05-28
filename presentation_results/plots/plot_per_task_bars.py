"""
Per-task primary metric bar chart: Oracle LoRA MoE vs Per-task LoRA vs Per-task FFT.
Run from any directory; saves to presentation_results/plots/per_task_bars.png
Data hardcoded from outputs/oracle_routing_metrics.json
"""
import os
import matplotlib.pyplot as plt
import numpy as np

TASKS = ["SST-2", "MRPC", "RTE", "MNLI\n(matched)", "QNLI", "QQP"]
TASK_KEYS = ["sst2", "mrpc", "rte", "mnli", "qnli", "qqp"]

ORACLE_LORA = [0.9427, 0.8709, 0.7184, 0.8780, 0.9235, 0.8804]
PER_TASK_LORA = [0.9427, 0.8709, 0.7184, 0.8780, 0.9235, 0.8804]
PER_TASK_FFT = [0.9518, 0.9105, 0.5632, 0.8760, 0.9200, 0.9021]

x = np.arange(len(TASKS))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5.5))

b1 = ax.bar(x - width, ORACLE_LORA, width=width, label="Oracle-Routed LoRA MoE",
            color="#1D39C4", zorder=3)
b2 = ax.bar(x, PER_TASK_LORA, width=width, label="Per-task LoRA (standalone)",
            color="#389E0D", zorder=3)
b3 = ax.bar(x + width, PER_TASK_FFT, width=width, label="Per-task Full Fine-Tuning",
            color="#D46B08", zorder=3)

# Annotate bars with values
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

ax.set_title("Per-task Performance: Oracle LoRA MoE vs LoRA vs Full Fine-Tuning\n"
             "(Primary metric: accuracy for most tasks; (acc+F1)/2 for MRPC and QQP)",
             fontsize=11)
ax.set_ylabel("Primary Metric Score", fontsize=11)
ax.set_ylim(0.0, 1.09)
ax.set_xticks(x)
ax.set_xticklabels(TASKS, fontsize=10)
ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.legend(fontsize=10)

# Annotate RTE standout
ax.annotate("RTE: LoRA +15.5pp\nover FFT",
            xy=(x[2] - width, ORACLE_LORA[2]),
            xytext=(x[2] - width + 0.05, 0.82),
            fontsize=8.5, color="#1D39C4",
            arrowprops=dict(arrowstyle="->", color="#1D39C4", lw=1.2))

plt.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "per_task_bars.png")
plt.savefig(out_path, dpi=180)
print(f"Saved: {out_path}")
plt.close()
