import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark_eval"

METHODS = ["snapkv", "expected_attention", "knorm", "tova", "streaming_llm"]
METHOD_LABELS = ["SnapKV", "ExpectedAttn", "KNorm", "TOVA", "StreamingLLM"]

TASK_GROUPS = {
    "NIAH": {"prefix": "niah", "baseline_dir": "niah_16k_baseline"},
    "Multi-Hop\nTracing":  {"prefix": "vt",   "baseline_dir": "vt_16k_baseline"},
    "Question\nAnswering": {"prefix": "qa",   "baseline_dir": "qa_16k_baseline"},
    "Aggregation":         {"prefix": ["cwe", "fwe"], "baseline_dir": "aggregation_16k_baseline"},
}


def avg_metrics(metrics: dict) -> float:
    vals = [v["string_match"] for v in metrics.values() if "string_match" in v]
    return np.mean(vals) if vals else 0.0


def load_score(dir_pattern: str) -> float | None:
    matches = [d for d in RESULTS_DIR.iterdir() if dir_pattern in d.name and "16384" in d.name and "Qwen2.5" not in d.name]
    if not matches:
        return None
    scores = []
    for d in matches:
        mf = d / "metrics.json"
        if mf.exists():
            scores.append(avg_metrics(json.loads(mf.read_text())))
    return np.mean(scores) if scores else None


# Build matrix: rows=tasks, cols=methods
task_labels = list(TASK_GROUPS.keys())
retained = np.full((len(task_labels), len(METHODS)), np.nan)
raw_scores = np.full((len(task_labels), len(METHODS)), np.nan)

for row_i, (task_label, cfg) in enumerate(TASK_GROUPS.items()):
    baseline = load_score(cfg["baseline_dir"])
    if baseline is None or baseline == 0:
        continue

    for col_i, method in enumerate(METHODS):
        # Find the run directory for this task/method/16k
        prefix = cfg["prefix"] if isinstance(cfg["prefix"], list) else [cfg["prefix"]]
        scenario_hints = {
            "niah": "needle_in_a_haystack",
            "vt": "multi_hop_tracing",
            "qa": "question_answering",
            "cwe": "aggregation",
            "fwe": "aggregation",
        }
        scenario = scenario_hints[prefix[0]]
        pattern = f"{scenario}_16k__ruler__16384__Qwen--Qwen3-8B__{method}__budget0.50"
        score = load_score(pattern)
        if score is not None:
            raw_scores[row_i, col_i] = score
            retained[row_i, col_i] = (score / baseline) * 100


# Colormap: red → yellow → green, centered at 100
cmap = mcolors.LinearSegmentedColormap.from_list(
    "rg", ["#d62728", "#ff7f0e", "#ffdd57", "#2ca02c"], N=256
)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(retained, cmap=cmap, vmin=40, vmax=110, aspect="auto")

ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels(METHOD_LABELS, fontsize=11)
ax.set_yticks(range(len(task_labels)))
ax.set_yticklabels(task_labels, fontsize=11)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

# Annotate cells
for r in range(len(task_labels)):
    for c in range(len(METHODS)):
        val = retained[r, c]
        raw = raw_scores[r, c]
        if not np.isnan(val):
            label = f"{val:.0f}%"
            color = "white" if val < 65 else "black"
            ax.text(c, r, label, ha="center", va="center", fontsize=8.5, color=color, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Quality Retained vs. Baseline (%)", fontsize=10)
cbar.ax.axhline(100, color="black", linewidth=1.5, linestyle="--")

ax.set_title("Quality Retained at 16k Context (budget = 0.5, Qwen3-8B)\nCell shows raw score / % of baseline retained",
             fontsize=11, pad=14)

plt.tight_layout()
out = Path(__file__).parent.parent / "results" / "quality_heatmap_16k.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
