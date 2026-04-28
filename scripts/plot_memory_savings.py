import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmark_eval"

METHOD_LABELS = {
    "snapkv": "SnapKV",
    "expected_attention": "ExpectedAttn",
    "knorm": "KNorm",
    "tova": "TOVA",
    "streaming_llm": "StreamingLLM",
}
CONTEXT_LENGTHS = [4096, 8192, 16384]
CONTEXT_LABELS = ["4k", "8k", "16k"]

# Collect peak_gpu_memory_mb keyed by (context_length, method)
baseline_memory = {cl: [] for cl in CONTEXT_LENGTHS}
method_memory = {cl: {m: [] for m in METHOD_LABELS} for cl in CONTEXT_LENGTHS}

for run_dir in RESULTS_DIR.iterdir():
    stats_file = run_dir / "run_stats.json"
    if not stats_file.exists():
        continue

    # Skip non-Qwen3-8B runs
    if "Qwen--Qwen2.5" in run_dir.name:
        continue

    with open(stats_file) as f:
        stats = json.load(f)
    mem = stats.get("peak_gpu_memory_mb")
    if mem is None:
        continue

    # Parse context length from dir name
    ctx_match = re.search(r"__(4096|8192|16384)__", run_dir.name)
    if not ctx_match:
        continue
    ctx = int(ctx_match.group(1))

    if "no_compression" in run_dir.name:
        baseline_memory[ctx].append(mem)
    else:
        for method in METHOD_LABELS:
            if f"__{method}__" in run_dir.name:
                method_memory[ctx][method].append(mem)
                break

# Average baselines per context length
avg_baseline = {ctx: np.mean(vals) for ctx, vals in baseline_memory.items() if vals}

# Compute average savings per (context, method)
savings = {cl: {} for cl in CONTEXT_LENGTHS}
for ctx in CONTEXT_LENGTHS:
    base = avg_baseline.get(ctx)
    if base is None:
        continue
    for method, vals in method_memory[ctx].items():
        if vals:
            savings[ctx][method] = base - np.mean(vals)

# Plot
methods = list(METHOD_LABELS.keys())
x = np.arange(len(CONTEXT_LABELS))
width = 0.15
colors = ["#2a6f97", "#e76f51", "#57cc99", "#f4a261", "#9b5de5"]

fig, ax = plt.subplots(figsize=(11, 6))

for i, (method, color) in enumerate(zip(methods, colors)):
    vals = [savings[ctx].get(method, 0) for ctx in CONTEXT_LENGTHS]
    offset = (i - len(methods) / 2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS[method], color=color, alpha=0.88)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#333333")

ax.set_xlabel("Context Length", fontsize=12)
ax.set_ylabel("Memory Saved vs. Baseline (MB)", fontsize=12)
ax.set_title("KV Cache Memory Savings by Press and Context Length\n(budget = 0.5, Qwen3-8B)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(CONTEXT_LABELS, fontsize=11)
ax.legend(title="Press", fontsize=9, title_fontsize=10)
ax.set_ylim(0, max(v for ctx in savings.values() for v in ctx.values()) * 1.25)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = Path(__file__).parent.parent / "results" / "memory_savings.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()