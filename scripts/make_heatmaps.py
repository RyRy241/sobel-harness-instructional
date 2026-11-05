#!/usr/bin/env python3
"""
Parse the output of run-cp5-cuda-configs-gpu-perlmutter.sh (slurm-*.out) and produce three
heatmaps (runtime, achieved occupancy, % peak DRAM bandwidth) using the same plotting
approach as heatmap_plot_hw5.py.

usage: python3 make_heatmaps.py <slurm-XXXX.out> [outdir]
"""
import re, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

threads_per_block = ['32', '64', '128', '256', '512', '1024']                # y axis
thread_blocks     = ["1", "4", "16", "64", "256", "1024", "4096"]             # x axis

log = open(sys.argv[1]).read()
outdir = sys.argv[2] if len(sys.argv) > 2 else "."

shape = (len(threads_per_block), len(thread_blocks))
runtime, occ, dram = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)

# split log into one chunk per configuration
chunks = re.split(r"Working on config N=(\d+), B=(\d+)", log)
for k in range(1, len(chunks), 3):
    N, B, body = chunks[k], chunks[k+1], chunks[k+2]
    i, j = threads_per_block.index(N), thread_blocks.index(B)
    m = re.search(r"gpu__time_duration\.avg\s+(\S+)\s+([\d.,]+)", body)
    if m:
        v = float(m.group(2).replace(",", ""))
        unit = m.group(1)
        to_ms = {"nsecond":1e-6,"ns":1e-6,"usecond":1e-3,"us":1e-3,"msecond":1.0,"ms":1.0,"second":1e3,"s":1e3}
        runtime[i, j] = v * to_ms[unit]
    m = re.search(r"Achieved Occupancy\s+%\s+([\d.,]+)", body)
    if m: occ[i, j] = float(m.group(1).replace(",", ""))
    m = re.search(r"dram__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.,]+)", body)
    if m: dram[i, j] = float(m.group(1).replace(",", ""))

def heatmap(data, title, cbar_label, fname, fmt):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(data, cmap="coolwarm")
    ax.set_xticks(np.arange(len(thread_blocks)));   ax.set_xticklabels(thread_blocks)
    ax.set_yticks(np.arange(len(threads_per_block))); ax.set_yticklabels(threads_per_block)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(threads_per_block)):
        for j in range(len(thread_blocks)):
            ax.text(j, i, "n/a" if np.isnan(data[i, j]) else fmt.format(data[i, j]),
                    ha="center", va="center", color="k", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Threads per block")
    ax.set_xlabel("Number of thread blocks")
    cb = fig.colorbar(im, ax=ax); cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=200)
    print("wrote", fname)

heatmap(runtime, "Sobel CUDA kernel runtime on Perlmutter A100", "Runtime (ms)", "heatmap_runtime.png", "{:.2f}")
heatmap(occ, "Achieved occupancy on Perlmutter A100", "Achieved occupancy (%)", "heatmap_occupancy.png", "{:.1f}")
heatmap(dram, "% of peak sustained DRAM bandwidth on Perlmutter A100", "% peak DRAM bandwidth", "heatmap_dram.png", "{:.1f}")

np.set_printoptions(precision=3, suppress=True)
print("runtime (ms)\n", runtime, "\noccupancy (%)\n", occ, "\ndram (%)\n", dram)
