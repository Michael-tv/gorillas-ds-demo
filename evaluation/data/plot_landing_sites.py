"""Scatter plot of throw landing sites (landing_distance_m vs landing_height_m).

Unlike run_*/plot_training_data_relationships.py (which plots the target
against each input feature for a hardcoded file), this plots physical landing
sites for any parquet pool passed via --data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, REPO_ROOT)
import params
from data_generation.contract import check_contract

# Board extent matches the original game's screen (640x350 EGA, ground at
# y=335), in the same meters/pixel scale both data producers use. Clamping
# axes to this box (rather than autoscaling) keeps injected outliers from
# stretching the plot until the real data is a thin band.
_METERS_PER_PIXEL = 0.2
BOARD_WIDTH_M  = 640 * _METERS_PER_PIXEL   # 128 m
BOARD_HEIGHT_M = 335 * _METERS_PER_PIXEL   # 67 m, ground (GroundY) to screen top

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=str, default=None,
                    help="path to a parquet pool (default: params.data_path(), "
                         "i.e. params.yaml's training_data)")
parser.add_argument("--max-points", type=int, default=5000,
                    help="random subsample cap so large pools don't overplot (default: 5000)")
parser.add_argument("--seed", type=int, default=42, help="subsample seed")
parser.add_argument("--save", type=str, default=None,
                    help="if given, save the figure here instead of showing it")
args = parser.parse_args()

data_path = args.data or params.data_path()
df = check_contract(pd.read_parquet(data_path), source=os.path.basename(data_path), verbose=False)

n_total = len(df)
if n_total > args.max_points:
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_total, size=args.max_points, replace=False)
    df = df.iloc[idx]

xs = df["landing_distance_m"]
zs = df["landing_height_m"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(xs, zs, alpha=0.3, s=8, color="steelblue", linewidths=0, rasterized=True)
ax.axhline(0, color="saddlebrown", linewidth=1.5)
ax.axvline(0, color="gray", linewidth=1, linestyle="--")
ax.set_xlim(0, BOARD_WIDTH_M)
ax.set_ylim(0, BOARD_HEIGHT_M)
ax.set_xlabel("landing_distance_m (m)")
ax.set_ylabel("landing_height_m (m)")
ax.set_title(f"Landing sites -- {os.path.basename(data_path)}  "
             f"({len(df):,} of {n_total:,} rows, board {BOARD_WIDTH_M:.0f}x{BOARD_HEIGHT_M:.0f} m)")
ax.grid(alpha=0.3)
plt.tight_layout()

if args.save:
    fig.savefig(args.save, dpi=150)
    print(f"Saved -> {args.save}")
else:
    plt.show()
