"""
Core feature evaluation logic for binary classification datasets.

Called by evaluate_features.py in each run folder:
    python classification/run_10k/evaluate_features.py
"""
import math
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeClassifier


def _show_scrollable(fig, title="Feature Space"):
    """Embed a matplotlib Figure in a native Tkinter window with vertical and horizontal scrollbars."""
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title(title)

    fig_w_px = int(fig.get_figwidth()  * fig.dpi)
    fig_h_px = int(fig.get_figheight() * fig.dpi)
    sw       = root.winfo_screenwidth()
    sh       = root.winfo_screenheight()
    win_w    = min(fig_w_px + 20, sw)
    win_h    = min(fig_h_px + 55, int(sh * 0.93))
    root.geometry(f"{win_w}x{win_h}+0+0")

    # ── Fixed toolbar at the top ──────────────────────────────────────────────
    toolbar_row = tk.Frame(root)
    toolbar_row.pack(side=tk.TOP, fill=tk.X)

    # ── Scrollable region below ───────────────────────────────────────────────
    scroll_frame = tk.Frame(root)
    scroll_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # hbar must be packed before the canvas so it claims the bottom strip first
    hbar    = tk.Scrollbar(scroll_frame, orient=tk.HORIZONTAL)
    hbar.pack(side=tk.BOTTOM, fill=tk.X)
    vbar    = tk.Scrollbar(scroll_frame, orient=tk.VERTICAL)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    tk_canv = tk.Canvas(scroll_frame, yscrollcommand=vbar.set, xscrollcommand=hbar.set)
    tk_canv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vbar.config(command=tk_canv.yview)
    hbar.config(command=tk_canv.xview)

    inner = tk.Frame(tk_canv)
    tk_canv.create_window((0, 0), window=inner, anchor="nw")

    mpl_canvas = FigureCanvasTkAgg(fig, master=inner)
    mpl_canvas.draw()
    mpl_canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(mpl_canvas, toolbar_row)
    toolbar.update()
    toolbar.pack(fill=tk.X)

    def _update_scroll(event=None):
        tk_canv.configure(scrollregion=tk_canv.bbox("all"))
    inner.bind("<Configure>", _update_scroll)

    def _on_wheel(event):
        tk_canv.yview_scroll(int(-1 * (event.delta / 120)), "units")
    tk_canv.bind_all("<MouseWheel>", _on_wheel)

    def _on_shift_wheel(event):
        tk_canv.xview_scroll(int(-1 * (event.delta / 120)), "units")
    tk_canv.bind_all("<Shift-MouseWheel>", _on_shift_wheel)

    root.mainloop()


def evaluate(df, features, target, label="", n_samples=None):
    """
    Print feature metrics for a binary classification dataset.

    Sections:
      1. Dataset summary
      2. Per-feature descriptive stats
      3. Point-biserial correlation with target  (sorted by |r|)
      4. Decision Tree feature importance        (fit on full sample, max_depth=8)
      5. Inter-feature correlation               (multicollinearity, pairs |r| > 0.7)
    """
    W = 64

    print(f"\n{'='*W}")
    print(f"  Feature Evaluation : {label}")
    print(f"{'='*W}")

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    X = df[features]
    y = df[target]

    OUTLIER_COL = "is_outlier"
    outlier_mask = df[OUTLIER_COL] != "none" if OUTLIER_COL in df.columns else None

    hits   = int(y.sum())
    misses = len(y) - hits

    total_missing = int(X.isnull().sum().sum())
    print(f"\n  Dataset")
    print(f"    Rows     : {len(df):,}")
    print(f"    Features : {len(features)}")
    print(f"    Target   : {target}  (hits={hits:,}  misses={misses:,}  balance={hits/len(y):.1%})")
    print(f"    Missing  : {total_missing}")
    if outlier_mask is not None and outlier_mask.any():
        for otype, cnt in df.loc[outlier_mask, OUTLIER_COL].value_counts().items():
            print(f"    Outliers [{otype}] : {int(cnt):,}  ({cnt / len(df):.1%})")

    print(f"\n  Feature Stats")
    print(f"  {'Feature':<28} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Skew':>7} {'NaN':>5}")
    print(f"  {'-'*83}")
    for col in features:
        s    = X[col]
        skew = float(s.skew())
        flag = "  **" if abs(skew) > 1 else ""
        print(f"  {col:<28} {s.mean():>10.3f} {s.std():>10.3f} "
              f"{s.min():>10.3f} {s.max():>10.3f} {skew:>7.3f} {int(s.isnull().sum()):>5}{flag}")

    print(f"\n  Correlation with '{target}'  [Pearson r == point-biserial, sorted by |r|]")
    print(f"  {'Feature':<28} {'r':>8}  Bar")
    print(f"  {'-'*55}")
    corrs = sorted(
        [(col, float(X[col].corr(y))) for col in features],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    for col, r in corrs:
        bar = "#" * int(abs(r) * 25)
        print(f"  {col:<28} {r:>8.4f}  {bar}")

    print(f"\n  Decision Tree Feature Importance  [max_depth=8, classifier]")
    print(f"  {'Feature':<28} {'Importance':>12}  Bar")
    print(f"  {'-'*55}")
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X.values, y.values)
    importances = sorted(
        zip(features, dt.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for col, imp in importances:
        bar = "#" * int(imp * 40)
        print(f"  {col:<28} {imp:>12.4f}  {bar}")

    print(f"\n  Inter-feature Correlation  [pairs with |r| > 0.7]")
    corr_matrix = X.corr().abs()
    high_pairs = [
        (a, b, float(corr_matrix.loc[a, b]))
        for i, a in enumerate(features)
        for b in features[i + 1:]
        if corr_matrix.loc[a, b] > 0.7
    ]
    if high_pairs:
        high_pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"  {'Feature A':<28} {'Feature B':<28} {'|r|':>6}")
        print(f"  {'-'*65}")
        for a, b, val in high_pairs:
            print(f"  {a:<28} {b:<28} {val:>6.4f}  ** high")
    else:
        print(f"  No pairs above 0.7 -- no obvious multicollinearity.")

    print(f"\n  Outlier Detection  [IQR method: < Q1-1.5·IQR  or  > Q3+1.5·IQR]")
    print(f"  {'Feature':<28} {'Fence Low':>12} {'Fence High':>12} {'N_out':>8} {'%':>7}")
    print(f"  {'-'*73}")
    outlier_rows = set()
    for col in features:
        s   = X[col].dropna()
        q1  = float(s.quantile(0.25))
        q3  = float(s.quantile(0.75))
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        mask   = (X[col] < lo) | (X[col] > hi)
        n_out  = int(mask.sum())
        pct    = n_out / len(X) * 100
        flag   = "  **" if pct > 5 else ""
        outlier_rows.update(X.index[mask].tolist())
        print(f"  {col:<28} {lo:>12.3f} {hi:>12.3f} {n_out:>8,} {pct:>6.2f}%{flag}")
    n_any = len(outlier_rows)
    print(f"\n  Rows with at least one outlier feature: {n_any:,} / {len(X):,}  ({n_any/len(X)*100:.2f}%)")
    if outlier_mask is not None and outlier_mask.any():
        print(f"\n  Outlier Recall  [% of labeled outliers caught by IQR]")
        print(f"  {'Type':<16} {'Actual':>8} {'Caught':>8} {'%':>8}")
        print(f"  {'-'*43}")
        for otype in sorted(df.loc[outlier_mask, OUTLIER_COL].unique()):
            type_mask = df[OUTLIER_COL] == otype
            type_idx  = set(X.index[type_mask].tolist())
            n_t = len(type_idx)
            n_c = len(type_idx & outlier_rows)
            print(f"  {otype:<16} {n_t:>8,} {n_c:>8,} {n_c/n_t*100:>7.1f}%")
        all_idx = set(X.index[outlier_mask].tolist())
        n_a     = len(all_idx)
        n_c_a   = len(all_idx & outlier_rows)
        print(f"  {'(total)':<16} {n_a:>8,} {n_c_a:>8,} {n_c_a/n_a*100:>7.1f}%")

    print()


def plot_outliers(df, features, target, label="", n_samples=None):
    """
    One subplot per feature: histogram with IQR fence lines, shaded outlier tails,
    and an outlier-count annotation. Displayed in a scrollable Tkinter window.
    """
    N_COLS = 4

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    X            = df[features]
    OUTLIER_COL  = "is_outlier"
    outlier_mask = df[OUTLIER_COL] != "none" if OUTLIER_COL in df.columns else None
    n       = len(features)
    n_rows  = math.ceil(n / N_COLS)

    fig  = Figure(figsize=(N_COLS * 5, n_rows * 4), dpi=100)
    axes = fig.subplots(n_rows, N_COLS, squeeze=False)

    fig.suptitle(
        f"Outlier Detection — {label}  ({len(df):,} samples)  [IQR method]",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(top=0.94, hspace=0.6, wspace=0.38)

    for ax in axes.flat:
        ax.set_visible(False)

    for feat_idx, col in enumerate(features):
        row_idx = feat_idx // N_COLS
        col_idx = feat_idx  % N_COLS
        ax = axes[row_idx][col_idx]
        ax.set_visible(True)

        vals = X[col].dropna().values
        q1   = float(np.percentile(vals, 25))
        q3   = float(np.percentile(vals, 75))
        iqr  = q3 - q1
        lo   = q1 - 1.5 * iqr
        hi   = q3 + 1.5 * iqr

        mask  = (X[col] < lo) | (X[col] > hi)
        n_out = int(mask.sum())
        pct   = n_out / len(X) * 100

        _, _, patches = ax.hist(vals, bins=50, color="steelblue", alpha=0.75, edgecolor="none")
        for patch in patches:
            x_l = patch.get_x()
            x_r = x_l + patch.get_width()
            if x_r <= lo or x_l >= hi:
                patch.set_facecolor("tomato")
                patch.set_alpha(0.9)

        ax.axvline(lo, color="darkred", linewidth=1.4, linestyle="--", label=f"lo {lo:.2f}")
        ax.axvline(hi, color="darkred", linewidth=1.4, linestyle="--", label=f"hi {hi:.2f}")

        xmin, xmax = float(vals.min()), float(vals.max())
        if xmin < lo:
            ax.axvspan(xmin, lo, alpha=0.10, color="red", zorder=0)
        if xmax > hi:
            ax.axvspan(hi, xmax, alpha=0.10, color="red", zorder=0)

        flag = "  **" if pct > 5 else ""
        ax.annotate(
            f"outliers: {n_out:,}  ({pct:.2f}%){flag}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            color="darkred" if pct > 5 else "dimgray",
        )
        ax.set_title(col, fontsize=10, fontweight="bold")
        if outlier_mask is not None:
            out_vals = X[col][outlier_mask].values
            if len(out_vals):
                ax.plot(out_vals, np.zeros(len(out_vals)), '|',
                        transform=ax.get_xaxis_transform(),
                        color="gold", markersize=8, markeredgewidth=1.5,
                        clip_on=False, label="outlier", zorder=5)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("Count" if col_idx == 0 else "")
        ax.tick_params(labelsize=8)

    _show_scrollable(fig, f"Outliers — {label}")


def plot_features(df, features, target, label="", n_samples=None, clean_only=False):
    """
    Multi-row figure: N_COLS features per row, two subplot rows per feature group
    (overlapping histograms on top, box plots below). Displayed in a scrollable
    Tkinter window — drag the right-hand scrollbar or use the mouse wheel.
    """
    N_COLS = 4

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    if clean_only and "is_outlier" in df.columns:
        df = df[df["is_outlier"] == "none"].reset_index(drop=True)

    miss         = df[df[target] == 0]
    hit          = df[df[target] == 1]
    n           = len(features)
    n_feat_rows = math.ceil(n / N_COLS)
    n_plot_rows = n_feat_rows * 2

    fig  = Figure(figsize=(N_COLS * 5, n_feat_rows * 8), dpi=100)
    axes = fig.subplots(n_plot_rows, N_COLS, squeeze=False)

    suffix = "  [outliers excluded]" if clean_only else ""
    fig.suptitle(
        f"Feature Space — {label}  ({len(df):,} samples){suffix}",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(top=0.96, hspace=0.45, wspace=0.38)

    for ax in axes.flat:
        ax.set_visible(False)

    for feat_idx, col in enumerate(features):
        feat_row  = feat_idx // N_COLS
        col_idx   = feat_idx  % N_COLS

        ax_hist = axes[feat_row * 2,     col_idx]
        ax_box  = axes[feat_row * 2 + 1, col_idx]
        ax_hist.set_visible(True)
        ax_box.set_visible(True)

        miss_vals = miss[col].values
        hit_vals  = hit[col].values

        # ── Overlapping histograms ────────────────────────────────────────────
        bins = np.linspace(
            min(miss_vals.min(), hit_vals.min()),
            max(miss_vals.max(), hit_vals.max()),
            50,
        )
        ax_hist.hist(miss_vals, bins=bins, density=True, alpha=0.5,
                     color="tomato", label="miss")
        ax_hist.hist(hit_vals,  bins=bins, density=True, alpha=0.5,
                     color="steelblue", label="hit")
        ax_hist.set_title(col, fontsize=10, fontweight="bold")
        ax_hist.legend(fontsize=8)
        ax_hist.set_ylabel("Density" if col_idx == 0 else "")
        ax_hist.tick_params(labelsize=8)

        # ── Box plots ─────────────────────────────────────────────────────────
        ax_box.boxplot(
            [miss_vals, hit_vals],
            labels=["miss", "hit"],
            patch_artist=True,
            boxprops=dict(facecolor="none"),
            medianprops=dict(color="black", linewidth=2),
        )
        ax_box.set_xlabel(col, fontsize=9)
        ax_box.set_ylabel(col if col_idx == 0 else "")
        ax_box.tick_params(labelsize=8)

    _show_scrollable(fig, label)
