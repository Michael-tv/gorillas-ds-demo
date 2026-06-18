"""
Core feature evaluation logic for regression datasets.

Called by evaluate_features.py in each run folder:
    python regression/run_raw_10k/evaluate_features.py
"""
import importlib.util
import math
import os
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeRegressor

_RHO = 1.225
_PI  = math.pi
_simulate_fn = None


def _get_simulate():
    global _simulate_fn
    if _simulate_fn is None:
        p    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "physics.py")
        spec = importlib.util.spec_from_file_location("_phys", p)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _simulate_fn = mod.simulate
    return _simulate_fn


def _bisect_velocity(el_deg, wind_x, mass, radius, Cd, height_diff, landing_dist,
                     v_lo=0.5, v_hi=100.0, tol=0.05, max_iter=50):
    sim = _get_simulate()
    def land(v):
        return sim(v, el_deg, wind_x, mass, radius, Cd, ground_z=height_diff)[-1][1]
    try:
        fl, fh = land(v_lo) - landing_dist, land(v_hi) - landing_dist
    except Exception:
        return None
    if fl * fh > 0:
        return None
    for _ in range(max_iter):
        mid = (v_lo + v_hi) / 2
        fm  = land(mid) - landing_dist
        if abs(fm) < tol:
            return mid
        if fl * fm <= 0:
            v_hi, fh = mid, fm
        else:
            v_lo, fl = mid, fm
    return (v_lo + v_hi) / 2


def _physics_curve(col, x_sweep, medians, features, v_hi=100.0):
    """Return (xs, ys) for the true physics velocity curve, sweeping `col` over x_sweep."""
    is_eng = "drag_param" in features
    is_raw = "wind_speed_ms" in features
    if not (is_eng or is_raw):
        return [], []
    vels = []
    for xv in x_sweep:
        p = dict(medians)
        p[col] = xv
        try:
            if is_eng:
                dp = max(float(p["drag_param"]), 1e-9)
                Cd = dp / (0.5 * _RHO * _PI)
                v  = _bisect_velocity(p["launch_angle_deg"], p["wind_x_ms"],
                                      1.0, 1.0, Cd,
                                      p["height_diff_m"], p["landing_distance_m"],
                                      v_hi=v_hi)
            else:
                wind_x = float(p["wind_speed_ms"]) * float(p["wind_direction_norm"])
                hd     = float(p["landing_height_m"]) - float(p["launch_height_m"])
                v      = _bisect_velocity(p["launch_angle_deg"], wind_x,
                                          p["mass_kg"], p["radius_m"], p["drag_coeff"],
                                          hd, p["landing_distance_m"],
                                          v_hi=v_hi)
        except Exception:
            v = None
        vels.append(v)
    valid = [(x, v) for x, v in zip(x_sweep, vels) if v is not None]
    if not valid:
        return [], []
    xs, vs = zip(*valid)
    return list(xs), list(vs)


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


def evaluate(data_path, features, target, label="", n_samples=None):
    """
    Print feature metrics for a regression dataset.

    Sections:
      1. Dataset summary
      2. Per-feature descriptive stats
      3. Pearson correlation with target  (sorted by |r|)
      4. Decision Tree feature importance (fit on full sample, max_depth=8)
      5. Inter-feature correlation        (multicollinearity, pairs |r| > 0.7)
      6. Outlier detection                (IQR method)
    """
    W = 64

    print(f"\n{'='*W}")
    print(f"  Feature Evaluation : {label}")
    print(f"  Data               : {data_path}")
    print(f"{'='*W}")

    df = pd.read_csv(data_path)
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    X = df[features]
    y = df[target]

    OUTLIER_COL  = "is_outlier"
    outlier_mask = df[OUTLIER_COL] != "none" if OUTLIER_COL in df.columns else None

    total_missing = int(X.isnull().sum().sum())
    print(f"\n  Dataset")
    print(f"    Rows     : {len(df):,}")
    print(f"    Features : {len(features)}")
    print(f"    Target   : {target}  (mean={float(y.mean()):.3f}  std={float(y.std()):.3f}  "
          f"min={float(y.min()):.3f}  max={float(y.max()):.3f})")
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

    print(f"\n  Pearson Correlation with '{target}'  [sorted by |r|]")
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

    print(f"\n  Decision Tree Feature Importance  [max_depth=8, regressor]")
    print(f"  {'Feature':<28} {'Importance':>12}  Bar")
    print(f"  {'-'*55}")
    dt = DecisionTreeRegressor(max_depth=8, random_state=42)
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


def plot_outliers(data_path, features, target, label="", n_samples=None):
    """
    One subplot per feature: histogram with IQR fence lines, shaded outlier tails,
    and an outlier-count annotation. Displayed in a scrollable Tkinter window.
    """
    N_COLS = 4

    df = pd.read_csv(data_path)
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


def plot_features(data_path, features, target, label="", n_samples=None, clean_only=False):
    """
    One scatter subplot per feature: feature on x-axis vs target on y-axis.
    Normal samples in blue, labeled outliers in red/orange overlay.
    Displayed in a scrollable Tkinter window.
    """
    N_COLS = 4

    df = pd.read_csv(data_path)
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    if clean_only and "is_outlier" in df.columns:
        df = df[df["is_outlier"] == "none"].reset_index(drop=True)

    OUTLIER_COL  = "is_outlier"
    outlier_mask = df[OUTLIER_COL] != "none" if OUTLIER_COL in df.columns and not clean_only else None

    X = df[features]
    y = df[target].values
    n       = len(features)
    n_rows  = math.ceil(n / N_COLS)

    fig  = Figure(figsize=(N_COLS * 5, n_rows * 4), dpi=100)
    axes = fig.subplots(n_rows, N_COLS, squeeze=False)

    suffix = "  [outliers excluded]" if clean_only else ""
    fig.suptitle(
        f"Feature vs Target — {label}  ({len(df):,} samples){suffix}   [target: {target}]",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(top=0.88, hspace=0.55, wspace=0.38)

    for ax in axes.flat:
        ax.set_visible(False)

    rng = np.random.default_rng(42)
    max_scatter = 2000
    idx = rng.choice(len(df), size=min(max_scatter, len(df)), replace=False)

    # physics curve range: use only the clean rows so outlier extremes don't distort the sweep
    if "is_outlier" in df.columns and not clean_only:
        clean_mask = df["is_outlier"] == "none"
        X_clean    = X[clean_mask]
        y_clean    = y[clean_mask.values]
    else:
        X_clean = X
        y_clean = y
    medians = {col: float(X_clean[col].median()) for col in features}
    v_cap   = float(np.percentile(y_clean, 99)) * 1.05

    for feat_idx, col in enumerate(features):
        row_idx = feat_idx // N_COLS
        col_idx = feat_idx  % N_COLS
        ax = axes[row_idx][col_idx]
        ax.set_visible(True)

        x_all = X[col].values

        if outlier_mask is not None:
            clean_idx  = np.where(~outlier_mask.values)[0]
            out_idx    = np.where(outlier_mask.values)[0]
            c_idx = rng.choice(clean_idx, size=min(max_scatter, len(clean_idx)), replace=False)
            ax.scatter(x_all[c_idx], y[c_idx],
                       alpha=0.20, s=6, color="steelblue", linewidths=0,
                       rasterized=True, label="clean")
            if len(out_idx):
                ax.scatter(x_all[out_idx], y[out_idx],
                           alpha=0.60, s=10, color="tomato", linewidths=0,
                           rasterized=True, label="outlier", zorder=3)
        else:
            ax.scatter(x_all[idx], y[idx],
                       alpha=0.25, s=6, color="steelblue", linewidths=0,
                       rasterized=True)

        x_clean_vals = X_clean[col].values
        x_p1  = float(np.percentile(x_clean_vals, 1))
        x_p99 = float(np.percentile(x_clean_vals, 99))
        x_sweep = np.linspace(x_p1, x_p99, 80)
        px, py  = _physics_curve(col, x_sweep, medians, features, v_hi=v_cap)
        if px:
            ax.plot(px, py, color="red", linewidth=2, zorder=6, label="physics")

        r = float(pd.Series(x_all).corr(pd.Series(y)))
        ax.set_title(f"{col}  (r={r:.3f})", fontsize=9, fontweight="bold")
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel(target if col_idx == 0 else "", fontsize=8)
        if outlier_mask is not None or px:
            ax.legend(fontsize=7, loc="best", handlelength=1.5)
        ax.tick_params(labelsize=7)

    _show_scrollable(fig, f"Features — {label}")
