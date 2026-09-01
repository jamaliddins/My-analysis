"""Five hypothesis tests on the Amazon sales export.

    python src/hypothesis_tests.py

Every test writes its figure to figures/. Requires the raw CSV in the
repository root — see README.
"""
from __future__ import annotations

import pathlib
from itertools import combinations

import matplotlib
matplotlib.use("Agg")            # write files; never block on a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.preprocessing import MissingDataError, build_dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RANDOM_STATE = 42
ALPHA = 0.05

BUCKET_ORDER = ["Single (1)", "Small (2-5)", "Medium (6-20)", "Bulk (20+)"]
MIN_GROUP_SIZE = 10


def _verdict(p_value: float) -> str:
    return "SUPPORTED" if p_value < ALPHA else "NOT SUPPORTED"


def _header(number: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"TEST {number}  {title}")
    print("=" * 72)


def rank_biserial(a: np.ndarray, b: np.ndarray, u_statistic: float) -> float:
    """Effect size for Mann-Whitney U, in [-1, 1].

    p-values alone are near-meaningless at n > 100,000 — almost any difference
    reaches significance. This reports how large the difference actually is.
    """
    return 1.0 - (2.0 * u_statistic) / (len(a) * len(b))


# --------------------------------------------------------------------- test 1
def test_quantity_amount_correlation(df: pd.DataFrame) -> dict:
    """Does order quantity track sale amount, and does that differ B2B vs B2C?"""
    _header(1, "Spearman — order quantity vs sale amount")

    rho, p_value = stats.spearmanr(df["Qty"], df["Amount"])
    print(f"  overall   rho = {rho:+.3f}   p = {p_value:.4g}   (n = {len(df):,})")

    per_segment = {}
    for label in ("B2B", "B2C"):
        subset = df[df["B2BLabel"] == label]
        r, p = stats.spearmanr(subset["Qty"], subset["Amount"])
        per_segment[label] = {"rho": float(r), "p": float(p), "n": len(subset)}
        print(f"  {label:<9} rho = {r:+.3f}   p = {p:.4g}   (n = {len(subset):,})")

    print(f"\n  {_verdict(p_value)}: quantity and amount are "
          f"{'related' if p_value < ALPHA else 'unrelated'}.")
    print("  Note: rho is modest — most orders are single items, so quantity")
    print("  explains little of the variation in amount by itself.")

    _plot_correlation(df, per_segment)
    return {"overall_rho": float(rho), "overall_p": float(p_value),
            "by_segment": per_segment}


def _plot_correlation(df: pd.DataFrame, per_segment: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, label, color in zip(axes, ("B2B", "B2C"), ("steelblue", "coral")):
        subset = df[df["B2BLabel"] == label]
        ax.scatter(subset["Qty"], subset["Amount"], alpha=0.3, s=15, color=color)

        slope, intercept = np.polyfit(subset["Qty"], subset["Amount"], 1)
        x_line = np.linspace(subset["Qty"].min(), subset["Qty"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=2)

        ax.set_title(f"{label}  (rho = {per_segment[label]['rho']:+.3f}, "
                     f"n = {per_segment[label]['n']:,})")
        ax.set_xlabel("Quantity")
    axes[0].set_ylabel("Amount")
    fig.suptitle("Order quantity vs sale amount")
    fig.tight_layout()
    _save(fig, "test1_spearman.png")


# --------------------------------------------------------------------- test 2
def test_revenue_per_unit(df: pd.DataFrame) -> dict:
    """Do B2B orders earn more per unit than B2C?"""
    _header(2, "Mann-Whitney U — revenue per unit, B2B vs B2C")

    b2b = df.loc[df["B2BLabel"] == "B2B", "RevenuePerUnit"].dropna().to_numpy()
    b2c = df.loc[df["B2BLabel"] == "B2C", "RevenuePerUnit"].dropna().to_numpy()

    u_statistic, p_value = stats.mannwhitneyu(b2b, b2c, alternative="two-sided")
    effect = rank_biserial(b2b, b2c, u_statistic)
    difference = (np.median(b2b) - np.median(b2c)) / np.median(b2c) * 100

    print(f"  B2B median  {np.median(b2b):>8,.2f}   (n = {len(b2b):,})")
    print(f"  B2C median  {np.median(b2c):>8,.2f}   (n = {len(b2c):,})")
    print(f"  difference  {difference:>+7.1f}%")
    print(f"  p = {p_value:.4g}   rank-biserial effect size = {effect:+.3f}")
    print(f"\n  {_verdict(p_value)}.")
    if p_value < ALPHA and abs(effect) < 0.1:
        print("  The effect size is negligible: statistically detectable at this")
        print("  sample size, but too small to act on commercially.")

    _plot_revenue_per_unit(b2b, b2c, p_value)
    return {"median_b2b": float(np.median(b2b)), "median_b2c": float(np.median(b2c)),
            "pct_difference": float(difference), "p": float(p_value),
            "effect_size": float(effect)}


def _plot_revenue_per_unit(b2b, b2c, p_value: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [np.clip(b2b, None, np.quantile(b2b, 0.95)),
         np.clip(b2c, None, np.quantile(b2c, 0.95))],
        tick_labels=["B2B", "B2C"], patch_artist=True,
        boxprops=dict(facecolor="mediumseagreen", alpha=0.6),
    )
    ax.set_ylabel("Revenue per unit")
    ax.set_title(f"Revenue per unit by segment  (p = {p_value:.4g})")
    fig.tight_layout()
    _save(fig, "test2_mannwhitney.png")


# --------------------------------------------------------------------- test 3
def test_category_differences(df: pd.DataFrame) -> dict:
    """Do sale amounts differ across product categories?"""
    _header(3, "Kruskal-Wallis — sale amount across categories")

    groups = {
        name: group["Amount"].dropna().to_numpy()
        for name, group in df.groupby("Category", observed=True)
        if len(group) >= MIN_GROUP_SIZE
    }
    statistic, p_value = stats.kruskal(*groups.values())
    print(f"  H = {statistic:,.1f}   p = {p_value:.4g}   "
          f"({len(groups)} categories)")
    print(f"\n  {_verdict(p_value)}.\n")

    print("  median amount per category:")
    ordered = sorted(groups.items(), key=lambda kv: -np.median(kv[1]))
    for name, values in ordered:
        print(f"    {name:<18} {np.median(values):>9,.2f}   (n = {len(values):,})")

    # Pairwise comparisons need correcting: 36 tests at alpha=0.05 would
    # produce false positives by chance alone.
    pairs = list(combinations(groups, 2))
    raw_p = [stats.mannwhitneyu(groups[a], groups[b],
                                alternative="two-sided")[1] for a, b in pairs]
    corrected = np.minimum(np.array(raw_p) * len(pairs), 1.0)   # Bonferroni

    n_different = int((corrected < ALPHA).sum())
    print(f"\n  pairwise: {n_different} of {len(pairs)} category pairs differ "
          f"(Bonferroni-corrected)")

    _plot_categories(groups, ordered)
    return {"p": float(p_value), "n_categories": len(groups),
            "n_pairs_different": n_different, "n_pairs": len(pairs)}


def _plot_categories(groups: dict, ordered: list) -> None:
    names = [name for name, _ in ordered]
    data = [np.clip(groups[n], None, np.quantile(groups[n], 0.95)) for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax.set_ylabel("Amount")
    ax.set_title("Sale amount by category, highest median first")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, "test3_kruskal_categories.png")


# --------------------------------------------------------------------- test 4
def test_order_size_spread(df: pd.DataFrame) -> dict:
    """Is the spread of sale amounts wider for larger orders?

    The claim is about spread as IQR, so it is tested on IQR via a bootstrap.
    Levene's test compares variance, which on this heavily right-skewed
    distribution is driven by a few extreme orders and answers a different
    question.
    """
    _header(4, "Bootstrap — amount spread across order-size buckets")

    buckets = {
        bucket: df.loc[df["OrderSizeBucket"] == bucket, "Amount"].dropna().to_numpy()
        for bucket in BUCKET_ORDER
    }
    buckets = {k: v for k, v in buckets.items() if len(v) >= MIN_GROUP_SIZE}

    print("  IQR and median per bucket:")
    for name, values in buckets.items():
        q75, q25 = np.percentile(values, [75, 25])
        print(f"    {name:<15} IQR {q75 - q25:>9,.2f}   "
              f"median {np.median(values):>9,.2f}   (n = {len(values):,})")

    if len(buckets) < 2:
        print("\n  Not enough populated buckets to compare.")
        return {"comparable": False}

    # Compare the smallest bucket against the largest populated one.
    names = list(buckets)
    small, large = buckets[names[0]], buckets[names[-1]]
    observed, (low, high) = _bootstrap_iqr_difference(large, small)

    print(f"\n  IQR({names[-1]}) - IQR({names[0]}) = {observed:,.2f}")
    print(f"  bootstrap 95% CI [{low:,.2f}, {high:,.2f}]")
    supported = low > 0
    print(f"\n  {'SUPPORTED' if supported else 'NOT SUPPORTED'}: the interval "
          f"{'excludes' if supported else 'includes'} zero.")

    _plot_order_size(buckets)
    return {"difference": float(observed), "ci": [float(low), float(high)],
            "supported": bool(supported), "compared": [names[0], names[-1]]}


def _bootstrap_iqr_difference(a, b, n_iter: int = 2_000, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)

    def iqr(values):
        q75, q25 = np.percentile(values, [75, 25])
        return q75 - q25

    observed = iqr(a) - iqr(b)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        diffs[i] = (iqr(rng.choice(a, len(a), replace=True))
                    - iqr(rng.choice(b, len(b), replace=True)))
    return observed, np.percentile(diffs, [2.5, 97.5])


def _plot_order_size(buckets: dict) -> None:
    names = list(buckets)
    data = [np.clip(buckets[n], None, np.quantile(buckets[n], 0.95)) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    box = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("darkorange")
        patch.set_alpha(0.6)
    ax.set_ylabel("Amount")
    ax.set_title("Sale amount spread by order size")
    fig.tight_layout()
    _save(fig, "test4_order_size_spread.png")


# --------------------------------------------------------------------- test 5
def test_cancellation_fulfilment(df: pd.DataFrame) -> dict:
    """Is cancellation associated with the fulfilment method?

    Tested on Fulfilment x IsCancelled. Crossing Fulfilment against the full
    Status column instead would test whether *any* status differs, which is a
    broader question than the one being asked.
    """
    _header(5, "Chi-square — cancellation vs fulfilment method")

    contingency = pd.crosstab(df["Fulfilment"], df["IsCancelled"])
    contingency.columns = ["Not cancelled", "Cancelled"][: contingency.shape[1]]
    print(contingency.to_string())

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    # Cramer's V: chi-square grows with n, so report the standardised strength.
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim else float("nan")

    print(f"\n  chi2 = {chi2:,.1f}   dof = {dof}   p = {p_value:.4g}")
    print(f"  Cramer's V = {cramers_v:.3f}")
    print(f"\n  {_verdict(p_value)}.")
    if p_value < ALPHA and cramers_v < 0.1:
        print("  Cramer's V below 0.1 means the association is very weak, even")
        print("  though it is statistically significant at this sample size.")

    rates = df.groupby("Fulfilment", observed=True)["IsCancelled"].mean() * 100
    print("\n  cancellation rate by fulfilment method:")
    for method, rate in rates.sort_values(ascending=False).items():
        print(f"    {method:<12} {rate:>5.2f}%")

    _plot_cancellation(rates, p_value)
    return {"chi2": float(chi2), "p": float(p_value),
            "cramers_v": float(cramers_v), "rates": rates.round(2).to_dict()}


def _plot_cancellation(rates: pd.Series, p_value: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ordered = rates.sort_values(ascending=False)
    ax.bar(ordered.index.astype(str), ordered.to_numpy(),
           color="indianred", alpha=0.8)
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_title(f"Cancellation rate by fulfilment method  (p = {p_value:.4g})")
    for i, value in enumerate(ordered.to_numpy()):
        ax.text(i, value, f"{value:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    _save(fig, "test5_cancellation.png")


# --------------------------------------------------------------------- shared
def _save(fig, filename: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)
    print(f"  saved figures/{filename}")


def main() -> None:
    try:
        df = build_dataset()
    except MissingDataError as exc:
        # A missing download is a setup problem, not a crash: say what to do.
        print(f"\n{exc}\n")
        raise SystemExit(1)

    test_quantity_amount_correlation(df)
    test_revenue_per_unit(df)
    test_category_differences(df)
    test_order_size_spread(df)
    test_cancellation_fulfilment(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
