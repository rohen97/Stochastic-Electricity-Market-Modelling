import os
from math import erf, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def normal_approx_pvalue(t_stat: float) -> float:
    z = abs(t_stat)
    cdf = 0.5 * (1 + erf(z / sqrt(2)))
    return 2 * (1 - cdf)


def extract_series(raw: pd.DataFrame, subcategory: str, variable: str) -> pd.Series:
    s = raw[
        (raw["Subcategory"] == subcategory) & (raw["Variable"] == variable)
    ][["Date", "Value"]].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.set_index("Date")["Value"].sort_index()


def build_total_series(raw: pd.DataFrame) -> pd.Series:
    # Try common total definitions first
    total_candidates = [
        ("Total", "Total Generation"),
        ("Aggregate fuel", "Total"),
        ("Aggregate fuel", "All sources"),
        ("Fuel", "Total"),
    ]

    for sub, var in total_candidates:
        s = extract_series(raw, sub, var)
        if not s.empty:
            return s.rename("Total")

    # Fallback: sum all Fuel rows by month
    fuel = raw[raw["Subcategory"] == "Fuel"][["Date", "Value"]].copy()
    if fuel.empty:
        pairs = raw[["Subcategory", "Variable"]].drop_duplicates().sort_values(["Subcategory", "Variable"])
        raise ValueError(
            "Could not construct Total. No known (Subcategory, Variable) total pair found, "
            "and Fuel fallback unavailable.\nAvailable pairs:\n"
            + pairs.to_string(index=False)
        )

    total = fuel.groupby("Date", as_index=True)["Value"].sum().sort_index()
    return total.rename("Total")


def main():
    input_csv = r"C:\Users\Rohen\Downloads\india_monthly_full_release_long_format.csv"
    output_dir = r"C:\Users\Rohen\Downloads\ets_output"
    os.makedirs(output_dir, exist_ok=True)

    usecols = ["State", "Date", "Category", "Subcategory", "Variable", "Unit", "Value"]
    raw = pd.read_csv(input_csv, usecols=usecols)

    raw = raw[
        (raw["State"] == "India Total")
        & (raw["Category"] == "Electricity generation")
        & (raw["Unit"] == "GWh")
    ].copy()

    raw["Date"] = pd.to_datetime(raw["Date"])

    # Core fuel series
    required = {
        "Coal": ("Fuel", "Coal"),
        "Gas": ("Fuel", "Gas"),
        "Solar": ("Fuel", "Solar"),
        "Wind": ("Fuel", "Wind"),
        "Hydro": ("Fuel", "Hydro"),
    }

    series = {}
    for name, (sub, var) in required.items():
        s = extract_series(raw, sub, var)
        if s.empty:
            pairs = raw[["Subcategory", "Variable"]].drop_duplicates().sort_values(["Subcategory", "Variable"])
            raise ValueError(
                f"Missing required series: ({sub}, {var}).\nAvailable pairs:\n{pairs.to_string(index=False)}"
            )
        series[name] = s.rename(name)

    # Total with robust detection
    total = build_total_series(raw)

    df = pd.concat([total] + list(series.values()), axis=1).sort_index()
    df = df.loc["2019-01-01":"2024-12-31"].copy()

    # Validation against silent failure risk
    total_non_na_share = df["Total"].notna().mean()
    if total_non_na_share <= 0.95:
        pairs = raw[["Subcategory", "Variable"]].drop_duplicates().sort_values(["Subcategory", "Variable"])
        raise ValueError(
            f"Total coverage too low: {total_non_na_share:.3f} (<= 0.95).\n"
            f"Available (Subcategory, Variable) pairs:\n{pairs.to_string(index=False)}"
        )

    # Requested pulled variables in TWh
    for c in ["Total", "Coal", "Gas", "Solar", "Wind", "Hydro"]:
        df[f"{c}_TWh"] = df[c] / 1000.0

    # Shares / HHI / volatility
    for c in ["Coal", "Gas", "Solar", "Wind", "Hydro"]:
        df[f"{c}Share"] = df[c] / df["Total"]

    df["REShare"] = (df["Solar"] + df["Wind"] + df["Hydro"]) / df["Total"]
    df["HHI"] = (
        df["CoalShare"] ** 2
        + df["GasShare"] ** 2
        + df["SolarShare"] ** 2
        + df["WindShare"] ** 2
        + df["HydroShare"] ** 2
    )

    df["g"] = np.log(df["Total"]) - np.log(df["Total"].shift(1))
    df["Vol12"] = df["g"].rolling(12).std()

    out_cols = [
        "Total_TWh", "Coal_TWh", "Gas_TWh", "Solar_TWh", "Wind_TWh", "Hydro_TWh",
        "CoalShare", "REShare", "HHI", "g", "Vol12"
    ]
    df[out_cols].to_csv(
        os.path.join(output_dir, "india_power_core_variables_2019_2024.csv"),
        index_label="Date"
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 1
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["CoalShare"] * 100, label="Coal share (%)", linewidth=2.2, color="#7f3b08")
    ax.plot(df.index, df["REShare"] * 100, label="RE share (Solar+Wind+Hydro, %)", linewidth=2.2, color="#1b7837")
    ax.set_title("Plot 1 - Structural Transition: Coal Share vs RE Share (2019-2024)")
    ax.set_ylabel("Share of total generation (%)")
    ax.legend(frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot1_structural_transition.png"), dpi=220)
    plt.close(fig)

    # Plot 2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["HHI"], linewidth=2.2, color="#2166ac")
    ax.set_title("Plot 2 - HHI Over Time (Higher = More Concentrated)")
    ax.set_ylabel("HHI")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot2_hhi_over_time.png"), dpi=220)
    plt.close(fig)

    # Plot 3
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["Vol12"], linewidth=2.2, color="#b2182b")
    ax.set_title("Plot 3 - Rolling 12-Month Demand Volatility")
    ax.set_ylabel("Volatility of log growth (std. dev.)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot3_rolling_demand_volatility.png"), dpi=220)
    plt.close(fig)

    # Plot 4
    markers = [
        ("2020-03-01", "COVID lockdown (Mar 2020)", "#d73027"),
        ("2022-02-01", "Ukraine invasion (Feb 2022)", "#4575b4"),
        ("2023-05-01", "Heatwave surge (May 2023)", "#1a9850"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["Vol12"], linewidth=2.2, color="#4d4d4d", label="Vol12")
    for d, label, color in markers:
        ax.axvline(pd.Timestamp(d), color=color, linestyle="--", linewidth=1.8, label=label)
    ax.set_title("Plot 4 - Crisis Markers Overlay on Demand Volatility")
    ax.set_ylabel("Volatility of log growth (std. dev.)")
    ax.legend(loc="upper right", frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot4_crisis_markers_overlay.png"), dpi=220)
    plt.close(fig)

    # Econometric model with one control:
    # Vol12_t = a + b1*HHI_t + b2*REShare_t + e_t
    model_df = df[["Vol12", "HHI", "REShare"]].dropna().copy()

    if HAS_STATSMODELS:
        X = sm.add_constant(model_df[["HHI", "REShare"]])
        model = sm.OLS(model_df["Vol12"], X).fit(cov_type="HC1")

        a = float(model.params["const"])
        b_hhi = float(model.params["HHI"])
        b_re = float(model.params["REShare"])

        se_a = float(model.bse["const"])
        se_hhi = float(model.bse["HHI"])
        se_re = float(model.bse["REShare"])

        p_a = float(model.pvalues["const"])
        p_hhi = float(model.pvalues["HHI"])
        p_re = float(model.pvalues["REShare"])

        r2 = float(model.rsquared)
        n = int(model.nobs)
        estimation_note = "OLS with HC1 robust SE"
    else:
        y = model_df["Vol12"].to_numpy()
        X = np.column_stack([
            np.ones(len(model_df)),
            model_df["HHI"].to_numpy(),
            model_df["REShare"].to_numpy(),
        ])
        b = np.linalg.inv(X.T @ X) @ (X.T @ y)
        resid = y - X @ b

        n = len(y)
        k = X.shape[1]
        s2 = (resid @ resid) / (n - k)
        vcv = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(vcv))
        t = b / se

        a, b_hhi, b_re = float(b[0]), float(b[1]), float(b[2])
        se_a, se_hhi, se_re = float(se[0]), float(se[1]), float(se[2])
        p_a, p_hhi, p_re = normal_approx_pvalue(t[0]), normal_approx_pvalue(t[1]), normal_approx_pvalue(t[2])

        ssr = float(resid @ resid)
        tss = float((y - y.mean()) @ (y - y.mean()))
        r2 = float(1 - ssr / tss)
        estimation_note = "OLS (normal-approx p-values; statsmodels unavailable)"

    summary = [
        "Model: Vol12_t = alpha + beta1*HHI_t + beta2*REShare_t + eps_t",
        f"Sample: {model_df.index.min().date()} to {model_df.index.max().date()} (n={n})",
        f"Estimation: {estimation_note}",
        f"alpha: {a:.6f} (SE {se_a:.6f}, p {p_a:.4f})",
        f"beta_HHI: {b_hhi:.6f} (SE {se_hhi:.6f}, p {p_hhi:.4f})",
        f"beta_REShare: {b_re:.6f} (SE {se_re:.6f}, p {p_re:.4f})",
        f"R^2: {r2:.6f}",
    ]

    with open(os.path.join(output_dir, "econometric_model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))
    print(f"\nSaved outputs in: {output_dir}")


if __name__ == "__main__":
    main()
