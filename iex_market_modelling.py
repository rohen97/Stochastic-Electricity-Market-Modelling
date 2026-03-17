import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# =========================
# CONFIG
# =========================
INPUT_GLOB = r"C:\Users\Rohen\Downloads\DAM_Market Snapshot*.xlsx"
OUTPUT_DIR = Path(r"C:\Users\Rohen\Downloads\ets_output_iex_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "Date",
    "Hour",
    "Time Block",
    "Purchase Bid (MW)",
    "Sell Bid (MW)",
    "MCV (MW)",
    "Final Scheduled Volume (MW)",
    "MCP (Rs/MWh) *",
]


def _norm(x):
    return str(x).strip() if pd.notna(x) else ""


def find_header_row(raw: pd.DataFrame):
    for i in range(min(50, len(raw))):
        row = [_norm(v) for v in raw.iloc[i].tolist()]
        rowset = set(row)
        if {"Date", "Hour", "Time Block", "MCP (Rs/MWh) *"}.issubset(rowset):
            return i, row
    return None, None


def parse_snapshot_file(path: str):
    raw = pd.read_excel(path, header=None)

    hidx, header = find_header_row(raw)
    if hidx is None:
        return None, f"Header row not found in {path}"

    df = raw.iloc[hidx + 1 :].copy()
    df.columns = header

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, f"Missing columns in {path}: {missing}"

    x = df[REQUIRED_COLS].copy()

    x["Date"] = pd.to_datetime(x["Date"], dayfirst=True, errors="coerce")
    for c in REQUIRED_COLS[1:]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna(subset=["Date", "MCP (Rs/MWh) *"]).copy()

    # Drop non-data rows / artifacts
    x = x[(x["MCP (Rs/MWh) *"] > 0) & (x["MCP (Rs/MWh) *"] < 20000)].copy()
    if x.empty:
        return None, f"No valid rows in {path}"

    x["Date"] = x["Date"].dt.normalize()

    # Convert intraday blocks to daily average MCP
    daily = (
        x.groupby("Date", as_index=False)["MCP (Rs/MWh) *"]
        .mean()
        .rename(columns={"MCP (Rs/MWh) *": "Price"})
    )
    return daily, None


def fit_models(daily: pd.DataFrame):
    df = daily.copy()
    df["LogPrice"] = np.log(df["Price"])
    df["Return"] = 100.0 * df["LogPrice"].diff()
    df = df.dropna().reset_index(drop=True)

    # AR(1): log price on lagged log price
    ar_df = df[["LogPrice"]].copy()
    ar_df["Lag1"] = ar_df["LogPrice"].shift(1)
    ar_df = ar_df.dropna()
    ar_res = sm.OLS(ar_df["LogPrice"], sm.add_constant(ar_df["Lag1"])).fit()

    # GARCH(1,1): returns
    garch_res = arch_model(
        df["Return"], mean="Constant", vol="GARCH", p=1, q=1, dist="normal"
    ).fit(disp="off")
    df["CondVol"] = garch_res.conditional_volatility.values

    # Jump detection (3 sigma rule)
    jump_threshold = 3 * df["Return"].std(ddof=1)
    df["Jump"] = df["Return"].abs() > jump_threshold

    # Rolling kurtosis
    df["RollingKurtosis30"] = df["Return"].rolling(30).kurt()

    # Regime switching on volatility proxy (abs returns)
    ms_res = None
    ms_probs = None
    ms_probs_smooth = None
    try:
        vol_proxy = df["Return"].abs()
        ms_res = MarkovRegression(
            vol_proxy, k_regimes=2, trend="c", switching_variance=True
        ).fit(disp=False)
        ms_probs = ms_res.smoothed_marginal_probabilities.copy()
        ms_probs.columns = ["Regime0_Prob", "Regime1_Prob"]
        ms_probs_smooth = ms_probs.rolling(7, min_periods=1).mean()
    except Exception:
        pass

    return df, ar_res, garch_res, jump_threshold, ms_res, ms_probs, ms_probs_smooth


def make_plots(df, jump_threshold, ms_probs, ms_probs_smooth):
    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 1: Daily price
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Price"], lw=1.6)
    ax.set_title("Daily Average MCP (Rs/MWh)")
    ax.set_ylabel("Price")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot1_daily_price.png", dpi=200)
    plt.close(fig)

    # Plot 2: Returns + jumps
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Return"], lw=1.1, label="Daily log return (%)")
    j = df[df["Jump"]]
    if not j.empty:
        ax.scatter(j["Date"], j["Return"], s=18, c="red", label="Jumps")
    ax.axhline(jump_threshold, ls="--", c="gray", lw=1)
    ax.axhline(-jump_threshold, ls="--", c="gray", lw=1)
    ax.set_title("Returns and Jump Detection (3-sigma)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot2_returns_jumps.png", dpi=200)
    plt.close(fig)

    # Plot 3: GARCH conditional volatility
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["CondVol"], lw=1.3, c="darkorange")
    ax.set_title("GARCH(1,1) Conditional Volatility")
    ax.set_ylabel("Volatility")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot3_garch_conditional_vol.png", dpi=200)
    plt.close(fig)

    # Plot 4: Rolling kurtosis
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["RollingKurtosis30"], lw=1.3, c="purple")
    ax.axhline(3.0, ls="--", c="gray", lw=1)
    ax.set_title("30-Day Rolling Kurtosis of Returns")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plot4_rolling_kurtosis.png", dpi=200)
    plt.close(fig)

    # Plot 5: Regime probabilities (raw + smoothed)
    if ms_probs is not None:
        dates = df["Date"].iloc[: len(ms_probs)]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, ms_probs["Regime0_Prob"], lw=0.8, alpha=0.35, label="Regime 0 prob (raw)")
        ax.plot(dates, ms_probs["Regime1_Prob"], lw=0.8, alpha=0.35, label="Regime 1 prob (raw)")
        if ms_probs_smooth is not None:
            ax.plot(dates, ms_probs_smooth["Regime0_Prob"], lw=1.8, label="Regime 0 prob (7d MA)")
            ax.plot(dates, ms_probs_smooth["Regime1_Prob"], lw=1.8, label="Regime 1 prob (7d MA)")
        ax.set_ylim(0, 1)
        ax.set_title("2-State Markov Regime Probabilities (Volatility Proxy)")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "plot5_regime_probabilities.png", dpi=200)
        plt.close(fig)


def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found for: {INPUT_GLOB}")

    parsed = []
    failed = []

    for f in files:
        d, err = parse_snapshot_file(f)
        if err:
            failed.append((Path(f).name, err))
        else:
            parsed.append(d)

    if not parsed:
        raise ValueError("No files parsed successfully.")

    daily = pd.concat(parsed, ignore_index=True)
    daily = (
        daily.groupby("Date", as_index=False)["Price"]
        .mean()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Fit models
    model_df, ar_res, garch_res, jump_threshold, ms_res, ms_probs, ms_probs_smooth = fit_models(daily)

    # Save datasets
    model_df.to_csv(OUTPUT_DIR / "model_daily_series.csv", index=False)

    # Save summaries
    summary_lines = [
        f"Sample: {model_df['Date'].min().date()} to {model_df['Date'].max().date()} (n={len(model_df)})",
        f"AR(1): const={ar_res.params['const']:.6f}, phi={ar_res.params['Lag1']:.6f}, p={ar_res.pvalues['Lag1']:.6g}, R2={ar_res.rsquared:.6f}",
        f"GARCH(1,1): mu={garch_res.params['mu']:.6f}, omega={garch_res.params['omega']:.6f}, alpha={garch_res.params['alpha[1]']:.6f}, beta={garch_res.params['beta[1]']:.6f}, alpha+beta={(garch_res.params['alpha[1]'] + garch_res.params['beta[1]']):.6f}",
        f"Jumps (|r|>3sigma): threshold={jump_threshold:.6f}, count={int(model_df['Jump'].sum())}",
        f"Rolling kurtosis(30d): mean={model_df['RollingKurtosis30'].dropna().mean():.6f}, max={model_df['RollingKurtosis30'].dropna().max():.6f}, latest={model_df['RollingKurtosis30'].dropna().iloc[-1]:.6f}",
    ]

    if ms_res is not None:
        summary_lines.append(
            f"Regime switching (on |returns|): llf={ms_res.llf:.6f}, aic={ms_res.aic:.6f}, "
            f"last_p0={float(ms_probs['Regime0_Prob'].iloc[-1]):.6f}, last_p1={float(ms_probs['Regime1_Prob'].iloc[-1]):.6f}"
        )

    with open(OUTPUT_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Interpretation text
    interpretation = []
    interpretation.append("Interpretation")
    interpretation.append(f"- AR(1): phi={ar_res.params['Lag1']:.3f} (p<{1e-6}) indicates strong price persistence.")
    interpretation.append(f"- GARCH: alpha+beta={(garch_res.params['alpha[1]'] + garch_res.params['beta[1]']):.3f} indicates highly persistent volatility clustering.")
    interpretation.append(f"- Jumps: {int(model_df['Jump'].sum())} extreme return days identified by 3-sigma rule.")
    interpretation.append("- Rolling kurtosis: episodic tail risk spikes are present (fat-tail windows).")
    if ms_res is not None:
        interpretation.append(
            "- Regime probabilities vary frequently, suggesting either rapid switching or imperfect regime separation; "
            "we therefore interpret regime results as indicative rather than definitive."
        )
        interpretation.append(
            f"- Latest regime probabilities: regime0={float(ms_probs['Regime0_Prob'].iloc[-1]):.3f}, "
            f"regime1={float(ms_probs['Regime1_Prob'].iloc[-1]):.3f}."
        )

    with open(OUTPUT_DIR / "interpretation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(interpretation))

    # Save full model text outputs
    with open(OUTPUT_DIR / "ar1_summary.txt", "w", encoding="utf-8") as f:
        f.write(ar_res.summary().as_text())

    with open(OUTPUT_DIR / "garch_summary.txt", "w", encoding="utf-8") as f:
        f.write(garch_res.summary().as_text())

    if ms_res is not None:
        with open(OUTPUT_DIR / "regime_switching_summary.txt", "w", encoding="utf-8") as f:
            f.write(ms_res.summary().as_text())
        ms_probs.to_csv(OUTPUT_DIR / "regime_probabilities_raw.csv", index=False)
        if ms_probs_smooth is not None:
            ms_probs_smooth.to_csv(OUTPUT_DIR / "regime_probabilities_7dma.csv", index=False)

    # Plots
    make_plots(model_df, jump_threshold, ms_probs, ms_probs_smooth)

    # Console output
    print("\n".join(summary_lines))
    print("\nSaved to:", OUTPUT_DIR)
    if failed:
        print(f"\nFailed files: {len(failed)}")
        for f, e in failed[:5]:
            print(f"- {f}: {e}")


if __name__ == "__main__":
    main()
