# ETS IEX Analysis

Python project for processing ETS-related power market datasets and analyzing Indian Energy Exchange market data with time-series and volatility models.

## Overview

This repository contains two standalone Python scripts:

- `ETS Data Final.py` builds monthly electricity generation indicators for India from a long-format generation dataset.
- `ETS IEX Data.py` parses IEX DAM market snapshot Excel files and fits market price models, including AR(1), GARCH(1,1), jump detection, and regime-switching analysis.

The project is designed for exploratory energy market analysis rather than as a packaged library. Both scripts currently use local Windows file paths and are intended to be run manually.

## Scripts

### `ETS Data Final.py`

This script:

- Loads a monthly long-format electricity generation CSV
- Filters to `India Total`, `Electricity generation`, and `GWh`
- Extracts core fuel series for coal, gas, solar, wind, and hydro
- Builds total generation using known total definitions or a fuel-sum fallback
- Creates derived indicators such as:
  - total generation in TWh
  - fuel generation in TWh
  - coal share
  - renewable energy share
  - HHI concentration
  - log-growth of total generation
  - 12-month rolling volatility
- Produces plots showing structural transition, concentration, volatility, and crisis markers
- Fits an econometric model of volatility on HHI and renewable share
- Saves a summary text file and output dataset

### `ETS IEX Data.py`

This script:

- Reads multiple IEX DAM market snapshot Excel files using a glob pattern
- Detects the correct header row automatically
- Validates and cleans required columns
- Aggregates intraday MCP values into a daily average price series
- Builds log-price and daily return series
- Fits:
  - AR(1) model on log prices
  - GARCH(1,1) model on returns
  - 3-sigma jump detection
  - 30-day rolling kurtosis
  - 2-state Markov regime-switching model on absolute returns
- Exports model summaries, interpretation notes, CSV outputs, and plots

## Repository Structure

```text
ets-project/
|-- ETS Data Final.py
|-- ETS IEX Data.py
|-- README.md
|-- .gitignore
```

## Requirements

Install Python 3.10+ and the following packages:

```bash
pip install pandas numpy matplotlib statsmodels openpyxl arch
```

## Input Files

The scripts currently expect local files at these hardcoded paths:

### For `ETS Data Final.py`

- Input CSV:
  - `C:\Users\Rohen\Downloads\india_monthly_full_release_long_format.csv`
- Output folder:
  - `C:\Users\Rohen\Downloads\ets_output`

### For `ETS IEX Data.py`

- Input Excel files:
  - `C:\Users\Rohen\Downloads\DAM_Market Snapshot*.xlsx`
- Output folder:
  - `C:\Users\Rohen\Downloads\ets_output_iex_models`

Before running the scripts on another machine, update the path constants near the top of each file.

## How To Run

From the project directory:

```bash
python "ETS Data Final.py"
python "ETS IEX Data.py"
```

## Outputs

### `ETS Data Final.py` outputs

The script writes files into `ets_output`, including:

- `india_power_core_variables_2019_2024.csv`
- `plot1_structural_transition.png`
- `plot2_hhi_over_time.png`
- `plot3_rolling_demand_volatility.png`
- `plot4_crisis_markers_overlay.png`
- `econometric_model_summary.txt`

### `ETS IEX Data.py` outputs

The script writes files into `ets_output_iex_models`, including:

- `model_daily_series.csv`
- `model_summary.txt`
- `interpretation.txt`
- `ar1_summary.txt`
- `garch_summary.txt`
- `regime_switching_summary.txt` if regime estimation succeeds
- `regime_probabilities_raw.csv` if regime estimation succeeds
- `regime_probabilities_7dma.csv` if smoothing is available
- `plot1_daily_price.png`
- `plot2_returns_jumps.png`
- `plot3_garch_conditional_vol.png`
- `plot4_rolling_kurtosis.png`
- `plot5_regime_probabilities.png` if regime estimation succeeds

## Notes

- The project is currently script-based and not yet refactored into reusable modules.
- File paths are hardcoded for a local Windows environment.
- `ETS Data Final.py` can fall back to a NumPy-based OLS implementation if `statsmodels` is unavailable.
- `ETS IEX Data.py` expects Excel inputs with the required DAM snapshot fields and may skip malformed files.
- Regime-switching estimation is wrapped in a `try/except`, so the rest of the pipeline can still complete if that model fails.

## Future Improvements

- Replace hardcoded paths with command-line arguments or a config file
- Add a `requirements.txt`
- Refactor scripts into reusable functions and modules
- Add unit tests for parsing and transformation logic
- Add sample input schemas and example outputs

## Author

Rohen
