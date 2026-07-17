import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from explanations.plot_explanations import export_graphs_all_vars


def load_viirs_scaled_dataframe(path="data/historic/viirs_inundation_temporal_scaled.csv"):
    """Load VIIRS scaled temporal data and prepare a date column for plotting."""
    df = pd.read_csv(path)

    if "period_start" in df.columns:
        df["date"] = pd.to_datetime(df["period_start"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Expected either 'period_start' or 'date' column in VIIRS CSV.")

    df = df.dropna(subset=["date"]).copy()

    # Plot the same target-style features used for inundation interpretation.
    plot_cols = [c for c in df.columns if c.startswith("percent_inundation")]
    if not plot_cols:
        raise ValueError("No 'percent_inundation*' columns found in VIIRS scaled CSV.")

    return df[["date"] + plot_cols], plot_cols


def run_visual_check():
    df, plot_cols = load_viirs_scaled_dataframe()
    export_graphs_all_vars(df, plot_cols)
    print(f"Generated {len(plot_cols)} VIIRS visual comparison plot(s).")


if __name__ == "__main__":
    run_visual_check()