import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.modeling.model_comparator import ModelComparator


def make_horizon_labels(df: pd.DataFrame, group_col: str, date_col: str, event_col: str, horizon_days: int) -> pd.Series:
    """
    Label each row as 1 if a maintenance event happens within the next `horizon_days`
    for the same borehole (group_col). Otherwise 0.
    """
    df = df.sort_values([group_col, date_col]).copy()
    y = np.zeros(len(df), dtype=int)

    # Precompute event dates per borehole
    for g, gdf in df.groupby(group_col, sort=False):
        idx = gdf.index.values
        dates = gdf[date_col].values.astype("datetime64[ns]")

        event_dates = gdf.loc[gdf[event_col] == 1, date_col].values.astype("datetime64[ns]")
        event_dates = np.sort(event_dates)

        if len(event_dates) == 0:
            continue

        # For each row date d, check if there is event_date in (d, d + horizon_days]
        # Using searchsorted for speed
        right = dates + np.timedelta64(horizon_days, "D")

        # first event strictly greater than d
        left_pos = np.searchsorted(event_dates, dates, side="right")
        # first event greater than right boundary
        right_pos = np.searchsorted(event_dates, right, side="right")

        y[idx] = (right_pos > left_pos).astype(int)

    return pd.Series(y, index=df.index).sort_index()


def main():
    # --- Load config ---
    with open("config/model_comparison.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- Load data ---
    data_path = Path("data/processed/thesis_data_cleaned.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            "CSV not found. Put it at: data/processed/thesis_data_cleaned.csv"
        )

    df = pd.read_csv(data_path)

    # --- Basic cleanup ---
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["monitor_num", "Date"]).reset_index(drop=True)

    # Drop useless column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop z_score (your file shows it’s all missing)
    if "z_score" in df.columns and df["z_score"].isna().all():
        df = df.drop(columns=["z_score"])

    # --- Feature selection (start simple and stable) ---
    # Use the “interpolated” litres/day plus weather + usage + location + population
    candidate_features = [
        "litres/day_interpolated",
        "Rainfall",
        "min_temp",
        "max_temp",
        "time_in_use",
        "wet_time",
        "dry_time_in_use",
        "dry_time_in_use_percentage",
        "longest_dry_time_in_use",
        "Population",
        "Latitude",
        "Longitude",
    ]
    features = [c for c in candidate_features if c in df.columns]

    # Fill missing numeric values with median (simple baseline)
    for c in features:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    X_all = df[features].to_numpy(dtype=float)

    # --- Time-based train/test split (no leakage) ---
    # Use last 20% of dates as test
    cutoff = df["Date"].quantile(0.80)
    train_mask = df["Date"] <= cutoff
    test_mask = df["Date"] > cutoff

    X_train = X_all[train_mask.values]
    X_test = X_all[test_mask.values]

    # --- Run per horizon and merge results ---
    all_results = {}

    for horizon in config["model_comparison"]["prediction_horizons"]:
        # Create horizon target (maintenance in next horizon days)
        y_h = make_horizon_labels(
            df=df,
            group_col="monitor_num",
            date_col="Date",
            event_col="maintenance",
            horizon_days=int(horizon),
        ).to_numpy(dtype=int)

        y_train = y_h[train_mask.values]
        y_test = y_h[test_mask.values]

        # Run comparator (we keep its horizon list but it won’t change y now)
        print(f"\n=== Running horizon {horizon} days ===")
        comparator = ModelComparator(config)
        results = comparator.run_comparison(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=features,
        )

        all_results[f"horizon_{horizon}"] = results.get("best_models", {}).get(horizon, {})

    print("\n✅ Done. Best models summary:")
    for k, v in all_results.items():
        print(k, "=>", v)


if __name__ == "__main__":
    main()
