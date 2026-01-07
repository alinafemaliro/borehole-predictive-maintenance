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

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["monitor_num", "Date"]).copy()

    # Basic time parts
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # Choose main signal
    col = "litres/day_interpolated"

    # Lags
    for lag in [1, 2, 7, 14]:
        df[f"{col}_lag_{lag}"] = df.groupby("monitor_num")[col].shift(lag)

    # Rolling windows (use past values only)
    for w in [7, 14, 30]:
        g = df.groupby("monitor_num")[col]
        df[f"{col}_roll_mean_{w}"] = g.shift(1).rolling(w).mean()
        df[f"{col}_roll_std_{w}"] = g.shift(1).rolling(w).std()

    # Trend (difference between recent mean and older mean)
    df[f"{col}_trend_7_14"] = df[f"{col}_roll_mean_7"] - df[f"{col}_roll_mean_14"]

    # Days since last maintenance
    df["days_since_maintenance"] = (
        df.groupby("monitor_num")["maintenance"]
          .apply(lambda s: (s.shift(1).fillna(0).eq(1)).cumsum())
          .reset_index(level=0, drop=True)
    )
    # The above gives "event block id". Convert to days since last event:
    last_event_date = df["Date"].where(df["maintenance"].shift(1) == 1)
    last_event_date = last_event_date.groupby(df["monitor_num"]).ffill()
    df["days_since_maintenance"] = (df["Date"] - last_event_date).dt.days
    df["days_since_maintenance"] = df["days_since_maintenance"].fillna(9999)

    return df


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
    
    df = add_time_features(df)

    # Drop useless column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop z_score (your file shows it’s all missing)
    if "z_score" in df.columns and df["z_score"].isna().all():
        df = df.drop(columns=["z_score"])

  
    # Use all numeric columns except identifiers, date, and target
    exclude = {"Unnamed: 0", "Date", "monitor_num", "maintenance"}
    features = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Fill missing numeric values with median
    for c in features:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    X_all = df[features].to_numpy(dtype=float)

    print("Number of features:", len(features))
    print("Example features:", features[:10])

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
