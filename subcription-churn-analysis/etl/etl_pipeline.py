import os
import pandas as pd
import numpy as np
from datetime import timedelta

os.makedirs("data", exist_ok=True)

AS_OF_DATE     = pd.Timestamp("2026-01-30")   # observation end from meta.json
PRED_HORIZON   = 14                            # days ahead to predict churn
OBS_WEEKS      = 4                             # feature window (weeks)

# ==========================================================
# 1. LOAD
# ==========================================================

def load_raw_data():
    print("Loading raw data...")

    users = pd.read_csv("raw_data/users.csv",
                        parse_dates=["signup_date"])

    plans = pd.read_csv("raw_data/plans.csv")

    subscriptions = pd.read_csv("raw_data/subscriptions.csv",
                                parse_dates=["start_date", "end_date", "cancel_date"])

    payments = pd.read_csv("raw_data/payments.csv",
                           parse_dates=["payment_date"])

    usage = pd.read_csv("raw_data/usage_daily.csv",
                        parse_dates=["date"])

    campaigns = pd.read_csv("raw_data/campaign_touchpoints.csv",
                            parse_dates=["touch_date"])

    return users, plans, subscriptions, payments, usage, campaigns


# ==========================================================
# 2. CLEAN & STANDARDIZE
# ==========================================================

def clean_data(users, plans, subscriptions, payments, usage, campaigns):
    print("Cleaning data...")

    # --- Normalize IDs to lowercase ---
    for df in [users, subscriptions, payments, usage, campaigns]:
        df["user_id"] = df["user_id"].str.strip().str.lower()

    # --- Text normalization ---
    subscriptions["status"]        = subscriptions["status"].str.strip().str.lower()
    payments["payment_status"]     = payments["payment_status"].str.strip().str.lower()
    payments["payment_method"]     = payments["payment_method"].str.strip().str.lower()
    campaigns["channel"]           = campaigns["channel"].str.strip().str.lower()
    campaigns["campaign_type"]     = campaigns["campaign_type"].str.strip().str.lower()
    plans["plan_name"]             = plans["plan_name"].str.strip().str.title()
    users["segment"]               = users["segment"].str.strip().str.lower()
    users["preferred_device"]      = users["preferred_device"].str.strip().str.lower()

    # --- Impute missing preferred_device ---
    users["preferred_device"] = users["preferred_device"].fillna("unknown")

    # --- Deduplicate (use .copy() to avoid SettingWithCopyWarning on subsequent assignments) ---
    users          = users.drop_duplicates().copy()
    subscriptions  = subscriptions.drop_duplicates().copy()
    payments       = payments.drop_duplicates().copy()
    usage          = usage.drop_duplicates().copy()
    campaigns      = campaigns.drop_duplicates().copy()

    # --- Flag usage outliers (minutes > 99th percentile) ---
    p99 = usage["minutes_used"].quantile(0.99)
    usage["is_outlier_minutes"] = (usage["minutes_used"] > p99).astype(int)
    print(f"  Usage outliers flagged (>{p99:.1f} min): {usage['is_outlier_minutes'].sum()}")

    # --- Flag payment outliers ---
    p99_pay = payments["amount"].quantile(0.99)
    payments["is_outlier_amount"] = (payments["amount"] > p99_pay).astype(int)

    return users, plans, subscriptions, payments, usage, campaigns


# ==========================================================
# 3. DIM_USERS_ENRICHED  (1 row per user)
# ==========================================================

def build_dim_users(users, plans, subscriptions, payments, usage):
    print("Building dim_users_enriched...")

    # -- Last active date --
    last_active = (
        usage.groupby("user_id")["date"]
        .max().reset_index()
        .rename(columns={"date": "last_active_date"})
    )

    # -- Avg daily minutes (non-outlier rows) --
    usage_clean = usage[usage["is_outlier_minutes"] == 0]
    usage_agg = (
        usage_clean.groupby("user_id")["minutes_used"]
        .mean().reset_index()
        .rename(columns={"minutes_used": "avg_daily_minutes"})
    )

    # -- Lifetime paid amount & lifetime_paid_months proxy --
    paid = payments[payments["payment_status"] == "success"]
    lifetime = (
        paid.groupby("user_id")
        .agg(
            lifetime_paid_amount=("amount", "sum"),
            successful_payments=("amount", "count")
        )
        .reset_index()
    )
    # lifetime_paid_months = successful payments (each payment ~ 1 billing cycle = 1 month)
    lifetime["lifetime_paid_months"] = lifetime["successful_payments"]
    lifetime = lifetime.drop(columns=["successful_payments"])

    # -- Latest subscription & plan join --
    latest_sub = (
        subscriptions.sort_values("end_date")
        .groupby("user_id").tail(1)
        [["user_id", "plan_id", "status", "end_date", "start_date"]]
    )
    latest_sub = latest_sub.merge(
        plans[["plan_id", "plan_name", "price", "tier"]],
        on="plan_id", how="left"
    )
    latest_sub = latest_sub.rename(columns={
        "price": "plan_price",
        "status": "sub_status",
        "end_date": "sub_end_date",
        "start_date": "sub_start_date"
    })

    # -- Merge all --
    dim = users.copy()
    dim = dim.merge(last_active,  on="user_id", how="left")
    dim = dim.merge(usage_agg,    on="user_id", how="left")
    dim = dim.merge(lifetime,     on="user_id", how="left")
    dim = dim.merge(latest_sub,   on="user_id", how="left")

    dim["avg_daily_minutes"]     = dim["avg_daily_minutes"].fillna(0)
    dim["lifetime_paid_amount"]  = dim["lifetime_paid_amount"].fillna(0)
    dim["lifetime_paid_months"]  = dim["lifetime_paid_months"].fillna(0).astype(int)

    # -- Tenure days (from signup to AS_OF_DATE) --
    dim["tenure_days"] = (AS_OF_DATE - dim["signup_date"]).dt.days

    # -- Days since last active (relative to feature snap date 2026-01-12, not AS_OF_DATE)
    # Using AS_OF_DATE causes ~99% to be 0 since usage runs until Jan 30.
    # Using the snap date gives meaningful signal about recent inactivity.
    FEATURE_SNAP_DATE = pd.Timestamp("2026-01-12")
    dim["days_since_last_active"] = (
        FEATURE_SNAP_DATE - pd.to_datetime(dim["last_active_date"])
    ).dt.days.clip(lower=0)

    # -- Engagement band (based on avg daily minutes, non-outlier) --
    q33 = dim["avg_daily_minutes"].quantile(0.33)
    q66 = dim["avg_daily_minutes"].quantile(0.66)
    dim["engagement_band"] = pd.cut(
        dim["avg_daily_minutes"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "medium", "high"]
    )

    # -- Price band --
    dim["price_band"] = pd.cut(
        dim["plan_price"].fillna(0),
        bins=[-1, 300, 500, 10000],
        labels=["low", "mid", "high"]
    )

    dim.to_csv("data/dim_users_enriched.csv", index=False)
    print(f"  dim_users_enriched: {dim.shape}")
    return dim


# ==========================================================
# 4. FACT_USER_WEEKLY  (1 row per user per week)
# ==========================================================

def build_fact_user_weekly(usage, payments, subscriptions, campaigns):
    print("Building fact_user_weekly...")

    # -- Weekly usage --
    usage["week_start"] = usage["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_usage = (
        usage.groupby(["user_id", "week_start"])
        .agg(
            active_days_week       =("date",           "nunique"),
            total_minutes_week     =("minutes_used",   "sum"),
            sessions_week          =("sessions_count", "sum"),
            feature_events_week    =("feature_events", "sum"),
            has_outlier_usage_week =("is_outlier_minutes", "max")
        )
        .reset_index()
    )

    # -- Weekly payments --
    payments["week_start"] = payments["payment_date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_payments = (
        payments.groupby(["user_id", "week_start"])
        .agg(
            payment_attempts_week =("payment_status", "count"),
            payment_failures_week =("payment_status", lambda x: (x != "success").sum())
        )
        .reset_index()
    )

    # -- Weekly campaign signals --
    campaigns["week_start"] = campaigns["touch_date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly_campaigns = (
        campaigns.groupby(["user_id", "week_start"])
        .agg(
            campaign_touches_week =("touch_id",  "count"),
            campaign_clicks_week  =("clicked",   "sum"),
            campaign_redeemed_week=("redeemed",  "sum")
        )
        .reset_index()
    )

    # -- Renewal due flag: subscription expiring within 14 days of week_end --
    sub_expiry = subscriptions[["user_id", "end_date"]].copy()
    sub_expiry = sub_expiry.dropna(subset=["end_date"])

    fact = weekly_usage.merge(weekly_payments,  on=["user_id", "week_start"], how="left")
    fact = fact.merge(weekly_campaigns,         on=["user_id", "week_start"], how="left")

    # Fill campaign & payment NAs
    for col in ["payment_attempts_week", "payment_failures_week",
                "campaign_touches_week", "campaign_clicks_week", "campaign_redeemed_week"]:
        fact[col] = fact[col].fillna(0)

    # Renewal due: flag weeks where the user's subscription expires within 14 days AFTER week_end
    # All active subs end on 2026-01-30; for earlier weeks this correctly flags the renewal window.
    # Using strict future-facing window: sub expires between week_end and week_end+14d
    fact["week_end"] = pd.to_datetime(fact["week_start"]) + timedelta(days=6)
    sub_exp_map = sub_expiry.groupby("user_id")["end_date"].max()

    week_ends   = pd.to_datetime(fact["week_start"]) + timedelta(days=6)
    user_expiry = fact["user_id"].map(sub_exp_map)
    days_to_exp = (user_expiry - week_ends).dt.days
    fact["renewal_due_flag"] = ((days_to_exp >= 0) & (days_to_exp <= 14)).astype(int)
    fact = fact.drop(columns=["week_end"])

    fact.to_csv("data/fact_user_weekly.csv", index=False)
    print(f"  fact_user_weekly: {fact.shape}")
    return fact


# ==========================================================
# 5. MODEL_CHURN_DATASET  (1 row per user, for ML)
# ==========================================================

def build_model_dataset(fact, dim, subscriptions):
    print("Building model_churn_dataset...")

    fact = fact.sort_values(["user_id", "week_start"])
    fact["week_start"] = pd.to_datetime(fact["week_start"])

    # Keep only data up to AS_OF_DATE (no future leakage)
    fact = fact[fact["week_start"] <= AS_OF_DATE].copy()

    FEATURE_SNAP_DATE = pd.Timestamp("2026-01-12")   # last week before label window
    PAY_WINDOW = 8                                    # weeks lookback for payment features

    # -- Rolling usage features (4w lookback) --
    for col, newcol in [
        ("total_minutes_week",  "minutes_4w_avg"),
        ("sessions_week",       "sessions_4w_avg"),
        ("active_days_week",    "active_days_4w_avg"),
    ]:
        fact[newcol] = (
            fact.groupby("user_id")[col]
            .transform(lambda x: x.rolling(OBS_WEEKS, min_periods=1).mean())
        )

    # -- Rolling payment features (8w lookback — payments are monthly so need wider window) --
    fact["failures_8w_sum"] = (
        fact.groupby("user_id")["payment_failures_week"]
        .transform(lambda x: x.rolling(PAY_WINDOW, min_periods=1).sum())
    )
    fact["attempts_8w_sum"] = (
        fact.groupby("user_id")["payment_attempts_week"]
        .transform(lambda x: x.rolling(PAY_WINDOW, min_periods=1).sum())
    )
    # Keep 4w alias for backward compatibility
    fact["failures_4w_sum"] = fact["failures_8w_sum"]

    # -- Trend features --
    fact["minutes_drop_pct"] = (
        (fact["total_minutes_week"] - fact["minutes_4w_avg"]) /
        (fact["minutes_4w_avg"] + 1e-6)
    )
    fact["sessions_drop_pct"] = (
        (fact["sessions_week"] - fact["sessions_4w_avg"]) /
        (fact["sessions_4w_avg"] + 1e-6)
    )
    fact["failure_rate_8w"] = (
        fact["failures_8w_sum"] / (fact["attempts_8w_sum"] + 1e-6)
    )
    fact["failure_rate_4w"] = fact["failure_rate_8w"]   # alias

    # -- One row per user: latest week at or before FEATURE_SNAP_DATE --
    snap_fact = fact[fact["week_start"] <= FEATURE_SNAP_DATE]
    model_df = (
        snap_fact.sort_values("week_start")
        .groupby("user_id")
        .tail(1)
        .copy()
    )
    print(f"  Feature snapshot week: {FEATURE_SNAP_DATE.date()} "
          f"({model_df['week_start'].nunique()} unique weeks in snapshot)")

    model_df["as_of_date"] = AS_OF_DATE

    # ----------------------------------------------------------
    # CHURN LABEL  (retrospective 14-day window — correct approach)
    #
    # Since this is a historical dataset (obs end = AS_OF_DATE = 2026-01-30)
    #
    # Churn event = subscription cancelled OR expired within the label window.
    # ----------------------------------------------------------

    subs = subscriptions.copy()
    subs["user_id"] = subs["user_id"].str.strip().str.lower()
    subs["status"]  = subs["status"].str.strip().str.lower()

    label_window_start = AS_OF_DATE - timedelta(days=PRED_HORIZON)   # 2026-01-16

    cancelled_mask = (
        (subs["status"] == "cancelled") &
        (subs["cancel_date"] >= label_window_start) &
        (subs["cancel_date"] <= AS_OF_DATE)
    )
    expired_mask = (
        (subs["status"] == "expired") &
        (subs["end_date"] >= label_window_start) &
        (subs["end_date"] <= AS_OF_DATE)
    )

    churning_users = subs[cancelled_mask | expired_mask]["user_id"].unique()
    print(f"  Label window: {label_window_start.date()} → {AS_OF_DATE.date()}")
    print(f"  Users labelled will_churn_14d=1: {len(churning_users)} "
          f"(churn rate: {len(churning_users)/2500:.3f})")

    model_df["will_churn_14d"] = model_df["user_id"].isin(churning_users).astype(int)

    # Drop old incorrect label column if present
    model_df = model_df.drop(columns=["churn_14d"], errors="ignore")

    # -- Merge dim features --
    dim_features = dim[[
        "user_id", "tenure_days", "days_since_last_active",
        "engagement_band", "lifetime_paid_amount", "lifetime_paid_months",
        "plan_name", "plan_price", "price_band", "sub_status"
    ]].copy()

    model_df = model_df.merge(dim_features, on="user_id", how="left")

    print(f"  model_churn_dataset: {model_df.shape}")
    print(f"  Churn rate: {model_df['will_churn_14d'].mean():.4f}")

    model_df.to_csv("data/model_churn_dataset.csv", index=False)
    return model_df


# ==========================================================
# MAIN
# ==========================================================

def main():
    users, plans, subscriptions, payments, usage, campaigns = load_raw_data()
    users, plans, subscriptions, payments, usage, campaigns = clean_data(
        users, plans, subscriptions, payments, usage, campaigns
    )

    dim  = build_dim_users(users, plans, subscriptions, payments, usage)
    fact = build_fact_user_weekly(usage, payments, subscriptions, campaigns)
    build_model_dataset(fact, dim, subscriptions)

    print("\nETL complete. Outputs in /data/")


if __name__ == "__main__":
    main()