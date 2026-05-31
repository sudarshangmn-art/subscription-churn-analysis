# Sudarshan_Capstone_SubscriptionChurn
## Subscription Churn Early-Warning & Retention Strategy

---

## Project Overview

This capstone builds an end-to-end churn prediction and retention system for a subscription-based digital product (learning app / OTT / SaaS). Starting from 6 raw CSV files, it produces curated analytics tables, a trained churn prediction model, a Tableau dashboard, and a final retention recommendation.

**Observation date:** 2026-01-30  
**Prediction horizon:** 14 days  
**Total users:** 2,500  
**Churners identified:** 121 (4.8% churn rate)  
**Users targeted for retention:** 218  
**Model:** Random Forest (ROC-AUC = 0.9046)

---

## Churn Definition

> A user is considered churned (`will_churn_14d = 1`) if their subscription was **cancelled OR expired** within the retrospective 14-day label window: **[2026-01-16, 2026-01-30]**.

**Label design rationale:**
- Forward-looking window not used: all active subscriptions share end_date = 2026-01-30 (auto-renewing), leaving no genuine forward signal
- All-time historical approach not used: labelling all 683 ever-cancelled users (27% rate) includes users who churned months ago — not predictable from recent behaviour
- Chosen 14-day retrospective window yields 121 churners (4.8% rate) — realistic, learnable signal with clean feature-label separation

**Feature snap date:** 2026-01-12 (features built from data up to this date; label from Jan 16–30)

---

## How to Run ETL End-to-End

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Step 1 — Place raw data files
Ensure these 6 files are in the `/data/raw/` directory:
```
users.csv
plans.csv
subscriptions.csv
payments.csv
usage_daily.csv
campaign_touchpoints.csv
```

### Step 2 — Run the ETL pipeline
```bash
cd etl/
python etl_pipeline.py
```

**Runtime:** ~30–60 seconds on a standard laptop

### Step 3 — Verify outputs
Check that `/data/` contains:
```
dim_users_enriched.csv   (2,500 rows × 11 columns)
fact_user_weekly.csv     (132,808 rows × 8 columns)
model_churn_dataset.csv  (2,500 rows × 22 columns)
targeting_list.csv       (218 rows × 4 columns)
```

---

## Curated Output Datasets

### `dim_users_enriched.csv` — 1 row per user
User-level enriched dimension table. Key columns:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `signup_date` | Original signup date |
| `city_tier` | City tier (1/2/3) |
| `segment` | User segment (value/standard/premium) |
| `preferred_device` | Most used device type |
| `acquisition_channel` | How user was acquired |
| `tenure_days` | Days since signup |
| `lifetime_paid_months` | Proxy for total paid months |
| `last_active_date` | Most recent active day |
| `avg_daily_minutes` | Average daily usage minutes |
| `engagement_band` | low / medium / high (based on usage quartiles) |

---

### `fact_user_weekly.csv` — 1 row per user per week
Weekly behavioural feature table. Key columns:

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `week_start` | Start of the week (Monday) |
| `active_days_week` | Days active that week |
| `total_minutes_week` | Total usage minutes that week |
| `sessions_week` | Number of sessions |
| `feature_events_week` | Feature interaction count |
| `payment_attempts_week` | Payment attempts that week |
| `payment_failures_week` | Failed payment attempts that week |

---

### `model_churn_dataset.csv` — 1 row per user (ML-ready)
Feature snapshot at 2026-01-12 with churn label. Key columns:

| Column | Description |
|--------|-------------|
| `will_churn_14d` | **Label** — 1 if churned in [Jan 16–30], else 0 |
| `minutes_drop_pct` | % change in minutes vs 4-week average |
| `failures_8w_sum` | Total payment failures in last 8 weeks |
| `failure_rate_8w` | Payment failure rate over 8 weeks |
| `renewal_due_flag` | 1 if renewal due within 14 days |
| `days_since_last_active` | Days inactive as of snap date |
| `plan_name` | Subscription plan (Basic/Standard/Premium) |
| `plan_price` | Monthly plan price (₹) |
| `engagement_band` | low / medium / high |
| `tenure_days` | Days since signup |

---

### `targeting_list.csv` — 218 targeted users
Output of the model scoring step. Columns: `user_id`, `churn_prob`, `risk_band`, `will_churn_14d`

---

## How to Run the Analysis

```bash
cd analysis/
python analysis.py
```

The script runs sequentially through 11 sections and prints all outputs to terminal:
1. Data loading & validation
2. Churn & retention diagnosis
3. Cohort retention table
4. KPI trends
5. Segment deep dive
6. Revenue at risk analysis
7. Churn investigation table
8. Model training (LR + RF)
9. Model evaluation & comparison
10. Feature importance & early warning signals
11. Impact estimation (30-day projection, 3 scenarios)

**Runtime:** ~2–3 minutes (Random Forest training)

---

## Dashboard

**Tool:** Tableau Public (Desktop Edition 2025.2)  
**File:** `dashboard/Sudarshan_Churn_Dashboard.twbx`

### How to open
1. Download and install [Tableau Public](https://public.tableau.com/en-us/s/download) (free)
2. Open `Sudarshan_Churn_Dashboard.twbx` — data is embedded, no external files needed
3. Navigate using the 4 dashboard tabs at the bottom

### Dashboard pages
| Dashboard | Contents |
|-----------|----------|
| **Executive Summary** | KPI tiles (Total Users, Churn Rate, Retention Rate, Revenue at Risk), Weekly Churn Trend, Revenue at Risk by Plan |
| **Cohorts & Segments** | Cohort Retention heatmap (M0–M6), Churn by Engagement Band, Revenue at Risk by Segment |
| **Driver Diagnostics** | Churn by Usage Drop Pattern, Payment Failure Impact, Renewal Risk Funnel, Churn Investigation Summary |
| **Model & Targeting** | Model Performance Summary, Risk Band Distribution, Who to Target table |

### Screenshots
See `dashboard/dashboard_screenshots/` for 4–6 PNG exports of each dashboard view.

---

## Folder Structure

```
Sudarshan_Capstone_SubscriptionChurn/
├── README.md
├── /data/
│   ├── dim_users_enriched.csv
│   ├── fact_user_weekly.csv
│   ├── model_churn_dataset.csv
│   └── targeting_list.csv
├── /etl/
│   └── etl_pipeline.py
├── /analysis/
│   └── analysis.py
├── /docs/
│   ├── Part_A_Problem_Framing.docx
│   ├── Part_E_Retention_Strategy.docx
│   └── Part_F_Impact_Estimation.docx
├── /dashboard/
│   ├── Sudarshan_Churn_Dashboard.twbx
│   └── /dashboard_screenshots/
│       ├── dashboard1_executive_summary.png
│       ├── dashboard2_cohorts_segments.png
│       ├── dashboard3_driver_diagnostics.png
│       └── dashboard4_model_targeting.png
└── /final_story/
    └── final_memo.pdf
```

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| Total users | 2,500 |
| Churn rate (14-day window) | 4.8% (121 users) |
| RF ROC-AUC | 0.9046 |
| LR ROC-AUC (baseline) | 0.8950 |
| Business threshold | 0.50 |
| Users targeted | 218 (170 high + 48 medium risk) |
| Precision @ 0.50 | 47% |
| Recall @ 0.50 | 67% |
| Revenue at risk | ₹87,982 |
| Base case net impact | ₹34,221 |

---

*Capstone Project*  
*Observation end date: 2026-01-30 | Model snap date: 2026-01-12*
