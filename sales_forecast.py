"""
Sales Forecast Model
=====================
Predicts monthly sales (Amount) from transaction-level data using a linear
trend + calendar-month seasonal index model, evaluated with a proper
train/test split (not just in-sample fit).

Input : raw_data.csv with columns [order_date, customer_id, payment_method,
        quantity, amount, profit]
Output: forecast_results.csv, monthly_sales.csv, sales_forecast_chart.png,
        and printed accuracy metrics (MAE, RMSE, MAPE, R^2)

Dependencies: pandas, numpy, scikit-learn, matplotlib (no statsmodels needed)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Load and aggregate to monthly sales
# ----------------------------------------------------------------------
df = pd.read_csv("raw_data.csv", parse_dates=["order_date"])

monthly = (
    df.set_index("order_date")
      .resample("MS")  # month start
      .agg(total_amount=("amount", "sum"),
           total_profit=("profit", "sum"),
           total_quantity=("quantity", "sum"),
           order_count=("amount", "count"))
      .reset_index()
      .rename(columns={"order_date": "month"})
)

# Fill any gap months with 0 so the time index has no missing periods
full_range = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
monthly = (monthly.set_index("month")
                   .reindex(full_range)
                   .fillna(0)
                   .rename_axis("month")
                   .reset_index())

monthly["t"] = np.arange(1, len(monthly) + 1)          # linear time index
monthly["calendar_month"] = monthly["month"].dt.month   # 1-12, for seasonality

monthly.to_csv("monthly_sales.csv", index=False)
print(f"Aggregated {len(df)} transactions into {len(monthly)} monthly periods "
      f"({monthly['month'].min().date()} to {monthly['month'].max().date()})\n")

# ----------------------------------------------------------------------
# 2. Train/test split (last 3 months held out — walk-forward style)
# ----------------------------------------------------------------------
TEST_MONTHS = 3
if len(monthly) <= TEST_MONTHS + 3:
    raise ValueError("Not enough history for a meaningful train/test split — "
                      "need at least 6+ months of data.")

train = monthly.iloc[:-TEST_MONTHS].copy()
test = monthly.iloc[-TEST_MONTHS:].copy()

# ----------------------------------------------------------------------
# 3. Fit linear trend on TRAIN only
# ----------------------------------------------------------------------
X_train = train[["t"]].values
y_train = train["total_amount"].values

trend_model = LinearRegression()
trend_model.fit(X_train, y_train)

train["trend_fit"] = trend_model.predict(X_train)

# ----------------------------------------------------------------------
# 4. Seasonal index from TRAIN residual ratio (actual / trend), by calendar month
#    Guard against divide-by-zero and months with too little data.
# ----------------------------------------------------------------------
train["ratio"] = np.where(train["trend_fit"] > 0,
                           train["total_amount"] / train["trend_fit"],
                           1.0)
seasonal_index = train.groupby("calendar_month")["ratio"].mean().to_dict()
overall_mean_ratio = train["ratio"].mean()

def get_seasonal_index(cal_month):
    return seasonal_index.get(cal_month, overall_mean_ratio)

# ----------------------------------------------------------------------
# 5. Predict on TEST (unseen) months: trend * seasonal index
# ----------------------------------------------------------------------
test["trend_fit"] = trend_model.predict(test[["t"]].values)
test["seasonal_idx"] = test["calendar_month"].map(get_seasonal_index)
test["forecast"] = test["trend_fit"] * test["seasonal_idx"]

# ----------------------------------------------------------------------
# 6. Evaluate on the held-out test months
# ----------------------------------------------------------------------
y_true = test["total_amount"].values
y_pred = test["forecast"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# MAPE guarding against zero actuals
nonzero = y_true != 0
mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) if nonzero.any() else np.nan
r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

# Naive baseline for comparison: "next month = same as last known month"
naive_pred = np.full_like(y_true, fill_value=train["total_amount"].iloc[-1], dtype=float)
naive_mae = mean_absolute_error(y_true, naive_pred)

print("=== Model Accuracy on Held-Out Test Months ===")
print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"MAPE : {mape:.1%}" if not np.isnan(mape) else "MAPE : n/a (zero actuals in test set)")
print(f"R^2  : {r2:.3f}" if not np.isnan(r2) else "R^2  : n/a")
print(f"\nNaive baseline MAE (last-value-carried-forward): ${naive_mae:,.2f}")
if mae < naive_mae:
    print("-> Model beats the naive baseline.")
else:
    print("-> Model does NOT beat the naive baseline — with this little data, "
          "treat forecasts as directional only.")

# ----------------------------------------------------------------------
# 7. Refit on FULL history, forecast next 3 months forward
# ----------------------------------------------------------------------
X_full = monthly[["t"]].values
y_full = monthly["total_amount"].values
final_model = LinearRegression().fit(X_full, y_full)

monthly["trend_fit_full"] = final_model.predict(X_full)
monthly["ratio_full"] = np.where(monthly["trend_fit_full"] > 0,
                                  monthly["total_amount"] / monthly["trend_fit_full"], 1.0)
seasonal_index_full = monthly.groupby("calendar_month")["ratio_full"].mean().to_dict()
overall_mean_ratio_full = monthly["ratio_full"].mean()

future_periods = 3
last_t = monthly["t"].max()
last_month = monthly["month"].max()
future_months = pd.date_range(last_month + pd.DateOffset(months=1), periods=future_periods, freq="MS")

future_df = pd.DataFrame({
    "month": future_months,
    "t": np.arange(last_t + 1, last_t + 1 + future_periods),
})
future_df["calendar_month"] = future_df["month"].dt.month
future_df["trend_forecast"] = final_model.predict(future_df[["t"]].values)
future_df["seasonal_idx"] = future_df["calendar_month"].map(
    lambda m: seasonal_index_full.get(m, overall_mean_ratio_full))
future_df["forecast"] = future_df["trend_forecast"] * future_df["seasonal_idx"]

print("\n=== Next 3 Months Forecast (trained on full history) ===")
for _, row in future_df.iterrows():
    print(f"{row['month'].strftime('%b-%Y')}: ${row['forecast']:,.2f}")

future_df[["month", "trend_forecast", "seasonal_idx", "forecast"]].to_csv(
    "forecast_results.csv", index=False)

# ----------------------------------------------------------------------
# 8. Chart: actual vs. fitted vs. forecast
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(monthly["month"], monthly["total_amount"], marker="o", label="Actual Sales", color="#1F2937")
ax.plot(monthly["month"], monthly["trend_fit_full"], linestyle="--", label="Trend (linear fit)", color="#9CA3AF")
ax.plot(future_df["month"], future_df["forecast"], marker="o", linestyle="--",
        label="Forecast (next 3 months)", color="#B45309")
ax.axvline(last_month, color="#D1D5DB", linestyle=":")
ax.set_title("Monthly Sales: Actual vs. Trend vs. Forecast")
ax.set_ylabel("Amount ($)")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("sales_forecast_chart.png", dpi=150)
print("\nSaved: monthly_sales.csv, forecast_results.csv, sales_forecast_chart.png")
