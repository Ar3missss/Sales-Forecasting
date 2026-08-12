"""
Sales Forecasting & BI Dashboard — Streamlit App
Author: Aryan Pathania
Reads the real sales_data.csv from the repo and replicates the
exact pipeline from sales-forecasting.ipynb:
  - Sale_Date -> datetime
  - Profit = (Unit_Price - Unit_Cost) * Quantity_Sold
  - Monthly revenue aggregation with a numeric time_index
  - Last 3 months held out as test set
  - LinearRegression fit on time_index
All metrics (MAE, RMSE, R2, MAPE) are computed live from the real data.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# 1. THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")

# 2. NOW we can add the CSS to hide Streamlit's default chrome for embedding
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load + prepare data (cached so it only runs once)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare():
    # Same path your notebook uses
    df = pd.read_csv("data/sales_data.csv")

    # --- Cleaning (identical to notebook) ---
    df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])          # to_datetime
    df["month"]      = df["Sale_Date"].dt.month
    df["month_name"] = df["Sale_Date"].dt.month_name()
    df["quarter"]    = df["Sale_Date"].dt.quarter
    df["year"]       = df["Sale_Date"].dt.year

    # --- Profit engineering (identical to notebook) ---
    df["Profit"] = (df["Unit_Price"] - df["Unit_Cost"]) * df["Quantity_Sold"]

    # --- Monthly time series with numeric time_index (identical to notebook) ---
    monthly_ts = df.groupby(df["Sale_Date"].dt.to_period("M"))["Sales_Amount"].sum().reset_index()
    monthly_ts.columns = ["Period", "Revenue"]
    monthly_ts["Date"]       = monthly_ts["Period"].dt.to_timestamp()
    monthly_ts["time_index"] = range(len(monthly_ts))
    return df, monthly_ts

df, monthly_ts = load_and_prepare()

# ------------------------------------------------------------------
# Train / test split + model (identical to notebook: last 3 months test)
# ------------------------------------------------------------------
split = len(monthly_ts) - 3
train = monthly_ts.iloc[:split]
test  = monthly_ts.iloc[split:]

X_train = train[["time_index"]]
y_train = train["Revenue"]
X_test  = test[["time_index"]]
y_test  = test["Revenue"]

model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluation metrics (computed live from real data) ---
y_pred       = model.predict(X_test)
mae          = mean_absolute_error(y_test, y_pred)
rmse         = np.sqrt(mean_squared_error(y_test, y_pred))
r2           = r2_score(y_test, y_pred)
mape         = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
slope        = model.coef_[0]
intercept    = model.intercept_

# ------------------------------------------------------------------
# Sidebar — user controls
# ------------------------------------------------------------------
st.sidebar.header("Forecast Parameters")
horizon = st.sidebar.slider("Forecast horizon (months)", 1, 12, 3)
region_filter = st.sidebar.selectbox(
    "Region (historical KPI filter)", ["All"] + sorted(df["Region"].unique().tolist())
)

# ------------------------------------------------------------------
# Title
# ------------------------------------------------------------------
st.title("📈 Sales Forecasting & Business Intelligence Dashboard")
st.markdown(
    "Interactive deployment of the linear-regression sales forecast built on the "
    "Kaggle sales dataset. Adjust the horizon in the sidebar to project future monthly revenue."
)

# ------------------------------------------------------------------
# KPI cards (from real data)
# ------------------------------------------------------------------
kpi_df = df if region_filter == "All" else df[df["Region"] == region_filter]
total_revenue = kpi_df["Sales_Amount"].sum()
total_profit  = kpi_df["Profit"].sum()
avg_order     = kpi_df["Sales_Amount"].mean()
profit_margin = (total_profit / total_revenue) * 100 if total_revenue else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${total_revenue:,.2f}")
c2.metric("Total Profit",  f"${total_profit:,.2f}")
c3.metric("Avg Order Value", f"${avg_order:,.2f}")
c4.metric("Profit Margin",   f"{profit_margin:.1f}%")
if region_filter != "All":
    st.caption(f"KPIs filtered to region: **{region_filter}**")

# ------------------------------------------------------------------
# Forecast for the chosen horizon
# ------------------------------------------------------------------
last_index   = monthly_ts["time_index"].max()
future_idx   = np.arange(last_index + 1, last_index + 1 + horizon).reshape(-1, 1)
future_pred  = model.predict(future_idx)
last_date    = monthly_ts["Date"].max()
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

forecast_df = pd.DataFrame({
    "Month":           future_dates.strftime("%B %Y"),
    "Predicted_Revenue": np.round(future_pred, 2)
})

# ------------------------------------------------------------------
# Historical + Forecast chart (Plotly, interactive)
# ------------------------------------------------------------------
st.subheader("Historical Revenue & Forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly_ts["Date"], y=monthly_ts["Revenue"],
    mode="lines+markers", name="Historical Revenue", line=dict(color="#2E8B57", width=2)
))
# Bridge between last actual and first forecast
bridge_x = [monthly_ts["Date"].iloc[-1], future_dates[0]]
bridge_y = [monthly_ts["Revenue"].iloc[-1], future_pred[0]]
fig.add_trace(go.Scatter(
    x=bridge_x, y=bridge_y,
    mode="lines", name="Forecast Bridge", line=dict(color="#E50914", dash="dash", width=2)
))
# Changed to lines+markers (removed text) to stop numbers from overlapping
fig.add_trace(go.Scatter(
    x=future_dates, y=future_pred,
    mode="lines+markers", name="Forecast",
    line=dict(color="#E50914", width=2),
    hovertemplate='<b>%{x|%b %Y}</b><br>Predicted: $%{y:,.2f}<extra></extra>'
))
fig.update_layout(
    xaxis_title="Month", yaxis_title="Revenue ($)",
    hovermode="x unified", height=450, 
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    margin=dict(l=20, r=20, t=40, b=20) # Tighter margins so it fits in the iframe better
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Forecast table
# ------------------------------------------------------------------
st.subheader(f"{horizon}-Month Revenue Forecast")
st.dataframe(forecast_df.style.format({"Predicted_Revenue": "${:,.2f}"}), use_container_width=True)

# ------------------------------------------------------------------
# Model evaluation (live, real numbers)
# ------------------------------------------------------------------
st.subheader("Model Evaluation (test period = last 3 months)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE",  f"${mae:,.2f}", help="Mean Absolute Error — avg dollar error per month")
m2.metric("RMSE", f"${rmse:,.2f}", help="Root Mean Squared Error on the 3-month test set")
m3.metric("R²",   f"{r2:.4f}",   help="How well the trend line explains revenue variance")
m4.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error vs actual revenue")
st.caption(f"Slope (monthly revenue change): **${slope:,.2f}** | "
           f"Intercept: **${intercept:,.2f}**")

# Actual vs Predicted on the test period
st.subheader("Actual vs Predicted — Test Period")
avp = pd.DataFrame({
    "Month":   test["Date"].dt.strftime("%B %Y"),
    "Actual":  y_test.values,
    "Predicted": np.round(y_pred, 2)
})
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=avp["Month"], y=avp["Actual"],  name="Actual",    marker_color="#2E8B57"))
fig2.add_trace(go.Bar(x=avp["Month"], y=avp["Predicted"], name="Predicted", marker_color="#E50914"))
fig2.update_layout(barmode="group", xaxis_title="Month", yaxis_title="Revenue ($)", height=380)
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(avp.style.format({"Actual": "${:,.2f}", "Predicted": "${:,.2f}"}), use_container_width=True)