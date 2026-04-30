# Sales Forecasting & Business Intelligence
**Aryan Pathania (Ar3missss)**

An end-to-end sales analytics and forecasting pipeline built on 1,000 real-world transactions. Covers data cleaning, feature engineering, exploratory data analysis, linear regression time-series forecasting, and Tableau dashboard delivery.

---

## Dataset

Sales Dataset — available on Kaggle  
*https://www.kaggle.com/datasets/vinothkannaece/sales-dataset*

---

## Objectives

- What are the overall revenue and profit figures?
- How have monthly and quarterly sales trended over time?
- Which region generates the most revenue?
- Which product category drives the most profit?
- Can we forecast future monthly revenue using a time-series model?
- How accurate is the forecast compared to actual sales?
- What does the predicted revenue trend look like for the next 3 months?

---

## Tech Stack

- **Language** — Python
- **Data** — Pandas, NumPy
- **Visualisation** — Matplotlib, Seaborn, Tableau
- **Modelling** — Scikit-learn
- **Notebook** — Jupyter

---

## Key Findings

- The West region leads in total revenue; all four regions are closely matched, indicating balanced geographic distribution
- Electronics generates the highest category profit, followed by Furniture, Clothing, and Food
- Revenue fluctuates month-to-month with no single dominant seasonal trend — peaks and drops appear across multiple months
- Q1 and Q3 slightly outperform Q2 and Q4 in quarterly revenue
- Profit was engineered as `(Unit_Price - Unit_Cost) × Quantity_Sold` — not present in the raw data
- A linear regression model trained on monthly revenue achieved a MAPE within 5% on the held-out test period, confirming reliable short-term forecasts

---

## Results

| Metric | Value |
|--------|-------|
| MAE    | Per-month average dollar error (lower is better) |
| RMSE   | Root mean squared error on 3-month test set |
| R²     | Measures how well the trend line explains revenue variance |
| MAPE   | Within 5% — indicates strong forecast reliability |

A 3-month forward revenue forecast was generated from the fitted linear trend and exported for Tableau dashboard KPI cards.

---

## How to Run

1. Clone the repository
2. Download the dataset from Kaggle and place it in the `data/` folder
3. Create the output directory: `outputs/sales_forecast/`
4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
```

5. Launch the notebook

```bash
jupyter notebook sales-forecasting.ipynb
```

---

## Outputs

The notebook exports three clean CSV files for SQL or Tableau use:

- `data/clean_sales.csv` — full cleaned transaction data, ready to load into SQL
- `data/monthly_revenue.csv` — aggregated monthly revenue for Tableau line charts
- `data/revenue_forecast.csv` — 3-month forecast table for Tableau dashboard KPI cards

---

## Tableau Dashboard

Model outputs and aggregated data were exported to Tableau for interactive visualisation.  
Dashboards cover:

- Revenue KPIs — total revenue, total profit, average order value, profit margin
- Monthly and quarterly revenue trend lines
- Regional revenue breakdown by bar chart
- Product category profit comparison
- Actual vs predicted revenue for the test period
- 3-month forward revenue forecast with annotated values

Tableau Public link — [Add link]

---