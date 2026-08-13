# 📊 Retail Sales EDA Dashboard

An interactive Exploratory Data Analysis project for a retail sales dataset.


## 🚀 Live Demo

👉 [Open the Retail Sales Dashboard](https://oibsip-retail-sales-eda.streamlit.app/)

## Features

- Dataset inspection and cleaning
- KPI cards for revenue, transactions, average order value and units sold
- Monthly and quarterly sales trends
- Gender and age-group analysis
- Revenue and quantity by product category
- Top 10 highest-value transactions
- Correlation heatmap
- CSV downloads
- Interactive category, gender and date filters

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit

## Project Structure

```text
retail_eda_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── retail_sales_dataset.csv
├── src/
│   ├── analysis.py
│   ├── data_cleaning.py
│   ├── data_loader.py
│   └── visualization.py
└── outputs/
    └── plots/
```

## Run Locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Important Dataset Note

The supplied dataset has 9 columns and contains `Product Category`, but it does not contain an individual product-name field. Therefore a true "Top 10 Products" analysis cannot be calculated from this dataset. The dashboard transparently provides a Top 10 Highest-Value Transactions view instead.

## Deployment

The project can be deployed using Streamlit Community Cloud after pushing the repository to GitHub.
