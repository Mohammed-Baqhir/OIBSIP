# 🏠 House Price Prediction — Linear Regression

**Oasis Infobyte — Level 2, Task 1**

End-to-end house-price prediction using the official **Kaggle House Prices: Advanced Regression Techniques (Ames Housing)** training data. Kaggle provides `train.csv`, `test.csv`, and `data_description.txt`; `SalePrice` is the target.

## Dataset
Official source: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Download the official `train.csv`, rename it to `housing.csv`, and put it at `data/housing.csv`.

## Structure
```text
DataAnalytics-L2-HousePricePrediction/
├── data/
│   └── housing.csv
├── house_price_prediction.ipynb
├── train.py
├── app.py
├── model.pkl          # generated after training
├── columns.pkl        # generated after training
├── README.md
└── requirements.txt
```

## Features used
- `GrLivArea` — living area
- `Neighborhood` — location
- `TotRmsAbvGrd` — rooms above grade
- `HouseAge` — engineered as `YrSold - YearBuilt`
- `OverallQual` — overall quality
- `FullBath` — bathrooms
- `GarageCars` — garage capacity
- `TotalBsmtSF` — basement area

This compact feature set directly matches the task wording while keeping the model explainable.

## Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Checklist
- EDA, null check and descriptive statistics
- Target distribution
- Feature-selection discussion
- Missing-value handling
- One-Hot Encoding
- Correlation heatmap
- 80/20 split
- Linear Regression
- MSE, RMSE and R²
- Actual vs predicted plot
- Residual plot
- Coefficient analysis
- Ridge comparison (bonus)
