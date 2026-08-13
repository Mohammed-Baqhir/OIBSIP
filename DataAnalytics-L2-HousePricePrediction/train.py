from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "housing.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"

NUMERIC_FEATURES = [
    "GrLivArea",
    "TotRmsAbvGrd",
    "FullBath",
    "GarageCars",
    "TotalBsmtSF",
    "OverallQual",
    "HouseAge",
]

CATEGORICAL_FEATURES = [
    "Neighborhood"
]

TARGET = "SalePrice"


def prepare_data(df):
    df = df.copy()

    # Create house age from the year the house was sold
    # and the year it was originally built.
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Prevent negative ages in case of unusual records.
    df["HouseAge"] = df["HouseAge"].clip(lower=0)

    return df


def build_pipeline():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    return pipeline


def main():

    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully: {df.shape}")

    df = prepare_data(df)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training Linear Regression model...")

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    # Save trained model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    # Save feature information
    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET
    }

    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print()
    print("=" * 55)
    print("HOUSE PRICE LINEAR REGRESSION")
    print("=" * 55)

    print(f"Rows          : {len(df):,}")
    print(f"Training rows : {len(X_train):,}")
    print(f"Testing rows  : {len(X_test):,}")

    print()
    print(f"MSE  : {mse:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    print()
    print("Saved successfully:")
    print(f"  {MODEL_PATH.name}")
    print(f"  {COLUMNS_PATH.name}")


if __name__ == "__main__":
    main()