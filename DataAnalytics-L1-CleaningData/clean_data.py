import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
INPUT = BASE / "data" / "retail_store_sales.csv"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(
        "Place the Kaggle file 'retail_store_sales.csv' inside the data folder first."
    )

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv(INPUT)
original_rows, original_cols = df.shape

# ---------------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES
# ---------------------------------------------------------
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# ---------------------------------------------------------
# 3. NORMALIZE NULL VALUES
# ---------------------------------------------------------
null_markers = {
    "", "None", "none", "NULL", "null",
    "N/A", "n/a", "NA", "na",
    "UNKNOWN", "unknown", "ERROR", "error"
}

for col in df.columns:
    if df[col].dtype == "object":
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.isin(null_markers))

# ---------------------------------------------------------
# 4. CONVERT DATA TYPES
# ---------------------------------------------------------
for col in ["Price Per Unit", "Quantity", "Total Spent"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "Transaction Date" in df.columns:
    df["Transaction Date"] = pd.to_datetime(
        df["Transaction Date"],
        errors="coerce"
    )

missing_before = int(df.isna().sum().sum())

# ---------------------------------------------------------
# 5. REMOVE DUPLICATES
# ---------------------------------------------------------
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# ---------------------------------------------------------
# 6. NORMALIZE CATEGORICAL DATA
# ---------------------------------------------------------
if "Category" in df.columns:

    df["Category"] = (
        df["Category"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    category_map = {
        "Electric Household Essentials":
            "Electric household essentials",

        "Computers And Electric Accessories":
            "Computers and electric accessories",

        "Milk Products": "Milk Products",
        "Patisserie": "Patisserie",
        "Beverages": "Beverages",
        "Furniture": "Furniture",
        "Food": "Food",
        "Butchers": "Butchers"
    }

    df["Category"] = df["Category"].replace(category_map)

if "Payment Method" in df.columns:
    df["Payment Method"] = (
        df["Payment Method"]
        .astype("string")
        .str.strip()
        .str.title()
    )

if "Location" in df.columns:
    df["Location"] = (
        df["Location"]
        .astype("string")
        .str.strip()
        .str.title()
    )

# ---------------------------------------------------------
# 7. RECOVER NUMERIC VALUES
# ---------------------------------------------------------
recovered_quantity = 0
recovered_price = 0
recovered_total = 0

if {
    "Quantity",
    "Price Per Unit",
    "Total Spent"
}.issubset(df.columns):

    # Quantity = Total Spent / Price
    mask = (
        df["Quantity"].isna()
        & df["Price Per Unit"].notna()
        & df["Total Spent"].notna()
        & df["Price Per Unit"].ne(0)
    )

    recovered_quantity = int(mask.sum())

    df.loc[mask, "Quantity"] = (
        df.loc[mask, "Total Spent"]
        / df.loc[mask, "Price Per Unit"]
    )

    # Price = Total Spent / Quantity
    mask = (
        df["Price Per Unit"].isna()
        & df["Quantity"].notna()
        & df["Total Spent"].notna()
        & df["Quantity"].ne(0)
    )

    recovered_price = int(mask.sum())

    df.loc[mask, "Price Per Unit"] = (
        df.loc[mask, "Total Spent"]
        / df.loc[mask, "Quantity"]
    )

    # Total = Quantity × Price
    mask = (
        df["Total Spent"].isna()
        & df["Quantity"].notna()
        & df["Price Per Unit"].notna()
    )

    recovered_total = int(mask.sum())

    df.loc[mask, "Total Spent"] = (
        df.loc[mask, "Quantity"]
        * df.loc[mask, "Price Per Unit"]
    )

# ---------------------------------------------------------
# 8. INFER MISSING ITEM VALUES
# ---------------------------------------------------------
item_recovered = 0

if {
    "Category",
    "Item",
    "Price Per Unit"
}.issubset(df.columns):

    lookup = (
        df.dropna(
            subset=[
                "Item",
                "Category",
                "Price Per Unit"
            ]
        )
        .drop_duplicates(
            ["Category", "Price Per Unit"]
        )
        [["Category", "Price Per Unit", "Item"]]
    )

    price_to_item = {
        (row["Category"], row["Price Per Unit"]):
        row["Item"]
        for _, row in lookup.iterrows()
    }

    mask = (
        df["Item"].isna()
        & df["Category"].notna()
        & df["Price Per Unit"].notna()
    )

    inferred = df.loc[mask].apply(
        lambda row:
        price_to_item.get(
            (
                row["Category"],
                row["Price Per Unit"]
            )
        ),
        axis=1
    )

    valid = inferred.notna()

    item_recovered = int(valid.sum())

    df.loc[
        inferred.index[valid],
        "Item"
    ] = inferred[valid]

# ---------------------------------------------------------
# 9. FILL REMAINING NUMERIC MISSING VALUES
# ---------------------------------------------------------
for col in [
    "Price Per Unit",
    "Quantity",
    "Total Spent"
]:

    if col in df.columns and df[col].isna().any():

        median = df[col].median()

        if pd.notna(median):
            df[col] = df[col].fillna(median)

# ---------------------------------------------------------
# 10. FILL REMAINING CATEGORICAL MISSING VALUES
# ---------------------------------------------------------
for col in [
    "Item",
    "Payment Method",
    "Location",
    "Category"
]:

    if col in df.columns and df[col].isna().any():

        mode = df[col].mode(dropna=True)

        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

# ---------------------------------------------------------
# 11. CLEAN DISCOUNT APPLIED
# ---------------------------------------------------------
if "Discount Applied" in df.columns:

    df["Discount Applied"] = (
        df["Discount Applied"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False
        })
        .fillna(False)
        .astype(bool)
    )

# ---------------------------------------------------------
# 12. FINAL TOTAL CALCULATION
# ---------------------------------------------------------
# IMPORTANT:
# Recalculate Total Spent AFTER all missing values
# have been filled.

if {
    "Quantity",
    "Price Per Unit",
    "Total Spent"
}.issubset(df.columns):

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["Price Per Unit"] = pd.to_numeric(
        df["Price Per Unit"],
        errors="coerce"
    )

    df["Total Spent"] = (
        df["Quantity"]
        * df["Price Per Unit"]
    )

# ---------------------------------------------------------
# 13. ROUND NUMERIC VALUES
# ---------------------------------------------------------
if "Quantity" in df.columns:
    df["Quantity"] = df["Quantity"].round(2)

if "Price Per Unit" in df.columns:
    df["Price Per Unit"] = df["Price Per Unit"].round(2)

# Recalculate AGAIN after rounding price/quantity.
# This guarantees mathematical consistency.

if {
    "Quantity",
    "Price Per Unit"
}.issubset(df.columns):

    df["Total Spent"] = (
        df["Quantity"]
        * df["Price Per Unit"]
    ).round(2)

# ---------------------------------------------------------
# 14. FORMAT DATE
# ---------------------------------------------------------
if "Transaction Date" in df.columns:

    df["Transaction Date"] = (
        df["Transaction Date"]
        .dt.strftime("%Y-%m-%d")
    )

# ---------------------------------------------------------
# 15. FINAL VALIDATION
# ---------------------------------------------------------
remaining_missing = int(
    df.isna().sum().sum()
)

remaining_duplicates = int(
    df.duplicated().sum()
)

validation_mismatches = 0

if {
    "Quantity",
    "Price Per Unit",
    "Total Spent"
}.issubset(df.columns):

    expected_total = (
        df["Quantity"]
        * df["Price Per Unit"]
    ).round(2)

    validation_mismatches = int(
        (
            ~np.isclose(
                df["Total Spent"],
                expected_total,
                atol=0.01,
                rtol=0
            )
        ).sum()
    )

# ---------------------------------------------------------
# 16. SAVE CLEAN DATASET
# ---------------------------------------------------------
out_csv = (
    OUTPUT /
    "retail_store_sales_cleaned.csv"
)

df.to_csv(
    out_csv,
    index=False
)

# ---------------------------------------------------------
# 17. CLEANING REPORT
# ---------------------------------------------------------
report = f"""
OASIS INFOBYTE - DATA CLEANING REPORT
========================================

Source: Retail Store Sales: Dirty for Data Cleaning

Original shape:
{original_rows} rows x {original_cols} columns

Final shape:
{len(df)} rows x {len(df.columns)} columns


QUALITY ACTIONS
----------------------------------------

- Exact duplicates removed: {duplicates_removed}
- Missing cells before cleaning: {missing_before}

- Quantity values recovered: {recovered_quantity}
- Price values recovered: {recovered_price}
- Total Spent values recovered: {recovered_total}

- Item values inferred from category + price: {item_recovered}

- Remaining missing cells: {remaining_missing}
- Remaining duplicate rows: {remaining_duplicates}

- Total formula mismatches after cleaning:
  {validation_mismatches}


OUTPUT
----------------------------------------

- {out_csv.name}

Cleaning completed successfully.
"""

(
    OUTPUT /
    "cleaning_report.txt"
).write_text(
    report,
    encoding="utf-8"
)

print(report)
