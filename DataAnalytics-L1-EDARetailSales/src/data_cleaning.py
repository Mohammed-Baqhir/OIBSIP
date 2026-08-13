import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and feature engineering for the retail dataset."""
    df = df.copy()
    df = df.drop_duplicates()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna()
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18–25", "26–35", "36–45", "46–55", "56–64"],
        include_lowest=True,
    )
    return df
