import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load the retail CSV dataset."""
    return pd.read_csv(path)
