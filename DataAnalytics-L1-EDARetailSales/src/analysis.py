def descriptive_statistics(df):
    """Return descriptive statistics for numerical columns."""
    numeric = df.select_dtypes(include="number")
    return {
        "mean": numeric.mean(),
        "median": numeric.median(),
        "mode": numeric.mode().iloc[0],
        "std": numeric.std(),
    }

def category_sales(df):
    return df.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False)

def monthly_sales(df):
    return df.set_index("Date")["Total Amount"].resample("MS").sum()

def quarterly_sales(df):
    return df.set_index("Date")["Total Amount"].resample("QS").sum()

def gender_sales(df):
    return df.groupby("Gender")["Total Amount"].sum().sort_values(ascending=False)

def age_group_sales(df):
    return df.groupby("Age Group", observed=False)["Total Amount"].sum()

def top_transactions(df, n=10):
    return df.nlargest(n, "Total Amount")
