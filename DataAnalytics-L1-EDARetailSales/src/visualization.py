import matplotlib.pyplot as plt
import seaborn as sns

def plot_category_sales(category_sales):
    fig, ax = plt.subplots(figsize=(8, 5))
    category_sales.plot(kind="bar", ax=ax)
    ax.set_title("Sales by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    return fig

def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig
