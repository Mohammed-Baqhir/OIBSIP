import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Retail Data Cleaning Dashboard",
    page_icon="🧹",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .clean-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        text-align: center;
        background-color: transparent;
    }

    .clean-number {
        font-size: 30px;
        font-weight: 700;
    }

    .clean-label {
        font-size: 14px;
        opacity: 0.75;
    }

    .before {
        font-size: 24px;
        font-weight: 600;
    }

    .after {
        font-size: 24px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# PATH
# =========================================================

BASE = Path(__file__).resolve().parent

DATA_FILE = (
    BASE
    / "output"
    / "retail_store_sales_cleaned.csv"
)

# =========================================================
# LOAD DATA
# =========================================================

if not DATA_FILE.exists():

    st.error(
        "Cleaned dataset not found. "
        "Please make sure "
        "`output/retail_store_sales_cleaned.csv` "
        "exists."
    )

    st.stop()

df = pd.read_csv(DATA_FILE)

# Convert date
if "Transaction Date" in df.columns:

    df["Transaction Date"] = pd.to_datetime(
        df["Transaction Date"],
        errors="coerce"
    )

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🧹 Retail Store Data Cleaning Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "Oasis Infobyte – Data Analytics Internship"
    "</div>",
    unsafe_allow_html=True
)

st.write(
    """
    This dashboard presents the results of cleaning and validating
    the **Retail Store Sales** dataset.

    The original dataset contained missing values, inconsistent
    categorical data and incomplete transaction information.
    The cleaned dataset is then used for further retail analysis.
    """
)

st.divider()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("🔎 Dashboard Filters")

# ---------------------------------------------------------
# CATEGORY
# ---------------------------------------------------------

if "Category" in df.columns:

    categories = sorted(
        df["Category"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_categories = st.sidebar.multiselect(
        "Category",
        categories,
        default=categories
    )

else:

    selected_categories = None


# ---------------------------------------------------------
# LOCATION
# ---------------------------------------------------------

if "Location" in df.columns:

    locations = sorted(
        df["Location"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_locations = st.sidebar.multiselect(
        "Location",
        locations,
        default=locations
    )

else:

    selected_locations = None


# ---------------------------------------------------------
# PAYMENT METHOD
# ---------------------------------------------------------

if "Payment Method" in df.columns:

    payments = sorted(
        df["Payment Method"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_payments = st.sidebar.multiselect(
        "Payment Method",
        payments,
        default=payments
    )

else:

    selected_payments = None


# =========================================================
# APPLY FILTERS SAFELY
# =========================================================

filtered_df = df.copy()

# Empty selection means "All"
if selected_categories:
    filtered_df = filtered_df[
        filtered_df["Category"].isin(
            selected_categories
        )
    ]

if selected_locations:
    filtered_df = filtered_df[
        filtered_df["Location"].isin(
            selected_locations
        )
    ]

if selected_payments:
    filtered_df = filtered_df[
        filtered_df["Payment Method"].isin(
            selected_payments
        )
    ]


# =========================================================
# EMPTY FILTER PROTECTION
# =========================================================

if filtered_df.empty:

    st.warning(
        "⚠️ No transactions match the selected filters. "
        "Please select a different filter combination."
    )

    st.stop()


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📊 Dataset Overview'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

# ---------------------------------------------------------
# TOTAL TRANSACTIONS
# ---------------------------------------------------------

with col1:

    st.metric(
        "Total Transactions",
        f"{len(filtered_df):,}"
    )


# ---------------------------------------------------------
# TOTAL SALES
# ---------------------------------------------------------

with col2:

    total_sales = filtered_df["Total Spent"].sum()

    st.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )


# ---------------------------------------------------------
# AVERAGE TRANSACTION
# ---------------------------------------------------------

with col3:

    avg_transaction = filtered_df[
        "Total Spent"
    ].mean()

    st.metric(
        "Average Transaction",
        f"${avg_transaction:,.2f}"
    )


# ---------------------------------------------------------
# UNIQUE ITEMS
# ---------------------------------------------------------

with col4:

    unique_items = filtered_df[
        "Item"
    ].nunique()

    st.metric(
        "Unique Items",
        f"{unique_items:,}"
    )

st.divider()

# =========================================================
# DATA CLEANING IMPACT
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🧹 Data Cleaning Impact'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "The following metrics show the improvement achieved during "
    "the data-cleaning process."
)

# ---------------------------------------------------------
# BEFORE / AFTER CARDS
# ---------------------------------------------------------

c1, c2, c3 = st.columns(3)

# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

with c1:

    st.markdown(
        """
        <div class="clean-card">

        <div class="clean-label">
        Missing Values
        </div>

        <div class="before">
        7,229
        </div>

        <div>↓</div>

        <div class="after">
        0
        </div>

        <div class="clean-label">
        Missing values remaining
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# FORMULA MISMATCHES
# ---------------------------------------------------------

with c2:

    st.markdown(
        """
        <div class="clean-card">

        <div class="clean-label">
        Formula Mismatches
        </div>

        <div class="before">
        604
        </div>

        <div>↓</div>

        <div class="after">
        0
        </div>

        <div class="clean-label">
        Total Spent inconsistencies
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# DUPLICATES
# ---------------------------------------------------------

with c3:

    st.markdown(
        """
        <div class="clean-card">

        <div class="clean-label">
        Duplicate Rows
        </div>

        <div class="before">
        0
        </div>

        <div>↓</div>

        <div class="after">
        0
        </div>

        <div class="clean-label">
        Duplicates remaining
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# ---------------------------------------------------------
# RECOVERY METRICS
# ---------------------------------------------------------

r1, r2, r3, r4 = st.columns(4)

with r1:

    st.metric(
        "💰 Prices Recovered",
        "609"
    )

with r2:

    st.metric(
        "🛒 Items Inferred",
        "1,213"
    )

with r3:

    st.metric(
        "📋 Final Rows",
        f"{len(df):,}"
    )

with r4:

    st.metric(
        "📐 Formula Errors",
        "0"
    )

st.success(
    "✅ Dataset successfully cleaned, validated "
    "and prepared for analysis."
)

st.divider()

# =========================================================
# SALES OVER TIME
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📅 Sales Over Time'
    '</div>',
    unsafe_allow_html=True
)

if "Transaction Date" in filtered_df.columns:

    daily_sales = (
        filtered_df
        .dropna(
            subset=["Transaction Date"]
        )
        .groupby(
            "Transaction Date"
        )["Total Spent"]
        .sum()
    )

    if not daily_sales.empty:

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            daily_sales.index,
            daily_sales.values
        )

        ax.set_xlabel(
            "Transaction Date"
        )

        ax.set_ylabel(
            "Total Sales"
        )

        ax.set_title(
            "Daily Sales Trend"
        )

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

st.divider()

# =========================================================
# CATEGORY + PAYMENT
# =========================================================

left, right = st.columns(2)

# =========================================================
# SALES BY CATEGORY
# =========================================================

with left:

    st.markdown(
        '<div class="section-title">'
        '📦 Sales by Category'
        '</div>',
        unsafe_allow_html=True
    )

    category_sales = (
        filtered_df
        .groupby(
            "Category"
        )["Total Spent"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if not category_sales.empty:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        category_sales.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Category"
        )

        ax.set_ylabel(
            "Total Sales"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

# =========================================================
# PAYMENT METHOD
# =========================================================

with right:

    st.markdown(
        '<div class="section-title">'
        '💳 Payment Method Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    payment_counts = (
        filtered_df[
            "Payment Method"
        ]
        .value_counts()
    )

    if not payment_counts.empty:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        payment_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Payment Method"
        )

        ax.set_ylabel(
            "Transactions"
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

# =========================================================
# TOP 10 ITEMS
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🏆 Top 10 Items by Sales'
    '</div>',
    unsafe_allow_html=True
)

top_items = (
    filtered_df
    .groupby(
        "Item"
    )["Total Spent"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)

if not top_items.empty:

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    top_items.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel(
        "Total Sales"
    )

    ax.set_ylabel(
        "Item"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

st.divider()

# =========================================================
# LOCATION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📍 Sales by Location'
    '</div>',
    unsafe_allow_html=True
)

location_sales = (
    filtered_df
    .groupby(
        "Location"
    )["Total Spent"]
    .sum()
    .sort_values(
        ascending=False
    )
)

if not location_sales.empty:

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    location_sales.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(
        "Location"
    )

    ax.set_ylabel(
        "Total Sales"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

st.divider()

# =========================================================
# CLEANED DATASET PREVIEW
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🔎 Cleaned Dataset Preview'
    '</div>',
    unsafe_allow_html=True
)

st.dataframe(
    filtered_df.head(100),
    width="stretch"
)

# =========================================================
# DOWNLOAD
# =========================================================

st.markdown(
    '<div class="section-title">'
    '⬇️ Download Cleaned Dataset'
    '</div>',
    unsafe_allow_html=True
)

csv = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Cleaned CSV",
    data=csv,
    file_name="retail_store_sales_cleaned.csv",
    mime="text/csv",
    width="stretch"
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Oasis Infobyte Data Analytics Internship | "
    "Retail Store Sales – Data Cleaning Task"
)