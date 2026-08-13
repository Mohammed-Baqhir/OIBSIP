import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Retail Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #000000;
        color: white;
    }

    .main {
        background-color: #000000;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Hero section */
    .hero {
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111111, #222222);
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #333333;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        color: white;
    }

    .hero p {
        margin: .35rem 0 0;
        color: #dddddd;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #000000;
        border: 1px solid #333333;
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(255,255,255,.05);
    }

    /* Metric labels */
    div[data-testid="stMetric"] label {
        color: white !important;
    }

    /* Metric numbers */
    div[data-testid="stMetricValue"] {
        color: white !important;
    }

    /* Metric delta */
    div[data-testid="stMetricDelta"] {
        color: white !important;
    }

    /* Normal text */
    p, span, label {
        color: white;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #050505;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("data/retail_sales_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Total Amount"])
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18–25", "26–35", "36–45", "46–55", "56–64"],
        include_lowest=True,
    )
    return df


df = load_data()

# ---------- Sidebar ----------
st.sidebar.title("🔎 Dashboard Filters")

categories = st.sidebar.multiselect(
    "Product Category",
    sorted(df["Product Category"].dropna().unique()),
    default=sorted(df["Product Category"].dropna().unique()),
)

genders = st.sidebar.multiselect(
    "Gender",
    sorted(df["Gender"].dropna().unique()),
    default=sorted(df["Gender"].dropna().unique()),
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered_df = df[
    df["Product Category"].isin(categories)
    & df["Gender"].isin(genders)
    & df["Date"].dt.date.between(start_date, end_date)
].copy()

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>📊 Retail Sales Analytics Dashboard</h1>
    <p>Explore revenue trends, customer behaviour, category performance and transaction patterns.</p>
</div>
""", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("No records match the selected filters. Please widen the filters.")
    st.stop()

# ---------- KPI cards ----------
total_sales = filtered_df["Total Amount"].sum()
total_orders = len(filtered_df)
avg_order = filtered_df["Total Amount"].mean()
total_quantity = filtered_df["Quantity"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Revenue", f"₹{total_sales:,.0f}")
c2.metric("🧾 Transactions", f"{total_orders:,}")
c3.metric("📈 Avg. Order Value", f"₹{avg_order:,.0f}")
c4.metric("📦 Units Sold", f"{total_quantity:,}")

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Trends", "👥 Customers", "📦 Products", "🔥 Correlations"]
)

with tab1:
    st.subheader("Monthly & Quarterly Sales Trends")

    monthly = (
        filtered_df.set_index("Date")["Total Amount"]
        .resample("MS")
        .sum()
    )
    quarterly = (
        filtered_df.set_index("Date")["Total Amount"]
        .resample("QS")
        .sum()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Monthly Revenue**")
        st.line_chart(monthly)

    with col2:
        st.markdown("**Quarterly Revenue**")
        st.line_chart(quarterly)

    st.info(
        "Observation: Monthly revenue shows short-term fluctuations, while "
        "quarterly aggregation makes broader business cycles easier to compare."
    )

with tab2:
    st.subheader("Customer Demographics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Revenue by Gender**")
        gender_sales = filtered_df.groupby("Gender")["Total Amount"].sum()
        st.bar_chart(gender_sales)

    with col2:
        st.markdown("**Customer Age Groups**")
        age_counts = filtered_df["Age Group"].value_counts().sort_index()
        st.bar_chart(age_counts)

    st.markdown("**Revenue by Age Group**")
    age_revenue = (
        filtered_df.groupby("Age Group", observed=False)["Total Amount"]
        .sum()
    )
    st.bar_chart(age_revenue)

    st.info(
        "Observation: The age-group view helps distinguish customer volume "
        "from revenue contribution, which can reveal high-value segments."
    )

with tab3:
    st.subheader("Product & Category Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Revenue by Product Category**")
        category_sales = (
            filtered_df.groupby("Product Category")["Total Amount"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(category_sales)

    with col2:
        st.markdown("**Quantity Sold by Category**")
        category_qty = (
            filtered_df.groupby("Product Category")["Quantity"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(category_qty)

    st.markdown("**Top 10 Highest-Value Transactions**")
    top10 = (
        filtered_df.nlargest(10, "Total Amount")
        [["Transaction ID", "Product Category", "Gender", "Age", "Quantity", "Total Amount"]]
        .reset_index(drop=True)
    )
    st.dataframe(top10, use_container_width=True)

    st.warning(
        "Dataset limitation: this dataset contains Product Category but no "
        "individual Product Name column. Therefore a true top-10 product "
        "ranking is not possible; the dashboard uses the top 10 transactions "
        "as a transparent substitute."
    )

with tab4:
    st.subheader("Correlation Heatmap")

    numeric_cols = [
        "Age", "Quantity", "Price per Unit", "Total Amount"
    ]
    corr = filtered_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.info(
        "Observation: Quantity and Total Amount are strongly connected because "
        "revenue is derived from units and unit price. Price per Unit can also "
        "help explain differences in transaction value."
    )

# ---------- Data preview and downloads ----------
st.markdown("---")
st.subheader("📄 Filtered Dataset")

st.dataframe(filtered_df.head(50), use_container_width=True)

download_df = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Filtered CSV",
    data=download_df,
    file_name="retail_sales_filtered.csv",
    mime="text/csv",
)

summary = (
    filtered_df.groupby("Product Category")["Total Amount"]
    .agg(["sum", "mean", "count"])
    .reset_index()
    .rename(
        columns={
            "sum": "Total Revenue",
            "mean": "Average Transaction",
            "count": "Transactions",
        }
    )
)
summary_csv = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "📊 Download Category Summary",
    data=summary_csv,
    file_name="category_sales_summary.csv",
    mime="text/csv",
)

st.caption("Retail Sales EDA • Python • Pandas • Matplotlib • Seaborn • Streamlit")
