import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
)


# =========================================================
# DATA PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "marketing_campaign.csv"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    # Check whether dataset exists
    if not DATA_PATH.exists():
        st.error(
            f"Dataset not found.\n\n"
            f"Expected file:\n`{DATA_PATH}`\n\n"
            f"Make sure `marketing_campaign.csv` is inside the "
            f"`data` folder in your GitHub repository."
        )
        st.stop()

    df = pd.read_csv(
        DATA_PATH,
        sep="\t"
    )

    df.columns = df.columns.str.strip()

    original_rows = len(df)

    duplicate_rows = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    # -----------------------------------------------------
    # Missing Income
    # -----------------------------------------------------

    missing_income = int(
        df["Income"].isna().sum()
    )

    df["Income"] = df["Income"].fillna(
        df["Income"].median()
    )

    # -----------------------------------------------------
    # Customer Date
    # -----------------------------------------------------

    if "Dt_Customer" in df.columns:

        df["Dt_Customer"] = pd.to_datetime(
            df["Dt_Customer"],
            dayfirst=True,
            errors="coerce"
        )

    # -----------------------------------------------------
    # Data Quality Checks
    # -----------------------------------------------------

    invalid_birth = int(
        (~df["Year_Birth"].between(1900, 2000)).sum()
    )

    invalid_income = int(
        (df["Income"] < 0).sum()
    )

    df = df[
        df["Year_Birth"].between(
            1900,
            2000
        )
    ].copy()

    df = df[
        df["Income"] >= 0
    ].copy()

    # -----------------------------------------------------
    # RFM FEATURES
    # -----------------------------------------------------

    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]

    purchase_cols = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]

    # Monetary = total historical spending
    df["Monetary"] = df[
        spend_cols
    ].sum(axis=1)

    # Frequency = total purchases
    df["Frequency"] = df[
        purchase_cols
    ].sum(axis=1)

    # Average purchase value
    df["AveragePurchaseValue"] = np.where(
        df["Frequency"] > 0,
        df["Monetary"] / df["Frequency"],
        0
    )

    # -----------------------------------------------------
    # CLV PROXY
    # -----------------------------------------------------

    # Exact CLV cannot be calculated because this dataset
    # does not provide customer lifespan, profit margin or
    # complete transaction history.

    df["CLV_Proxy"] = df["Monetary"]

    # -----------------------------------------------------
    # DATA QUALITY SUMMARY
    # -----------------------------------------------------

    quality = {
        "original_rows": original_rows,
        "duplicate_rows": duplicate_rows,
        "missing_income_filled": missing_income,
        "invalid_birth_removed": invalid_birth,
        "invalid_income_removed": invalid_income,
        "final_rows": len(df),
        "features": len(df.columns),
    }

    return df, quality


# =========================================================
# MODEL FEATURES
# =========================================================

def make_model_features(df):

    x = df[
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    ].copy()

    # Log transformation reduces skewness
    x["Frequency"] = np.log1p(
        x["Frequency"]
    )

    x["Monetary"] = np.log1p(
        x["Monetary"]
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(x)

    return X, scaler


# =========================================================
# K-MEANS MODEL
# =========================================================

def run_kmeans(df, k):

    X, scaler = make_model_features(df)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    result = df.copy()

    result["Cluster"] = labels

    score = silhouette_score(
        X,
        labels
    )

    return (
        result,
        scaler,
        model,
        X,
        score
    )


# =========================================================
# CLUSTER LABELING
# =========================================================

def cluster_label(row, medians):

    r = row["Recency"]
    f = row["Frequency"]
    m = row["Monetary"]

    if (
        r <= medians["Recency"]
        and f >= medians["Frequency"]
        and m >= medians["Monetary"]
    ):
        return "Loyal High-Value"

    if (
        r <= medians["Recency"]
        and f < medians["Frequency"]
        and m >= medians["Monetary"]
    ):
        return "Recent Big Spenders"

    if (
        r > medians["Recency"]
        and f >= medians["Frequency"]
        and m >= medians["Monetary"]
    ):
        return "At-Risk High-Value"

    if (
        r > medians["Recency"]
        and f < medians["Frequency"]
    ):
        return "Low-Engagement"

    return "Regular Customers"


# =========================================================
# MARKETING ACTION
# =========================================================

def marketing_action(segment):

    actions = {

        "Loyal High-Value":
            "VIP rewards, early access, premium bundles and loyalty benefits.",

        "Recent Big Spenders":
            "Cross-sell and upsell complementary products while engagement is high.",

        "At-Risk High-Value":
            "Win-back campaigns, personalised offers and reminders to reduce churn risk.",

        "Low-Engagement":
            "Low-cost reactivation campaigns, discovery content and limited-time offers.",

        "Regular Customers":
            "Encourage repeat purchases with bundles, loyalty points and personalised recommendations.",
    }

    return actions.get(
        segment,
        "Use a personalised campaign based on the segment profile."
    )


# =========================================================
# LOAD DATA
# =========================================================

df, quality = load_data()


# =========================================================
# HEADER
# =========================================================

st.title(
    "🎯 Customer Segmentation Analysis"
)

st.markdown(
    "### RFM-based K-Means clustering for targeted e-commerce marketing"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ Controls"
)

k_selected = st.sidebar.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
)


if "run_clustering" not in st.session_state:

    st.session_state.run_clustering = False


if "active_k" not in st.session_state:

    st.session_state.active_k = 4


if st.sidebar.button(
    "🚀 Run Clustering",
    width="stretch"
):

    st.session_state.active_k = k_selected

    st.session_state.run_clustering = True


if not st.session_state.run_clustering:

    st.info(
        "Select K in the sidebar and click "
        "**Run Clustering** to generate the customer segments."
    )

    st.stop()


# =========================================================
# RUN CLUSTERING
# =========================================================

k = st.session_state.active_k

(
    clustered,
    scaler,
    model,
    X,
    silhouette
) = run_kmeans(
    df,
    k
)


# =========================================================
# CLUSTER PROFILE
# =========================================================

raw_profile = (
    clustered
    .groupby("Cluster")
    .agg(
        Customers=("ID", "count"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        AveragePurchaseValue=(
            "AveragePurchaseValue",
            "mean"
        ),
        CLV_Proxy=(
            "CLV_Proxy",
            "mean"
        ),
    )
    .reset_index()
)


medians = {

    "Recency":
        clustered["Recency"].median(),

    "Frequency":
        clustered["Frequency"].median(),

    "Monetary":
        clustered["Monetary"].median(),
}


name_map = {

    int(row["Cluster"]):
        cluster_label(
            row,
            medians
        )

    for _, row in raw_profile.iterrows()
}


clustered["Segment"] = clustered[
    "Cluster"
].map(name_map)


profile = raw_profile.copy()

profile["Segment"] = profile[
    "Cluster"
].map(name_map)

profile["Marketing_Action"] = profile[
    "Segment"
].map(marketing_action)


# =========================================================
# KPI ROW
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "Customers",
    f"{len(clustered):,}"
)


c2.metric(
    "Avg Purchase Value",
    f"${clustered['AveragePurchaseValue'].mean():,.2f}"
)


c3.metric(
    "Avg Frequency",
    f"{clustered['Frequency'].mean():.2f}"
)


c4.metric(
    "Avg Monetary",
    f"${clustered['Monetary'].mean():,.2f}"
)


c5.metric(
    "Silhouette Score",
    f"{silhouette:.3f}"
)


# =========================================================
# TABS
# =========================================================

tabs = st.tabs(
    [
        "🔍 Dataset & RFM",
        "📐 Elbow Method",
        "🔵 Cluster Analysis",
        "🎯 Cluster Profiles",
        "🔮 Predict Segment",
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tabs[0]:

    st.subheader(
        "Dataset Inspection & Data Quality"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Original Rows",
        f"{quality['original_rows']:,}"
    )

    b.metric(
        "Final Rows",
        f"{quality['final_rows']:,}"
    )

    c.metric(
        "Features",
        quality["features"]
    )

    d.metric(
        "Duplicate Rows Removed",
        quality["duplicate_rows"]
    )


    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Missing Income Filled",
        quality["missing_income_filled"]
    )

    q2.metric(
        "Invalid Birth Years Removed",
        quality["invalid_birth_removed"]
    )

    q3.metric(
        "Invalid Income Rows Removed",
        quality["invalid_income_removed"]
    )


    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        width="stretch"
    )


    st.subheader(
        "RFM & Purchase Descriptive Statistics"
    )

    stats = clustered[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "AveragePurchaseValue",
            "CLV_Proxy",
        ]
    ].describe().T.round(2)


    st.dataframe(
        stats,
        width="stretch"
    )


    st.info(
        "RFM features used for clustering: Recency, Frequency and Monetary. "
        "Frequency is the total number of web, catalog and store purchases. "
        "Monetary is total historical product spend."
    )


    st.warning(
        "CLV note: this dataset is customer-level and does not provide complete "
        "transaction history, customer lifespan or profit margin. Therefore "
        "CLV Proxy equals historical Monetary spend; it is not presented "
        "as an exact lifetime-value model."
    )


# =========================================================
# TAB 2
# =========================================================

with tabs[1]:

    st.subheader(
        "📐 Elbow Method — Optimal K"
    )


    X_elbow, _ = make_model_features(df)

    k_values = list(
        range(2, 9)
    )

    inertias = []

    sil_scores = []


    for candidate in k_values:

        km = KMeans(
            n_clusters=candidate,
            random_state=42,
            n_init=10
        )

        labels = km.fit_predict(
            X_elbow
        )

        inertias.append(
            km.inertia_
        )

        sil_scores.append(
            silhouette_score(
                X_elbow,
                labels
            )
        )


    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )


    ax.plot(
        k_values,
        inertias,
        marker="o"
    )


    ax.axvline(
        k,
        linestyle="--",
        alpha=0.7,
        label=f"Selected K = {k}"
    )


    ax.set_xlabel(
        "Number of Clusters (K)"
    )

    ax.set_ylabel(
        "Inertia"
    )

    ax.set_title(
        "Elbow Method"
    )


    ax.legend()

    ax.grid(
        alpha=0.25
    )


    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


    elbow_table = pd.DataFrame({

        "K": k_values,

        "Inertia": np.round(
            inertias,
            2
        ),

        "Silhouette Score": np.round(
            sil_scores,
            3
        ),

    })


    st.dataframe(
        elbow_table,
        width="stretch"
    )


    st.success(
        f"Current clustering uses K = {k}. "
        f"The silhouette score for this selection is {silhouette:.3f}. "
        "The final K should consider both the elbow point "
        "and business interpretability."
    )


# =========================================================
# TAB 3
# =========================================================

with tabs[2]:

    st.subheader(
        "🔵 Customer Cluster Visualisations"
    )


    left, right = st.columns(2)


    with left:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        sns.scatterplot(
            data=clustered,
            x="Frequency",
            y="Monetary",
            hue="Segment",
            palette="tab10",
            s=55,
            alpha=0.75,
            ax=ax,
        )


        ax.set_title(
            "Frequency vs Monetary"
        )

        ax.set_xlabel(
            "Purchase Frequency"
        )

        ax.set_ylabel(
            "Total Spend"
        )


        ax.legend(
            title="Segment",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )


        st.pyplot(
            fig,
            width="stretch"
        )

        plt.close(fig)


    with right:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        sns.scatterplot(
            data=clustered,
            x="Recency",
            y="Monetary",
            hue="Segment",
            palette="tab10",
            s=55,
            alpha=0.75,
            ax=ax,
        )


        ax.set_title(
            "Recency vs Monetary"
        )

        ax.set_xlabel(
            "Recency (days)"
        )

        ax.set_ylabel(
            "Total Spend"
        )


        ax.legend(
            title="Segment",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )


        st.pyplot(
            fig,
            width="stretch"
        )

        plt.close(fig)


    st.subheader(
        "Customers per Cluster"
    )


    counts = (
        clustered
        .groupby(
            [
                "Cluster",
                "Segment"
            ]
        )
        .size()
        .reset_index(
            name="Customers"
        )
        .sort_values(
            "Customers",
            ascending=False
        )
    )


    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )


    sns.barplot(
        data=counts,
        x="Segment",
        y="Customers",
        ax=ax
    )


    ax.set_title(
        "Number of Customers by Segment"
    )

    ax.set_xlabel(
        "Customer Segment"
    )

    ax.set_ylabel(
        "Customers"
    )


    ax.tick_params(
        axis="x",
        rotation=25
    )


    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# =========================================================
# TAB 4
# =========================================================

with tabs[3]:

    st.subheader(
        "🎯 Cluster Profiles"
    )


    display = profile[
        [
            "Cluster",
            "Segment",
            "Customers",
            "Recency",
            "Frequency",
            "Monetary",
            "AveragePurchaseValue",
            "CLV_Proxy",
            "Marketing_Action",
        ]
    ].copy()


    st.dataframe(
        display.round(2),
        width="stretch"
    )


    for _, row in profile.sort_values(
        "Customers",
        ascending=False
    ).iterrows():

        st.markdown(

            f"#### Cluster {int(row['Cluster'])} — "
            f"{row['Segment']}\n"

            f"- **Customers:** "
            f"{int(row['Customers']):,}\n"

            f"- **Average Recency:** "
            f"{row['Recency']:.1f} days\n"

            f"- **Average Frequency:** "
            f"{row['Frequency']:.1f} purchases\n"

            f"- **Average Monetary:** "
            f"${row['Monetary']:,.2f}\n"

            f"- **Marketing Action:** "
            f"{row['Marketing_Action']}"
        )


# =========================================================
# TAB 5
# =========================================================

with tabs[4]:

    st.subheader(
        "🔮 Predict Customer Segment"
    )


    st.write(
        "Enter a customer's RFM behaviour. The dashboard "
        "standardises the values with the same transformation "
        "used for K-Means and assigns the nearest learned cluster."
    )


    p1, p2, p3 = st.columns(3)


    with p1:

        recency_input = st.number_input(
            "Recency (days)",
            min_value=0.0,
            value=30.0,
            step=1.0
        )


    with p2:

        frequency_input = st.number_input(
            "Purchase Frequency",
            min_value=0.0,
            value=10.0,
            step=1.0
        )


    with p3:

        monetary_input = st.number_input(
            "Monetary Value ($)",
            min_value=0.0,
            value=500.0,
            step=50.0
        )


    if st.button(
        "🎯 Predict Segment",
        width="stretch"
    ):

        user = pd.DataFrame(
            [
                {
                    "Recency": recency_input,
                    "Frequency": frequency_input,
                    "Monetary": monetary_input,
                }
            ]
        )


        user["Frequency"] = np.log1p(
            user["Frequency"]
        )

        user["Monetary"] = np.log1p(
            user["Monetary"]
        )


        user_scaled = scaler.transform(
            user[
                [
                    "Recency",
                    "Frequency",
                    "Monetary",
                ]
            ]
        )


        predicted_cluster = int(
            model.predict(
                user_scaled
            )[0]
        )


        predicted_segment = name_map[
            predicted_cluster
        ]


        st.success(
            f"Predicted Segment: **Cluster "
            f"{predicted_cluster} — "
            f"{predicted_segment}**"
        )


        st.info(
            f"Recommended action: "
            f"{marketing_action(predicted_segment)}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(
    "Oasis Infobyte — Data Analytics Task 2 | "
    "Customer Segmentation Analysis"
)