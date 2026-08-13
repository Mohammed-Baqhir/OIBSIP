from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


st.title("🏠 House Price Prediction")

st.write(
    "Predict house sale prices using a Linear Regression model "
    "trained on the Kaggle Ames Housing dataset."
)


# Check model files
if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():

    st.error(
        "Model files are missing. Please run `python train.py` first."
    )

    st.stop()


# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# Load feature information
with open(COLUMNS_PATH, "rb") as f:
    metadata = pickle.load(f)


st.subheader("Enter House Details")


col1, col2 = st.columns(2)


with col1:

    area = st.number_input(
        "Living Area (sq ft)",
        min_value=200,
        max_value=6000,
        value=1500,
        step=50
    )

    neighborhood = st.text_input(
        "Neighborhood",
        value="CollgCr"
    )

    rooms = st.number_input(
        "Rooms Above Grade",
        min_value=1,
        max_value=20,
        value=6,
        step=1
    )

    age = st.number_input(
        "House Age (years)",
        min_value=0,
        max_value=150,
        value=20,
        step=1
    )


with col2:

    quality = st.slider(
        "Overall Quality (1–10)",
        min_value=1,
        max_value=10,
        value=6
    )

    baths = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=6,
        value=2,
        step=1
    )

    garage = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    basement = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=800,
        step=50
    )


st.divider()


if st.button(
    "💰 Predict House Price",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame([
        {
            "GrLivArea": area,
            "TotRmsAbvGrd": rooms,
            "FullBath": baths,
            "GarageCars": garage,
            "TotalBsmtSF": basement,
            "OverallQual": quality,
            "HouseAge": age,
            "Neighborhood": neighborhood
        }
    ])


    prediction = float(
        model.predict(input_data)[0]
    )


    st.success(
        f"### Estimated Sale Price: ${prediction:,.0f}"
    )


    st.info(
        "This is a machine-learning estimate based on patterns "
        "learned from the training dataset. It is not a professional appraisal."
    )


st.divider()


st.subheader("Model Information")


info1, info2, info3 = st.columns(3)


info1.metric(
    "Algorithm",
    "Linear Regression"
)


info2.metric(
    "Train / Test",
    "80% / 20%"
)


info3.metric(
    "R² Score",
    "0.8263"
)