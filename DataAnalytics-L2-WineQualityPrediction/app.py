from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "models.pkl", "rb") as f:
    bundle = pickle.load(f)
with open(BASE_DIR / "metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

models = bundle["models"]
features = bundle["features"]
best_model = bundle["best_model"]

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Prediction")
st.caption("Oasis Infobyte — Data Analytics Task 2")
st.write(
    "Predict whether a wine is **Good (quality ≥ 7)** using "
    "its physicochemical properties."
)

st.sidebar.header("Wine Properties")

defaults = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

ranges = {
    "fixed acidity": (3.0, 16.0, 0.1),
    "volatile acidity": (0.0, 2.0, 0.01),
    "citric acid": (0.0, 1.0, 0.01),
    "residual sugar": (0.5, 16.0, 0.1),
    "chlorides": (0.0, 0.7, 0.001),
    "free sulfur dioxide": (0.0, 80.0, 1.0),
    "total sulfur dioxide": (0.0, 300.0, 1.0),
    "density": (0.98, 1.04, 0.0001),
    "pH": (2.5, 4.5, 0.01),
    "sulphates": (0.2, 2.0, 0.01),
    "alcohol": (8.0, 15.0, 0.1)
}

values = {}
for feature in features:
    lo, hi, step = ranges[feature]
    values[feature] = st.sidebar.number_input(
        feature.title(),
        min_value=float(lo),
        max_value=float(hi),
        value=float(defaults[feature]),
        step=float(step)
    )

model_choice = st.selectbox(
    "Select Classification Model",
    list(models.keys()),
    index=list(models.keys()).index(best_model)
)

if st.button("🍷 Predict Wine Quality", type="primary", use_container_width=True):
    input_data = pd.DataFrame([values], columns=features)
    prediction = int(models[model_choice].predict(input_data)[0])

    if prediction == 1:
        st.success("🍷 GOOD WINE — predicted quality is 7 or higher")
    else:
        st.info("Wine predicted as LOWER QUALITY — predicted quality is below 7")

    st.caption(f"Model used: {model_choice}")

st.divider()

st.subheader("📊 Model Comparison")
comparison = pd.DataFrame({
    name: {"Accuracy": result["accuracy"]}
    for name, result in metrics["results"].items()
}).T
comparison.index.name = "Model"
st.dataframe(
    comparison.style.format({"Accuracy": "{:.2%}"}),
    use_container_width=True
)

st.subheader("🌟 Random Forest Feature Importance")
importance = pd.Series(metrics["feature_importance"]).sort_values(ascending=False)
st.bar_chart(importance.head(10))

st.subheader("🎯 Target Definition")
st.write(
    "Quality scores are converted into a binary classification target: "
    "**quality ≥ 7 = Good Wine**, while **quality < 7 = Lower Quality**. "
    "The `Id` column is excluded because it is only an identifier."
)
