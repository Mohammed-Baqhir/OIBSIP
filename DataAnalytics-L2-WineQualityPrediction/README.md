# 🍷 Wine Quality Prediction — Oasis Infobyte Task 2

## 🚀 Live Demo

👉 [Open Wine Quality Prediction App](https://oibsip-wine-quality-prediction-l2.streamlit.app/)

## Objective
Train and compare **Random Forest, SGD, and SVC** classifiers to predict whether a wine is Good (quality ≥ 7) from physicochemical properties.

## Dataset
This project uses the uploaded **WineQT.csv** dataset.

- Rows: 1,143
- Physicochemical features: 11
- `quality`: target
- `Id`: identifier, excluded from modelling

## Target Engineering
- `quality >= 7` → Good Wine (1)
- `quality < 7` → Lower Quality Wine (0)

Binary classification is used because the original quality classes are strongly imbalanced and the deployment goal is to identify higher-quality wines.

## Models
1. Random Forest Classifier
2. Stochastic Gradient Descent (SGD) Classifier
3. Support Vector Classifier (SVC)

All models use class balancing, and the train/test split is stratified.

## Evaluation
- Accuracy
- Classification report
- Confusion matrix
- Random Forest feature importance
- Model comparison table

## Streamlit App
The app allows users to enter wine physicochemical properties and select a classifier to predict wine quality.

## Run locally

```powershell
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Project Structure

```text
Wine-Quality-Prediction/
├── data/
│   └── WineQT.csv
├── app.py
├── train.py
├── wine_quality_prediction.ipynb
├── models.pkl
├── metrics.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Streamlit Deployment
Add the deployed Streamlit URL here after deployment.
