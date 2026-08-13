from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WineQT.csv"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
print("Loading dataset...")
print(f"Dataset loaded successfully: {df.shape}")

df["good_wine"] = (df["quality"] >= 7).astype(int)
X = df.drop(columns=["quality", "good_wine", "Id"])
y = df["good_wine"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1
    ),
    "SGD": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SGDClassifier(
            loss="log_loss", max_iter=2000,
            random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ]),
    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", class_weight="balanced"))
    ])
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }
    print(f"\n{name}")
    print(f"Accuracy: {results[name]['accuracy']:.4f}")
    print(classification_report(y_test, pred, zero_division=0))
    print(confusion_matrix(y_test, pred))

best_model = max(results, key=lambda n: results[n]["accuracy"])
importance = pd.Series(
    models["Random Forest"].feature_importances_, index=X.columns
).sort_values(ascending=False).to_dict()

with open(BASE_DIR / "models.pkl", "wb") as f:
    pickle.dump({
        "models": models,
        "features": list(X.columns),
        "best_model": best_model
    }, f)

with open(BASE_DIR / "metrics.pkl", "wb") as f:
    pickle.dump({
        "results": results,
        "feature_importance": importance,
        "best_model": best_model
    }, f)

print("\nSaved successfully: models.pkl and metrics.pkl")
print("Best model:", best_model)
