import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

users = pd.read_csv("data/users.csv").rename(columns={"code": "userCode"})
flights = pd.read_csv("data/flights.csv")
hotels = pd.read_csv("data/hotels.csv")

f = flights.groupby("userCode").agg(
    n_flights=("price", "count"),
    avg_flight_price=("price", "mean"),
    avg_distance=("distance", "mean"),
)
h = hotels.groupby("userCode").agg(
    n_hotels=("total", "count"),
    avg_stay_days=("days", "mean"),
    avg_hotel_total=("total", "mean"),
)

df = users.merge(f, on="userCode", how="left").merge(h, on="userCode", how="left").fillna(0)

NUM = ["age", "n_flights", "avg_flight_price", "avg_distance",
       "n_hotels", "avg_stay_days", "avg_hotel_total"]
X = pd.get_dummies(df[NUM + ["company"]])
y = df["gender"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("gender_classification")
best = None

candidates = [
    ("logistic_regression", LogisticRegression(max_iter=2000), {"max_iter": 2000}),
    ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42),
     {"n_estimators": 100}),
]

for name, model, params in candidates:
    with mlflow.start_run(run_name=name):
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc = float(accuracy_score(y_te, pred))
        f1 = float(f1_score(y_te, pred, average="macro"))
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(
            model, name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(name, "accuracy:", round(acc, 4), "f1_macro:", round(f1, 4))
        print(classification_report(y_te, pred, zero_division=0))
        if best is None or f1 > best[0]:
            best = (f1, name, model)

joblib.dump(best[2], "models/gender_model.pkl")
joblib.dump(list(X.columns), "models/gender_columns.pkl")
print("Saved best model:", best[1])