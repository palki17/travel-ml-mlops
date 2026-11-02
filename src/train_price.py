import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = ["from", "to", "flightType", "agency"]

flights = pd.read_csv("data/flights.csv")
X = pd.get_dummies(flights[FEATURES])
y = flights["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("flight_price")
best = None

candidates = [
    ("linear_regression", LinearRegression(), {}),
    ("random_forest", RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42), {"n_estimators": 50}),
]

for name, model, params in candidates:
    with mlflow.start_run(run_name=name):
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        r2 = float(r2_score(y_te, pred))
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])  
        print(name, "RMSE:", round(rmse, 3), "R2:", round(r2, 4))
        if best is None or rmse < best[0]:
            best = (rmse, name, model)

joblib.dump(best[2], "models/price_model.pkl")
joblib.dump(list(X.columns), "models/price_columns.pkl")
print("Saved best model:", best[1])