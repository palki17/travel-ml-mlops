import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Voyage Analytics API")

price_model = joblib.load("models/price_model.pkl")
price_columns = joblib.load("models/price_columns.pkl")


class FlightInput(BaseModel):
    from_city: str
    to: str
    flightType: str
    agency: str


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/predict-price")
def predict_price(d: FlightInput):
    row = pd.DataFrame([{
        "from": d.from_city,
        "to": d.to,
        "flightType": d.flightType,
        "agency": d.agency,
    }])
    row = pd.get_dummies(row).reindex(columns=price_columns, fill_value=0)
    return {"predicted_price": round(float(price_model.predict(row)[0]), 2)}