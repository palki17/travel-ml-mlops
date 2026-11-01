import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Voyage Analytics API")

price_model = joblib.load("models/price_model.pkl")
price_columns = joblib.load("models/price_columns.pkl")
gender_model = joblib.load("models/gender_model.pkl")
gender_columns = joblib.load("models/gender_columns.pkl")


class FlightInput(BaseModel):
    from_city: str
    to: str
    flightType: str
    agency: str


class GenderInput(BaseModel):
    age: int
    company: str
    n_flights: int
    avg_flight_price: float
    avg_distance: float
    n_hotels: int
    avg_stay_days: float
    avg_hotel_total: float


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


@app.post("/predict-gender")
def predict_gender(d: GenderInput):
    row = pd.DataFrame([d.model_dump()])
    row = pd.get_dummies(row).reindex(columns=gender_columns, fill_value=0)
    probs = gender_model.predict_proba(row)[0]
    classes = gender_model.classes_
    return {
        "predicted_gender": str(classes[probs.argmax()]),
        "probabilities": {str(c): round(float(p), 3) for c, p in zip(classes, probs)},
        "note": "Demo only: model accuracy is ~34%, close to random guessing.",
    }