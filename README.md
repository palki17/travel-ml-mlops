# Voyage Analytics - ML API

Backend for the travel ML project: flight price prediction, gender classification and hotel recommendation.

## Run the API locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- API runs at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Run it from the project root (the `models/` folder must be next to `api/`)
- CORS is open (`*`) for development

## Endpoints

### GET `/`
Health check. Returns `{"status": "ok"}`.

### POST `/predict-price`

Request:
```json
{
  "from_city": "Recife (PE)",
  "to": "Florianopolis (SC)",
  "flightType": "firstClass",
  "agency": "FlyingDrops"
}
```
Response:
```json
{ "predicted_price": 1434.38 }
```

Valid values (use these for dropdowns):
- `from_city` and `to` (pick two different cities): Aracaju (SE), Brasilia (DF), Campo Grande (MS), Florianopolis (SC), Natal (RN), Recife (PE), Rio de Janeiro (RJ), Salvador (BH), Sao Paulo (SP)
- `flightType`: economic, premium, firstClass
- `agency`: CloudFy, FlyingDrops, Rainbow

### POST `/predict-gender`

Request:
```json
{
  "age": 30,
  "company": "4You",
  "n_flights": 200,
  "avg_flight_price": 950,
  "avg_distance": 500,
  "n_hotels": 30,
  "avg_stay_days": 3,
  "avg_hotel_total": 700
}
```
Response:
```json
{
  "predicted_gender": "none",
  "probabilities": { "female": 0.3, "male": 0.29, "none": 0.41 },
  "note": "Demo only: model accuracy is ~34%, close to random guessing."
}
```

Valid `company` values: 4You, Acme Factory, Monsters CYA, Umbrella LTDA, Wonka Company.
Show the `note` field in the UI. The model is not reliable (see Results).

### POST `/recommend-hotel`

Request:
```json
{ "user_code": 5, "n": 3 }
```
Response:
```json
{
  "user_code": 5,
  "method": "collaborative_filtering",
  "recommended_hotels": ["Hotel K", "Hotel AF", "Hotel CB"]
}
```
Unknown users get `"method": "popularity_fallback"` with the most popular hotels.

## Results

| Model | Result |
|---|---|
| Flight price (Random Forest) | R2 = 1.0. Price is fully determined by from, to, flightType and agency in this synthetic dataset. Linear Regression baseline: R2 = 0.64 |
| Gender classification | Accuracy ~34% (random guessing for 3 classes is 33%). No signal between travel behaviour and gender in this data |
| Hotel recommendation | Hit@3 = 37.3% vs popularity baseline 35.3% (only 9 hotels) |

## Project structure

```
api/main.py            FastAPI app
src/train_*.py         training scripts (log to MLflow)
notebooks/01_eda.ipynb exploratory analysis
models/                trained models (.pkl)
data/                  CSV files (not in git)
```

## Retrain

```bash
python src/train_price.py
python src/train_gender.py
python src/train_recommender.py
mlflow ui
```