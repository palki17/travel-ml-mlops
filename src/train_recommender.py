import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

hotels = pd.read_csv("data/hotels.csv")
hotels["date"] = pd.to_datetime(hotels["date"], format="%m/%d/%Y")
hotels = hotels.sort_values("date")


def user_hotel_matrix(df):
    return df.pivot_table(index="userCode", columns="name", values="travelCode",
                          aggfunc="count", fill_value=0)


# Evaluation: each user's last booking is the test, the rest is train
test = hotels.groupby("userCode").tail(1)
train = hotels.drop(test.index)

M = user_hotel_matrix(train)
names = list(M.columns)
sim = cosine_similarity(M.T.values)
scores = M.values @ sim

actual = test.set_index("userCode")["name"].reindex(M.index)
top3 = np.argsort(-scores, axis=1)[:, :3]
hit_cf = float(np.mean([actual.iloc[i] in [names[j] for j in top3[i]] for i in range(len(M))]))

pop3 = train["name"].value_counts().index[:3].tolist()
hit_pop = float(np.mean([a in pop3 for a in actual]))

print("Hotels:", len(names), "| Users evaluated:", len(M))
print("Hit@3 collaborative filtering:", round(hit_cf, 4))
print("Hit@3 popularity baseline    :", round(hit_pop, 4))

mlflow.set_experiment("hotel_recommendation")
with mlflow.start_run(run_name="item_based_cf"):
    mlflow.log_param("method", "item-based cosine similarity")
    mlflow.log_metric("hit_at_3", hit_cf)
    mlflow.log_metric("hit_at_3_popularity_baseline", hit_pop)

# Final model: fit on all data
M_all = user_hotel_matrix(hotels)
joblib.dump({
    "names": list(M_all.columns),
    "users": [int(u) for u in M_all.index],
    "matrix": M_all.values,
    "similarity": cosine_similarity(M_all.T.values),
    "popular": hotels["name"].value_counts().index.tolist(),
}, "models/recommender.pkl")
print("Saved models/recommender.pkl")