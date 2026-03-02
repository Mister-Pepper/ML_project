import duckdb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.sparse import hstack

#load from motherduck
con = duckdb.connect("md:raw_reddit_listings")

df = con.execute("""
SELECT
  p.item_key AS clean_key,
  p.price,
  r.num_comments,
  r.vote_score,
  r.created_utc
FROM parsed_items p
JOIN raw_posts r
  ON p.post_id = r.post_id
WHERE p.price IS NOT NULL
  AND p.is_bundle = FALSE
  AND p.is_sold = FALSE
  AND LENGTH(p.item_key) > 3
""").fetchdf()

print("Loaded rows:", len(df))

# easiest for now
df = df.dropna()

# remove extreme outliers (top 1%) - misread> but kohakus yk?
df = df[df["price"] < df["price"].quantile(0.99)]

# log features (helps model stability)
df["log_comments"] = np.log1p(df["num_comments"])
df["log_score"] = np.log1p(df["vote_score"])

vectorizer = TfidfVectorizer(max_features=500)
X_text = vectorizer.fit_transform(df["clean_key"])

# numeric features
X_num = df[["log_comments", "log_score"]].values

# combine
X = hstack([X_text, X_num])
y = df["price"].values

# break up data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RR

model1 = Ridge()
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

mae1 = mean_absolute_error(y_test, pred1)
rmse1 = np.sqrt(mean_squared_error(y_test, pred1))

print("\nModel 1 (Ridge)")
print("MAE:", round(mae1, 2))
print("RMSE:", round(rmse1, 2))

# random forest

model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

mae2 = mean_absolute_error(y_test, pred2)
rmse2 = np.sqrt(mean_squared_error(y_test, pred2))

print("\nModel 2 (Random Forest)")
print("MAE:", round(mae2, 2))
print("RMSE:", round(rmse2, 2))

#model selection
best_model = model2 if mae2 < mae1 else model1

print("\nBest model:", "Random Forest" if best_model == model2 else "Ridge")

def predict_price(item_name):
    text_vec = vectorizer.transform([item_name.lower()])
    num_vec = np.array([[0, 0]])  # no metadata for now

    X_input = hstack([text_vec, num_vec])

    pred = best_model.predict(X_input)[0]
    return round(pred, 2)

print("\n--- SAMPLE PREDICTIONS ---")


#TESTINGGG
test_items = [
    "neo65",
    "bauer r2",
    "kohaku",
    "gmk darling",
    "keychron q1"
]

for item in test_items:
    print(f"{item}: ${predict_price(item)}")
