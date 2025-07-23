import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
mlflow.sklearn.autolog()
df = pd.read_csv(r"C:\Users\nagarajuhome\OneDrive\fullstack\mlops\airbnb_listings.csv")
x = df.drop(columns=["PricePerNight", "ListingID"])
y = df["PricePerNight"]
categorical = ["City", "RoomType"]
numeric = ["Bedroom", "Bathroom", "GuestsCapacity", "HasWifi", "HasAC", "DistanceFromCityCenter"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough"
)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
with mlflow.start_run():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")
    print(f"Model Saved with MAE = {mae}, R² = {r2}")
#afer executing in the termial type mlflow ui
