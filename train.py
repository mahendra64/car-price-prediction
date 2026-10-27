from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cars.csv"
MODEL = ROOT / "car_price_model.joblib"

CATEGORICAL = [
    "brand",
    "model",
    "fuel",
    "transmission",
    "condition",
    "city",
]
NUMERIC = ["year", "mileage_km", "owners", "engine_cc"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "sale_price"


def train(data_path=DATA):
    df = pd.read_csv(data_path)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing)}"
        )

    df = df[required].dropna()

    if len(df) < 10:
        raise ValueError("Add at least 10 complete car sale records before training.")

    preprocess = ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL,
            ),
            ("numbers", "passthrough", NUMERIC),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    joblib.dump(pipeline, MODEL)

    print(f"Saved model: {MODEL.name}")
    print(f"Validation MAE: ₹{mean_absolute_error(y_test, predictions):,.0f}")
    print(f"Validation R²: {r2_score(y_test, predictions):.2f}")
    print("Metrics from sample data are only a demonstration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the used-car price regression model"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA,
        help="CSV file containing historical car sales",
    )
    args = parser.parse_args()
    train(args.data)
