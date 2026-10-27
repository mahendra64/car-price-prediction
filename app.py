import joblib
import pandas as pd

from train import FEATURES, MODEL

CHOICES = {
    "fuel": ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"],
    "transmission": ["Manual", "Automatic"],
    "condition": ["Poor", "Fair", "Good", "Excellent"],
}


def ask(label, cast=str):
    while True:
        value = input(f"{label}: ").strip()
        try:
            return cast(value)
        except ValueError:
            print("Please enter a valid value.")


def main():
    if not MODEL.exists():
        print("Model not found. First train it with: python train.py")
        return

    print("\nUsed Car Price Quotation")
    print("Enter the car details below.")

    car = {
        "brand": ask("Make / brand"),
        "model": ask("Model / variant"),
        "year": ask("Model year", int),
        "mileage_km": ask("Odometer reading (km)", int),
        "fuel": ask("Fuel type"),
        "transmission": ask("Transmission"),
        "owners": ask("Number of previous owners", int),
        "condition": ask("Condition (Poor/Fair/Good/Excellent)"),
        "city": ask("City"),
        "engine_cc": ask("Engine size (cc)", int),
    }

    model = joblib.load(MODEL)
    estimate = float(model.predict(pd.DataFrame([car], columns=FEATURES))[0])

    print(f"\nEstimated sale price: ₹{estimate:,.0f}")
    print("This is an estimate based on the available training data.")


if __name__ == "__main__":
    main()
