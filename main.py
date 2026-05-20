from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# cargar modelo
model = joblib.load("modelo.pkl")

# cargar encoders
encoders = joblib.load("encoders.pkl")

@app.get("/")
def home():
    return {
        "status": "IA funcionando"
    }

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # transformar texto a números
    df["Brand"] = encoders["Brand"].transform([data["Brand"]])

    df["Model"] = encoders["Model"].transform([data["Model"]])

    df["Fuel_Type"] = encoders["Fuel_Type"].transform([data["Fuel_Type"]])

    df["Transmiss"] = encoders["Transmiss"].transform([data["Transmiss"]])

    # columnas exactas del entrenamiento
    X = df[
        [
            "Brand",
            "Model",
            "Model_Ye",
            "Kilometer",
            "Fuel_Type",
            "Transmiss"
        ]
    ]

    prediction = model.predict(X)

    price_mxn = int(prediction[0] * 18)

    price_mxn = max(
        50000,
        min(price_mxn, 3000000)
    )

    return {
        "price": price_mxn
    }