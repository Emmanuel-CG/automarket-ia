import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# cargar dataset
df = pd.read_csv("entrenamiento_autos_25k.csv")

# columnas categóricas
categorical = [
    "Brand",
    "Model",
    "Fuel_Type",
    "Transmiss"
]

encoders = {}

for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# variables para entrenar
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

# precio objetivo
y = df["Price_USD"]

# dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# modelo IA
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

# entrenar
model.fit(X_train, y_train)

# guardar modelo
joblib.dump(model, "modelo.pkl")

# guardar encoders
joblib.dump(encoders, "encoders.pkl")

print("MODELO ENTRENADO")