from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "IA funcionando"}

@app.post("/predict")
def predict(data: dict):

    return {
        "price": 250000
    }