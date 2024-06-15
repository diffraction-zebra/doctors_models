import io

import pandas as pd
from fastapi import FastAPI

import models

app = FastAPI()


@app.on_event("startup")
def load():
    models.load_models()


@app.get("/models/predict")
def predict(start: str, end: str) -> str:
    index = pd.date_range(start, end, freq='W').to_period(freq="W")
    print('alive')
    return models.predict(index).to_json()


@app.post("/models/update")
def update(data: str) -> None:
    data = pd.read_json(io.StringIO(data))
    data.index = pd.PeriodIndex(data.index, freq='W')
    models.update(data)
