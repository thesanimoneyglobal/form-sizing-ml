from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pickle

app = FastAPI()


class EstimationItem(BaseModel):
    number_of_items: int
    number_of_scores: int
    form_type: Literal['10', '11', '12']
    branching_logic: bool
    complex_functionality: bool
    normative_scoring: bool
    form_mapping: bool
    web_service_setup: bool


model_path = 'models'

with open(f'{model_path}/model_Form Complexity 1.pkl', 'rb') as f:
    model_Form_Complexity_1 = pickle.load(f)

with open(f'{model_path}/model_Form Complexity 2.pkl', 'rb') as f:
    model_Form_Complexity_2 = pickle.load(f)

with open(f'{model_path}/model_Form Complexity 3.pkl', 'rb') as f:
    model_Form_Complexity_3 = pickle.load(f)

with open(f'{model_path}/model_Form Complexity 4.pkl', 'rb') as f:
    model_Form_Complexity_4 = pickle.load(f)


def preprocess_input(data: EstimationItem) -> np.ndarray:
    x = [
        data.number_of_items,
        data.branching_logic,
        data.complex_functionality,
        data.number_of_scores,
        data.normative_scoring,
        data.form_mapping,
        data.web_service_setup,
        int(data.form_type)
    ]
    return np.array(x).reshape(1, -1)


@app.get("/")
async def root():
    return {"Hello": "World"}


@app.post("/")
async def predicting_endpoint(item: EstimationItem):
    x = preprocess_input(item)

    # Predict probabilities
    predict_form_complexity_1 = model_Form_Complexity_1.predict_proba(x)[:, 1]
    predict_form_complexity_2 = model_Form_Complexity_2.predict_proba(x)[:, 1]
    predict_form_complexity_3 = model_Form_Complexity_3.predict_proba(x)[:, 1]
    predict_form_complexity_4 = model_Form_Complexity_4.predict_proba(x)[:, 1]

    # Return predictions as a JSON response
    labels = ["S", "M", "L", "XL"]
    # Probabilities for each form complexity
    data = [
        predict_form_complexity_1[0],
        predict_form_complexity_2[0],
        predict_form_complexity_3[0],
        predict_form_complexity_4[0]
    ]

    # Return predictions as a JSON response
    return dict(zip(labels, data))
