from typing import Optional
import pandas as pd

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

# import relative libs from gradient_boosting_model pack
from gradient_boosting_model.train_pipeline import run_training
from gradient_boosting_model.processing_utils.data_management import load_dataset
from gradient_boosting_model.config.core import config

# Train the model
run_training()


app = FastAPI()

from gradient_boosting_model.predict import make_prediction
from app.input_schem import InputSchema

@app.get("/")
async def root():
    return "Defaulting Estimation"


@app.post("/predict")
def predict(input_data: list[InputSchema]):

	# parse json request to dataframe
	input_data = pd.DataFrame(jsonable_encoder(input_data))

	prediction = make_prediction(input_data=input_data)

	return str(prediction) 


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

