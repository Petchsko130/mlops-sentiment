import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 👉 ชี้ไปยัง MLflow Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

# ✅ โหลด model จาก Model Registry
model = mlflow.sklearn.load_model("runs:/efc68fc0101943b2bce129f7902df05e/model")

app = FastAPI()


# 🧠 ใส่ feature name ให้ตรงกับตอนเทรน (Iris dataset)
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"message": "🎯 Model is ready to predict"}


@app.post("/predict")
def predict(input_data: InputData):
    input_df = pd.DataFrame(
        [
            {
                "sepal length (cm)": input_data.sepal_length,
                "sepal width (cm)": input_data.sepal_width,
                "petal length (cm)": input_data.petal_length,
                "petal width (cm)": input_data.petal_width,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}
