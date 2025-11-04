from fastapi import FastAPI
from Schema.pydantic_model import UserInput
import pandas as pd
import pickle
from fastapi.responses import JSONResponse

with open('model/model.pkl','rb') as f:
    model = pickle.load(f)

encoder = model["Lable_encoder"]
scaler = model["Standard_scaler"]
model = model["Logistic_regression"]

app = FastAPI(title="Student Success Prediction API")

@app.get("/")
def default():
    return {"message": "Welcome to the Student Success Prediction API"}

@app.post("/prediction")
def prediction_output(prediction:UserInput):
    df = pd.DataFrame([{
        "age": prediction.age,
        "gender": prediction.gender,
        "previous_grades": prediction.previous_grades,
        "attendance_percentage": prediction.attendance_percentage,
        "study_hours_per_week": prediction.study_hours_per_week,
        "assignments_submitted": prediction.assignments_submitted,
        "exam_scores": prediction.exam_scores
    }])

    df = scaler.transform(df)

    predicted_output = model.predict(df)

    output = "Pass" if predicted_output[0] == 1 else "Fail"

    return JSONResponse(status_code=200,content={
        "Prediction of the Student is":output
    })