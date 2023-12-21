import os
from fastapi import FastAPI
from preprocess import recommendation_function
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/test")
def index():
    return f"Hello test"

@app.post("/recommend_foods")
def recommend_foods_endpoint(sex: str,
    age: int,
    weight: float,
    height: float,
    activity_level: float,
    weight_target: float,
    pace: str,
    vegetarian: str):
    # Call your recommendation function from preprocessing.py
    recommended_foods = recommendation_function(sex, age, weight, height, activity_level, weight_target, pace, vegetarian)

    return {"message": "Recommended Foods", "recommended_foods": recommended_foods}


port = os.environ.get("PORT", 8090)
print(f"Listening to this http://0.0.0.0:{port}/test")
uvicorn.run(app, host='0.0.0.0',port=port)