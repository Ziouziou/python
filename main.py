import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
app = FastAPI()

origins = [

    "http://localhost:4200",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Request(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    x7: float
    x8: float
    x9: float
    x10: float
    x11: float
    x12: float
    x13: float
    x14: float
    x15: float
    x16: float
    x17: float
    x18: float
    x19: float
    x20: float
    x21: float
    x22: float
    x23: float
    def to_numpy(self):
        return np.array(
            [self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7, self.x8, self.x9, self.x10, self.x11,
             self.x12, self.x13, self.x14, self.x15, self.x16, self.x17, self.x18, self.x19, self.x20, self.x21,
             self.x22, self.x23])

@app.get("/")
async def root():
    return {"message": "Hgggello Wojkkkjkjjkrld"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/predict")
async def read_item(request: Request):
    data = request.to_numpy().reshape(1, -1)

    with open('modelv2.pkl', 'rb') as f:
        model = pickle.load(f)
        prediction = int(model.predict(data)[0])

        return {'prediction': prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)