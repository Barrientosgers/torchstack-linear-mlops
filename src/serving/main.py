import os
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "/models/linear.pt"

app = FastAPI(title="TorchStack Linear API", version="0.1.0")

class PredictRequest(BaseModel):
    x: float

def load_model():
    model = nn.Linear(1, 1)
    if not os.path.exists(MODEL_PATH):
        return None
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found. Run trainer first.")
    x = torch.tensor([[req.x]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x).item()
    return {"x": req.x, "y": y}