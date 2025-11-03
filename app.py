from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, joblib, os

# Load model bundle
MODEL_PATH = os.getenv("MODEL_PATH", "formula_model.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
base_cols = bundle["base_cols"]

app = FastAPI()

# Allow your GitHub Pages site
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "bases": len(base_cols)}

@app.get("/predict")
def predict(L: float = Query(...), a: float = Query(...), b: float = Query(...)):
    try:
        X = np.array([[L, a, b]], dtype=float)
        y = model.predict(X)
        y = np.clip(y, 0, None)
        y = (y / (y.sum(axis=1, keepdims=True) + 1e-9)) * 100.0
        mix = {k: float(v) for k, v in zip(base_cols, y[0])}
        # Display cleanup: round & hide zeros
        display = {k: round(v, 2) for k, v in mix.items() if v > 0.0}
        return {"input": {"L*": L, "a*": a, "b*": b}, "formula": display}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
