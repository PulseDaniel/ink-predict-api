# app.py
import os, json, sqlite3
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict_core import (
    add_core_lab_features, build_per_base_tint_features, postprocess_from_config,
    lock_to_neighbor_inkset
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_PATH = DATA_DIR / "formula_model.joblib"
FORMULAS_CSV = DATA_DIR / "formulas.csv"
TINT_CSV = DATA_DIR / "tint_features.csv"
DB_PATH = APP_DIR / "db.sqlite"

# ---- load model bundle ----
if not MODEL_PATH.exists():
    raise RuntimeError("Missing model bundle at data/formula_model.joblib")
bundle = joblib.load(MODEL_PATH)
reg       = bundle["model"]
selector  = bundle["selector"]
scaler    = bundle["scaler"]
base_cols = bundle["base_cols"]
tint_map  = bundle["tint_map"]      # dict per base
config    = bundle.get("config", {})
extenders = tuple(config.get("extender_names", ("Extender","DC21-001 Extender")))
try:
    max_C_train = float(config.get("max_C_train", None))
except Exception:
    max_C_train = None

# library for ink-set lock
try:
    lib_df = pd.read_csv(FORMULAS_CSV)
except Exception:
    lib_df = None

# ---- DB ----
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now')),
            pantone TEXT,
            target_L REAL, target_a REAL, target_b REAL,
            suggested_json TEXT,
            approved_json TEXT,
            measured_L REAL, measured_a REAL, measured_b REAL,
            notes TEXT
        )
    """)
    return conn

# ---- API ----
app = FastAPI(title="Ink Formula Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # set to your GitHub Pages origin later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    L: float = Field(..., description="Target L*")
    a: float = Field(..., description="Target a*")
    b: float = Field(..., description="Target b*")
    lock_neighbours: bool = Field(True, description="Restrict inks to nearest approved ink-sets")

class PredictOut(BaseModel):
    input: Dict[str, float]
    formula: Dict[str, float]

class LogIn(BaseModel):
    pantone: Optional[str] = None
    target_L: float
    target_a: float
    target_b: float
    suggested: Dict[str, float]
    approved: Optional[Dict[str, float]] = None
    measured_L: Optional[float] = None
    measured_a: Optional[float] = None
    measured_b: Optional[float] = None
    notes: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    L, a, b = float(inp.L), float(inp.a), float(inp.b)
    core = add_core_lab_features(L, a, b, max_C_train=max_C_train)
    tint_vec = build_per_base_tint_features(a, b, base_cols, tint_map, extenders)
    X = np.hstack([core, tint_vec]).reshape(1, -1)
    Xs = scaler.transform(X)

    y = np.clip(reg.predict(Xs), 0, None).ravel()
    sel_raw = np.array(selector.predict(Xs)).ravel()
    sel = (sel_raw > 0.5).astype(float)
    if sel.shape[0] != y.shape[0]:
        sel = np.ones_like(y)

    y = y * sel
    s = y.sum() or 1.0
    y = (y / s) * 100.0

    if inp.lock_neighbours:
        y = lock_to_neighbor_inkset(L, a, b, base_cols, y, lib_df, k=3, de_gate=3.0, min_pct=0.5)

    display = postprocess_from_config(base_cols, y, config)
    return {"input": {"L*": L, "a*": a, "b*": b}, "formula": display}

@app.post("/log_approved")
def log_approved(payload: LogIn):
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (pantone, target_L, target_a, target_b, suggested_json, approved_json,
                              measured_L, measured_a, measured_b, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload.pantone,
        payload.target_L, payload.target_a, payload.target_b,
        json.dumps(payload.suggested or {}, ensure_ascii=False),
        json.dumps(payload.approved or {}, ensure_ascii=False),
        payload.measured_L, payload.measured_a, payload.measured_b,
        payload.notes
    ))
    conn.commit(); conn.close()
    return {"ok": True, "id": cur.lastrowid}

@app.get("/export_training")
def export_training():
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT * FROM feedback WHERE approved_json IS NOT NULL AND length(approved_json) > 2",
        conn
    )
    conn.close()
    if df.empty:
        return {"rows": 0, "csv": ""}
    out_rows = []
    for _, r in df.iterrows():
        row = {"Pantone": r.get("pantone") or "",
               "L*": float(r["target_L"]), "a*": float(r["target_a"]), "b*": float(r["target_b"])}
        mix = json.loads(r["approved_json"] or "{}")
        for k, v in mix.items(): row[k] = float(v)
        out_rows.append(row)
    out = pd.DataFrame(out_rows).fillna(0.0)
    return {"rows": int(len(out)), "csv": out.to_csv(index=False)}
