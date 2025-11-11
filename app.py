# app.py — API for ink formula prediction + feedback logging

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict_core import (
    add_core_lab_features,
    build_per_base_tint_features,
    postprocess_from_config,
    lock_to_neighbor_inkset,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODEL_PATH = DATA_DIR / "formula_model.joblib"
FORMULAS_CSV = DATA_DIR / "formulas.csv"
DB_PATH = APP_DIR / "db.sqlite"

# ---------- load model bundle ----------
if not MODEL_PATH.exists():
    raise RuntimeError("Missing model bundle at data/formula_model.joblib")

bundle = joblib.load(MODEL_PATH)
reg = bundle["model"]
selector = bundle["selector"]
scaler = bundle["scaler"]
base_cols = bundle["base_cols"]
tint_map = bundle["tint_map"]
config = bundle.get("config", {})

extenders = tuple(config.get("extender_names", ("Extender", "DC21-001 Extender")))

try:
    max_C_train = float(config.get("max_C_train", None))
except Exception:
    max_C_train = None

# ---------- reference library for neighbour lock ----------
try:
    lib_df = pd.read_csv(FORMULAS_CSV)
except Exception:
    lib_df = None

# ---------- HARD GROUP MAP (built-in) ----------
# Short names → DC21 ink names in your model
ALIASES = {
    "Rubine": "DC21-102 Rubine",
    "Bright Red": "DC21-101 Bright Red",
    "Rhodamine": "DC21-114 Rhodamine",
    "Yellow": "DC21-304 Yellow",
    "Orange": "DC21-201 Orange",
    "Pro Blue": "DC21-501 Process Blue",
    "Process Blue": "DC21-501 Process Blue",
    "Green": "DC21-402 Green",
    "Violet": "DC21-604 Violet",
    "Black": "DC21-802 Black",
    # if/when you have a real warm red base, change this:
    "Warm Red": "DC21-101 Bright Red",
}

_GROUPS_SHORT = {
    "Reds":              ["Rubine", "Bright Red", "Rhodamine", "Yellow", "Orange", "Pro Blue", "Black"],
    "Yellows & Oranges": ["Yellow", "Orange", "Green", "Pro Blue", "Bright Red", "Rubine", "Black"],
    "Blues":             ["Pro Blue", "Violet", "Green", "Rubine", "Yellow", "Black"],
    "Greens":            ["Pro Blue", "Yellow", "Green", "Orange", "Black"],
    "Browns":            ["Rubine", "Yellow", "Orange", "Black"],
    "Pinks & Purples":   ["Rubine", "Rhodamine", "Violet", "Warm Red", "Black"],
    "Greys":             ["Pro Blue", "Yellow", "Rubine", "Black"],
}


def to_dc_names(short_list):
    out = []
    for s in short_list:
        name = ALIASES.get(s)
        if name and name in base_cols:
            out.append(name)
    return sorted(set(out))


GROUPS = {g: to_dc_names(names) for g, names in _GROUPS_SHORT.items()}
ALL_GROUP_NAMES = sorted(GROUPS.keys())


def allowed_inks_for_group(group: Optional[str]) -> set:
    """
    Returns the set of allowed ink names for the given group.

    - If group is None/empty: all inks allowed.
    - Extenders are ALWAYS allowed.
    - If mapping resolves to nothing: at least extenders are allowed.
    """
    if not group:
        return set(base_cols)

    allowed = set(GROUPS.get(group, []))

    # always allow extenders present in base_cols
    for ex in extenders:
        for b in base_cols:
            if ex.lower() in b.lower():
                allowed.add(b)

    if not allowed:
        # fallback: extenders only
        allowed = {b for b in base_cols if any(ex.lower() in b.lower() for ex in extenders)}

    return allowed


# ---------- DB ----------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
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
        """
    )
    return conn


# ---------- API ----------
app = FastAPI(title="Ink Formula Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you like
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictIn(BaseModel):
    L: float
    a: float
    b: float
    lock_neighbours: bool = True
    group: Optional[str] = Field(None, description="One of /groups; restrict inks to this group (+Extender)")


class PredictOut(BaseModel):
    input: Dict[str, float]
    formula: Dict[str, float]
    group: Optional[str] = None
    allowed_inks: Optional[list] = None


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


@app.get("/groups")
def groups():
    """List available groups + their DC inks."""
    return {"groups": ALL_GROUP_NAMES, "group_inks": GROUPS}


@app.get("/inks")
def inks():
    """All base inks the model knows (for dropdowns)."""
    return {"inks": sorted(list(base_cols))}


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    L, a, b = float(inp.L), float(inp.a), float(inp.b)

    # Build feature vector
    core = add_core_lab_features(L, a, b, max_C_train=max_C_train)
    tint_vec = build_per_base_tint_features(a, b, base_cols, tint_map, extenders)
    X = np.hstack([core, tint_vec]).reshape(1, -1)
    Xs = scaler.transform(X)

    # Raw predictions
    y = np.clip(reg.predict(Xs), 0, None).ravel()

    # Selector → 0/1 mask
    sel_raw = np.array(selector.predict(Xs)).ravel().astype(float)
    sel = (sel_raw > 0.5).astype(float)

    # HARD GROUP FILTER
    allowed = allowed_inks_for_group(inp.group)
    mask = np.array([1.0 if base_cols[i] in allowed else 0.0 for i in range(len(base_cols))], dtype=float)
    sel = sel * mask

    # Apply selection
    y = y * sel
    s = y.sum() or 1.0
    y = (y / s) * 100.0

    # Optional neighbour lock
    if inp.lock_neighbours:
        y = lock_to_neighbor_inkset(
            L,
            a,
            b,
            base_cols,
            y,
            lib_df,
            k=3,
            de_gate=3.0,
            min_pct=0.5,
        )

    # Re-apply group mask for safety
    for i, name in enumerate(base_cols):
        if name not in allowed:
            y[i] = 0.0
    s = y.sum() or 1.0
    y = (y / s) * 100.0

    display = postprocess_from_config(base_cols, y, config)

    return {
        "input": {"L*": L, "a*": a, "b*": b},
        "formula": display,
        "group": inp.group,
        "allowed_inks": sorted(list(allowed)),
    }


@app.post("/log_approved")
def log_approved(payload: LogIn):
    """
    Save an approved/tweaked formula into a local sqlite DB.

    This is what the web UI calls when you click "Save edited formula".
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (
            pantone,
            target_L, target_a, target_b,
            suggested_json,
            approved_json,
            measured_L, measured_a, measured_b,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.pantone,
            payload.target_L,
            payload.target_a,
            payload.target_b,
            json.dumps(payload.suggested or {}, ensure_ascii=False),
            json.dumps(payload.approved or {}, ensure_ascii=False),
            payload.measured_L,
            payload.measured_a,
            payload.measured_b,
            payload.notes,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return {"ok": True, "id": int(new_id)}


@app.get("/export_training")
def export_training():
    """
    Export all logged, approved formulas as a CSV string embedded in JSON.

    For retraining: call this, parse CSV, merge with your base formulas.
    """
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT * FROM feedback WHERE approved_json IS NOT NULL AND length(approved_json) > 2",
        conn,
    )
    conn.close()

    if df.empty:
        return {"rows": 0, "csv": ""}

    out_rows = []
    for _, r in df.iterrows():
        row = {
            "Pantone": r.get("pantone") or "",
            "L*": float(r["target_L"]),
            "a*": float(r["target_a"]),
            "b*": float(r["target_b"]),
        }
        mix = json.loads(r["approved_json"] or "{}")
        for k, v in mix.items():
            try:
                row[k] = float(v)
            except Exception:
                continue
        out_rows.append(row)

    out = pd.DataFrame(out_rows).fillna(0.0)
    return {"rows": int(len(out)), "csv": out.to_csv(index=False)}
