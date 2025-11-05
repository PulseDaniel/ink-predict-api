# train_model.py — tint-aware, gamut-aware trainer (no kNN, sparse, shop-safe)
import argparse, math, numpy as np, pandas as pd, joblib, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# -------- CONFIG (edit if you need) --------
MIN_DOSE = 0.5              # % below this → 0 in labels
TOP_N_CHROMA = 4            # keep ≤ N chromatic inks (+ extender)
EXTENDER_NAMES = ("Extender", "DC21-001 Extender")
FAMILY_SETS = (
    ("DC21-114 Rhodamine", "DC21-102 Rubine", "DC21-101 Bright Red"),
    # add more exclusive families if needed
)
# XGB sparsity/regularisation
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=900,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=1.0,   # L1 → sparsity
    reg_lambda=1.5,  # L2
    n_jobs=-1
)
TINT_SUBSTRATE = "white"    # use white-substrate tint priors
# -------------------------------------------

def is_extender(name:str)->bool:
    n = name.lower()
    return any(e.lower() in n for e in EXTENDER_NAMES)

def add_core_lab_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    L, a, b = df["L*"].values, df["a*"].values, df["b*"].values
    C = np.sqrt(a*a + b*b)
    h = np.arctan2(b, a)
    df["C*"] = C
    df["sin_h"] = np.sin(h)
    df["cos_h"] = np.cos(h)
    df["L_over_C"] = L / (C + 1e-6)
    df["a_over_C"] = a / (C + 1e-6)
    df["b_over_C"] = b / (C + 1e-6)
    # gamut proximity (safe, monotonic): C* / max_C*
    max_C = float(np.nanmax(C)) if np.isfinite(C).all() else 1.0
    df["C_norm"] = C / (max_C if max_C > 0 else 1.0)
    return df

def load_tint_features(path: str) -> pd.DataFrame:
    tf = pd.read_csv(path)
    tf["base"] = tf["base"].astype(str).str.strip()
    tf["substrate"] = tf["substrate"].astype(str).str.lower()
    tf = tf[tf["substrate"] == TINT_SUBSTRATE.lower()]
    required = {"base","dir_ab_cos","dir_ab_sin","slope_C_per_pct"}
    missing = required - set(tf.columns)
    if missing:
        raise ValueError(f"tint_features.csv missing columns: {sorted(missing)}")
    return tf.set_index("base")[["dir_ab_cos","dir_ab_sin","slope_C_per_pct"]]

def per_base_tint_feat(a: float, b: float, base_cols: list, tint_map: pd.DataFrame):
    """For each base: alignment with target a*b direction, low-dose slope, and their product."""
    v = np.array([a, b], dtype=float)
    n = float(np.linalg.norm(v))
    tgt_cos, tgt_sin = (1.0, 0.0) if n == 0 else (v[0]/n, v[1]/n)
    feats = []
    for base in base_cols:
        if base in tint_map.index and (not is_extender(base)):
            bcos = float(tint_map.loc[base, "dir_ab_cos"])
            bsin = float(tint_map.loc[base, "dir_ab_sin"])
            slope = float(tint_map.loc[base, "slope_C_per_pct"])
            align = bcos * tgt_cos + bsin * tgt_sin  # cosine similarity in a*b plane
            feats.extend([align, slope, align * slope])
        else:
            feats.extend([0.0, 0.0, 0.0])  # neutral for extenders / missing bases
    return np.array(feats, dtype=float)

def clean_labels_row(row: dict, base_cols: list) -> dict:
    y = {b: float(row.get(b, 0.0)) for b in base_cols}
    # 1) zero sub-min-dose
    for b in base_cols:
        if y[b] < MIN_DOSE:
            y[b] = 0.0
    # 2) family exclusivity (keep dominant)
    for fam in FAMILY_SETS:
        if not any(b in y for b in fam): 
            continue
        vals = {b: y.get(b, 0.0) for b in fam}
        if sum(vals.values()) <= 0:
            continue
        keep = max(vals, key=vals.get)
        for b in fam:
            if b != keep:
                y[b] = 0.0
    # 3) top-N chroma (extenders exempt)
    chroma = [(b, y[b]) for b in base_cols if not is_extender(b)]
    chroma.sort(key=lambda kv: kv[1], reverse=True)
    for b, _ in chroma[TOP_N_CHROMA:]:
        y[b] = 0.0
    # 4) renorm to 100
    s = sum(y.values())
    if s > 0:
        for b in base_cols:
            y[b] = 100.0 * y[b] / s
    return y

def build_matrices(formulas_csv: str, tint_csv: str):
    df = pd.read_csv(formulas_csv)
    feature_cols = ["L*","a*","b*"]
    base_cols = [c for c in df.columns if c not in ["Pantone"] + feature_cols]

    # label cleanup
    Y_list = []
    for _, r in df.iterrows():
        y = clean_labels_row({b: r[b] for b in base_cols}, base_cols)
        Y_list.append([y[b] for b in base_cols])
    Y = np.array(Y_list, dtype=float)          # cleaned percentages
    Y_sel = (Y > 0.0).astype(int)              # inclusion labels

    # core LAB + gamut features
    df_feat = add_core_lab_features(df)
    X_core = df_feat[["L*","a*","b*","C*","sin_h","cos_h","L_over_C","a_over_C","b_over_C","C_norm"]].values

    # tint features (white substrate)
    tint_map = load_tint_features(tint_csv)
    tint_rows = []
    for _, r in df.iterrows():
        tint_rows.append(per_base_tint_feat(float(r["a*"]), float(r["b*"]), base_cols, tint_map))
    X_tint = np.vstack(tint_rows)

    # concatenate and scale
    X_all = np.hstack([X_core, X_tint])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    return X_scaled, Y, Y_sel, base_cols, scaler, tint_map

def train(csv_path: str, tint_csv: str, out_path: str):
    X, Y, Y_sel, base_cols, scaler, tint_map = build_matrices(csv_path, tint_csv)
    Xtr, Xte, Ytr, Yte, YStr, YSte = train_test_split(X, Y, Y_sel, test_size=0.2, random_state=42)

    # Stage 1 — selector (balanced)
    selector = MultiOutputRegressor(
        LogisticRegression(max_iter=400, solver="lbfgs", n_jobs=1, class_weight="balanced")
    )
    selector.fit(Xtr, YStr)

    # Stage 2 — regressor (sparse XGB)
    reg = MultiOutputRegressor(XGBRegressor(**XGB_PARAMS))
    reg.fit(Xtr, Ytr)

    # Eval (mask by selector, renorm)
    Yp = np.clip(reg.predict(Xte), 0, None)
    sel_mask = (selector.predict(Xte) > 0.5).astype(float)
    Yp = Yp * sel_mask
    denom = Yp.sum(axis=1, keepdims=True); denom[denom==0] = 1.0
    Yp = (Yp / denom) * 100.0

    mae_per_base = mean_absolute_error(Yte, Yp, multioutput="raw_values")
    mae_mean = float(mae_per_base.mean())
    comp_counts = (Yp > 0.0).sum(axis=1)
    pct_le_topN = float((comp_counts <= (TOP_N_CHROMA + 1)).mean() * 100.0)  # +1 if extender present

    print("Per-base MAE (%):")
    for col, m in zip(base_cols, mae_per_base):
        print(f"  {col:>28}: {m:5.2f}")
    print(f"Mean MAE (%): {mae_mean:.2f}")
    print(f"Predictions using ≤ {TOP_N_CHROMA} chroma inks (+extender): {pct_le_topN:.1f}%")

    bundle = {
        "model": reg,
        "selector": selector,
        "scaler": scaler,
        "feature_cols": ["L*","a*","b*"],
        "base_cols": base_cols,
        "tint_map": tint_map.to_dict(orient="index"),
        "config": {
            "min_dose": MIN_DOSE,
            "top_n_chroma": TOP_N_CHROMA,
            "extender_names": EXTENDER_NAMES,
            "family_sets": FAMILY_SETS,
            "xgb_params": XGB_PARAMS,
            "tint_substrate": TINT_SUBSTRATE,
            "eval": {"mae_mean": mae_mean, "pct_le_topN": pct_le_topN}
        }
    }
    joblib.dump(bundle, out_path)
    print(f"Saved model bundle → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="formulas.csv")
    ap.add_argument("--tint_csv", default="tint_features.csv")
    ap.add_argument("--out", default="formula_model.joblib")
    args = ap.parse_args()
    train(args.csv, args.tint_csv, args.out)
