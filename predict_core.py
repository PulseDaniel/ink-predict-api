# predict_core.py
import math, numpy as np, pandas as pd

def is_extender(name: str, extender_names):
    n = name.lower()
    return any(e.lower() in n for e in extender_names)

def add_core_lab_features(L, a, b, max_C_train=None):
    C = (a*a + b*b) ** 0.5
    h = math.atan2(b, a)
    if max_C_train is None or max_C_train <= 0:
        max_C_train = 100.0
    C_norm = C / max_C_train
    return np.array([
        L, a, b,
        C,
        math.sin(h), math.cos(h),
        L/(C+1e-6), a/(C+1e-6), b/(C+1e-6),
        C_norm
    ], dtype=float)

def build_per_base_tint_features(a, b, base_cols, tint_map, extender_names):
    v = np.array([a, b], dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0:
        tgt_cos, tgt_sin = 1.0, 0.0
    else:
        tgt_cos, tgt_sin = v[0]/n, v[1]/n

    feats = []
    for base in base_cols:
        if (base not in tint_map) or is_extender(base, extender_names):
            align = 0.0; slope = 0.0
        else:
            bcos = float(tint_map[base].get("dir_ab_cos", 0.0))
            bsin = float(tint_map[base].get("dir_ab_sin", 0.0))
            slope = float(tint_map[base].get("slope_C_per_pct", 0.0))
            align = bcos * tgt_cos + bsin * tgt_sin
        feats.extend([align, slope, align * slope])
    return np.array(feats, dtype=float)

def postprocess_from_config(base_cols, vec, config):
    min_pct       = float(config.get("min_dose", 0.5))
    top_n_chroma  = int(config.get("top_n_chroma", 4))
    extender_names= tuple(config.get("extender_names", ("Extender","DC21-001 Extender")))
    family_sets   = tuple(tuple(x) for x in config.get("family_sets", []))

    mix = {k: float(v) for k, v in zip(base_cols, vec)}
    s = sum(mix.values()) or 1.0
    for k in mix: mix[k] = 100.0 * mix[k] / s

    for fam in family_sets:
        present = {k: mix.get(k, 0.0) for k in fam}
        if sum(present.values()) <= 0: continue
        keep = max(present, key=present.get)
        for k in fam:
            if k != keep: mix[k] = 0.0

    chroma = sorted([(k,v) for k,v in mix.items() if k and not is_extender(k, extender_names)],
                    key=lambda kv: kv[1], reverse=True)
    for k,_ in chroma[top_n_chroma:]:
        mix[k] = 0.0

    if chroma:
        largest_key = chroma[0][0]
        for k,v in list(mix.items()):
            if (not is_extender(k, extender_names)) and k != largest_key and v < min_pct:
                mix[k] = 0.0

    s = sum(mix.values()) or 1.0
    for k in mix: mix[k] = 100.0 * mix[k] / s

    display = {k: round(v, 2) for k, v in mix.items() if v > 0.0}
    t = sum(display.values()) or 1.0
    for k in list(display): display[k] = round(display[k] * 100.0 / t, 2)
    return display

# ΔE00 for neighbour ink-set lock
def ciede2000(p, q):
    L1,a1,b1 = p; L2,a2,b2 = q
    kL=kC=kH=1.0; deg=math.pi/180.0
    C1=(a1*a1+b1*b1)**0.5; C2=(a2*a2+b2*b2)**0.5
    Cm=(C1+C2)/2; G=0.5*(1-((Cm**7)/((Cm**7)+(25**7)))**0.5)
    a1p=(1+G)*a1; a2p=(1+G)*a2; C1p=(a1p*a1p+b1*b1)**0.5; C2p=(a2p*a2p+b1*b1)**0.5
    h1p=(math.atan2(b1,a1p)/deg)%360; h2p=(math.atan2(b2,a2p)/deg)%360
    dLp=L2-L1; dCp=C2p-C1p
    dh=h2p-h1p; dh=dh-360 if dh>180 else (dh+360 if dh<-180 else dh)
    dHp=2*(C1p*C2p)**0.5*math.sin((dh*deg)/2)
    Lpm=(L1+L2)/2; Cpm=(C1p+C2p)/2
    hpm=(h1p+h2p+360)/2 if abs(h1p-h2p)>180 else (h1p+h2p)/2
    T=1-0.17*math.cos((hpm-30)*deg)+0.24*math.cos((2*hpm)*deg)+0.32*math.cos((3*hpm+6)*deg)-0.20*math.cos((4*hpm-63)*deg)
    Sl=1+(0.015*(Lpm-50)**2)/math.sqrt(20+(Lpm-50)**2)
    Sc=1+0.045*Cpm; Sh=1+0.015*Cpm*T
    Rc=2*math.sqrt((Cpm**7)/((Cpm**7)+(25**7)))
    Rt=-Rc*math.sin(2*(30*math.exp(-(((hpm-275)/25)**2)))*deg)
    return math.sqrt((dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2 + Rt*(dCp/Sc)*(dHp/Sh))

def lock_to_neighbor_inkset(L, a, b, base_cols, y_vec, lib_df, k=3, de_gate=3.0, min_pct=0.5):
    if lib_df is None or not {"L*","a*","b*"}.issubset(lib_df.columns):
        return y_vec
    feats = lib_df[["L*","a*","b*"]].to_numpy(dtype=float)
    ds = np.array([ciede2000((L,a,b), tuple(row)) for row in feats])
    idx = np.argsort(ds)[:k]
    if float(ds[idx].mean()) > de_gate:
        return y_vec
    allow = set()
    for i in idx:
        for bname in base_cols:
            if bname in lib_df.columns and float(lib_df.iloc[i][bname]) > min_pct:
                allow.add(bname)
    y_masked = y_vec.copy()
    for j, bname in enumerate(base_cols):
        if bname not in allow:
            y_masked[j] = 0.0
    s = y_masked.sum() or 1.0
    return (y_masked / s) * 100.0
