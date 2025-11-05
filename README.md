\# Ink Formula Predictor API



FastAPI service that serves an ink formula ML model and logs approved formulas for retraining.



\## Endpoints



\- `GET /health` → `{status:"ok"}`

\- `POST /predict` → body `{ L, a, b, lock\_neighbours=true }` → `{ input, formula }`

\- `POST /log\_approved` → body `{ pantone?, target\_L, target\_a, target\_b, suggested, approved?, measured\_L?, measured\_a?, measured\_b?, notes? }`

\- `GET /export\_training` → `{ rows, csv }` (CSV of approved rows: target → approved mix)



\## Setup (local)



```bash

python -m venv .venv

. .venv/Scripts/activate  # Windows

pip install -r requirements.txt



\# Put your model + data in ./data

\#   data/formula\_model.joblib

\#   data/formulas.csv

\#   data/tint\_features.csv



uvicorn app:app --reload



