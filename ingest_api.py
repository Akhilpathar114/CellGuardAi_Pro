
from fastapi import FastAPI
import pandas as pd
from cellguardai.data_cleaning import clean_bms_dataframe
from cellguardai.rules_engine import evaluate_safety_rules
from cellguardai.ai_model import compute_risk_scores

app = FastAPI()

@app.post("/ingest")
def ingest(data: dict):
    df = pd.DataFrame([data])
    df,_ = clean_bms_dataframe(df)
    df = evaluate_safety_rules(df)
    df = compute_risk_scores(df)
    return df.to_dict(orient="records")[0]
