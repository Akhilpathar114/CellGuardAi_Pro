
import numpy as np
def compute_risk_scores(df):
    df = df.copy()
    df["risk_score"] = np.clip(
        (df.get("Temp1",0)/80 + df.get("C_Diff",0)/100)/2,0,1
    )
    return df
