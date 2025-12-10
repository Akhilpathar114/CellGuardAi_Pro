
import pandas as pd
import numpy as np
from . import config

def clean_bms_dataframe(df):
    df = df.copy()
    if "Soc" in df.columns:
        df["Soc"] = df["Soc"].astype(str).str.replace("%","")
        df["Soc"] = pd.to_numeric(df["Soc"], errors="coerce").clip(config.SOC_MIN, config.SOC_MAX)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].interpolate().fillna(df[numeric].median())
    return df, {"rows": len(df)}
