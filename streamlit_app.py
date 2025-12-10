
import streamlit as st
import pandas as pd
from cellguardai.data_cleaning import clean_bms_dataframe
from cellguardai.rules_engine import evaluate_safety_rules
from cellguardai.ai_model import compute_risk_scores

st.title("CellGuardAI Live + CSV")

mode = st.selectbox("Mode",["CSV","Live Demo"])

if mode=="CSV":
    f=st.file_uploader("Upload CSV")
    if f:
        df=pd.read_csv(f)
        df,_=clean_bms_dataframe(df)
        df=evaluate_safety_rules(df)
        df=compute_risk_scores(df)
        st.dataframe(df.head())
else:
    st.info("Live mode uses /ingest API")
