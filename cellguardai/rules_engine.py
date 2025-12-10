
def evaluate_safety_rules(df):
    df = df.copy()
    levels = []
    for _, r in df.iterrows():
        level = "LOW"
        if r.get("Temp1",0) > 60: level = "CRITICAL"
        elif r.get("C_Diff",0) > 80: level = "HIGH"
        levels.append(level)
    df["rule_level"] = levels
    return df
