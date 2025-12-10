
def explain_row(row):
    msgs=[]
    if row.get("Temp1",0)>60: msgs.append("High temperature")
    if row.get("C_Diff",0)>80: msgs.append("Cell imbalance")
    return {"messages":msgs}
