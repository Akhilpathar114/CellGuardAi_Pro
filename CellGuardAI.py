import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import re

st.set_page_config(page_title="CellGuard.AI - Shell Skills4Future", layout="wide")

# ----------------------------
# Demo data (for testing only)
# ----------------------------
def gen_sample_data(n=800, seed=42, scenario="Generic"):
    np.random.seed(seed)
    t = np.arange(n)
    base_v = 3.7
    base_i = 1.5
    base_temp = 30.0
    soc_base = 80.0

    if scenario == "Generic":
        v = base_v + 0.05 * np.sin(t / 50) + np.random.normal(0, 0.005, n)
        i = base_i + 0.3 * np.sin(t / 30) + np.random.normal(0, 0.05, n)
        temp = base_temp + 3 * np.sin(t / 60) + np.random.normal(0, 0.3, n)
        soc = np.clip(soc_base + 10 * np.sin(t / 80) + np.random.normal(0, 1, n), 0, 100)
        cycle = t // 50
    elif scenario == "EV":
        v = base_v + 0.03 * np.sin(t / 40) - 0.0005 * t / n + np.random.normal(0, 0.008, n)
        i = 2.5 + 0.4 * np.sin(t / 20) + np.random.normal(0, 0.07, n)
        temp = base_temp + 4 * np.sin(t / 120) + 0.01 * (t / n) * 10 + np.random.normal(0, 0.5, n)
        soc = np.clip(90 - 20 * (t / n) + np.random.normal(0, 1.5, n), 0, 100)
        cycle = t // 10
    elif scenario == "Drone":
        v = base_v + 0.04 * np.sin(t / 30) + np.random.normal(0, 0.006, n)
        i = base_i + 0.6 * np.sin(t / 10) + np.random.normal(0, 0.2, n)
        temp = base_temp + 2 * np.sin(t / 80) + np.random.normal(0, 0.4, n)
        soc = np.clip(85 + 6 * np.sin(t / 40) + np.random.normal(0, 2, n), 0, 100)
        cycle = t // 30
    else:
        v = base_v + 0.02 * np.sin(t / 80) + np.random.normal(0, 0.002, n)
        i = 0.8 + 0.1 * np.sin(t / 60) + np.random.normal(0, 0.02, n)
        temp = base_temp + 1.5 * np.sin(t / 120) + np.random.normal(0, 0.15, n)
        soc = np.clip(95 + 3 * np.sin(t / 160) + np.random.normal(0, 0.5, n), 0, 100)
        cycle = t // 200

    df = pd.DataFrame({
        "time": t,
        "voltage": v,
        "current": i,
        "temperature": temp,
        "soc": soc,
        "cycle": cycle
    })
    df["timestamp"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["time"], unit="s")
    return df

# ----------------------------
# Column helpers
# ----------------------------
def normalize_cols(df):
    df = df.copy()
    simple = {c: "".join(ch for ch in str(c).lower() if ch.isalnum()) for c in df.columns}
    patt = {
        "voltage": ["volt", "vcell", "cellv", "packv", "packvol"],
        "current": ["curr", "amp", "amps", "ichg", "idis", "current", "curent"],
        "temperature": ["temp", "temperature", "celltemp", "packtemp"],
        "soc": ["soc", "stateofcharge"],
        "cycle": ["cycle", "cyclecount", "chargecycle"],
        "time": ["time", "timestamp", "t", "index"],
    }
    cmap = {}
    used = set()
    for target, keys in patt.items():
        for orig, simplified in simple.items():
            if orig in used:
                continue
            if any(k in simplified for k in keys):
                cmap[target] = orig
                used.add(orig)
                break
    rename = {orig: targ for targ, orig in cmap.items()}
    df = df.rename(columns=rename)
    return df, cmap

def ensure_cols_exist(df, needed):
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ----------------------------
# Unique column name helper
# ----------------------------
def make_unique_columns(df):
    df = df.copy()
    seen = {}
    new_cols = []
    for c in df.columns:
        name = str(c)
        if name not in seen:
            seen[name] = 0
            new_cols.append(name)
        else:
            seen[name] += 1
            new_cols.append(f"{name}_{seen[name]}")
    df.columns = new_cols
    return df

# ----------------------------
# Shell_BMS specific pre-clean
# ----------------------------
def preclean_shell_bms(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    time_col = None
    for c in df.columns:
        cl = c.lower()
        if cl == "date":
            date_col = c
        if cl == "time":
            time_col = c
    if date_col is not None and time_col is not None:
        ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
        if ts.notna().any():
            df["timestamp"] = ts
            t0 = ts.min()
            df["time"] = (ts - t0).dt.total_seconds()

    rename_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if cl in ("packvol", "packvolt", "packvoltage"):
            rename_map[c] = "voltage"
        elif cl in ("curent", "current"):
            rename_map[c] = "current"
        elif cl == "soc":
            rename_map[c] = "soc"
        elif cl == "cycle":
            rename_map[c] = "cycle"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# ----------------------------
# Strong sanitizer (silent)
# ----------------------------
def strong_sanitize(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def looks_like_header(row):
        cnt = 0
        for col in df.columns:
            try:
                if str(row.get(col, "")).strip().lower() == str(col).strip().lower():
                    cnt += 1
            except Exception:
                pass
        return cnt >= max(2, len(df.columns) // 4)

    header_mask = df.apply(looks_like_header, axis=1)
    if header_mask.any():
        df = df.loc[~header_mask].reset_index(drop=True)

    force_numeric = ["voltage", "current", "temperature", "temp", "soc", "cycle", "time"]
    lower_map = {c.lower(): c for c in df.columns}
    mapped = {}
    for want in force_numeric:
        for clower, orig in lower_map.items():
            if want in clower or (any(k in clower for k in ["volt", "vcell", "packv", "pack vol"]) and "volt" in want):
                if want not in mapped:
                    mapped[want] = orig

    for core in ["voltage", "current", "temperature", "soc", "cycle", "time"]:
        if core in df.columns and core not in mapped:
            mapped[core] = core

    def clean_num_str(s):
        if pd.isnull(s):
            return s
        s = str(s).strip()
        s = re.sub(r'(?i)\b(v|volts?|a|amps?|%|degc|c|khz|hz|mv|ma|ah)\b', '', s)
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'[^\d\.\-eE,]+', '', s)
        s = s.replace(',', '')
        s = s.strip()
        return s if s != '' else None

    for key, orig_col in mapped.items():
        try:
            raw = df[orig_col].astype(object)
        except Exception:
            continue
        cleaned = raw.map(clean_num_str)
        coerced = pd.to_numeric(cleaned, errors="coerce")
        df[orig_col] = coerced
        if df[orig_col].notna().sum() > 0:
            med = df[orig_col].median(skipna=True)
            df[orig_col] = df[orig_col].fillna(med).ffill().bfill()
        else:
            df[orig_col] = df[orig_col].astype(float)

    return df

# ----------------------------
# Feature engineering
# ----------------------------
def make_features(df, window=10):
    df = df.copy()
    df = ensure_cols_exist(df, ["voltage", "current", "temperature", "soc", "cycle", "time"])

    # Voltage
    if df["voltage"].notna().sum() > 0:
        df["voltage_ma"] = df["voltage"].rolling(window, min_periods=1).mean()
        df["voltage_roc"] = df["voltage"].diff().fillna(0)
        df["voltage_var"] = df["voltage"].rolling(window, min_periods=1).var().fillna(0)
    else:
        df["voltage_ma"] = np.nan
        df["voltage_roc"] = np.nan
        df["voltage_var"] = np.nan

    # Temperature
    if df["temperature"].notna().sum() > 0:
        df["temp_ma"] = df["temperature"].rolling(window, min_periods=1).mean()
        df["temp_roc"] = df["temperature"].diff().fillna(0)
    else:
        df["temp_ma"] = np.nan
        df["temp_roc"] = np.nan

    # SOC
    if df["soc"].notna().sum() > 0:
        df["soc_ma"] = df["soc"].rolling(window, min_periods=1).mean()
        df["soc_roc"] = df["soc"].diff().fillna(0)
    else:
        df["soc_ma"] = np.nan
        df["soc_roc"] = np.nan

    # Cell-level
    cell_cols = [c for c in df.columns if str(c).lower().startswith("cell")]
    if cell_cols:
        cell_vals = df[cell_cols].apply(pd.to_numeric, errors="coerce")
        df["cell_v_max"] = cell_vals.max(axis=1)
        df["cell_v_min"] = cell_vals.min(axis=1)
        df["cell_v_diff"] = df["cell_v_max"] - df["cell_v_min"]
        df["cell_v_std"] = cell_vals.std(axis=1)
    else:
        df["cell_v_max"] = np.nan
        df["cell_v_min"] = np.nan
        df["cell_v_diff"] = np.nan
        df["cell_v_std"] = np.nan

    # Multi-temp (Temp1..N)
    temp_cols = [c for c in df.columns if str(c).lower().startswith("temp")]
    if temp_cols:
        temps = df[temp_cols].apply(pd.to_numeric, errors="coerce")
        df["temp_max"] = temps.max(axis=1)
        df["temp_min"] = temps.min(axis=1)
        df["temp_diff"] = df["temp_max"] - df["temp_min"]
        df["temperature"] = temps.mean(axis=1)
        df["temp_ma"] = df["temperature"].rolling(window, min_periods=1).mean()
        df["temp_roc"] = df["temperature"].diff().fillna(0)
    else:
        df["temp_max"] = df.get("temp_max", np.nan)
        df["temp_min"] = df.get("temp_min", np.nan)
        df["temp_diff"] = df.get("temp_diff", np.nan)

    # Capacity / SOH
    full_cap_col = None
    rem_ah_col = None
    for c in df.columns:
        cl = str(c).lower().replace(" ", "")
        if "fullcap" in cl or "fullcapacity" in cl:
            full_cap_col = c
        if "remah" in cl or "rema" in cl or "remain" in cl:
            rem_ah_col = c

    if full_cap_col is not None:
        fc = pd.to_numeric(df[full_cap_col], errors="coerce")
        if fc.notna().sum() > 0:
            med = fc.median()
            if med > 1000:
                fc_ah = fc / 1000.0
            else:
                fc_ah = fc
        else:
            fc_ah = fc
        df["full_cap_ah"] = fc_ah
        if fc_ah.notna().sum() > 0 and fc_ah.max() > 0:
            df["soh_est"] = (fc_ah / fc_ah.max()) * 100.0
        else:
            df["soh_est"] = np.nan
    else:
        df["full_cap_ah"] = np.nan
        df["soh_est"] = np.nan

    if rem_ah_col is not None:
        ra = pd.to_numeric(df[rem_ah_col], errors="coerce")
        if ra.notna().sum() > 0:
            if ra.median() > 1000:
                ra_ah = ra / 1000.0
            else:
                ra_ah = ra
        else:
            ra_ah = ra
        df["rem_ah"] = ra_ah
    else:
        df["rem_ah"] = np.nan

    if "full_cap_ah" in df.columns and df["full_cap_ah"].notna().sum() > 0 and df["current"].notna().sum() > 0:
        cap_sanit = df["full_cap_ah"].replace(0, np.nan)
        df["c_rate"] = df["current"] / cap_sanit
    else:
        df["c_rate"] = np.nan

    # Risk label
    if df["voltage"].notna().sum() > 0:
        volt_drop_thresh = -0.03
        cond = pd.Series(False, index=df.index)

        if df["temperature"].notna().sum() > 0:
            tmean = df["temperature"].mean()
            tstd = df["temperature"].std()
            tth = tmean + 2 * tstd if not np.isnan(tmean) and not np.isnan(tstd) else np.nan
            if not np.isnan(tth):
                cond = cond | (df["temperature"] > tth)

        if "voltage_roc" in df.columns:
            cond = cond | (df["voltage_roc"] < volt_drop_thresh)

        if "soc_roc" in df.columns:
            cond = cond | (df["soc_roc"] < -5)

        if "cell_v_diff" in df.columns:
            im_diff = df["cell_v_diff"]
            if im_diff.notna().sum() > 0:
                thresh = im_diff.mean() + 2 * im_diff.std()
                cond = cond | (im_diff > thresh)
        df["risk_label"] = np.where(cond, 1, 0)
    else:
        df["risk_label"] = 0

    return df

# ----------------------------
# Models + scoring
# ----------------------------
def run_models(df, contamination=0.05):
    df = df.copy()
    possible = [
        "voltage", "current", "temperature", "soc",
        "voltage_ma", "voltage_roc", "soc_roc",
        "voltage_var", "temp_ma", "cycle",
        "cell_v_diff", "cell_v_std", "temp_diff", "c_rate", "soh_est"
    ]
    features = [f for f in possible if f in df.columns and df[f].notna().sum() > 0]

    df["anomaly_flag"] = 0
    df["risk_pred"] = 0
    df["battery_health_score"] = 50.0

    if len(features) >= 2 and df[features].dropna().shape[0] >= 30:
        try:
            iso = IsolationForest(n_estimators=120, contamination=contamination, random_state=42)
            X = df[features].fillna(df[features].median())
            iso.fit(X)
            df["anomaly_flag"] = iso.predict(X).map({1: 0, -1: 1})
        except Exception:
            df["anomaly_flag"] = 0

    if "risk_label" in df.columns and df["risk_label"].nunique() > 1:
        clf_feats = [f for f in features if f in df.columns]
        if len(clf_feats) >= 2:
            try:
                Xc = df[clf_feats].fillna(df[clf_feats].median())
                yc = df["risk_label"]
                tree = DecisionTreeClassifier(max_depth=4, random_state=42)
                tree.fit(Xc, yc)
                df["risk_pred"] = tree.predict(Xc)
            except Exception:
                df["risk_pred"] = df["risk_label"]
        else:
            df["risk_pred"] = df["risk_label"]
    else:
        tseries = df.get("temperature", pd.Series(np.nan, index=df.index))
        tmean = tseries.mean() if hasattr(tseries, "mean") else np.nan
        tstd = tseries.std() if hasattr(tseries, "std") else np.nan
        tth = tmean + 2 * tstd if not np.isnan(tmean) and not np.isnan(tstd) else np.nan
        cond_temp = (tseries > tth) if not np.isnan(tth) else False
        df["risk_pred"] = np.where((df.get("anomaly_flag", 0) == 1) | cond_temp, 1, 0)

    base = pd.Series(0.0, index=df.index)
    if "voltage_ma" in df.columns and df["voltage_ma"].notna().sum() > 0:
        vm = df["voltage_ma"].fillna(method="ffill").fillna(df["voltage"].median() if "voltage" in df.columns else 3.7)
        base += (vm.max() - vm)
    elif "voltage" in df.columns:
        v = df["voltage"].fillna(df["voltage"].median())
        base += (v.max() - v)
    else:
        base += 0.5

    if "temperature" in df.columns and df["temperature"].notna().sum() > 0:
        t = df["temperature"].fillna(df["temperature"].median())
        base += (t - t.min()) / 10.0

    if "cell_v_diff" in df.columns and df["cell_v_diff"].notna().sum() > 0:
        cd = df["cell_v_diff"].fillna(0)
        base += cd * 20.0

    if "temp_diff" in df.columns and df["temp_diff"].notna().sum() > 0:
        td = df["temp_diff"].fillna(0)
        base += td / 2.0

    base = base + df.get("anomaly_flag", 0) * 1.0 + df.get("risk_pred", 0) * 0.8

    trend_feats = [f for f in ["voltage_ma", "voltage_var", "temp_ma", "cycle", "anomaly_flag", "cell_v_diff", "soh_est"] if f in df.columns]
    if len(trend_feats) >= 2 and df[trend_feats].dropna().shape[0] >= 20:
        try:
            Xtr = df[trend_feats].fillna(0)
            reg = LinearRegression()
            reg.fit(Xtr, base)
            hp = reg.predict(Xtr)
        except Exception:
            hp = base.values
    else:
        hp = base.values

    hp = np.array(hp, dtype=float)
    hp_norm = (hp - hp.min()) / (hp.max() - hp.min() + 1e-9)
    health_comp = 1 - hp_norm

    score = (
        0.55 * health_comp +
        0.2 * (1 - df.get("risk_pred", 0)) +
        0.15 * (1 - df.get("anomaly_flag", 0)) +
        0.1 * (df.get("soh_est", 100) / 100.0).fillna(1.0)
    )
    df["battery_health_score"] = (score * 100).clip(0, 100)

    return df

# ----------------------------
# Recommendations and labels
# ----------------------------
def simple_recommend(row):
    sc = row.get("battery_health_score", 50)
    rp = row.get("risk_pred", 0)
    an = row.get("anomaly_flag", 0)
    imb = row.get("cell_v_diff", 0)

    if sc > 85 and rp == 0 and an == 0 and (pd.isna(imb) or imb < 0.03):
        return "Healthy — normal operation."
    elif 70 < sc <= 85:
        return "Watch — avoid deep discharge & fast-charge in this window."
    elif 50 < sc <= 70:
        return "Caution — reduce fast charging, allow cooling, monitor cell imbalance."
    else:
        return "Critical — reduce load, avoid fast charge, schedule inspection and pack-level testing."

def pack_label(score):
    if score >= 85:
        return "HEALTHY", "green"
    elif score >= 60:
        return "WATCH", "orange"
    else:
        return "CRITICAL", "red"

def make_gauge_figure(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Battery Health Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightcoral"},
                {'range': [60, 85], 'color': "gold"},
                {'range': [85, 100], 'color': "lightgreen"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
    return fig

def basic_alerts(df):
    alerts = []
    if "temp_max" in df.columns and df["temp_max"].notna().sum() > 0:
        tmean = df["temp_max"].mean()
        tstd = df["temp_max"].std()
        recent = df["temp_max"].iloc[-1]
        if recent > (tmean + 2 * tstd):
            alerts.append({
                "title": "Thermal drift",
                "detail": "Temperature well above normal — hotspot risk. Allow cooling and inspect cooling path.",
                "severity": "high"
            })

    if "voltage_roc" in df.columns and "voltage_var" in df.columns:
        last_roc = df["voltage_roc"].rolling(5).mean().iloc[-1]
        last_var = df["voltage_var"].rolling(10).mean().iloc[-1]
        if last_roc < -0.01:
            alerts.append({
                "title": "Voltage sag pattern",
                "detail": "Sustained negative voltage trend — internal resistance may be rising.",
                "severity": "medium"
            })
        if last_var > df["voltage_var"].mean() + df["voltage_var"].std():
            alerts.append({
                "title": "Voltage variance rising",
                "detail": "Cell-to-cell voltage variance is increasing — pack imbalance risk.",
                "severity": "medium"
            })

    if "current" in df.columns and df["current"].notna().sum() > 0:
        spike_pct = (df["current"] > (df["current"].mean() + 2 * df["current"].std())).mean()
        if spike_pct > 0.02:
            alerts.append({
                "title": "Current spikes",
                "detail": "Frequent high current spikes — mechanical / connection stress likely.",
                "severity": "medium"
            })

    if "cell_v_diff" in df.columns and df["cell_v_diff"].notna().sum() > 0:
        latest = df["cell_v_diff"].iloc[-1]
        mean_diff = df["cell_v_diff"].mean()
        std_diff = df["cell_v_diff"].std()
        if latest > mean_diff + 2 * std_diff and latest > 0.05:
            alerts.append({
                "title": "Cell imbalance",
                "detail": f"Latest cell voltage spread is {latest*1000:.0f} mV — balancing or cell replacement may be needed.",
                "severity": "high"
            })

    if "anomaly_flag" in df.columns:
        p = df["anomaly_flag"].mean()
        if p > 0.05:
            alerts.append({
                "title": "Anomaly rate high",
                "detail": f"{p * 100:.1f}% readings flagged as anomalous — investigate patterns and operating conditions.",
                "severity": "medium"
            })

    if "risk_pred" in df.columns and df["risk_pred"].iloc[-1] == 1:
        alerts.append({
            "title": "Immediate risk",
            "detail": "Model predicts elevated risk on latest measurement. Operate conservatively.",
            "severity": "high"
        })
    return alerts

def top_recs_from_df(df, n=5):
    out = []
    try:
        if "battery_health_score" in df.columns:
            worst = df.nsmallest(n, "battery_health_score")
            if "recommendation" in worst.columns:
                rec_counts = worst["recommendation"].value_counts()
                for rec, cnt in rec_counts.items():
                    out.append({"recommendation": rec, "count": int(cnt)})
    except Exception:
        pass
    return out

# ----------------------------
# Cycle failure estimation
# ----------------------------
def estimate_failure_cycle(df, target_score=60.0):
    if "cycle" not in df.columns or "battery_health_score" not in df.columns:
        return None
    if df["cycle"].notna().nunique() < 5:
        return None
    sub = df[["cycle", "battery_health_score"]].dropna()
    if len(sub) < 25:
        return None
    X = sub["cycle"].values.reshape(-1, 1)
    y = sub["battery_health_score"].values
    try:
        reg = LinearRegression()
        reg.fit(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        if slope >= 0:
            return None
        cyc_pred = (target_score - intercept) / slope
        if cyc_pred <= sub["cycle"].max():
            return None
        return float(cyc_pred)
    except Exception:
        return None

# ----------------------------
# PDF report
# ----------------------------
def make_pdf(df_out, avg_score, anomaly_pct, alerts, recs, verdict_text):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    margin = 18 * mm
    x = margin
    y = h - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "CellGuard.AI — Diagnostic Report (Shell Skills4Future)")
    y -= 8 * mm

    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Avg Health Score: {avg_score:.1f}/100")
    c.drawString(x + 80 * mm, y, f"Anomaly Rate: {anomaly_pct:.2f}%")
    y -= 6 * mm
    c.drawString(x, y, f"Data points: {len(df_out)}")
    y -= 8 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Combined Verdict:")
    c.setFont("Helvetica", 11)
    c.drawString(x + 30, y, verdict_text)
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Top AI Alerts:")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    if alerts:
        for a in alerts[:6]:
            c.drawString(x + 6, y, f"- {a['title']}: {a['detail']}")
            y -= 5 * mm
            if y < margin + 40 * mm:
                c.showPage()
                y = h - margin
                c.setFont("Helvetica", 10)
    else:
        c.drawString(x + 6, y, "- None")
        y -= 6 * mm

    y -= 4 * mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Top Recommendations:")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    if recs:
        for r in recs[:6]:
            c.drawString(x + 6, y, f"- {r['recommendation']} (observed {r['count']} times)")
            y -= 5 * mm
            if y < margin + 20 * mm:
                c.showPage()
                y = h - margin
                c.setFont("Helvetica", 10)
    else:
        c.drawString(x + 6, y, "- None")
        y -= 6 * mm

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(margin, margin, "Generated by CellGuard.AI — Shell Skills4Future Demo")
    c.save()
    buf.seek(0)
    return buf.read()

# ----------------------------
# Helpers for plots
# ----------------------------
def safe_plot(fig, key):
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as e:
        st.warning(f"Chart '{key}' failed: {e}")

def chart_explainer(title, text):
    with st.expander(f"ℹ️ About: {title}"):
        st.write(text)

# ----------------------------
# Suggestion box
# ----------------------------
def build_suggestions(df, avg_score, anomaly_pct):
    suggestions = []
    if "cell_v_diff" in df.columns and df["cell_v_diff"].notna().sum() > 0:
        latest_diff = df["cell_v_diff"].iloc[-1]
        mean_diff = df["cell_v_diff"].mean()
        if latest_diff > 0.06:
            suggestions.append("High cell imbalance detected (>{:.0f} mV). Plan balancing and check for a weak cell.".format(latest_diff * 1000))
        elif latest_diff > mean_diff * 1.5:
            suggestions.append("Cell imbalance trending upward. Monitor and avoid aggressive charge/discharge for some cycles.")

    if "soh_est" in df.columns and df["soh_est"].notna().sum() > 0:
        avg_soh = df["soh_est"].mean()
        if avg_soh < 80:
            suggestions.append("Estimated SOH is below 80%. Plan for pack derating or replacement in medium term.")
        elif avg_soh < 90:
            suggestions.append("SOH slightly reduced. Use conservative charging profiles to slow down degradation.")

    if "temp_max" in df.columns and df["temp_max"].notna().sum() > 0:
        tmax = df["temp_max"].max()
        if tmax > 45:
            suggestions.append("Peak temperature above 45°C. Improve cooling or reduce continuous high load.")
        elif tmax > 40:
            suggestions.append("Temperature occasionally high. Avoid charging immediately after heavy discharge.")

    if "c_rate" in df.columns and df["c_rate"].notna().sum() > 0:
        max_c = df["c_rate"].abs().max()
        if max_c > 1.5:
            suggestions.append("High C-rate usage observed (>1.5C). This accelerates ageing — consider limiting peak current.")
        elif max_c > 1.0:
            suggestions.append("Moderate C-rate usage. Keeping C-rate closer to 1C or below will extend life.")

    if anomaly_pct > 5.0:
        suggestions.append("Anomaly rate is >5%. There may be unstable operating conditions or sensor issues.")

    if avg_score < 60:
        suggestions.append("Overall health score is low. Operate in a safe envelope (no fast charge, reduced depth of discharge) and schedule detailed diagnostics.")

    return suggestions

# ----------------------------
# Main app
# ----------------------------
def main():
    st.title("CELLGUARD.AI — Battery Intelligence Dashboard")
    st.caption("Skills4Future x Shell | AI-driven cell analysis, health scoring, and actionable recommendations.")

    st.sidebar.header("Configuration")
    data_mode = st.sidebar.radio("Data source", ["Sample data", "Upload CSV (Shell_BMS style)"])
    scenario = st.sidebar.selectbox("Demo scenario (sample data)", ["Generic", "EV", "Drone", "Phone"])
    contamination = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.2, 0.05, 0.01)
    window = st.sidebar.slider("Rolling window (for features)", 5, 30, 10)
    st.sidebar.markdown("**Default input:** Shell_BMS-style CSV (Date, Time, Pack Vol, Curent, Soc, Cell1..N, Temp1..N, Cycle, etc.)")

    if data_mode == "Sample data":
        df_raw = gen_sample_data(n=800, seed=42, scenario=scenario)
        st.sidebar.success(f"Using simulated data: {scenario}")
    else:
        uploaded = st.sidebar.file_uploader("Upload Shell_BMS-style CSV", type=["csv"])
        if uploaded is None:
            st.warning("Upload a CSV or switch to Sample data.")
            st.stop()
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception:
            try:
                df_raw = pd.read_csv(uploaded, encoding="latin1")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()
        st.sidebar.success("CSV loaded (Shell_BMS format).")
        df_raw = preclean_shell_bms(df_raw)
        df_raw = strong_sanitize(df_raw)

    # Ensure unique columns early
    df_raw = make_unique_columns(df_raw)

    df_raw, col_map = normalize_cols(df_raw)
    df_raw = ensure_cols_exist(df_raw, ["voltage", "current", "temperature", "soc", "cycle", "time"])

    # Make sure timestamp exists
    if "timestamp" in df_raw.columns:
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
        if not df_raw["timestamp"].notna().any():
            df_raw.drop(columns=["timestamp"], inplace=True)
    if "timestamp" not in df_raw.columns:
        try:
            df_raw["timestamp"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df_raw["time"], unit="s")
        except Exception:
            pass

    df_feat = make_features(df_raw, window=window)
    df_out = run_models(df_feat, contamination=contamination)
    df_out["recommendation"] = df_out.apply(simple_recommend, axis=1)

    # Make sure df_out columns are unique too
    df_out = make_unique_columns(df_out)

    # ---------------- Cycle / index filter (no datetime slider) ----------------
    df_view = df_out.copy()
    st.sidebar.subheader("Analysis window")

    if "cycle" in df_view.columns and df_view["cycle"].notna().any():
        cmin = int(df_view["cycle"].min())
        cmax = int(df_view["cycle"].max())
        if cmin == cmax:
            c_start, c_end = cmin, cmax
            st.sidebar.info(f"Only single cycle {cmin} in data.")
        else:
            c_start, c_end = st.sidebar.slider(
                "Select cycle range",
                min_value=cmin,
                max_value=cmax,
                value=(cmin, cmax),
            )
        mask = df_view["cycle"].between(c_start, c_end)
        df_view = df_view.loc[mask].reset_index(drop=True)
        st.sidebar.write(f"Rows in selected cycle window: {len(df_view)}")
    else:
        n_rows = len(df_view)
        if n_rows > 1:
            start_idx, end_idx = st.sidebar.slider(
                "Select row range",
                min_value=0,
                max_value=n_rows - 1,
                value=(0, n_rows - 1),
            )
            df_view = df_view.iloc[start_idx:end_idx+1].reset_index(drop=True)
            st.sidebar.write(f"Rows in selected window: {len(df_view)}")
        else:
            st.sidebar.info("Not enough rows for range selection. Using full data.")

    if len(df_view) == 0:
        st.error("No data in selected window. Widen the range.")
        st.stop()

    # Ensure df_view has unique column names before plotting (for Narwhals)
    df_view = make_unique_columns(df_view)

    avg_score = float(df_view["battery_health_score"].mean()) if "battery_health_score" in df_view.columns and not df_view["battery_health_score"].isnull().all() else 50.0
    anomaly_pct = float(df_view["anomaly_flag"].mean() * 100) if "anomaly_flag" in df_view.columns else 0.0
    label, color = pack_label(avg_score)

    alerts = basic_alerts(df_view)
    recs = top_recs_from_df(df_view, n=8)
    pdf_bytes = make_pdf(df_view, avg_score, anomaly_pct, alerts, recs, "Auto-generated verdict for selected window")
    predicted_cycle = estimate_failure_cycle(df_out)

    # ---------------- Header cards ----------------
    left, mid, right = st.columns([1.4, 1.4, 1.2])
    with left:
        st.markdown("### Battery Health")
        gauge = make_gauge_figure(avg_score)
        safe_plot(gauge, key="gauge_health")
    with mid:
        st.markdown("### Pack Status")
        badge_color = "#2ecc71" if label == "HEALTHY" else ("#f39c12" if label == "WATCH" else "#e74c3c")
        st.markdown(
            f"<span style='background:{badge_color};color:#fff;padding:6px 10px;border-radius:8px;font-weight:600'>{label}</span>",
            unsafe_allow_html=True
        )
        st.metric("Avg Health Score (selected)", f"{avg_score:.1f}/100", delta=f"{(avg_score - 85):.1f} vs ideal")
        st.write(f"- Data points in view: **{len(df_view)}**")
        st.write(f"- AI anomaly rate: **{anomaly_pct:.2f}%**")
        st.write(f"- Mapped columns: {', '.join(list(col_map.keys())) if col_map else 'auto-map not found'}")
    with right:
        st.markdown("### Export")
        st.download_button(
            "⬇️ Download processed CSV (full)",
            df_out.to_csv(index=False).encode("utf-8"),
            "CellGuardAI_Output_Full.csv",
            "text/csv",
            key="download_processed_csv_header_full"
        )
        st.download_button(
            "⬇️ Download window CSV (selected range)",
            df_view.to_csv(index=False).encode("utf-8"),
            "CellGuardAI_Output_SelectedWindow.csv",
            "text/csv",
            key="download_processed_csv_header_window"
        )

    # ---------------- Suggestion box ----------------
    st.markdown("## 🧠 Next-Action Suggestion Box")
    suggs = build_suggestions(df_view, avg_score, anomaly_pct)
    if predicted_cycle is not None:
        suggs.append(
            "Based on trend, health score may drop near the warning threshold around cycle {:.0f}. "
            "This is an estimate — useful for planning preventive maintenance.".format(predicted_cycle)
        )
    if not suggs:
        st.success("No strong issues detected. Maintain current operating pattern and monitor trends regularly.")
    else:
        for s in suggs:
            st.markdown(f"- ✅ {s}")

    # ---------------- Verdict + PDF ----------------
    st.subheader("Combined Verdict and Reports")
    st.write("### Final Verdict (for selected window)")
    if avg_score < 60:
        st.error("Combined verdict: Immediate action recommended. Operate conservatively and schedule inspection.")
    elif avg_score < 75:
        st.warning("Combined verdict: Monitor closely. Use conservative charge/discharge limits.")
    else:
        st.success("Combined verdict: Pack is healthy in this window. Keep monitoring trends.")

    st.download_button(
        "📄 Download PDF Report (selected window)",
        data=pdf_bytes,
        file_name="CellGuardAI_Report_SelectedWindow.pdf",
        mime="application/pdf",
        key="download_pdf_report"
    )

    st.info(
        "For live / real-time use with a running BMS: stream data into a Shell_BMS-style CSV file and let this app reload it. "
        "On Streamlit Cloud, this usually means your logger keeps updating the file and you refresh or rerun the app periodically."
    )

    st.markdown("---")

    # ---------------- Summary metrics ----------------
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        if "temp_max" in df_view.columns and df_view["temp_max"].notna().sum() > 0:
            st.metric("Max Temp (°C)", f"{df_view['temp_max'].max():.2f}")
        else:
            st.metric("Max Temp (°C)", "N/A")
    with s2:
        if "cell_v_diff" in df_view.columns and df_view["cell_v_diff"].notna().sum() > 0:
            st.metric("Avg Cell Imbalance (mV)", f"{(df_view['cell_v_diff'].mean()*1000):.1f}")
        else:
            st.metric("Avg Cell Imbalance (mV)", "N/A")
    with s3:
        if "cycle" in df_view.columns and df_view["cycle"].notna().sum() > 0:
            st.metric("Cycle (max in view)", f"{int(df_view['cycle'].max())}")
        else:
            st.metric("Cycle (max in view)", "N/A")
    with s4:
        st.metric("Anomaly % (view)", f"{anomaly_pct:.2f}%")
    with s5:
        if "risk_pred" in df_view.columns:
            st.metric("Last Risk Pred", "HIGH" if df_view["risk_pred"].iloc[-1] == 1 else "NORMAL")
        else:
            st.metric("Last Risk Pred", "N/A")

    st.markdown("---")

    # ---------------- Tabs for charts ----------------
    tab_ai, tab_trad, tab_cells, tab_table = st.tabs(
        ["CellGuard.AI View", "Traditional BMS View", "Cell-Level View", "Data Table & Export"]
    )

    time_axis = "timestamp" if "timestamp" in df_view.columns else "time"

    with tab_ai:
        st.subheader("AI-Based Battery Insights")
        if "battery_health_score" in df_view.columns:
            fig_h = px.line(df_view, x=time_axis, y="battery_health_score",
                            labels={time_axis: "Time", "battery_health_score": "Health Score"},
                            title="Health Score Over Time")
            safe_plot(fig_h, key="ai_health_timeline")
            chart_explainer(
                "Health Score Over Time",
                "Shows the AI-computed battery health score across the selected window. Drops or spikes highlight stress events, imbalance, or anomalies."
            )

        if "voltage_var" in df_view.columns:
            fig_vv = px.line(df_view, x=time_axis, y="voltage_var",
                             labels={time_axis: "Time", "voltage_var": "Voltage Variance"},
                             title="Voltage Variance (imbalance proxy)")
            safe_plot(fig_vv, key="ai_voltage_var")
            chart_explainer(
                "Voltage Variance",
                "Approximate measure of how stable the voltage is. Higher variance can mean stronger oscillations or cell-to-cell imbalance."
            )

        if "soc" in df_view.columns:
            fig_soc_ai = px.line(df_view, x=time_axis, y="soc",
                                 labels={time_axis: "Time", "soc": "SOC (%)"},
                                 title="SOC Trend")
            safe_plot(fig_soc_ai, key="ai_soc_chart")
            chart_explainer(
                "SOC Trend",
                "State of charge evolution over time. Very aggressive swings between high and low SOC can speed up ageing."
            )

        if "current" in df_view.columns:
            fig_cur = px.line(df_view, x=time_axis, y="current",
                              labels={time_axis: "Time", "current": "Current (A)"},
                              title="Current Flow Over Time")
            safe_plot(fig_cur, key="ai_current_plot")
            chart_explainer(
                "Current Flow",
                "Displays the current drawn or charged. Spikes and long periods at high current indicate higher C-rate stress."
            )

        if "temperature" in df_view.columns:
            fig_temp_hist = px.histogram(df_view, x="temperature",
                                         labels={"temperature": "Temperature (°C)"},
                                         title="Temperature Distribution")
            safe_plot(fig_temp_hist, key="ai_temp_hist")
            chart_explainer(
                "Temperature Distribution",
                "Shows how often the battery operates at different temperatures. Staying within a moderate band is better for lifetime."
            )

        corr_cols = [c for c in ["voltage", "current", "temperature", "soc", "battery_health_score", "cell_v_diff", "soh_est"] if c in df_view.columns]
        if len(corr_cols) >= 2:
            corr = df_view[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            safe_plot(fig_corr, key="ai_corr_heatmap")
            chart_explainer(
                "Correlation Heatmap",
                "Correlation between key parameters. Helps understand how current, temperature, SOC and imbalance relate to health score."
            )

    with tab_trad:
        st.subheader("Traditional BMS View")
        if "voltage" in df_view.columns:
            fig_v = px.line(df_view, x=time_axis, y="voltage",
                            labels={time_axis: "Time", "voltage": "Voltage (V)"},
                            title="Pack Voltage Over Time")
            safe_plot(fig_v, key="trad_voltage_chart")
            chart_explainer(
                "Pack Voltage Over Time",
                "Standard BMS view of pack voltage. On its own it misses deeper imbalance and degradation patterns."
            )

        if "temperature" in df_view.columns:
            fig_t = px.line(df_view, x=time_axis, y="temperature",
                            labels={time_axis: "Time", "temperature": "Temperature (°C)"},
                            title="Average Temperature Over Time")
            safe_plot(fig_t, key="trad_temp_chart")
            chart_explainer(
                "Average Temperature Over Time",
                "Shows the average of available temperature sensors. Peaks indicate thermal stress moments."
            )

        if "soc" in df_view.columns:
            fig_soc_trad = px.line(df_view, x=time_axis, y="soc",
                                   labels={time_axis: "Time", "soc": "SOC (%)"},
                                   title="SOC Over Time (Traditional)")
            safe_plot(fig_soc_trad, key="trad_soc_chart")
            chart_explainer(
                "SOC Over Time (Traditional)",
                "Classic SOC plot a BMS would show. Combined with AI views it gives a more complete picture."
            )

    with tab_cells:
        st.subheader("Cell-Level View")
        cell_cols = [c for c in df_view.columns if str(c).lower().startswith("cell")]
        if cell_cols:
            fig_cells = px.line(df_view, x=time_axis, y=cell_cols,
                                labels={time_axis: "Time", "value": "Cell Voltage"},
                                title="Per-Cell Voltage Over Time")
            safe_plot(fig_cells, key="cell_voltage_chart")
            chart_explainer(
                "Per-Cell Voltage Over Time",
                "Each line is a cell. Diverging lines or one cell consistently lower is a sign of weak or ageing cells."
            )
        else:
            st.info("No explicit cell voltage columns (Cell1..N) found in this dataset.")

        if "cell_v_diff" in df_view.columns:
            fig_imb = px.line(df_view, x=time_axis, y="cell_v_diff",
                               labels={time_axis: "Time", "cell_v_diff": "Cell V Diff (V)"},
                               title="Cell Imbalance Over Time")
            safe_plot(fig_imb, key="cell_imbalance_chart")
            chart_explainer(
                "Cell Imbalance Over Time",
                "Difference between highest and lowest cell voltage in the pack. High or growing imbalance is bad for long-term health."
            )

    with tab_table:
        st.header("Processed Data & Export")
        st.download_button(
            "⬇️ Download full report CSV (all rows)",
            df_out.to_csv(index=False).encode("utf-8"),
            "CellGuardAI_FullReport_AllRows.csv",
            "text/csv",
            key="download_full_report_all"
        )
        st.download_button(
            "⬇️ Download current window CSV (filtered)",
            df_view.to_csv(index=False).encode("utf-8"),
            "CellGuardAI_FullReport_SelectedWindow.csv",
            "text/csv",
            key="download_full_report_window"
        )
        st.dataframe(df_view.head(500), use_container_width=True)
        st.caption("Showing first 500 rows of the current selected window. Download full CSVs above for complete analysis.")

    st.caption("CellGuard.AI — Designed for EV users, battery manufacturers, and academic labs as part of the Shell Skills4Future program.")

if __name__ == "__main__":
    main()
