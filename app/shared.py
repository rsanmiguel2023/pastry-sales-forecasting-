from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RAW = ROOT / "data" / "raw"

ACCENT = {
    "banner_border": "#7B61FF",
    "banner_text": "#f0c040",
    "banner_bg_start": "#0f2440",
    "banner_bg_end": "#1a3660",
}

def load_css(prefix: str) -> None:
    st.markdown(
        f'''
        <style>
        .{prefix}-tip-title {{
            display:flex; align-items:center; gap:0.45rem; margin-bottom:0.35rem;
        }}
        .{prefix}-tip-title h2, .{prefix}-tip-title h3 {{
            margin:0; padding:0; letter-spacing:-0.01em;
        }}
        .{prefix}-tip-title h2 {{ font-size:1.48rem; font-weight:700; }}
        .{prefix}-tip-title h3 {{ font-size:1.28rem; font-weight:650; }}
        .{prefix}-tip {{
            position:relative; display:inline-flex; align-items:center; cursor:help;
        }}
        .{prefix}-tip-icon {{
            width:1.15rem; height:1.15rem; border-radius:999px; display:inline-flex;
            align-items:center; justify-content:center; font-size:0.78rem;
            color:#566; background:#eef2ff; border:1px solid #d8def7;
            user-select:none;
        }}
        .{prefix}-tip-box {{
            visibility:hidden; opacity:0; width:320px;
            background-color:rgba(22, 24, 35, 0.98); color:#f2f4f8;
            text-align:left; border-radius:10px; padding:12px 14px;
            font-size:0.88rem; line-height:1.55;
            position:absolute; z-index:9999;
            bottom:calc(100% + 10px); left:50%; transform:translateX(-50%);
            transition:opacity 0.18s ease; box-shadow:0 10px 30px rgba(0,0,0,0.30);
            pointer-events:none; white-space:normal;
        }}
        .{prefix}-tip-box::after {{
            content:""; position:absolute; top:100%; left:50%; margin-left:-6px;
            border:6px solid transparent; border-top-color:rgba(22, 24, 35, 0.98);
        }}
        .{prefix}-tip:hover .{prefix}-tip-box {{
            visibility:visible; opacity:1;
        }}
        .{prefix}-step-badge {{
            display:inline-block; background:#f6f7ff; border:1px solid #e1e5fb;
            border-radius:999px; padding:7px 12px; margin-bottom:8px;
            font-size:0.72rem; font-weight:700; color:#364b85; letter-spacing:0.07em;
        }}
        .{prefix}-subtle {{
            color:#5f6b7a; font-size:0.95rem;
        }}
        </style>
        ''',
        unsafe_allow_html=True,
    )

def tip_header(label: str, tooltip: str, prefix: str, level: int = 3) -> None:
    parts = tooltip.split("**")
    tip_html = "".join(f"<strong>{p}</strong>" if i % 2 == 1 else p for i, p in enumerate(parts))
    st.markdown(
        f'<div class="{prefix}-tip-title"><h{level}>{label}</h{level}>'
        f'<span class="{prefix}-tip"><span class="{prefix}-tip-icon">i</span>'
        f'<span class="{prefix}-tip-box">{tip_html}</span></span></div>',
        unsafe_allow_html=True,
    )

def what_it_means(text: str, label: str = "What does this mean?") -> None:
    with st.expander(label):
        st.write(text)

def summary_banner(title: str, body: str, border: str | None = None) -> None:
    border = border or ACCENT["banner_border"]
    st.markdown(
        f'''
        <div style="
            background: linear-gradient(135deg, {ACCENT['banner_bg_start']} 0%, {ACCENT['banner_bg_end']} 100%);
            border-left: 5px solid {border};
            border-radius: 12px;
            padding: 22px 26px;
            margin-bottom: 10px;
        ">
            <p style="
                color: {ACCENT['banner_text']};
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin: 0 0 10px 0;
            ">{title}</p>
            <p style="color: #edf1f7; font-size: 1rem; line-height: 1.75; margin: 0;">{body}</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def safe_csv(name: str) -> pd.DataFrame:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_train_df() -> pd.DataFrame:
    path = RAW / "train.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "store" in df.columns and "store_id" not in df.columns:
        df = df.rename(columns={"store": "store_id"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    if "store_id" in df.columns and "store_num" not in df.columns:
        df["store_num"] = pd.factorize(df["store_id"])[0] + 1
    # holiday encodings
    state_map = {"normal_day": 0, "state_holiday": 1, "day_before": 1, "day_after": 1}
    school_map = {"normal_day": 0, "school_holiday": 2}
    special_map = {"normal_day": 0, "special_day": 3, "day_before": 3}
    if "is_state_holiday" in df.columns:
        df["is_state_holiday_num"] = df["is_state_holiday"].map(state_map).fillna(0).astype(int)
    if "is_school_holiday" in df.columns:
        df["is_school_holiday_num"] = df["is_school_holiday"].map(school_map).fillna(0).astype(int)
    if "is_special_day" in df.columns:
        df["is_special_day_num"] = df["is_special_day"].map(special_map).fillna(0).astype(int)
    holiday_cols = [c for c in ["is_state_holiday_num", "is_school_holiday_num", "is_special_day_num"] if c in df.columns]
    if holiday_cols:
        df["is_holiday"] = df[holiday_cols].sum(axis=1)
        df["is_holiday_binary"] = (df["is_holiday"] > 0).astype(int)
    # selected imputations and transforms
    for c in ["unsold", "ordered"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in ["precipitation_sum", "unsold", "ordered"]:
        if c in df.columns:
            series = df[c].astype(float).copy()
            min_val = series.min()
            if pd.notna(min_val) and min_val <= 0:
                series = series + abs(min_val) + 1.0
            df[f"{c}_boxcox"] = (series + 1e-6).map(lambda x: pd.np.log1p(x)) if hasattr(pd, "np") else None
    # lags/rolling
    if "sales" in df.columns:
        if "store_id" in df.columns:
            df = df.sort_values(["store_id", "date"]).copy()
            df["sales_lag_1"] = df.groupby("store_id")["sales"].shift(1)
            df["sales_lag_7"] = df.groupby("store_id")["sales"].shift(7)
            df["sales_rolling_mean_7"] = df.groupby("store_id")["sales"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
            df["sales_rolling_mean_14"] = df.groupby("store_id")["sales"].transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        else:
            df["sales_lag_1"] = df["sales"].shift(1)
            df["sales_lag_7"] = df["sales"].shift(7)
            df["sales_rolling_mean_7"] = df["sales"].shift(1).rolling(7, min_periods=1).mean()
            df["sales_rolling_mean_14"] = df["sales"].shift(1).rolling(14, min_periods=1).mean()
    return df

def metric_card_cols(n: int):
    return st.columns(n)

def footer(text: str) -> None:
    st.caption(text)
