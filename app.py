# app.py
# =============================================================================
# PO Dashboard – Observed Assigned User Analytics (Session-Log Dataset)
#
# Design goals:
# - PO-friendly: prioritize reality and visibility over statistical "cleaning"
# - Visual-first landing (time series + composition)
# - Segmentation & engagement exploration
# - A/B comparison tools separated from operational views
#
# Methodology alignment with your patched A/B approach:
# - Unit of analysis: user_id (randomization unit)
# - "Per user" means per OBSERVED assigned user (users present in session logs)
# - Unconditional metrics (inactive in a window contribute zeros via left join)
# - Windows are computed by days_since_assignment (D7 = 0–6, D30 = 0–29)
# - Winsorization is exposed ONLY in A/B Compare for robustness/inference
#
# Notes on dataset limitations:
# - This is session-log population, not true assignment logs.
# - "Retention / active rate among all assigned users" cannot be fully identified.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Interactive charts with hover tooltips
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    WINDOWS = {
        "d1": (0, 0),     # day 0 only
        "d3": (0, 2),     # days 0-2
        "d7": (0, 6),     # days 0-6
        "d14": (0, 13),   # days 0-13
        "d30": (0, 29),   # days 0-29
    }

    CONTROL = "control"
    TEST = "test"

    DEFAULT_WINSOR_Q = 0.999
    DEFAULT_BOOTSTRAP_N = 2000
    RNG_SEED = 42


# =============================================================================
# HELPERS (safe parsing / formatting)
# =============================================================================

def _lower_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _safe_mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]

def _pct(x: float) -> str:
    return f"{x * 100:,.2f}%"

def _money(x: float, decimals: int = 6) -> str:
    return f"${x:,.{decimals}f}"

def _num(x: float, decimals: int = 2) -> str:
    return f"{x:,.{decimals}f}"


# =============================================================================
# DATA LOADING & CLEANING (patched to reflect your NaN profile)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_clean_data(file_bytes: bytes) -> pd.DataFrame:
    """
    Load CSV and apply robust cleaning based on observed missingness:

    Missingness notes you provided:
    - country has some nulls -> fill with "unknown"
    - first_app_launch_date has a few nulls -> keep, but downstream features must be tolerant
    - ad_revenue/rv_revenue/fs_revenue have many nulls -> fill with 0.0
    - install_store has many nulls -> infer from platform when possible
    - iap_revenue/transactions exist (non-null) but still fill defensively
    """

    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))

    # Hard requirements (minimal set for this dashboard)
    required = ["user_id", "session_id", "open_at", "assignment_date", "cohort", "platform"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize categorical fields
    df["cohort"] = _lower_str(df["cohort"])
    df["platform"] = _lower_str(df["platform"])

    # Country: fill nulls
    if "country" not in df.columns:
        df["country"] = "unknown"
    df["country"] = df["country"].fillna("unknown").astype(str)

    # install_store: create if missing; infer if null
    if "install_store" not in df.columns:
        df["install_store"] = np.nan

    is_ios = df["platform"].eq("ios")
    is_android = df["platform"].eq("android")

    df.loc[is_ios & df["install_store"].isna(), "install_store"] = "apple_app_store"
    df.loc[is_android & df["install_store"].isna(), "install_store"] = "unknown_android_store"
    df["install_store"] = df["install_store"].fillna("unknown").astype(str)

    # Revenue columns: fill NaN with 0
    for col in ["ad_revenue", "rv_revenue", "fs_revenue", "iap_revenue", "iap_transactions"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Engagement columns: fill NaN with 0
    for col in ["session_length", "game_count", "session_number", "payer_status"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    # Watched counts: fill NaN with 0 and cast int if safe
    for col in ["rv_watched", "fs_watched"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int64")

    # Optional categorical fields that are useful for segmentation
    for col in ["campaign", "manufacturer", "model"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    # first_app_launch_date may exist; keep as-is (parsed later)
    if "first_app_launch_date" not in df.columns:
        df["first_app_launch_date"] = pd.NaT

    return df


@st.cache_data(show_spinner=False)
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps in UTC and create day-level features.
    We prioritize assignment_date / open_at for window logic (robust).
    """

    d = df.copy()

    # Parse times
    d["open_at"] = pd.to_datetime(d["open_at"], utc=True, errors="coerce", cache=True)

    # assignment_date is a date-like field in your earlier code; parse tolerant
    d["assignment_date"] = pd.to_datetime(d["assignment_date"], utc=True, errors="coerce", cache=True)

    # first_app_launch_date: can be ISO string with Z; parse tolerant
    d["first_app_launch_date"] = pd.to_datetime(d["first_app_launch_date"], utc=True, errors="coerce", cache=True)

    # Day-level
    d["active_day"] = d["open_at"].dt.floor("D")
    d["assign_day"] = d["assignment_date"].dt.floor("D")
    d["install_day"] = d["first_app_launch_date"].dt.floor("D")

    # Fallback for missing install_day: use first observed active_day
    # (Not perfect, but harmless for this dashboard; windows are assignment-based)
    d.loc[d["install_day"].isna(), "install_day"] = d.loc[d["install_day"].isna(), "active_day"]

    # days since assignment
    d["days_since_assignment"] = (d["active_day"] - d["assign_day"]).dt.days.astype("Int64")

    # post-assignment flag
    d["is_post_assignment"] = (d["active_day"] >= d["assign_day"]).fillna(False).astype(bool)

    return d


# =============================================================================
# ASSIGNED UNIVERSE (patched: deterministic and contamination-safe)
# =============================================================================

@st.cache_data(show_spinner=False)
def build_assigned_universe(sessions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Build observed assigned universe from session logs.
    Drop users with multiple cohort labels to prevent contamination.
    """

    d = sessions.copy()

    cohort_n = d.groupby("user_id")["cohort"].nunique(dropna=True)
    bad_users = cohort_n[cohort_n > 1].index

    n_users_raw = int(d["user_id"].nunique())
    n_bad = int(len(bad_users))

    info = {
        "users_raw": n_users_raw,
        "users_dropped_multi_cohort": n_bad,
        "pct_dropped": (n_bad / n_users_raw * 100) if n_users_raw else 0.0,
    }

    if n_bad > 0:
        d = d[~d["user_id"].isin(bad_users)].copy()

    universe = (
        d.groupby("user_id", as_index=False)
        .agg(
            cohort=("cohort", "first"),
            platform=("platform", _safe_mode),
            country=("country", _safe_mode),
            install_store=("install_store", _safe_mode),
            campaign=("campaign", _safe_mode),
            manufacturer=("manufacturer", _safe_mode),
            model=("model", _safe_mode),
            assignment_date=("assignment_date", "min"),
            first_app_launch_date=("first_app_launch_date", "min"),
        )
    )

    return universe, info


def create_window_mask(df: pd.DataFrame, window: str) -> pd.Series:
    """
    Window mask based on days_since_assignment, inclusive.
    Example: D7 = 0..6
    """
    a, b = Config.WINDOWS[window]
    m = df["is_post_assignment"] & df["days_since_assignment"].between(a, b)
    return m.fillna(False).astype(bool)


# =============================================================================
# USER-LEVEL KPI AGGREGATION (unconditional per observed assigned user)
# =============================================================================

@st.cache_data(show_spinner=False)
def aggregate_user_kpis_unconditional(
    sessions: pd.DataFrame,
    universe: pd.DataFrame,
    window: str
) -> pd.DataFrame:
    """
    Aggregate session-level to user-level KPIs for a given window.
    Unconditional (includes inactive users via left join; inactive users get 0s).
    """

    mask = create_window_mask(sessions, window)
    s = sessions.loc[mask].copy()

    agg = (
        s.groupby("user_id")
        .agg(
            sessions=("session_id", "count"),
            total_session_length=("session_length", "sum"),
            total_game_count=("game_count", "sum"),
            rv_rev=("rv_revenue", "sum"),
            fs_rev=("fs_revenue", "sum"),
            rv_views=("rv_watched", "sum"),
            fs_views=("fs_watched", "sum"),
            iap_rev=("iap_revenue", "sum"),
            iap_transactions=("iap_transactions", "sum"),
            payer_status_max=("payer_status", "max"),
        )
        .reset_index()
    )

    out = universe.merge(agg, on="user_id", how="left")

    numeric_cols = [
        "sessions", "total_session_length", "total_game_count",
        "rv_rev", "fs_rev", "rv_views", "fs_views",
        "iap_rev", "iap_transactions", "payer_status_max"
    ]
    for c in numeric_cols:
        out[c] = out[c].fillna(0)

    # Derived revenue metrics
    out["ad_rev"] = out["rv_rev"] + out["fs_rev"]
    out["iap_rev_net"] = out["iap_rev"] * 0.7
    out["total_rev_net"] = out["ad_rev"] + out["iap_rev_net"]

    # Convenience flags (for segmentation)
    out["is_active"] = (out["sessions"] > 0).astype(int)
    out["rv_adopter"] = (out["rv_views"] > 0).astype(int)
    out["fs_adopter"] = (out["fs_views"] > 0).astype(int)
    out["is_payer"] = (out["payer_status_max"] > 0).astype(int)

    # Intensity metrics (defined only for active users)
    out["avg_session_length"] = out["total_session_length"] / out["sessions"].replace(0, np.nan)
    out["avg_games_per_session"] = out["total_game_count"] / out["sessions"].replace(0, np.nan)

    # eCPM proxies (revenue per watched count)
    out["rv_ecpm"] = 1000 * out["rv_rev"] / out["rv_views"].replace(0, np.nan)
    out["fs_ecpm"] = 1000 * out["fs_rev"] / out["fs_views"].replace(0, np.nan)

    out["window"] = window
    return out


# =============================================================================
# DAILY AGGREGATION (visual-first landing: time series)
# =============================================================================

@st.cache_data(show_spinner=False)
def build_daily_metrics(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily metrics from session logs (no A/B window needed).
    These are operational visuals and do NOT rely on winsorization.
    """

    d = sessions.copy()

    # Filter to rows with a valid active_day
    d = d[d["active_day"].notna()].copy()

    daily = (
        d.groupby("active_day")
        .agg(
            dau=("user_id", "nunique"),  # DAU proxy (observed)
            sessions=("session_id", "count"),
            total_session_length=("session_length", "sum"),
            total_game_count=("game_count", "sum"),
            rv_rev=("rv_revenue", "sum"),
            fs_rev=("fs_revenue", "sum"),
            iap_rev=("iap_revenue", "sum"),
            rv_views=("rv_watched", "sum"),
            fs_views=("fs_watched", "sum"),
        )
        .reset_index()
        .sort_values("active_day")
    )

    daily["ad_rev"] = daily["rv_rev"] + daily["fs_rev"]
    daily["iap_rev_net"] = daily["iap_rev"] * 0.7
    daily["total_rev_net"] = daily["ad_rev"] + daily["iap_rev_net"]

    # Per-user rates (safe)
    daily["sessions_per_dau"] = daily["sessions"] / daily["dau"].replace(0, np.nan)
    daily["rev_per_dau"] = daily["total_rev_net"] / daily["dau"].replace(0, np.nan)
    daily["rv_rev_per_dau"] = daily["rv_rev"] / daily["dau"].replace(0, np.nan)
    daily["fs_rev_per_dau"] = daily["fs_rev"] / daily["dau"].replace(0, np.nan)

    return daily


# =============================================================================
# A/B COMPARE UTILITIES (winsorization + bootstrap CI)
# =============================================================================

def apply_winsorization(df: pd.DataFrame, quantile: float, cols: List[str]) -> pd.DataFrame:
    """
    Dynamic winsorization with zero-inflation safeguard:
    - If the quantile threshold collapses to <=0 while positive values exist,
      recompute threshold on positive values only.
    """
    d = df.copy()
    for col in cols:
        if col not in d.columns:
            continue
        thr = d[col].quantile(quantile)
        if thr <= 0 and (d[col] > 0).any():
            thr = d.loc[d[col] > 0, col].quantile(quantile)
        d[col + "_w"] = d[col].clip(upper=thr)
    return d


def bootstrap_mean_diff_ci(x_c: np.ndarray, x_t: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap CI for mean difference (test - control).
    Returns (diff, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n_c, n_t = len(x_c), len(x_t)
    if n_c == 0 or n_t == 0:
        return np.nan, np.nan, np.nan

    obs = x_t.mean() - x_c.mean()
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        c_s = rng.choice(x_c, size=n_c, replace=True)
        t_s = rng.choice(x_t, size=n_t, replace=True)
        diffs[i] = t_s.mean() - c_s.mean()

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return obs, ci_low, ci_high


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="PO Dashboard (Session-Log A/B Dataset)", layout="wide")

st.title("PO Dashboard – Session-Log A/B Dataset")
st.caption(
    "Operational views show raw business reality (no winsorization). "
    "Winsorization is available only in A/B Compare for robustness/inference."
)

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if not uploaded:
    st.info("Upload the CSV dataset to load the dashboard.")
    st.stop()

# Read bytes for caching stability
file_bytes = uploaded.getvalue()

with st.spinner("Loading and preprocessing data..."):
    raw = load_and_clean_data(file_bytes)
    sessions = create_temporal_features(raw)
    universe, universe_info = build_assigned_universe(sessions)
    daily = build_daily_metrics(sessions)

# Sidebar filters (operational)
st.sidebar.header("Global Filters")

# Date range from daily table
min_day = pd.to_datetime(daily["active_day"].min()).date()
max_day = pd.to_datetime(daily["active_day"].max()).date()
date_range = st.sidebar.date_input("Date range (active_day)", value=(min_day, max_day), min_value=min_day, max_value=max_day)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_day, max_day

# Categorical filters
cohort_opts = sorted(universe["cohort"].astype(str).unique().tolist())
platform_opts = sorted(universe["platform"].astype(str).unique().tolist())
country_opts = sorted(universe["country"].astype(str).unique().tolist())
store_opts = sorted(universe["install_store"].astype(str).unique().tolist())
campaign_opts = sorted(universe["campaign"].astype(str).unique().tolist())

cohort_f = st.sidebar.multiselect("Cohort", cohort_opts, default=cohort_opts)
platform_f = st.sidebar.multiselect("Platform", platform_opts, default=platform_opts)
store_f = st.sidebar.multiselect("Install store", store_opts, default=store_opts)

# Country can be huge; default to top 25 by user count
top_country = (
    universe.groupby("country")["user_id"].nunique().sort_values(ascending=False).head(25).index.tolist()
)
country_f = st.sidebar.multiselect("Country (default: top 25)", country_opts, default=top_country)

campaign_f = st.sidebar.multiselect("Campaign", campaign_opts, default=campaign_opts)

st.sidebar.divider()

# Window selection (used for user-level KPI aggregation)
window = st.sidebar.selectbox("A/B window (days since assignment)", list(Config.WINDOWS.keys()), index=list(Config.WINDOWS.keys()).index("d7"))

# Build user KPIs for the selected window (unconditional per observed assigned user)
with st.spinner("Aggregating user KPIs for selected window..."):
    user_kpis = aggregate_user_kpis_unconditional(sessions, universe, window)

# Apply user-level filters
u = user_kpis.copy()
u = u[
    u["cohort"].astype(str).isin(cohort_f) &
    u["platform"].astype(str).isin(platform_f) &
    u["install_store"].astype(str).isin(store_f) &
    u["country"].astype(str).isin(country_f) &
    u["campaign"].astype(str).isin(campaign_f)
].copy()

# Apply date filter to daily metrics (operational time series)
d = daily.copy()
d = d[
    (d["active_day"].dt.date >= start_date) &
    (d["active_day"].dt.date <= end_date)
].copy()

# Data health / glossary
with st.expander("Data Health, Assumptions, and Column Glossary"):
    st.subheader("Data health summary")
    st.write("**Assigned-universe consistency** (users with multiple cohort labels are dropped):")
    st.json(universe_info)

    st.write("**Raw missing values (top 20 columns)**:")
    st.dataframe(raw.isna().sum().sort_values(ascending=False).head(20), use_container_width=True)

    st.subheader("Key equations used in this dashboard")
    st.markdown(
        """
- **Ad Revenue (user/window)** = `rv_revenue + fs_revenue` (summed over sessions in window)  
- **IAP Net Revenue (user/window)** = `0.7 * iap_revenue` (70% after store fee)  
- **Total Net Revenue (user/window)** = `Ad Revenue + IAP Net Revenue`  
- **DAU (daily proxy)** = `distinct user_id per active_day`  
- **Sessions per DAU** = `sessions / dau`  
- **Revenue per DAU** = `total_rev_net / dau`  
        """
    )

    st.subheader("Raw data column glossary (provided)")
    st.markdown(
        """
- **country**: user location (IP or store region)  
- **first_app_launch_date**: first time user launched the app  
- **fs_revenue**: revenue from full-screen/interstitial ads  
- **fs_watched**: number of full-screen ads watched  
- **game_count**: number of waves played in the session  
- **iap_revenue**: gross in-app purchase revenue (net = 0.7 × gross)  
- **iap_transactions**: number of IAP transactions  
- **install_store**: store where the app was installed  
- **manufacturer / model**: device information  
- **open_at**: session start/open timestamp  
- **platform**: iOS / Android  
- **ad_revenue**: overall ad revenue (multiple formats)  
- **rv_revenue**: rewarded video revenue  
- **rv_watched**: rewarded videos watched  
- **session_id**: unique session identifier  
- **session_length**: session length  
- **session_number**: session sequence for the user  
- **user_id**: unique user identifier  
- **campaign**: acquisition campaign type  
- **assignment_date**: date/time assigned to A/B cohort  
- **cohort**: control or test  
- **payer_status**: 0 non-payer, 1 payer  
        """
    )

# Tabs
tabs = st.tabs(["Home (Visual Overview)", "Monetization", "Engagement", "Segments", "A/B Compare (Robustness)"])


# =============================================================================
# TAB 1: HOME (VISUAL OVERVIEW)
# =============================================================================

with tabs[0]:
    st.subheader("Home – Visual Overview (Operational Time Series)")

    cA, cB, cC, cD = st.columns(4)

    # Summary metrics from filtered daily range (operational)
    dau_avg = float(d["dau"].mean()) if len(d) else 0.0
    sess_per_dau = float(d["sessions_per_dau"].mean()) if len(d) else 0.0
    rev_per_dau = float(d["rev_per_dau"].mean()) if len(d) else 0.0
    total_rev = float(d["total_rev_net"].sum()) if len(d) else 0.0

    cA.metric("Avg DAU (proxy)", f"{dau_avg:,.0f}")
    with cA.popover("ℹ️"):
        st.markdown("**DAU (proxy)** = distinct `user_id` per `active_day`.")

    cB.metric("Avg Sessions / DAU", _num(sess_per_dau, 2))
    with cB.popover("ℹ️"):
        st.markdown("**Sessions / DAU** = `sessions / dau` (daily), averaged over selected dates.")

    cC.metric("Avg Revenue / DAU (Net)", _money(rev_per_dau, 6))
    with cC.popover("ℹ️"):
        st.markdown("**Revenue / DAU (Net)** = `total_rev_net / dau`, where:")
        st.markdown("- `total_rev_net = (rv_rev + fs_rev) + 0.7 * iap_rev`")

    cD.metric("Total Revenue (Net)", _money(total_rev, 2))
    with cD.popover("ℹ️"):
        st.markdown("Sum of daily **Total Net Revenue** over selected dates.")

    st.divider()

    if not PLOTLY_OK:
        st.warning("Plotly is not available. Install `plotly` for interactive charts with hover tooltips.")
    else:
        # Time series: DAU & sessions
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=d["active_day"], y=d["dau"], mode="lines", name="DAU (proxy)"))
        fig1.add_trace(go.Scatter(x=d["active_day"], y=d["sessions"], mode="lines", name="Sessions"))
        fig1.update_layout(
            title="DAU (proxy) and Sessions over time",
            xaxis_title="active_day",
            yaxis_title="count",
            hovermode="x unified"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Time series: Revenue per DAU
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=d["active_day"], y=d["rev_per_dau"], mode="lines", name="Revenue/DAU (Net)"))
        fig2.add_trace(go.Scatter(x=d["active_day"], y=d["sessions_per_dau"], mode="lines", name="Sessions/DAU"))
        fig2.update_layout(
            title="Revenue per DAU and Sessions per DAU over time",
            xaxis_title="active_day",
            yaxis_title="value",
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Stacked area: revenue components
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=d["active_day"], y=d["rv_rev"], stackgroup="one", name="RV revenue"))
        fig3.add_trace(go.Scatter(x=d["active_day"], y=d["fs_rev"], stackgroup="one", name="FS revenue"))
        fig3.add_trace(go.Scatter(x=d["active_day"], y=d["iap_rev_net"], stackgroup="one", name="IAP net revenue"))
        fig3.update_layout(
            title="Revenue composition over time (stacked)",
            xaxis_title="active_day",
            yaxis_title="revenue",
            hovermode="x unified"
        )
        st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# TAB 2: MONETIZATION (RAW, PO-REALITY)
# =============================================================================

with tabs[1]:
    st.subheader(f"Monetization – User/window view (Window: {window.upper()})")

    # Core: show both mean and median to expose heavy-tail reality (instead of hiding it)
    users_n = int(u["user_id"].nunique())

    mean_total = float(u["total_rev_net"].mean())
    med_total = float(u["total_rev_net"].median())
    mean_ad = float(u["ad_rev"].mean())
    mean_iap = float(u["iap_rev_net"].mean())

    zero_total = float((u["total_rev_net"] == 0).mean())
    zero_rv = float((u["rv_rev"] == 0).mean())
    zero_fs = float((u["fs_rev"] == 0).mean())
    zero_iap = float((u["iap_rev_net"] == 0).mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users (filtered)", f"{users_n:,}")
    c2.metric("Total Rev/User (mean)", _money(mean_total, 6))
    with c2.popover("ℹ️"):
        st.markdown("**Total Rev/User (mean)** = mean over users of:")
        st.markdown("`total_rev_net = (rv_rev + fs_rev) + 0.7 * iap_rev`")

    c3.metric("Total Rev/User (median)", _money(med_total, 6))
    with c3.popover("ℹ️"):
        st.markdown("Median of per-user `total_rev_net`.")
        st.markdown("Useful when distributions are heavy-tailed or zero-inflated.")

    c4.metric("Ad Rev/User (mean)", _money(mean_ad, 6))
    with c4.popover("ℹ️"):
        st.markdown("**Ad Rev/User** = mean over users of:")
        st.markdown("`ad_rev = rv_rev + fs_rev`")

    c5.metric("IAP Rev/User (net mean)", _money(mean_iap, 6))
    with c5.popover("ℹ️"):
        st.markdown("**IAP Net Rev/User** = mean over users of:")
        st.markdown("`iap_rev_net = 0.7 * iap_rev`")

    st.divider()

    z1, z2, z3, z4 = st.columns(4)
    z1.metric("% users with total_rev=0", _pct(zero_total))
    z2.metric("% users with rv_rev=0", _pct(zero_rv))
    z3.metric("% users with fs_rev=0", _pct(zero_fs))
    z4.metric("% users with iap_net=0", _pct(zero_iap))

    st.divider()

    if PLOTLY_OK:
        # Distribution: total_rev_net and ad_rev
        left, right = st.columns(2)
        with left:
            fig = px.histogram(u, x="total_rev_net", nbins=60, title="Distribution: Total Net Revenue per User")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.histogram(u, x="ad_rev", nbins=60, title="Distribution: Ad Revenue per User (RV + FS)")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)

        # Revenue mix pie (sum)
        mix = pd.DataFrame({
            "component": ["RV", "FS", "IAP (Net)"],
            "value": [u["rv_rev"].sum(), u["fs_rev"].sum(), u["iap_rev_net"].sum()]
        })
        fig = px.pie(mix, values="value", names="component", title="Revenue Mix (sum over filtered users)")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Install plotly for interactive charts with hover tooltips: `pip install plotly`.")


# =============================================================================
# TAB 3: ENGAGEMENT (RAW, PO-REALITY)
# =============================================================================

with tabs[2]:
    st.subheader(f"Engagement – User/window view (Window: {window.upper()})")
    st.caption("Operational engagement should be shown raw. Heavy-tail effects are visible via mean vs median where relevant.")

    mean_sessions = float(u["sessions"].mean())
    med_sessions = float(u["sessions"].median())
    active_share = float(u["is_active"].mean())

    mean_len = float(u["total_session_length"].mean())
    mean_games = float(u["total_game_count"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active share (proxy)", _pct(active_share))
    with c1.popover("ℹ️"):
        st.markdown("**Active share (proxy)** = % of users with `sessions > 0` in the selected window.")

    c2.metric("Sessions/User (mean)", _num(mean_sessions, 2))
    c3.metric("Sessions/User (median)", _num(med_sessions, 2))
    c4.metric("Total Session Length/User (mean)", _num(mean_len, 2))
    c5.metric("Game Count/User (mean)", _num(mean_games, 2))

    st.divider()

    if PLOTLY_OK:
        # Sessions distribution
        fig = px.histogram(u, x="sessions", nbins=60, title="Distribution: Sessions per User")
        st.plotly_chart(fig, use_container_width=True)

        # Session length distribution
        fig = px.histogram(u, x="total_session_length", nbins=60, title="Distribution: Total Session Length per User")
        st.plotly_chart(fig, use_container_width=True)

        # Behavior vs monetization
        scatter = u[["sessions", "total_rev_net"]].copy()
        fig = px.scatter(scatter, x="sessions", y="total_rev_net", title="Sessions vs Total Net Revenue (per user)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

        corr = scatter.corr().iloc[0, 1] if len(scatter) > 2 else np.nan
        st.metric("Correlation: sessions vs total_rev_net", f"{corr:.4f}")

    else:
        st.info("Install plotly for interactive charts with hover tooltips: `pip install plotly`.")


# =============================================================================
# TAB 4: SEGMENTS (RAW, PO-REALITY + VISUAL)
# =============================================================================

with tabs[3]:
    st.subheader(f"Segments – Explore drivers (Window: {window.upper()})")
    st.caption("Segment tables show raw means/medians + zero share. This is usually what a PO needs.")

    dim = st.selectbox(
        "Segment dimension",
        ["country", "platform", "install_store", "campaign", "manufacturer", "model"],
        index=0
    )

    metric = st.selectbox(
        "Metric",
        ["total_rev_net", "ad_rev", "rv_rev", "fs_rev", "iap_rev_net", "sessions", "total_session_length", "total_game_count"],
        index=0
    )

    seg = (
        u.groupby(dim)
        .agg(
            users=("user_id", "nunique"),
            mean=(metric, "mean"),
            median=(metric, "median"),
            p90=(metric, lambda x: x.quantile(0.9)),
            zero_share=(metric, lambda x: (x == 0).mean()),
            sum=(metric, "sum"),
        )
        .reset_index()
        .sort_values("users", ascending=False)
    )

    st.dataframe(seg, use_container_width=True)

    if PLOTLY_OK:
        topn = st.slider("Top N segments for charts", 5, 50, 20)

        # Bar: mean metric
        seg_top = seg.sort_values("mean", ascending=False).head(topn)
        fig = px.bar(seg_top, x=dim, y="mean", title=f"Top {topn} segments by MEAN of {metric}", hover_data=["users", "median", "zero_share"])
        fig.update_layout(xaxis_tickangle=-35, hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

        # Bubble: users vs mean metric
        fig = px.scatter(
            seg.head(300),  # cap for performance if dimension is huge
            x="users",
            y="mean",
            size="sum",
            hover_name=dim,
            hover_data=["median", "zero_share", "sum"],
            title=f"Segment map: users vs mean({metric}) (bubble size = sum)"
        )
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Install plotly for interactive charts with hover tooltips: `pip install plotly`.")


# =============================================================================
# TAB 5: A/B COMPARE (ROBUSTNESS / INFERENCE)
# =============================================================================

with tabs[4]:
    st.subheader(f"A/B Compare – Robustness tools (Window: {window.upper()})")
    st.caption(
        "This tab is the only place where winsorization is used, because it is an inference/robustness tool. "
        "Operational tabs should remain raw."
    )

    kpi = st.selectbox(
        "KPI for comparison",
        ["total_rev_net", "ad_rev", "rv_rev", "fs_rev", "iap_rev_net", "sessions", "total_session_length", "total_game_count", "iap_transactions"],
        index=0
    )

    colA, colB, colC = st.columns([1, 1, 2])

    use_w = colA.checkbox("Use winsorization", value=True)
    winsor_q = colB.slider("Winsor quantile", 0.95, 0.9999, Config.DEFAULT_WINSOR_Q, step=0.0001)
    n_boot = colC.slider("Bootstrap iterations (CI)", 500, 5000, Config.DEFAULT_BOOTSTRAP_N, step=500)

    metric_col = kpi

    df_ab = u[["cohort", metric_col]].copy()

    if use_w:
        df_ab = apply_winsorization(df_ab, winsor_q, cols=[metric_col])
        metric_col = metric_col + "_w"

    # Summary table
    ab = (
        df_ab.groupby("cohort")
        .agg(
            n=(metric_col, "count"),
            mean=(metric_col, "mean"),
            median=(metric_col, "median"),
            p90=(metric_col, lambda x: x.quantile(0.9)),
            zero_share=(metric_col, lambda x: (x == 0).mean()),
        )
        .reset_index()
    )

    st.dataframe(ab, use_container_width=True)

    # Lift + CI
    cohorts_present = set(ab["cohort"].astype(str).tolist())
    if Config.CONTROL in cohorts_present and Config.TEST in cohorts_present:
        c = df_ab[df_ab["cohort"] == Config.CONTROL][metric_col].to_numpy()
        t = df_ab[df_ab["cohort"] == Config.TEST][metric_col].to_numpy()

        mean_c = float(np.mean(c)) if len(c) else np.nan
        mean_t = float(np.mean(t)) if len(t) else np.nan
        diff = mean_t - mean_c
        rel = (diff / mean_c * 100) if (mean_c is not None and mean_c != 0) else np.nan

        diff_obs, ci_low, ci_high = bootstrap_mean_diff_ci(c, t, n_boot=n_boot, seed=Config.RNG_SEED)

        st.metric("Mean lift (Test - Control)", f"{diff:+.6f}", delta=f"{rel:+.2f}%" if np.isfinite(rel) else "NA")

        st.write("**Bootstrap 95% CI (mean diff)**")
        st.write(f"[{ci_low:+.6f}, {ci_high:+.6f}]")

        with st.popover("ℹ️"):
            st.markdown(
                """
**Comparison definition**
- Lift = `mean(test) - mean(control)` on selected KPI  
- If winsorization is enabled, the KPI is clipped at the chosen quantile (per selected population)  
- CI is bootstrap percentile CI on mean differences  
                """
            )

        if PLOTLY_OK:
            # Visual: distribution by cohort
            fig = px.histogram(df_ab, x=metric_col, color="cohort", barmode="overlay", nbins=60,
                               title=f"Distribution of {kpi} by cohort ({'winsorized' if use_w else 'raw'})")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Both control and test cohorts are not present after filtering. Adjust filters to compare.")


# Footer note
st.divider()
st.caption(
    "Implementation note: This dashboard uses the session-log population. "
    "True assignment logs would be required to measure retention/active rate on the full assigned universe."
)
