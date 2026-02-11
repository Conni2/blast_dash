# app.py
# =============================================================================
# 💣 This is Blast! – PO Dashboard (Session-Log A/B Dataset)
#
# Local-first version:
# - No upload (large CSV)
# - Reads from a local relative path by default
# - Optional Parquet cache for faster reloads (safe fallback if unavailable)
#
# How to read this dashboard:
# 🏠 Home (Overview)
#   - Uses FULL dataset
#   - Only one filter: date range (active_day)
#   - Purpose: quick “what’s going on?” snapshot for PO
#
# 📊 Other tabs (Monetization / Engagement / Segments / A/B Compare)
#   - Uses Global Filters (country/store/platform/campaign/payer/cohort)
#   - User-level KPIs computed by assignment window (D7 etc.)
#
# Robustness:
# - Winsorization is available ONLY in A/B Compare (inference / robustness tool)
#
# Dataset limitation reminder:
# - This is session-log population, not true assignment logs.
# - True retention on full assigned universe cannot be identified without assignment logs.
# =============================================================================

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Interactive charts with hover tooltips
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    DEFAULT_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "Voodoo Case Study - Game Analyst.csv")

    WINDOWS = {
        "d1": (0, 0),
        "d3": (0, 2),
        "d7": (0, 6),
        "d14": (0, 13),
        "d30": (0, 29),
    }

    CONTROL = "control"
    TEST = "test"

    DEFAULT_WINSOR_Q = 0.999
    DEFAULT_BOOTSTRAP_N = 2000
    RNG_SEED = 42

    # For Home breakdown charts
    HOME_TOPN_DEFAULT = 10


# =============================================================================
# HELPERS
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

def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def _parquet_path_for(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return base + ".parquet"


# =============================================================================
# CSV READ SETTINGS (memory-friendly)
# =============================================================================

USECOLS = [
    "country", "first_app_launch_date",
    "fs_revenue", "fs_watched",
    "game_count",
    "iap_revenue", "iap_transactions",
    "install_store",
    "manufacturer", "model",
    "open_at",
    "platform",
    "ad_revenue",
    "rv_revenue", "rv_watched",
    "session_id",
    "session_length",
    "session_number",
    "user_id",
    "campaign",
    "assignment_date",
    "cohort",
    "payer_status"
]

DTYPE_HINTS = {
    "country": "string",
    "install_store": "string",
    "manufacturer": "string",
    "model": "string",
    "platform": "string",
    "cohort": "string",
    "campaign": "string",
    "session_id": "string",
    "user_id": "string",
    "fs_revenue": "float64",
    "rv_revenue": "float64",
    "ad_revenue": "float64",
    "iap_revenue": "float64",
    "iap_transactions": "float64",
    "session_length": "float64",
    "game_count": "float64",
    "session_number": "float64",
    "fs_watched": "float64",
    "rv_watched": "float64",
    "payer_status": "float64",
}


# =============================================================================
# LOAD + CACHE (CSV -> optional Parquet)
# =============================================================================

@st.cache_data(show_spinner=False)
def read_csv_from_path(path: str) -> pd.DataFrame:
    """Read CSV with column pruning + dtype hints."""
    compression = "gzip" if path.lower().endswith(".gz") else None

    header = pd.read_csv(path, nrows=0, compression=compression)
    existing_cols = header.columns.tolist()

    usecols = [c for c in USECOLS if c in existing_cols]
    dtype = {k: v for k, v in DTYPE_HINTS.items() if k in usecols}

    df = pd.read_csv(
        path,
        compression=compression,
        usecols=usecols,
        dtype=dtype,
        low_memory=False
    )
    return df


def try_read_parquet(path_parquet: str) -> Tuple[bool, pd.DataFrame]:
    """
    Try to read parquet. Returns (ok, df).
    Must not crash the app if pyarrow/parquet isn't available.
    """
    try:
        df = pd.read_parquet(path_parquet)
        return True, df
    except Exception:
        return False, pd.DataFrame()


def try_write_parquet(df: pd.DataFrame, path_parquet: str) -> bool:
    """Try to write parquet safely (no crash if pyarrow isn't available)."""
    try:
        df.to_parquet(path_parquet, index=False)
        return True
    except Exception:
        return False


def load_raw_with_parquet_cache(csv_path: str, rebuild: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Parquet caching strategy:
    - If parquet exists and is newer than csv and rebuild=False -> load parquet
    - Else -> read csv and (try) write parquet
    Safe fallback: if parquet read/write fails, keep working with csv.
    """
    pq_path = _parquet_path_for(csv_path)
    csv_mtime = _file_mtime(csv_path)
    pq_mtime = _file_mtime(pq_path)

    info = {
        "csv_path": csv_path,
        "parquet_path": pq_path,
        "loaded_from": None,
        "parquet_written": False,
        "parquet_available": False,
    }

    # Load from parquet if valid
    if (not rebuild) and os.path.exists(pq_path) and (pq_mtime >= csv_mtime):
        ok, df = try_read_parquet(pq_path)
        if ok and not df.empty:
            info["loaded_from"] = "parquet"
            info["parquet_available"] = True
            return df, info

    # Fallback to CSV
    df = read_csv_from_path(csv_path)
    info["loaded_from"] = "csv"

    # Try write parquet
    if try_write_parquet(df, pq_path):
        info["parquet_written"] = True
        info["parquet_available"] = True

    return df, info


# =============================================================================
# CLEAN + FEATURE ENGINEERING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    required = ["user_id", "session_id", "open_at", "assignment_date", "cohort", "platform"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d["cohort"] = _lower_str(d["cohort"])
    d["platform"] = _lower_str(d["platform"])

    if "country" not in d.columns:
        d["country"] = "unknown"
    d["country"] = d["country"].fillna("unknown").astype(str)

    if "install_store" not in d.columns:
        d["install_store"] = np.nan

    is_ios = d["platform"].eq("ios")
    is_android = d["platform"].eq("android")
    d.loc[is_ios & d["install_store"].isna(), "install_store"] = "apple_app_store"
    d.loc[is_android & d["install_store"].isna(), "install_store"] = "unknown_android_store"
    d["install_store"] = d["install_store"].fillna("unknown").astype(str)

    for col in ["ad_revenue", "rv_revenue", "fs_revenue", "iap_revenue", "iap_transactions"]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = d[col].fillna(0.0)

    for col in ["session_length", "game_count", "session_number", "payer_status"]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = d[col].fillna(0.0)

    for col in ["rv_watched", "fs_watched"]:
        if col not in d.columns:
            d[col] = 0
        d[col] = d[col].fillna(0).astype("int64", errors="ignore")

    for col in ["campaign", "manufacturer", "model"]:
        if col not in d.columns:
            d[col] = "unknown"
        d[col] = d[col].fillna("unknown").astype(str)

    if "first_app_launch_date" not in d.columns:
        d["first_app_launch_date"] = pd.NaT

    return d


@st.cache_data(show_spinner=False)
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["open_at"] = pd.to_datetime(d["open_at"], utc=True, errors="coerce", cache=True)
    d["assignment_date"] = pd.to_datetime(d["assignment_date"], utc=True, errors="coerce", cache=True)
    d["first_app_launch_date"] = pd.to_datetime(d["first_app_launch_date"], utc=True, errors="coerce", cache=True)

    d["active_day"] = d["open_at"].dt.floor("D")
    d["assign_day"] = d["assignment_date"].dt.floor("D")
    d["install_day"] = d["first_app_launch_date"].dt.floor("D")

    # install_day fallback: first observed active_day
    d.loc[d["install_day"].isna(), "install_day"] = d.loc[d["install_day"].isna(), "active_day"]

    d["days_since_assignment"] = (d["active_day"] - d["assign_day"]).dt.days.astype("Int64")
    d["is_post_assignment"] = (d["active_day"] >= d["assign_day"]).fillna(False).astype(bool)

    return d


# =============================================================================
# ASSIGNED UNIVERSE (drop multi-cohort users)
# =============================================================================

@st.cache_data(show_spinner=False)
def build_assigned_universe(sessions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
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
            install_day=("install_day", "min"),
        )
    )

    payer_user = d.groupby("user_id", as_index=False).agg(payer_status_max=("payer_status", "max"))
    universe = universe.merge(payer_user, on="user_id", how="left")
    universe["payer_status_max"] = universe["payer_status_max"].fillna(0)
    universe["payer_segment"] = np.where(universe["payer_status_max"] > 0, "payer", "non_payer")

    return universe, info


def create_window_mask(df: pd.DataFrame, window: str) -> pd.Series:
    a, b = Config.WINDOWS[window]
    m = df["is_post_assignment"] & df["days_since_assignment"].between(a, b)
    return m.fillna(False).astype(bool)


# =============================================================================
# USER KPI (unconditional per observed assigned user)
# =============================================================================

@st.cache_data(show_spinner=False)
def aggregate_user_kpis_unconditional(sessions: pd.DataFrame, universe: pd.DataFrame, window: str) -> pd.DataFrame:
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
            payer_status_max_window=("payer_status", "max"),
        )
        .reset_index()
    )

    out = universe.merge(agg, on="user_id", how="left")

    numeric_cols = [
        "sessions", "total_session_length", "total_game_count",
        "rv_rev", "fs_rev", "rv_views", "fs_views",
        "iap_rev", "iap_transactions", "payer_status_max_window"
    ]
    for c in numeric_cols:
        out[c] = out[c].fillna(0)

    out["ad_rev"] = out["rv_rev"] + out["fs_rev"]
    out["iap_rev_net"] = out["iap_rev"] * 0.7
    out["total_rev_net"] = out["ad_rev"] + out["iap_rev_net"]

    out["is_active"] = (out["sessions"] > 0).astype(int)
    out["rv_adopter"] = (out["rv_views"] > 0).astype(int)
    out["fs_adopter"] = (out["fs_views"] > 0).astype(int)

    out["avg_session_length"] = out["total_session_length"] / out["sessions"].replace(0, np.nan)
    out["avg_games_per_session"] = out["total_game_count"] / out["sessions"].replace(0, np.nan)

    out["rv_ecpm"] = 1000 * out["rv_rev"] / out["rv_views"].replace(0, np.nan)
    out["fs_ecpm"] = 1000 * out["fs_rev"] / out["fs_views"].replace(0, np.nan)

    out["payer_segment"] = out["payer_segment"].fillna("non_payer")
    out["window"] = window
    return out


# =============================================================================
# OVERVIEW (daily metrics + breakdown + installs)
# =============================================================================

@st.cache_data(show_spinner=False)
def build_daily_metrics(sessions: pd.DataFrame) -> pd.DataFrame:
    d = sessions.copy()
    d = d[d["active_day"].notna()].copy()

    daily = (
        d.groupby("active_day")
        .agg(
            dau=("user_id", "nunique"),
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
    daily["iap_rev_net"] = 0.7 * daily["iap_rev"]
    daily["total_rev_net"] = daily["ad_rev"] + daily["iap_rev_net"]

    daily["sessions_per_dau"] = daily["sessions"] / daily["dau"].replace(0, np.nan)
    daily["rev_per_dau"] = daily["total_rev_net"] / daily["dau"].replace(0, np.nan)

    return daily


@st.cache_data(show_spinner=False)
def build_daily_breakdown_active(sessions: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """
    Daily breakdown: active users (by active_day) split by user attributes.
    """
    s = sessions[sessions["active_day"].notna()][["active_day", "user_id"]].drop_duplicates()
    s = s.merge(
        universe[["user_id", "country", "install_store", "platform", "campaign", "payer_segment"]],
        on="user_id",
        how="left"
    )

    out = (
        s.groupby(["active_day", "country", "install_store", "platform", "campaign", "payer_segment"])
         .agg(active_users=("user_id", "nunique"))
         .reset_index()
    )
    return out


@st.cache_data(show_spinner=False)
def build_daily_installs(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Install proxy: distinct users by install_day (from first_app_launch_date, with fallback).
    """
    u = universe.copy()
    u = u[u["install_day"].notna()].copy()

    daily = (
        u.groupby("install_day")
         .agg(new_users=("user_id", "nunique"))
         .reset_index()
         .rename(columns={"install_day": "day"})
         .sort_values("day")
    )
    daily["cum_new_users"] = daily["new_users"].cumsum()
    return daily


@st.cache_data(show_spinner=False)
def build_daily_install_breakdown(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Install proxy breakdown: new users per install_day by dimension.
    """
    u = universe.copy()
    u = u[u["install_day"].notna()].copy()
    out = (
        u.groupby(["install_day", "country", "install_store", "platform", "campaign", "payer_segment"])
         .agg(new_users=("user_id", "nunique"))
         .reset_index()
         .rename(columns={"install_day": "day"})
    )
    return out


# =============================================================================
# A/B COMPARE UTILITIES
# =============================================================================

def apply_winsorization(df: pd.DataFrame, quantile: float, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for col in cols:
        if col not in d.columns:
            continue
        thr = d[col].quantile(quantile)
        if thr <= 0 and (d[col] > 0).any():
            thr = d.loc[d[col] > 0, col].quantile(quantile)
        d[col + "_w"] = d[col].clip(upper=thr)
    return d


def bootstrap_mean_diff_ci(x_c: np.ndarray, x_t: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
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
# UI
# =============================================================================

st.set_page_config(page_title="💣 This is Blast! Dashboard", layout="wide")

st.title("💣 This is Blast! – PO Analytics Dashboard")
st.caption("✨ Home is your quick snapshot. Other tabs are for deeper dives with filters. (Hover charts to see numbers!)")

# ----------------------------
# Sidebar: data source + parquet cache
# ----------------------------
st.sidebar.header("📁 Data Source")

path = st.sidebar.text_input(
    "Local CSV path (.csv or .csv.gz)",
    value=Config.DEFAULT_LOCAL_PATH,
    help="Tip: Keep the CSV next to app.py, or provide an absolute path."
)

path = os.path.abspath(os.path.expanduser(path))
if not os.path.exists(path):
    st.error("File not found. Please check the path.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("⚡ Fast Reload (Parquet Cache)")

rebuild_parquet = st.sidebar.button(
    "Rebuild Parquet Cache",
    help="Forces re-creation of the parquet file from CSV. Useful if you suspect cache is stale."
)

# Load raw (parquet preferred, safe fallback)
with st.spinner("Loading data… (first run can be slow, later runs are much faster 💨)"):
    df_raw, cache_info = load_raw_with_parquet_cache(path, rebuild=rebuild_parquet)
    raw = load_and_clean_data(df_raw)
    sessions = create_temporal_features(raw)
    universe, universe_info = build_assigned_universe(sessions)

    daily_all = build_daily_metrics(sessions)
    bd_active_all = build_daily_breakdown_active(sessions, universe)
    installs_all = build_daily_installs(universe)
    bd_install_all = build_daily_install_breakdown(universe)

with st.sidebar.expander("Cache status", expanded=False):
    st.write(cache_info)

# ----------------------------
# Sidebar: Data Health
# ----------------------------
with st.sidebar.expander("🧪 Data Health", expanded=False):
    st.write("Multi-cohort users are dropped to avoid contamination.")
    st.json(universe_info)
    st.write("Raw missing values (top 20 columns):")
    st.dataframe(raw.isna().sum().sort_values(ascending=False).head(20), use_container_width=True)

st.sidebar.divider()

# ----------------------------
# Sidebar: Global Filters (NOT used in Home)
# ----------------------------
st.sidebar.header("🎛️ Global Filters (Non-Home tabs)")

window = st.sidebar.selectbox(
    "User KPI window (days since assignment)",
    list(Config.WINDOWS.keys()),
    index=list(Config.WINDOWS.keys()).index("d7"),
    help="Defines post-assignment window: e.g., D7 means days_since_assignment ∈ [0, 6]."
)

cohort_opts = sorted(universe["cohort"].astype(str).unique().tolist())
platform_opts = sorted(universe["platform"].astype(str).unique().tolist())
store_opts = sorted(universe["install_store"].astype(str).unique().tolist())
campaign_opts = sorted(universe["campaign"].astype(str).unique().tolist())
country_opts = sorted(universe["country"].astype(str).unique().tolist())
payer_opts = ["payer", "non_payer"]

cohort_f = st.sidebar.multiselect("Cohort", cohort_opts, default=cohort_opts, help="Applied to Monetization/Engagement/Segments/A-B only.")
platform_f = st.sidebar.multiselect("Platform", platform_opts, default=platform_opts, help="Applied to Monetization/Engagement/Segments/A-B only.")
store_f = st.sidebar.multiselect("Install store", store_opts, default=store_opts, help="Applied to Monetization/Engagement/Segments/A-B only.")
campaign_f = st.sidebar.multiselect("Campaign", campaign_opts, default=campaign_opts, help="Applied to Monetization/Engagement/Segments/A-B only.")

top_country = (
    universe.groupby("country")["user_id"].nunique()
    .sort_values(ascending=False).head(25).index.tolist()
)
country_f = st.sidebar.multiselect(
    "Country (default: top 25)",
    country_opts,
    default=top_country,
    help="Applied to Monetization/Engagement/Segments/A-B only."
)

payer_f = st.sidebar.multiselect(
    "Payer status",
    payer_opts,
    default=payer_opts,
    help="payer if max(payer_status) > 0 across observed logs (user-level proxy)."
)

# Build user KPIs for non-Home tabs
with st.spinner("Preparing user-level metrics for filtered tabs…"):
    user_kpis = aggregate_user_kpis_unconditional(sessions, universe, window)

u = user_kpis[
    user_kpis["cohort"].astype(str).isin(cohort_f) &
    user_kpis["platform"].astype(str).isin(platform_f) &
    user_kpis["install_store"].astype(str).isin(store_f) &
    user_kpis["campaign"].astype(str).isin(campaign_f) &
    user_kpis["country"].astype(str).isin(country_f) &
    user_kpis["payer_segment"].astype(str).isin(payer_f)
].copy()


# =============================================================================
# TABS
# =============================================================================

tabs = st.tabs([
    "🏠 Home (Overview)",
    "🪙 Monetization",
    "🎮 Engagement",
    "🔍 Segments",
    "👩🏻‍🔬 A/B Compare"
])


# =============================================================================
# HOME (Overview) – FULL DATASET + DATE FILTER ONLY
# =============================================================================

with tabs[0]:
    st.subheader("🏠 Overview")
    st.caption("This page is intentionally simple: one date filter, full dataset, and lots of visual signals ✨")

    if daily_all.empty:
        st.warning("No daily data available (check open_at parsing).")
        st.stop()

    if not PLOTLY_OK:
        st.warning("Plotly is not available. Install it for interactive charts: `pip install plotly`.")
        st.stop()

    # Home-only date filter (active_day)
    min_day = pd.to_datetime(daily_all["active_day"].min()).date()
    max_day = pd.to_datetime(daily_all["active_day"].max()).date()

    home_date_range = st.date_input(
        "📅 Date range (active_day)",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
        help="Applies to ALL charts on Home only."
    )
    if isinstance(home_date_range, tuple) and len(home_date_range) == 2:
        home_start, home_end = home_date_range
    else:
        home_start, home_end = min_day, max_day

    d = daily_all[
        (daily_all["active_day"].dt.date >= home_start) &
        (daily_all["active_day"].dt.date <= home_end)
    ].copy()

    bd = bd_active_all[
        (bd_active_all["active_day"].dt.date >= home_start) &
        (bd_active_all["active_day"].dt.date <= home_end)
    ].copy()

    # KPI cards
    cA, cB, cC, cD = st.columns(4)

    dau_avg = float(d["dau"].mean()) if len(d) else 0.0
    sess_per_dau = float(d["sessions_per_dau"].mean()) if len(d) else 0.0
    rev_per_dau = float(d["rev_per_dau"].mean()) if len(d) else 0.0
    total_rev = float(d["total_rev_net"].sum()) if len(d) else 0.0

    cA.metric("Avg DAU (proxy)", f"{dau_avg:,.0f}",
              help="DAU = |{ user_id }| per day (from session logs).")
    cB.metric("Avg Sessions / DAU", _num(sess_per_dau, 2),
              help="sessions_per_dau = sessions / dau (daily), then averaged over selected days.")
    cC.metric("Avg Revenue / DAU (Net)", _money(rev_per_dau, 6),
              help="rev_per_dau = total_rev_net / dau\nwhere total_rev_net = (rv_rev + fs_rev) + 0.7 * iap_rev")
    cD.metric("Total Revenue (Net)", _money(total_rev, 2),
              help="Total = Σ total_rev_net over selected days.")

    st.divider()

    # Separate charts: DAU / Sessions
    fig_dau = go.Figure()
    fig_dau.add_trace(go.Scatter(x=d["active_day"], y=d["dau"], mode="lines", name="DAU"))
    fig_dau.update_layout(title="DAU (proxy) over time", xaxis_title="active_day", yaxis_title="users", hovermode="x unified")
    st.plotly_chart(fig_dau, use_container_width=True)

    fig_sess = go.Figure()
    fig_sess.add_trace(go.Scatter(x=d["active_day"], y=d["sessions"], mode="lines", name="Sessions"))
    fig_sess.update_layout(title="Sessions over time", xaxis_title="active_day", yaxis_title="sessions", hovermode="x unified")
    st.plotly_chart(fig_sess, use_container_width=True)

    # Rates
    fig_rates = go.Figure()
    fig_rates.add_trace(go.Scatter(x=d["active_day"], y=d["rev_per_dau"], mode="lines", name="Revenue / DAU (Net)"))
    fig_rates.add_trace(go.Scatter(x=d["active_day"], y=d["sessions_per_dau"], mode="lines", name="Sessions / DAU"))
    fig_rates.update_layout(title="Per-user rates over time", xaxis_title="active_day", yaxis_title="value", hovermode="x unified")
    st.plotly_chart(fig_rates, use_container_width=True)

    # Revenue composition (stacked)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["rv_rev"], stackgroup="one", name="RV revenue"))
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["fs_rev"], stackgroup="one", name="FS revenue"))
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["iap_rev_net"], stackgroup="one", name="IAP net revenue"))
    fig_comp.update_layout(title="Revenue composition over time (stacked)", xaxis_title="active_day", yaxis_title="revenue", hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # Breakdown selector
    dim_map = {
        "🌍 Country": "country",
        "🏬 Install store": "install_store",
        "📱 Platform": "platform",
        "🎯 Campaign": "campaign",
        "💳 Payer status": "payer_segment",
    }

    dim_label = st.selectbox(
        "Breakdown dimension (Active users)",
        list(dim_map.keys()),
        index=0,
        help="Shows how daily active users are distributed across this dimension."
    )
    dim = dim_map[dim_label]

    topn = st.slider(
        "Top N categories (others → 'Other')",
        min_value=5,
        max_value=30,
        value=Config.HOME_TOPN_DEFAULT,
        step=1,
        help="Keeps charts readable."
    )

    if bd.empty:
        st.info("No breakdown data available for the selected dates.")
    else:
        tmp = bd.groupby(["active_day", dim]).agg(active_users=("active_users", "sum")).reset_index()
        totals = tmp.groupby(dim)["active_users"].sum().sort_values(ascending=False)
        top_cats = totals.head(topn).index.tolist()

        tmp[dim] = np.where(tmp[dim].isin(top_cats), tmp[dim], "Other")

        tmp2 = tmp.groupby(["active_day", dim]).agg(active_users=("active_users", "sum")).reset_index()
        day_total = tmp2.groupby("active_day")["active_users"].sum().reset_index(name="day_total")
        tmp2 = tmp2.merge(day_total, on="active_day", how="left")
        tmp2["share"] = tmp2["active_users"] / tmp2["day_total"].replace(0, np.nan)

        fig_stack = px.bar(
            tmp2, x="active_day", y="share", color=dim,
            title=f"Daily active user share by {dim_label} (stacked)",
            hover_data={"active_users": True, "day_total": True, "share": ":.2%"}
        )
        fig_stack.update_layout(barmode="stack", hovermode="x unified", yaxis_tickformat=".0%")
        st.plotly_chart(fig_stack, use_container_width=True)

        comp = tmp2.groupby(dim).agg(active_users=("active_users", "sum")).reset_index().sort_values("active_users", ascending=False)
        fig_top = px.bar(
            comp, x=dim, y="active_users",
            title=f"Total active users by {dim_label} (selected date range)",
            hover_data={"active_users": True}
        )
        fig_top.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_top, use_container_width=True)

        # Cute + useful summary table
        st.markdown("#### ✨ Quick summary (Top categories)")
        comp["share"] = comp["active_users"] / comp["active_users"].sum()
        st.dataframe(
            comp.assign(share=comp["share"].map(lambda x: f"{x*100:.2f}%")),
            use_container_width=True
        )

    st.divider()

    # -------------------------
    # Install proxy section
    # -------------------------
    st.subheader("🆕 New users (Install proxy)")
    st.caption("Based on first_app_launch_date (fallback to first observed active_day when missing).")

    # Filter install_day to the same Home date range (but on install axis)
    if installs_all.empty:
        st.info("No install proxy data available.")
    else:
        inst = installs_all[
            (installs_all["day"].dt.date >= home_start) &
            (installs_all["day"].dt.date <= home_end)
        ].copy()

        fig_inst = go.Figure()
        fig_inst.add_trace(go.Bar(x=inst["day"], y=inst["new_users"], name="New users"))
        fig_inst.update_layout(title="Daily new users (install proxy)", xaxis_title="install_day", yaxis_title="users", hovermode="x unified")
        st.plotly_chart(fig_inst, use_container_width=True)

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=inst["day"], y=inst["cum_new_users"], mode="lines", name="Cumulative new users"))
        fig_cum.update_layout(title="Cumulative new users (install proxy)", xaxis_title="install_day", yaxis_title="users", hovermode="x unified")
        st.plotly_chart(fig_cum, use_container_width=True)

        # Breakdown for installs
        dim_label2 = st.selectbox(
            "Breakdown dimension (New users)",
            list(dim_map.keys()),
            index=0,
            help="Shows how daily new users are distributed across this dimension."
        )
        dim2 = dim_map[dim_label2]

        bd_i = bd_install_all[
            (bd_install_all["day"].dt.date >= home_start) &
            (bd_install_all["day"].dt.date <= home_end)
        ].copy()

        if not bd_i.empty:
            tmp = bd_i.groupby(["day", dim2]).agg(new_users=("new_users", "sum")).reset_index()
            totals = tmp.groupby(dim2)["new_users"].sum().sort_values(ascending=False)
            top_cats = totals.head(topn).index.tolist()
            tmp[dim2] = np.where(tmp[dim2].isin(top_cats), tmp[dim2], "Other")

            tmp2 = tmp.groupby(["day", dim2]).agg(new_users=("new_users", "sum")).reset_index()
            day_total = tmp2.groupby("day")["new_users"].sum().reset_index(name="day_total")
            tmp2 = tmp2.merge(day_total, on="day", how="left")
            tmp2["share"] = tmp2["new_users"] / tmp2["day_total"].replace(0, np.nan)

            fig_stack = px.bar(
                tmp2, x="day", y="share", color=dim2,
                title=f"Daily new user share by {dim_label2} (stacked)",
                hover_data={"new_users": True, "day_total": True, "share": ":.2%"}
            )
            fig_stack.update_layout(barmode="stack", hovermode="x unified", yaxis_tickformat=".0%")
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.info("No install breakdown data available for this period.")


# =============================================================================
# Monetization (global filters apply)
# =============================================================================

with tabs[1]:
    st.subheader("🪙 Monetization")
    st.caption("Global filters apply. Metrics are raw (no winsorization).")

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
    c1.metric("Users", f"{users_n:,}", help="N_users = distinct user_id in filtered user table.")
    c2.metric("Total revenue / user (mean, net)", _money(mean_total, 6),
              help="total_rev_net = ad_rev + iap_rev_net\nad_rev = rv_rev + fs_rev\niap_rev_net = 0.7 * iap_rev")
    c3.metric("Total revenue / user (median, net)", _money(med_total, 6),
              help="median(total_rev_net) across users (robust to heavy tail).")
    c4.metric("Ad revenue / user (mean)", _money(mean_ad, 6),
              help="ad_rev = rv_rev + fs_rev")
    c5.metric("IAP revenue / user (mean, net)", _money(mean_iap, 6),
              help="iap_rev_net = 0.7 * iap_rev")

    st.divider()

    z1, z2, z3, z4 = st.columns(4)
    z1.metric("% users with total_rev=0", _pct(zero_total), help="P(total_rev_net = 0) over filtered users.")
    z2.metric("% users with rv_rev=0", _pct(zero_rv), help="P(rv_rev = 0) over filtered users.")
    z3.metric("% users with fs_rev=0", _pct(zero_fs), help="P(fs_rev = 0) over filtered users.")
    z4.metric("% users with iap_net=0", _pct(zero_iap), help="P(iap_rev_net = 0) over filtered users.")

    if PLOTLY_OK:
        left, right = st.columns(2)
        with left:
            fig = px.histogram(u, x="total_rev_net", nbins=60, title="Distribution: total_rev_net per user")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.histogram(u, x="ad_rev", nbins=60, title="Distribution: ad_rev per user")
            st.plotly_chart(fig, use_container_width=True)

        mix = pd.DataFrame({
            "component": ["RV", "FS", "IAP (Net)"],
            "value": [u["rv_rev"].sum(), u["fs_rev"].sum(), u["iap_rev_net"].sum()]
        })
        fig = px.pie(mix, values="value", names="component", title="Revenue mix (sum over filtered users)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install plotly for interactive charts: `pip install plotly`.")


# =============================================================================
# Engagement (global filters apply)
# =============================================================================

with tabs[2]:
    st.subheader("🎮 Engagement")
    st.caption("Global filters apply. Metrics are raw (no winsorization).")

    mean_sessions = float(u["sessions"].mean())
    med_sessions = float(u["sessions"].median())
    active_share = float(u["is_active"].mean())
    mean_len = float(u["total_session_length"].mean())
    mean_games = float(u["total_game_count"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active share (proxy)", _pct(active_share),
              help="active_share = mean( 1[sessions > 0] ) over filtered users.")
    c2.metric("Sessions / user (mean)", _num(mean_sessions, 2), help="mean(sessions) per user (windowed).")
    c3.metric("Sessions / user (median)", _num(med_sessions, 2), help="median(sessions) per user (windowed).")
    c4.metric("Total session length / user (mean)", _num(mean_len, 2), help="mean(Σ session_length) per user (windowed).")
    c5.metric("Game count / user (mean)", _num(mean_games, 2), help="mean(Σ game_count) per user (windowed).")

    if PLOTLY_OK:
        fig = px.histogram(u, x="sessions", nbins=60, title="Distribution: sessions per user")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(u, x="total_session_length", nbins=60, title="Distribution: total_session_length per user")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(u, x="sessions", y="total_rev_net", title="Sessions vs total_rev_net (per user)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install plotly for interactive charts: `pip install plotly`.")


# =============================================================================
# Segments (global filters apply)
# =============================================================================

with tabs[3]:
    st.subheader("🔍 Segments")
    st.caption("Global filters apply. Tables show mean/median + zero share (raw).")

    dim = st.selectbox(
        "Segment dimension",
        ["country", "platform", "install_store", "campaign", "payer_segment", "manufacturer", "model"],
        index=0,
        help="Groups are formed on user-level attributes (from universe)."
    )

    metric = st.selectbox(
        "Metric",
        ["total_rev_net", "ad_rev", "rv_rev", "fs_rev", "iap_rev_net",
         "sessions", "total_session_length", "total_game_count"],
        index=0,
        help="Metric is computed at user-level within the chosen assignment window."
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
        topn = st.slider("Top N segments for charts", 5, 50, 20, help="High-cardinality dims can be heavy.")
        seg_top = seg.sort_values("mean", ascending=False).head(topn)

        fig = px.bar(
            seg_top, x=dim, y="mean",
            title=f"Top {topn} segments by mean({metric})",
            hover_data=["users", "median", "zero_share", "sum"]
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            seg.head(300),
            x="users", y="mean", size="sum",
            hover_name=dim,
            hover_data=["median", "zero_share", "sum"],
            title=f"Segment map: users vs mean({metric}) (bubble size = sum)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install plotly for interactive charts: `pip install plotly`.")


# =============================================================================
# A/B Compare (global filters apply, winsor optional)
# =============================================================================

with tabs[4]:
    st.subheader("👩🏻‍🔬 A/B Compare")
    st.caption("Yes — Global filters apply here too ✅ (only winsorization is special to this tab).")

    kpi = st.selectbox(
        "KPI",
        ["total_rev_net", "ad_rev", "rv_rev", "fs_rev", "iap_rev_net",
         "sessions", "total_session_length", "total_game_count", "iap_transactions"],
        index=0,
        help="Comparison is done on the filtered user table."
    )

    colA, colB, colC = st.columns([1, 1, 2])
    use_w = colA.checkbox("Winsorize", value=True,
                          help="Optional robustness tool: clips extreme values at a quantile threshold.")
    winsor_q = colB.slider("Winsor quantile", 0.95, 0.9999, Config.DEFAULT_WINSOR_Q, step=0.0001,
                           help="If threshold collapses to 0 but positives exist, recompute threshold on positives.")
    n_boot = colC.slider("Bootstrap iterations", 500, 5000, Config.DEFAULT_BOOTSTRAP_N, step=500,
                         help="Used for percentile 95% CI of mean difference.")

    metric_col = kpi
    df_ab = u[["cohort", metric_col]].copy()

    if use_w:
        df_ab = apply_winsorization(df_ab, winsor_q, cols=[metric_col])
        metric_col = metric_col + "_w"

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

    cohorts_present = set(ab["cohort"].astype(str).tolist())
    if Config.CONTROL in cohorts_present and Config.TEST in cohorts_present:
        c = df_ab[df_ab["cohort"] == Config.CONTROL][metric_col].to_numpy()
        t = df_ab[df_ab["cohort"] == Config.TEST][metric_col].to_numpy()

        mean_c = float(np.mean(c)) if len(c) else np.nan
        mean_t = float(np.mean(t)) if len(t) else np.nan

        diff_obs, ci_low, ci_high = bootstrap_mean_diff_ci(c, t, n_boot=n_boot, seed=Config.RNG_SEED)

        diff = mean_t - mean_c
        rel = (diff / mean_c * 100) if (mean_c is not None and mean_c != 0) else np.nan

        st.metric(
            "Mean lift (Test - Control)",
            f"{diff:+.6f}",
            delta=f"{rel:+.2f}%" if np.isfinite(rel) else "NA",
            help="lift = mean(test) - mean(control)\nCI = bootstrap percentile interval of lift"
        )
        st.write(f"Bootstrap 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}]")

        if PLOTLY_OK:
            fig = px.histogram(
                df_ab, x=metric_col, color="cohort",
                barmode="overlay", nbins=60,
                title=f"Distribution of {kpi} by cohort ({'winsorized' if use_w else 'raw'})"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Both control and test cohorts are not present after filtering. Adjust filters to compare.")


# =============================================================================
# FOOTER: equations + glossary
# =============================================================================

st.divider()

with st.expander("📌 Key Equations (Reference)", expanded=False):
    st.markdown(
        """
- `ad_rev = rv_rev + fs_rev`  
- `iap_rev_net = 0.7 * iap_rev`  
- `total_rev_net = ad_rev + iap_rev_net`  
- `dau = |{ user_id }| per active_day`  
- `sessions_per_dau = sessions / dau`  
- `rev_per_dau = total_rev_net / dau`  
        """
    )

with st.expander("📚 Column Glossary (Reference)", expanded=False):
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

st.caption("📝 Note: This dashboard uses session logs. True assignment logs are needed for full-universe retention.")
