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
#   - Only one filter: date range (active_day)  -> LOCAL to Home only
#
# 📊 Other tabs (Monetization / Engagement / Segments / A/B Compare)
#   - Uses Global Filters (date/country/platform/campaign/payer)
#   - User-level KPIs computed by assignment window (D7 etc.)
#
# 🆕 New Users & Retention
#   - Install-day based cohort (D0 = install_day)
#   - Strict retention (day-K) shown as MAIN (downward-looking curve)
#   - Rolling retention shown as OPTIONAL (cumulative "ever returned by K" curve)
#   - Kernel curves: D0..D30 intensity (sessions / playtime / revenue per user)
#   - D0 selection is a SINGLE day (not a range) to match classic cohort definitions
#
# Robustness:
# - Winsorization is available ONLY in A/B Compare (inference / robustness tool)
#
# Dataset limitation reminder:
# - This is session-log population, not true assignment logs.
# - “Assigned universe” is “observed assigned users” (users appearing in session logs).
# =============================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Interactive charts
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

    # Assignment windows (days post-assignment, inclusive)
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

    HOME_TOPN_DEFAULT = 10
    SEG_TOPN_DEFAULT = 20
    MAX_TOPN = 50

    RETENTION_MAX_DAYS = 30


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
    """Try reading parquet safely."""
    try:
        df = pd.read_parquet(path_parquet)
        return True, df
    except Exception:
        return False, pd.DataFrame()


def try_write_parquet(df: pd.DataFrame, path_parquet: str) -> bool:
    """Try writing parquet safely."""
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

    if (not rebuild) and os.path.exists(pq_path) and (pq_mtime >= csv_mtime):
        ok, df = try_read_parquet(pq_path)
        if ok and not df.empty:
            info["loaded_from"] = "parquet"
            info["parquet_available"] = True
            return df, info

    df = read_csv_from_path(csv_path)
    info["loaded_from"] = "csv"

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

    # install_store exists but we intentionally DO NOT use it in UI (platform is enough)
    if "install_store" not in d.columns:
        d["install_store"] = np.nan

    # Revenue / engagement numeric fills
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

    # Install-based day index (for retention)
    d["days_since_install"] = (d["active_day"] - d["install_day"]).dt.days.astype("Int64")

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
# DAILY METRICS (calendar-day time series)
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

    # Per-user rates (per DAU)
    daily["sessions_per_dau"] = daily["sessions"] / daily["dau"].replace(0, np.nan)
    daily["rev_per_dau"] = daily["total_rev_net"] / daily["dau"].replace(0, np.nan)
    daily["playtime_per_dau"] = daily["total_session_length"] / daily["dau"].replace(0, np.nan)
    daily["gamecount_per_dau"] = daily["total_game_count"] / daily["dau"].replace(0, np.nan)

    return daily


@st.cache_data(show_spinner=False)
def build_daily_breakdown_active(sessions: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """
    Daily breakdown: active users (by active_day) split by user attributes.
    (platform only — install_store removed intentionally)
    """
    s = sessions[sessions["active_day"].notna()][["active_day", "user_id"]].drop_duplicates()
    s = s.merge(
        universe[["user_id", "country", "platform", "campaign", "payer_segment"]],
        on="user_id",
        how="left"
    )

    out = (
        s.groupby(["active_day", "country", "platform", "campaign", "payer_segment"])
         .agg(active_users=("user_id", "nunique"))
         .reset_index()
    )
    return out


@st.cache_data(show_spinner=False)
def build_daily_active_payer_share(
    sessions: pd.DataFrame,
    universe: pd.DataFrame,
    country_f: List[str],
    platform_f: List[str],
    campaign_f: List[str]
) -> pd.DataFrame:
    """
    Daily payer share among ACTIVE users (calendar-day), after applying filters
    (excluding payer filter; otherwise share becomes trivial).
    """
    s = sessions[sessions["active_day"].notna()][["active_day", "user_id"]].drop_duplicates()
    s = s.merge(
        universe[["user_id", "country", "platform", "campaign", "payer_segment"]],
        on="user_id",
        how="left"
    )

    s = s[
        s["country"].astype(str).isin(country_f) &
        s["platform"].astype(str).isin(platform_f) &
        s["campaign"].astype(str).isin(campaign_f)
    ].copy()

    daily = (
        s.groupby(["active_day", "payer_segment"])
         .agg(active_users=("user_id", "nunique"))
         .reset_index()
    )

    pivot = daily.pivot_table(index="active_day", columns="payer_segment", values="active_users", aggfunc="sum").fillna(0)
    pivot["active_total"] = pivot.sum(axis=1)
    pivot["active_payers"] = pivot.get("payer", 0.0)
    pivot["payer_share_active"] = pivot["active_payers"] / pivot["active_total"].replace(0, np.nan)

    out = pivot.reset_index()[["active_day", "active_total", "active_payers", "payer_share_active"]].sort_values("active_day")
    return out


# =============================================================================
# INSTALL-COHORT RETENTION + KERNEL CURVES (D0 = install_day)
# =============================================================================

@st.cache_data(show_spinner=False)
def build_install_cohort_kernel(
    sessions: pd.DataFrame,
    universe: pd.DataFrame,
    cohort_start: pd.Timestamp,
    cohort_end: pd.Timestamp,
    max_days: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    Install-cohort analysis (D0 = install_day):
    - Strict retention curve (day-K active rate) for K=0..max_days
    - Rolling retention curve (cumulative ever-returned by day K) for K=0..max_days
      Note: Rolling retention is non-decreasing by definition.
    - Kernel curves for sessions/playtime/revenue per user by day_since_install (0..max_days)
    - D30 LTV + IAP contribution computed over day_since_install in [0,29]
    """
    u = universe.copy()
    u = u[u["install_day"].notna()].copy()

    cohort_users = u[
        (u["install_day"] >= cohort_start) &
        (u["install_day"] <= cohort_end)
    ][["user_id", "install_day", "payer_segment"]].copy()

    n_users = int(cohort_users["user_id"].nunique())

    if n_users == 0:
        empty = pd.DataFrame()
        return {
            "cohort_users": cohort_users,
            "retention": empty,
            "kernel": empty,
            "ltv_summary": pd.DataFrame([{
                "cohort_users": 0,
                "d30_ltv_net": np.nan,
                "d30_iap_contribution": np.nan
            }])
        }

    # Sessions restricted to cohort users, compute day_since_install int
    s = sessions[
        sessions["user_id"].isin(cohort_users["user_id"]) &
        sessions["active_day"].notna() &
        sessions["install_day"].notna()
    ].copy()

    s["day_since_install"] = (s["active_day"] - s["install_day"]).dt.days.astype("Int64")
    s = s[s["day_since_install"].between(0, max_days)].copy()

    # --- Retention (strict + rolling) ---
    active_days = s[["user_id", "day_since_install"]].drop_duplicates()

    # strict retention: active on day k
    strict = (
        active_days.groupby("day_since_install")
        .agg(active_users=("user_id", "nunique"))
        .reset_index()
        .sort_values("day_since_install")
    )
    strict["strict_retention"] = strict["active_users"] / n_users

    # rolling retention: returned at least once in days 1..k
    returns = active_days[active_days["day_since_install"] >= 1].copy()
    first_return = returns.groupby("user_id")["day_since_install"].min().reset_index(name="first_return_day")

    all_k = pd.DataFrame({"day_since_install": list(range(0, max_days + 1))})
    rolling_vals = []
    for k in range(0, max_days + 1):
        if k < 1:
            rolling_vals.append(0.0)
        else:
            rolling_vals.append(float((first_return["first_return_day"] <= k).mean()) if len(first_return) else 0.0)
    all_k["rolling_retention"] = rolling_vals

    retention = all_k.merge(
        strict[["day_since_install", "strict_retention"]],
        on="day_since_install",
        how="left"
    ).fillna({"strict_retention": 0.0})
    retention["cohort_users"] = n_users

    # --- Kernel curves (unconditional per cohort user) ---
    s["ad_rev"] = s["rv_revenue"].fillna(0.0) + s["fs_revenue"].fillna(0.0)
    s["iap_rev_net"] = 0.7 * s["iap_revenue"].fillna(0.0)
    s["total_rev_net"] = s["ad_rev"] + s["iap_rev_net"]

    kernel = (
        s.groupby("day_since_install")
        .agg(
            sessions=("session_id", "count"),
            playtime=("session_length", "sum"),
            ad_rev=("ad_rev", "sum"),
            iap_rev_net=("iap_rev_net", "sum"),
            total_rev_net=("total_rev_net", "sum"),
        )
        .reset_index()
        .sort_values("day_since_install")
    )
    kernel["cohort_users"] = n_users
    kernel["sessions_per_user"] = kernel["sessions"] / n_users
    kernel["playtime_per_user"] = kernel["playtime"] / n_users
    kernel["rev_per_user"] = kernel["total_rev_net"] / n_users

    # Ensure all days 0..max_days exist
    all_days = pd.DataFrame({"day_since_install": list(range(0, max_days + 1))})
    kernel = all_days.merge(kernel, on="day_since_install", how="left").fillna({
        "sessions": 0, "playtime": 0, "ad_rev": 0, "iap_rev_net": 0, "total_rev_net": 0,
        "sessions_per_user": 0, "playtime_per_user": 0, "rev_per_user": 0,
        "cohort_users": n_users
    })

    # --- D30 LTV (0..29) ---
    s_d30 = s[s["day_since_install"].between(0, 29)].copy()
    ltv_total = float(s_d30["total_rev_net"].sum()) / n_users
    ltv_iap = float(s_d30["iap_rev_net"].sum()) / max(1, n_users)
    ltv_iap_share = (ltv_iap / ltv_total) if ltv_total > 0 else np.nan

    ltv_summary = pd.DataFrame([{
        "cohort_users": n_users,
        "d30_ltv_net": ltv_total,
        "d30_iap_contribution": ltv_iap_share
    }])

    return {
        "cohort_users": cohort_users,
        "retention": retention,
        "kernel": kernel,
        "ltv_summary": ltv_summary
    }


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
st.caption("Home = snapshot. Other tabs = deeper dives with GLOBAL filters. Hover charts to see exact values.")

# ----------------------------
# Sidebar: data source + parquet cache
# ----------------------------
st.sidebar.header("📁 Data Source")

path = st.sidebar.text_input(
    "Local CSV path (.csv or .csv.gz)",
    value=Config.DEFAULT_LOCAL_PATH,
    help="Keep the CSV next to app.py or provide an absolute path."
)
path = os.path.abspath(os.path.expanduser(path))
if not os.path.exists(path):
    st.error("File not found. Please check the path.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("⚡ Fast Reload (Parquet Cache)")

rebuild_parquet = st.sidebar.button(
    "Rebuild Parquet Cache",
    help="Forces re-creation of the parquet file from CSV (use if cache seems stale)."
)

# Load raw
with st.spinner("Loading data… (first run can be slow, later runs are much faster)"):
    df_raw, cache_info = load_raw_with_parquet_cache(path, rebuild=rebuild_parquet)
    raw = load_and_clean_data(df_raw)
    sessions_all = create_temporal_features(raw)
    universe, universe_info = build_assigned_universe(sessions_all)

    daily_all = build_daily_metrics(sessions_all)
    bd_active_all = build_daily_breakdown_active(sessions_all, universe)

with st.sidebar.expander("Cache status", expanded=False):
    st.write(cache_info)

# ----------------------------
# Sidebar: Data Health
# ----------------------------
with st.sidebar.expander("🧪 Data Health", expanded=False):
    st.write("Users with multiple cohorts are dropped to avoid contamination.")
    st.json(universe_info)
    st.write("Raw missing values (top 20 columns):")
    st.dataframe(raw.isna().sum().sort_values(ascending=False).head(20), use_container_width=True)

st.sidebar.divider()

# ----------------------------
# Sidebar: Global Filters (NOT used in Home)
# ----------------------------
st.sidebar.header("🎛️ Global Filters (Non-Home tabs)")

if daily_all.empty:
    st.error("No daily data available (check open_at parsing).")
    st.stop()

global_min_day = pd.to_datetime(daily_all["active_day"].min()).date()
global_max_day = pd.to_datetime(daily_all["active_day"].max()).date()

global_date_range = st.sidebar.date_input(
    "📅 Global date range (active_day)",
    value=(global_min_day, global_max_day),
    min_value=global_min_day,
    max_value=global_max_day,
    help="Applies to Monetization / Engagement / Segments / A/B Compare / New Users & Retention. (Home ignores this.)"
)
if isinstance(global_date_range, tuple) and len(global_date_range) == 2:
    g_start, g_end = global_date_range
else:
    g_start, g_end = global_min_day, global_max_day

window = st.sidebar.selectbox(
    "User KPI window (days since assignment)",
    list(Config.WINDOWS.keys()),
    index=list(Config.WINDOWS.keys()).index("d7"),
    help="Example: D7 means days_since_assignment ∈ [0, 6]. Windowed KPIs are computed per user."
)

platform_opts = sorted(universe["platform"].astype(str).unique().tolist())
campaign_opts = sorted(universe["campaign"].astype(str).unique().tolist())
country_opts = sorted(universe["country"].astype(str).unique().tolist())
payer_opts = ["payer", "non_payer"]

platform_f = st.sidebar.multiselect("Platform", platform_opts, default=platform_opts)
campaign_f = st.sidebar.multiselect("Campaign", campaign_opts, default=campaign_opts)

top_country = (
    universe.groupby("country")["user_id"].nunique()
    .sort_values(ascending=False).head(25).index.tolist()
)
country_f = st.sidebar.multiselect(
    "Country (default: top 25)",
    country_opts,
    default=top_country,
)

payer_f = st.sidebar.multiselect(
    "Payer status",
    payer_opts,
    default=payer_opts,
    help="Proxy payer segment: payer if max(payer_status) > 0 in observed logs."
)

# Apply GLOBAL DATE FILTER to sessions for non-Home tabs
sessions_g = sessions_all[
    (sessions_all["active_day"].dt.date >= g_start) &
    (sessions_all["active_day"].dt.date <= g_end)
].copy()

# Build user KPIs for non-Home tabs (with global date filter applied)
with st.spinner("Preparing user-level metrics for filtered tabs…"):
    user_kpis_g = aggregate_user_kpis_unconditional(sessions_g, universe, window)

u = user_kpis_g[
    user_kpis_g["platform"].astype(str).isin(platform_f) &
    user_kpis_g["campaign"].astype(str).isin(campaign_f) &
    user_kpis_g["country"].astype(str).isin(country_f) &
    user_kpis_g["payer_segment"].astype(str).isin(payer_f)
].copy()

# Daily metrics for global-filtered period (for time series in non-Home tabs)
daily_g = daily_all[
    (daily_all["active_day"].dt.date >= g_start) &
    (daily_all["active_day"].dt.date <= g_end)
].copy()


# =============================================================================
# TABS
# =============================================================================

tabs = st.tabs([
    "🏠 Home (Overview)",
    "🪙 Monetization",
    "🎮 Engagement",
    "🔍 Segments",
    "👩🏻‍🔬 A/B Compare",
    "🆕 New Users & Retention",
])


# =============================================================================
# HOME (Overview) – FULL DATASET + LOCAL DATE FILTER ONLY
# =============================================================================

with tabs[0]:
    st.subheader("🏠 Overview")
    st.caption("Home uses full data and a local date filter (sidebar GLOBAL filters do not apply here).")

    if not PLOTLY_OK:
        st.warning("Plotly is not available. Install it for interactive charts: `pip install plotly`.")
        st.stop()

    # Home-only date filter (active_day)
    min_day = pd.to_datetime(daily_all["active_day"].min()).date()
    max_day = pd.to_datetime(daily_all["active_day"].max()).date()

    home_date_range = st.date_input(
        "📅 Date range (Home only)",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
        help="Local to Home only. Other tabs use the GLOBAL date filter in the sidebar."
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

    # KPI cards (user-friendly labels + formulas in help)
    cA, cB, cC, cD = st.columns(4)

    dau_avg = float(d["dau"].mean()) if len(d) else 0.0
    sess_per_dau = float(d["sessions_per_dau"].mean()) if len(d) else 0.0
    rev_per_dau = float(d["rev_per_dau"].mean()) if len(d) else 0.0
    total_rev = float(d["total_rev_net"].sum()) if len(d) else 0.0

    cA.metric(
        "Daily Active Users (Avg)",
        f"{dau_avg:,.0f}",
        help="DAU = unique users per day (from session logs).\n\nFormula:\ndau = |{user_id}| per active_day"
    )
    cB.metric(
        "Sessions per Active User",
        _num(sess_per_dau, 2),
        help="Average sessions per active user.\n\nFormula:\nsessions_per_dau = sessions / dau"
    )
    cC.metric(
        "Revenue per Active User (Net)",
        _money(rev_per_dau, 6),
        help="Average net revenue per active user.\n\n"
             "Formulas:\n"
             "ad_rev = rv_rev + fs_rev\n"
             "iap_rev_net = 0.7 × iap_rev\n"
             "total_rev_net = ad_rev + iap_rev_net\n"
             "rev_per_dau = total_rev_net / dau"
    )
    cD.metric(
        "Total Revenue (Net)",
        _money(total_rev, 2),
        help="Total net revenue in the selected period.\n\n"
             "Formula:\nTotal = Σ total_rev_net over selected days"
    )

    st.divider()

    # Total daily playtime + total revenue (one row)
    r1, r2 = st.columns(2)

    fig_play = go.Figure()
    fig_play.add_trace(go.Scatter(x=d["active_day"], y=d["total_session_length"], mode="lines", name="Total playtime"))
    fig_play.update_layout(
        title="Total Player Time Spent per Day",
        xaxis_title="Date",
        yaxis_title="Total Playtime",
        hovermode="x unified"
    )
    r1.plotly_chart(fig_play, use_container_width=True)

    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(x=d["active_day"], y=d["total_rev_net"], mode="lines", name="Total net revenue"))
    fig_tr.update_layout(
        title="Total Revenue per Day (Net)",
        xaxis_title="Date",
        yaxis_title="Net Revenue",
        hovermode="x unified"
    )
    r2.plotly_chart(fig_tr, use_container_width=True)

    st.divider()

    # DAU + Sessions side-by-side
    g1, g2 = st.columns(2)

    fig_dau = go.Figure()
    fig_dau.add_trace(go.Scatter(x=d["active_day"], y=d["dau"], mode="lines", name="DAU"))
    fig_dau.update_layout(
        title="Daily Active Users Over Time",
        xaxis_title="Date",
        yaxis_title="Users",
        hovermode="x unified"
    )
    g1.plotly_chart(fig_dau, use_container_width=True)

    fig_sess = go.Figure()
    fig_sess.add_trace(go.Scatter(x=d["active_day"], y=d["sessions"], mode="lines", name="Sessions"))
    fig_sess.update_layout(
        title="Total Sessions Over Time",
        xaxis_title="Date",
        yaxis_title="Sessions",
        hovermode="x unified"
    )
    g2.plotly_chart(fig_sess, use_container_width=True)

    # Per-user rates
    fig_rates = go.Figure()
    fig_rates.add_trace(go.Scatter(
        x=d["active_day"], y=d["rev_per_dau"], mode="lines", name="Revenue per Active User (Net)"
    ))
    fig_rates.add_trace(go.Scatter(
        x=d["active_day"], y=d["sessions_per_dau"], mode="lines", name="Sessions per Active User"
    ))
    fig_rates.update_layout(
        title="Per-User Performance Trends",
        xaxis_title="Date",
        yaxis_title="Per-User Metric",
        hovermode="x unified"
    )
    st.plotly_chart(fig_rates, use_container_width=True)

    # Revenue composition (stacked)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["rv_rev"], stackgroup="one", name="Rewarded Video (RV)"))
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["fs_rev"], stackgroup="one", name="Fullscreen / Interstitial (FS)"))
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["iap_rev_net"], stackgroup="one", name="IAP (Net)"))
    fig_comp.update_layout(
        title="Revenue Mix Over Time (Daily)",
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # Breakdown selector (platform only; install_store removed)
    dim_map = {
        "🌍 Country": "country",
        "📱 Platform": "platform",
        "🎯 Campaign": "campaign",
        "💳 Payer status": "payer_segment",
    }

    dim_label = st.selectbox(
        "Break down active users by",
        list(dim_map.keys()),
        index=0,
        help="Shows how daily active users are distributed across this dimension."
    )
    dim = dim_map[dim_label]

    max_available = int(bd[dim].nunique()) if (not bd.empty and dim in bd.columns) else Config.MAX_TOPN
    topn = st.number_input(
        f"Top N categories (max: {max_available})",
        min_value=1,
        max_value=max(1, max_available),
        value=min(Config.HOME_TOPN_DEFAULT, max_available),
        step=1,
        help="Other categories are grouped into 'Other' to keep charts readable."
    )

    if bd.empty:
        st.info("No breakdown data available for the selected dates.")
    else:
        tmp = bd.groupby(["active_day", dim]).agg(active_users=("active_users", "sum")).reset_index()
        totals = tmp.groupby(dim)["active_users"].sum().sort_values(ascending=False)
        top_cats = totals.head(int(topn)).index.tolist()

        tmp[dim] = np.where(tmp[dim].isin(top_cats), tmp[dim], "Other")

        tmp2 = tmp.groupby(["active_day", dim]).agg(active_users=("active_users", "sum")).reset_index()
        day_total = tmp2.groupby("active_day")["active_users"].sum().reset_index(name="day_total")
        tmp2 = tmp2.merge(day_total, on="active_day", how="left")
        tmp2["share"] = tmp2["active_users"] / tmp2["day_total"].replace(0, np.nan)

        fig_stack = px.bar(
            tmp2, x="active_day", y="share", color=dim,
            title=f"Active User Share by {dim_label} (Daily)",
            hover_data={"active_users": True, "day_total": True, "share": ":.2%"}
        )
        fig_stack.update_layout(barmode="stack", hovermode="x unified", yaxis_tickformat=".0%")
        st.plotly_chart(fig_stack, use_container_width=True)

        comp = tmp2.groupby(dim).agg(active_users=("active_users", "sum")).reset_index().sort_values("active_users", ascending=False)
        fig_top = px.bar(
            comp, x=dim, y="active_users",
            title=f"Total Active Users by {dim_label} (Selected Period)",
            hover_data={"active_users": True}
        )
        fig_top.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown("#### Top categories summary")
        comp["share"] = comp["active_users"] / comp["active_users"].sum()
        st.dataframe(
            comp.assign(share=comp["share"].map(lambda x: f"{x*100:.2f}%")),
            use_container_width=True
        )


# =============================================================================
# Monetization (global filters apply) — replace distributions with over-time lines
# =============================================================================

with tabs[1]:
    st.subheader("🪙 Monetization")
    st.caption("Global filters apply. Focus on revenue trends over time.")

    if not PLOTLY_OK:
        st.warning("Plotly is required here. Install it: `pip install plotly`.")
        st.stop()

    users_n = int(u["user_id"].nunique())
    mean_total = float(u["total_rev_net"].mean())
    med_total = float(u["total_rev_net"].median())
    mean_ad = float(u["ad_rev"].mean())
    mean_iap = float(u["iap_rev_net"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Users (Filtered)",
        f"{users_n:,}",
        help="Distinct users in the filtered user table (windowed KPIs)."
    )
    c2.metric(
        "Avg Revenue per User (Net)",
        _money(mean_total, 6),
        help="Mean net revenue per user (windowed).\n\n"
             "Formulas:\n"
             "ad_rev = rv_rev + fs_rev\n"
             "iap_rev_net = 0.7 × iap_rev\n"
             "total_rev_net = ad_rev + iap_rev_net"
    )
    c3.metric(
        "Median Revenue per User",
        _money(med_total, 6),
        help="Median net revenue per user (more robust when revenue is heavy-tailed)."
    )
    c4.metric(
        "Avg Ad Revenue per User",
        _money(mean_ad, 6),
        help="Mean ad revenue per user (windowed).\n\nFormula:\nad_rev = rv_rev + fs_rev"
    )
    c5.metric(
        "Avg IAP Revenue per User (Net)",
        _money(mean_iap, 6),
        help="Mean IAP revenue per user after store fee.\n\nFormula:\niap_rev_net = 0.7 × iap_rev"
    )

    st.divider()

    # Over-time revenue lines (IAP net, Ad, Total net)
    s = sessions_g[sessions_g["active_day"].notna()][["active_day", "user_id", "rv_revenue", "fs_revenue", "iap_revenue"]].copy()
    s = s.merge(universe[["user_id", "country", "platform", "campaign", "payer_segment"]], on="user_id", how="left")

    s = s[
        s["country"].astype(str).isin(country_f) &
        s["platform"].astype(str).isin(platform_f) &
        s["campaign"].astype(str).isin(campaign_f) &
        s["payer_segment"].astype(str).isin(payer_f)
    ].copy()

    s["ad_rev"] = s["rv_revenue"].fillna(0.0) + s["fs_revenue"].fillna(0.0)
    s["iap_rev_net"] = 0.7 * s["iap_revenue"].fillna(0.0)
    s["total_rev_net"] = s["ad_rev"] + s["iap_rev_net"]

    daily_rev = (
        s.groupby("active_day")
        .agg(ad_rev=("ad_rev", "sum"), iap_rev_net=("iap_rev_net", "sum"), total_rev_net=("total_rev_net", "sum"))
        .reset_index()
        .sort_values("active_day")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_rev["active_day"], y=daily_rev["ad_rev"], mode="lines", name="Ad revenue (RV + FS)"))
    fig.add_trace(go.Scatter(x=daily_rev["active_day"], y=daily_rev["iap_rev_net"], mode="lines", name="IAP revenue (Net)"))
    fig.add_trace(go.Scatter(x=daily_rev["active_day"], y=daily_rev["total_rev_net"], mode="lines", name="Total revenue (Net)"))
    fig.update_layout(
        title="Revenue Performance Over Time (Filtered)",
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Revenue mix + payer share over time (side-by-side)
    left, right = st.columns(2)

    mix = pd.DataFrame({
        "component": ["Rewarded Video (RV)", "Fullscreen / Interstitial (FS)", "IAP (Net)"],
        "value": [
            float(u["rv_rev"].sum()),
            float(u["fs_rev"].sum()),
            float(u["iap_rev_net"].sum()),
        ]
    })
    fig_mix = px.pie(mix, values="value", names="component", title="Revenue Mix (Selected Users)")
    left.plotly_chart(fig_mix, use_container_width=True)

    payer_ts = build_daily_active_payer_share(
        sessions=sessions_g,
        universe=universe,
        country_f=country_f,
        platform_f=platform_f,
        campaign_f=campaign_f
    )
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=payer_ts["active_day"], y=payer_ts["payer_share_active"], mode="lines", name="Payer share"))
    fig_p.update_layout(
        title="Share of Payers Among Active Users (Over Time)",
        xaxis_title="Date",
        yaxis_title="Payer Share",
        hovermode="x unified"
    )
    fig_p.update_yaxes(tickformat=".0%")
    right.plotly_chart(fig_p, use_container_width=True)


# =============================================================================
# Engagement (global filters apply)
# =============================================================================

with tabs[2]:
    st.subheader("🎮 Engagement")
    st.caption("Global filters apply. Distributions are compact, plus daily engagement trends.")

    if not PLOTLY_OK:
        st.warning("Plotly is required here. Install it: `pip install plotly`.")
        st.stop()

    mean_sessions = float(u["sessions"].mean())
    med_sessions = float(u["sessions"].median())
    active_share = float(u["is_active"].mean())
    mean_len = float(u["total_session_length"].mean())
    mean_games = float(u["total_game_count"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Active User Rate (Proxy)",
        _pct(active_share),
        help="Share of users who had at least one session in the selected assignment window.\n\n"
             "Formula:\nis_active = 1 if sessions > 0"
    )
    c2.metric("Avg Sessions per User (Mean)", _num(mean_sessions, 2),
              help="Windowed sessions aggregated per user, then averaged across users.")
    c3.metric("Avg Sessions per User (Median)", _num(med_sessions, 2),
              help="Median sessions per user in the assignment window.")
    c4.metric("Avg Playtime per User", _num(mean_len, 2),
              help="Playtime = Σ session_length per user (windowed), averaged across users.")
    c5.metric("Avg Games per User", _num(mean_games, 2),
              help="Games = Σ game_count per user (windowed), averaged across users.")

    # Distributions: one row
    colL, colR = st.columns(2)
    fig1 = px.histogram(u, x="sessions", nbins=60, title="Distribution of Sessions per User")
    colL.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(u, x="total_session_length", nbins=60, title="Distribution of Total Playtime per User")
    colR.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Over-time engagement rates
    s = sessions_g[sessions_g["active_day"].notna()][["active_day", "user_id", "session_id", "game_count", "session_length"]].copy()
    s = s.merge(universe[["user_id", "country", "platform", "campaign", "payer_segment"]], on="user_id", how="left")
    s = s[
        s["country"].astype(str).isin(country_f) &
        s["platform"].astype(str).isin(platform_f) &
        s["campaign"].astype(str).isin(campaign_f) &
        s["payer_segment"].astype(str).isin(payer_f)
    ].copy()

    daily_eng = (
        s.groupby("active_day")
        .agg(
            dau=("user_id", "nunique"),
            sessions=("session_id", "count"),
            total_game_count=("game_count", "sum"),
        )
        .reset_index()
        .sort_values("active_day")
    )
    daily_eng["sessions_per_user"] = daily_eng["sessions"] / daily_eng["dau"].replace(0, np.nan)
    daily_eng["gamecount_per_user"] = daily_eng["total_game_count"] / daily_eng["dau"].replace(0, np.nan)

    t1, t2 = st.columns(2)

    fig_su = go.Figure()
    fig_su.add_trace(go.Scatter(x=daily_eng["active_day"], y=daily_eng["sessions_per_user"], mode="lines", name="Sessions per user"))
    fig_su.update_layout(
        title="Sessions per Active User (Daily Trend)",
        xaxis_title="Date",
        yaxis_title="Sessions / Active User",
        hovermode="x unified"
    )
    t1.plotly_chart(fig_su, use_container_width=True)

    fig_gu = go.Figure()
    fig_gu.add_trace(go.Scatter(x=daily_eng["active_day"], y=daily_eng["gamecount_per_user"], mode="lines", name="Games per user"))
    fig_gu.update_layout(
        title="Games per Active User (Daily Trend)",
        xaxis_title="Date",
        yaxis_title="Games / Active User",
        hovermode="x unified"
    )
    t2.plotly_chart(fig_gu, use_container_width=True)


# =============================================================================
# Segments (global filters apply)
# =============================================================================

with tabs[3]:
    st.subheader("🔍 Segments")
    st.caption("Global filters apply. Segment table is windowed (per-user) based on the selected assignment window.")

    dim = st.selectbox(
        "Segment dimension",
        ["country", "platform", "campaign", "payer_segment", "manufacturer", "model"],
        index=0,
        help="Segments are created from user attributes (from assigned universe)."
    )

    metric = st.selectbox(
        "Metric",
        ["total_rev_net", "ad_rev", "rv_rev", "fs_rev", "iap_rev_net",
         "sessions", "total_session_length", "total_game_count"],
        index=0,
        help="Metric is computed at user-level within the selected assignment window."
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

    max_available = int(seg.shape[0])
    topn = st.number_input(
        f"Top N segments for charts (max: {max_available})",
        min_value=1,
        max_value=max(1, max_available),
        value=min(Config.SEG_TOPN_DEFAULT, max_available),
        step=1,
        help="Keeps charts readable when there are many segments."
    )

    seg_top = seg.sort_values("mean", ascending=False).head(int(topn))

    fig = px.bar(
        seg_top, x=dim, y="mean",
        title=f"Top Segments by Average {metric}",
        hover_data=["users", "median", "zero_share", "sum"]
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        seg.head(300),
        x="users", y="mean", size="sum",
        hover_name=dim,
        hover_data=["median", "zero_share", "sum"],
        title="Segment Landscape (Users vs Average Metric)"
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# A/B Compare (global filters apply; cohort filter removed; both cohorts always included)
# =============================================================================

with tabs[4]:
    st.subheader("👩🏻‍🔬 A/B Compare")
    st.caption("Global filters apply. Cohort is NOT a filter — control vs test comparison happens here only.")

    kpi = st.selectbox(
        "KPI to compare (control vs test)",
        ["total_rev_net", "ad_rev", "rv_rev", "fs_rev", "iap_rev_net",
         "sessions", "total_session_length", "total_game_count", "iap_transactions"],
        index=0,
        help="Lift is computed as mean(Test) − mean(Control) using the filtered user table."
    )

    colA, colB, colC = st.columns([1, 1, 2])
    use_w = colA.checkbox(
        "Winsorize (robustness)",
        value=True,
        help="Clips extreme values at a chosen quantile to reduce heavy-tail impact."
    )
    winsor_q = colB.slider(
        "Winsor quantile",
        0.95, 0.9999, Config.DEFAULT_WINSOR_Q, step=0.0001,
        help="If threshold collapses to 0 but positive values exist, threshold is recomputed on positives."
    )
    n_boot = colC.slider(
        "Bootstrap iterations",
        500, 5000, Config.DEFAULT_BOOTSTRAP_N, step=500,
        help="Used to compute the 95% bootstrap confidence interval for the mean lift."
    )

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
            "Average Lift (Test − Control)",
            f"{diff:+.6f}",
            delta=f"{rel:+.2f}%" if np.isfinite(rel) else "NA",
            help="Lift definition:\nmean(Test) − mean(Control)\n\n"
                 "CI:\nBootstrap percentile interval (95%)"
        )
        st.write(f"Bootstrap 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}]")

        fig = px.histogram(
            df_ab, x=metric_col, color="cohort",
            barmode="overlay", nbins=60,
            title=f"Distribution of {kpi} (Control vs Test)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Both control and test cohorts are not present after filtering. Adjust filters to compare.")


# =============================================================================
# New Users & Retention (install-day based; strict + rolling + kernel curves)
# =============================================================================

with tabs[5]:
    st.subheader("🆕 New Users & Retention (Install Cohort)")
    st.caption("D0 = install day. Main chart = strict day-K retention. Optional chart = rolling (cumulative) retention. Kernel = intensity from D0 to D30.")

    if not PLOTLY_OK:
        st.warning("Plotly is required here. Install it: `pip install plotly`.")
        st.stop()

    install_min = pd.to_datetime(universe["install_day"].min()).date() if universe["install_day"].notna().any() else g_start
    install_max = pd.to_datetime(universe["install_day"].max()).date() if universe["install_day"].notna().any() else g_end

    st.markdown("#### 📌 Choose D0 (Install day) — single-day cohort")
    d0 = st.date_input(
        "D0 (Install day)",
        value=max(install_min, g_start),
        min_value=install_min,
        max_value=install_max,
        help="Classic cohort definition: users who installed on this exact day."
    )
    c_start, c_end = d0, d0

    u_cohort_filtered = universe[
        (universe["install_day"].dt.date >= c_start) &
        (universe["install_day"].dt.date <= c_end) &
        universe["country"].astype(str).isin(country_f) &
        universe["platform"].astype(str).isin(platform_f) &
        universe["campaign"].astype(str).isin(campaign_f) &
        universe["payer_segment"].astype(str).isin(payer_f)
    ][["user_id", "install_day", "payer_segment"]].copy()

    result = build_install_cohort_kernel(
        sessions=sessions_g,
        universe=universe[universe["user_id"].isin(u_cohort_filtered["user_id"])].copy(),
        cohort_start=pd.to_datetime(c_start).tz_localize("UTC"),
        cohort_end=pd.to_datetime(c_end).tz_localize("UTC"),
        max_days=Config.RETENTION_MAX_DAYS
    )

    ltv = result["ltv_summary"].iloc[0].to_dict()
    n_users = int(ltv.get("cohort_users", 0))

    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Cohort Size (Users)",
        f"{n_users:,}",
        help="Users whose install_day is exactly D0 (after GLOBAL filters)."
    )
    k2.metric(
        "D30 LTV (Net)",
        _money(float(ltv.get("d30_ltv_net", np.nan)), 6),
        help="D30 LTV (net) = Σ total_rev_net over day_since_install ∈ [0,29] / cohort_users"
    )
    k3.metric(
        "IAP Contribution (D30)",
        _pct(float(ltv.get("d30_iap_contribution", np.nan)) if pd.notna(ltv.get("d30_iap_contribution")) else 0.0),
        help="IAP contribution = (Σ iap_rev_net) / (Σ total_rev_net) over day_since_install ∈ [0,29]"
    )

    st.divider()

    retention = result["retention"]
    kernel = result["kernel"]

    if n_users == 0 or retention.empty or kernel.empty:
        st.info("No cohort users for the selected D0 + filters.")
    else:
        st.markdown("#### 📉 Day-K Retention (Strict)")
        fig_strict = go.Figure()
        fig_strict.add_trace(go.Scatter(
            x=retention["day_since_install"],
            y=retention["strict_retention"],
            mode="lines",
            name="Strict retention"
        ))
        fig_strict.update_layout(
            title="Day-K Retention: 해당 Day에 접속한 유저 비율",
            xaxis_title="Days Since Install (K)",
            yaxis_title="Retention",
            hovermode="x unified"
        )
        fig_strict.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_strict, use_container_width=True)

        with st.expander("🔁 Optional: Rolling retention"):
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(
                x=retention["day_since_install"],
                y=retention["rolling_retention"],
                mode="lines",
                name="Rolling retention"
            ))
            fig_ret.update_layout(
                title="Rolling Retention (Cumulative)",
                xaxis_title="Days Since Install (K)",
                yaxis_title="Retention",
                hovermode="x unified"
            )
            fig_ret.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_ret, use_container_width=True)

        st.divider()

        st.markdown("#### 📈 Kernel curves (Intensity from D0 to D30)")
        a, b = st.columns(2)

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=kernel["day_since_install"],
            y=kernel["sessions_per_user"],
            mode="lines",
            name="Sessions per user"
        ))
        fig_s.update_layout(
            title="Sessions per User by Days Since Install",
            xaxis_title="Days Since Install",
            yaxis_title="Sessions / User",
            hovermode="x unified"
        )
        a.plotly_chart(fig_s, use_container_width=True)

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=kernel["day_since_install"],
            y=kernel["playtime_per_user"],
            mode="lines",
            name="Playtime per user"
        ))
        fig_p.update_layout(
            title="Playtime per User by Days Since Install",
            xaxis_title="Days Since Install",
            yaxis_title="Playtime / User",
            hovermode="x unified"
        )
        b.plotly_chart(fig_p, use_container_width=True)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=kernel["day_since_install"],
            y=kernel["rev_per_user"],
            mode="lines",
            name="Revenue per user (net)"
        ))
        fig_r.update_layout(
            title="Net Revenue per User by Days Since Install",
            xaxis_title="Days Since Install",
            yaxis_title="Revenue / User",
            hovermode="x unified"
        )
        st.plotly_chart(fig_r, use_container_width=True)


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
- **Strict retention (install-based, D0 = install_day)**:  
  `strict_retention(K) = P( active on day K )`  
- **Rolling retention (install-based, cumulative)**:  
  `rolling_retention(K) = P( active at least once in days 1..K )`  
        """
    )

with st.expander("📚 Column Glossary (Reference)", expanded=False):
    st.markdown(
        """
- **country**: user location (IP or store region)  
- **first_app_launch_date**: first time user launched the app (used for install_day)  
- **fs_revenue / fs_watched**: fullscreen/interstitial ads revenue / views  
- **rv_revenue / rv_watched**: rewarded video revenue / views  
- **iap_revenue / iap_transactions**: gross IAP revenue / transactions (net = 0.7 × gross)  
- **open_at**: session start timestamp  
- **active_day**: day bucket of open_at  
- **install_day**: day bucket of first_app_launch_date (fallback: first observed active_day)  
- **platform**: iOS / Android  
- **campaign**: acquisition campaign type  
- **assignment_date / assign_day**: experiment assignment date / day bucket  
- **cohort**: control or test (not a global filter; used in A/B Compare)  
- **payer_status**: 0 non-payer, 1 payer  
        """
    )

st.caption("📝 Note: This dashboard uses session logs. True assignment logs are needed for full-universe retention.")
