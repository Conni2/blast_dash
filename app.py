# app.py
# =============================================================================
# 💣 This is Blast! – PO Dashboard (Session-Log A/B Dataset)
#
# Local-first:
# - Reads from local CSV path
# - Optional Parquet cache for faster reload
#
# Key UX rules (this version):
# - Home uses ONLY its own date filter (local)
# - All other tabs use GLOBAL filters (including global date)
# - Cohort is NOT a global filter (only used for comparisons)
# - “Stiff explanations” moved into Streamlit help tooltips where possible
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
# CSV READ SETTINGS
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
# LOAD + CACHE
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
    try:
        df = pd.read_parquet(path_parquet)
        return True, df
    except Exception:
        return False, pd.DataFrame()


def try_write_parquet(df: pd.DataFrame, path_parquet: str) -> bool:
    try:
        df.to_parquet(path_parquet, index=False)
        return True
    except Exception:
        return False


def load_raw_with_parquet_cache(csv_path: str, rebuild: bool = False) -> Tuple[pd.DataFrame, Dict]:
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

    if "install_store" not in d.columns:
        d["install_store"] = np.nan

    # Numeric fills
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

    # install_day fallback: first observed active_day (only for missing launch date)
    d.loc[d["install_day"].isna(), "install_day"] = d.loc[d["install_day"].isna(), "active_day"]

    d["days_since_assignment"] = (d["active_day"] - d["assign_day"]).dt.days.astype("Int64")
    d["is_post_assignment"] = (d["active_day"] >= d["assign_day"]).fillna(False).astype(bool)

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

    daily["sessions_per_dau"] = daily["sessions"] / daily["dau"].replace(0, np.nan)
    daily["rev_per_dau"] = daily["total_rev_net"] / daily["dau"].replace(0, np.nan)
    daily["playtime_per_dau"] = daily["total_session_length"] / daily["dau"].replace(0, np.nan)

    return daily


@st.cache_data(show_spinner=False)
def build_daily_breakdown_active(sessions: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
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
# INSTALL-COHORT RETENTION + KERNEL CURVES (with cohort comparison)
# =============================================================================

@st.cache_data(show_spinner=False)
def build_install_cohort_kernel_by_cohort(
    sessions: pd.DataFrame,
    universe: pd.DataFrame,
    cohort_start: pd.Timestamp,
    cohort_end: pd.Timestamp,
    max_days: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    Install-day cohort analysis (D0=install_day), computed per experiment cohort (control/test).

    Outputs:
    - retention: rolling + strict per day_since_install, with cohort column
    - kernel: sessions/playtime/revenue per user per day_since_install, with cohort column
    - ltv_summary: per cohort D30 LTV (net) + IAP contribution
    """
    u = universe.copy()
    u = u[u["install_day"].notna()].copy()

    cohort_users = u[
        (u["install_day"] >= cohort_start) &
        (u["install_day"] <= cohort_end)
    ][["user_id", "install_day", "payer_segment", "cohort"]].copy()

    if cohort_users.empty:
        return {
            "cohort_users": cohort_users,
            "retention": pd.DataFrame(),
            "kernel": pd.DataFrame(),
            "ltv_summary": pd.DataFrame()
        }

    # Sessions restricted to cohort users
    s = sessions[
        sessions["user_id"].isin(cohort_users["user_id"]) &
        sessions["active_day"].notna() &
        sessions["install_day"].notna()
    ].copy()

    s = s.merge(cohort_users[["user_id", "cohort"]], on="user_id", how="left")
    s["day_since_install"] = (s["active_day"] - s["install_day"]).dt.days.astype("Int64")
    s = s[s["day_since_install"].between(0, max_days)].copy()

    # Precompute revenue components
    s["ad_rev"] = s["rv_revenue"].fillna(0.0) + s["fs_revenue"].fillna(0.0)
    s["iap_rev_net"] = 0.7 * s["iap_revenue"].fillna(0.0)
    s["total_rev_net"] = s["ad_rev"] + s["iap_rev_net"]

    retention_rows = []
    kernel_rows = []
    ltv_rows = []

    for coh in [Config.CONTROL, Config.TEST]:
        u_coh = cohort_users[cohort_users["cohort"] == coh].copy()
        n_users = int(u_coh["user_id"].nunique())
        if n_users == 0:
            continue

        s_coh = s[s["cohort"] == coh].copy()

        # Strict retention: active on day k
        active_days = s_coh[["user_id", "day_since_install"]].drop_duplicates()

        strict = (
            active_days.groupby("day_since_install")
            .agg(active_users=("user_id", "nunique"))
            .reset_index()
            .sort_values("day_since_install")
        )
        strict["cohort"] = coh
        strict["cohort_users"] = n_users
        strict["strict_retention"] = strict["active_users"] / n_users

        # Rolling retention: returned at least once by day K (days 1..K)
        returns = active_days[active_days["day_since_install"] >= 1].copy()
        first_return = returns.groupby("user_id")["day_since_install"].min().reset_index(name="first_return_day")

        ks = pd.DataFrame({"day_since_install": list(range(1, max_days + 1))})
        ks["rolling_retention"] = ks["day_since_install"].apply(
            lambda k: float((first_return["first_return_day"] <= k).mean()) if len(first_return) else 0.0
        )
        ks["cohort"] = coh
        ks["cohort_users"] = n_users

        ret = ks.merge(
            strict[["day_since_install", "strict_retention"]],
            on="day_since_install",
            how="left"
        )
        retention_rows.append(ret)

        # Kernel curves (unconditional per cohort user)
        ker = (
            s_coh.groupby("day_since_install")
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
        ker["cohort"] = coh
        ker["cohort_users"] = n_users
        ker["sessions_per_user"] = ker["sessions"] / n_users
        ker["playtime_per_user"] = ker["playtime"] / n_users
        ker["rev_per_user"] = ker["total_rev_net"] / n_users

        # Ensure all days exist
        all_days = pd.DataFrame({"day_since_install": list(range(0, max_days + 1))})
        ker = all_days.merge(ker, on="day_since_install", how="left")
        ker["cohort"] = ker["cohort"].fillna(coh)
        ker["cohort_users"] = ker["cohort_users"].fillna(n_users)
        for c in ["sessions", "playtime", "ad_rev", "iap_rev_net", "total_rev_net", "sessions_per_user", "playtime_per_user", "rev_per_user"]:
            ker[c] = ker[c].fillna(0)
        kernel_rows.append(ker)

        # D30 LTV net + IAP contribution (0..29)
        s_d30 = s_coh[s_coh["day_since_install"].between(0, 29)].copy()
        ltv_total = float(s_d30["total_rev_net"].sum()) / n_users
        ltv_iap = float(s_d30["iap_rev_net"].sum()) / n_users
        ltv_iap_share = (ltv_iap / ltv_total) if ltv_total > 0 else np.nan

        ltv_rows.append({
            "cohort": coh,
            "cohort_users": n_users,
            "d30_ltv_net": ltv_total,
            "d30_iap_contribution": ltv_iap_share
        })

    retention_out = pd.concat(retention_rows, ignore_index=True) if retention_rows else pd.DataFrame()
    kernel_out = pd.concat(kernel_rows, ignore_index=True) if kernel_rows else pd.DataFrame()
    ltv_out = pd.DataFrame(ltv_rows)

    return {
        "cohort_users": cohort_users,
        "retention": retention_out,
        "kernel": kernel_out,
        "ltv_summary": ltv_out
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
st.caption("Home = quick snapshot. Other tabs follow the GLOBAL filters in the sidebar.")

# ----------------------------
# Sidebar: data source + cache
# ----------------------------
st.sidebar.header("📁 Data Source")

path = st.sidebar.text_input(
    "Local CSV path (.csv or .csv.gz)",
    value=Config.DEFAULT_LOCAL_PATH,
    help="Use a local path. If the file is large, Parquet cache can speed up reload."
)
path = os.path.abspath(os.path.expanduser(path))
if not os.path.exists(path):
    st.error("File not found. Please check the path.")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("⚡ Fast Reload")

rebuild_parquet = st.sidebar.button(
    "Rebuild Parquet Cache",
    help="Forces re-creation of parquet from CSV."
)

with st.spinner("Loading data…"):
    df_raw, cache_info = load_raw_with_parquet_cache(path, rebuild=rebuild_parquet)
    raw = load_and_clean_data(df_raw)
    sessions_all = create_temporal_features(raw)
    universe, universe_info = build_assigned_universe(sessions_all)

    daily_all = build_daily_metrics(sessions_all)
    bd_active_all = build_daily_breakdown_active(sessions_all, universe)

with st.sidebar.expander("Cache status"):
    st.write(cache_info)

with st.sidebar.expander("🧪 Data health"):
    st.write("Users with multiple cohort values are dropped (to avoid contamination).")
    st.json(universe_info)

st.sidebar.divider()

# ----------------------------
# Sidebar: Global Filters
# ----------------------------
st.sidebar.header("🎛️ Global Filters (Non-Home tabs)")

if daily_all.empty:
    st.error("No daily data available (check open_at parsing).")
    st.stop()

global_min_day = pd.to_datetime(daily_all["active_day"].min()).date()
global_max_day = pd.to_datetime(daily_all["active_day"].max()).date()

global_date_range = st.sidebar.date_input(
    "Global date range",
    value=(global_min_day, global_max_day),
    min_value=global_min_day,
    max_value=global_max_day,
    help="Applies to all tabs EXCEPT Home."
)
if isinstance(global_date_range, tuple) and len(global_date_range) == 2:
    g_start, g_end = global_date_range
else:
    g_start, g_end = global_min_day, global_max_day

window = st.sidebar.selectbox(
    "Assignment window for per-user KPIs",
    list(Config.WINDOWS.keys()),
    index=list(Config.WINDOWS.keys()).index("d7"),
    help="Defines the post-assignment window for user-level KPIs (e.g., d7 = days 0–6)."
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
    default=top_country
)

payer_f = st.sidebar.multiselect(
    "Payer segment",
    payer_opts,
    default=payer_opts,
    help="payer if max(payer_status)>0 across observed logs (proxy)."
)

# Apply global date to session logs for non-Home tabs
sessions_g = sessions_all[
    (sessions_all["active_day"].dt.date >= g_start) &
    (sessions_all["active_day"].dt.date <= g_end)
].copy()

# User KPIs for non-Home tabs
with st.spinner("Preparing user KPIs…"):
    user_kpis_g = aggregate_user_kpis_unconditional(sessions_g, universe, window)

u = user_kpis_g[
    user_kpis_g["platform"].astype(str).isin(platform_f) &
    user_kpis_g["campaign"].astype(str).isin(campaign_f) &
    user_kpis_g["country"].astype(str).isin(country_f) &
    user_kpis_g["payer_segment"].astype(str).isin(payer_f)
].copy()

daily_g = daily_all[
    (daily_all["active_day"].dt.date >= g_start) &
    (daily_all["active_day"].dt.date <= g_end)
].copy()

# =============================================================================
# TABS
# =============================================================================

tabs = st.tabs([
    "🏠 Home",
    "🪙 Monetization",
    "🎮 Engagement",
    "🔍 Segments",
    "👩🏻‍🔬 A/B Compare",
    "🆕 New Users & Retention",
])

# =============================================================================
# HOME
# =============================================================================

with tabs[0]:
    st.subheader("🏠 Home snapshot")
    if not PLOTLY_OK:
        st.warning("Plotly is required. Install it: `pip install plotly`.")
        st.stop()

    min_day = pd.to_datetime(daily_all["active_day"].min()).date()
    max_day = pd.to_datetime(daily_all["active_day"].max()).date()

    home_date_range = st.date_input(
        "Home date range",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
        help="Home only (does not affect other tabs)."
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

    cA, cB, cC, cD = st.columns(4)
    dau_avg = float(d["dau"].mean()) if len(d) else 0.0
    sess_per_dau = float(d["sessions_per_dau"].mean()) if len(d) else 0.0
    rev_per_dau = float(d["rev_per_dau"].mean()) if len(d) else 0.0
    total_rev = float(d["total_rev_net"].sum()) if len(d) else 0.0

    cA.metric("Avg daily active users", f"{dau_avg:,.0f}", help="Daily unique users from session logs.")
    cB.metric("Avg sessions per active user", _num(sess_per_dau, 2), help="Daily sessions / DAU, averaged over selected days.")
    cC.metric("Avg net revenue per active user", _money(rev_per_dau, 6), help="(RV+FS + 0.7×IAP) / DAU.")
    cD.metric("Total net revenue", _money(total_rev, 2), help="Sum of net revenue over selected days.")

    st.divider()

    r1, r2 = st.columns(2)
    fig_play = go.Figure()
    fig_play.add_trace(go.Scatter(x=d["active_day"], y=d["total_session_length"], mode="lines", name="Playtime"))
    fig_play.update_layout(title="Total playtime per day", xaxis_title="Date", yaxis_title="Total playtime", hovermode="x unified")
    r1.plotly_chart(fig_play, use_container_width=True)

    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(x=d["active_day"], y=d["total_rev_net"], mode="lines", name="Net revenue"))
    fig_tr.update_layout(title="Total net revenue per day", xaxis_title="Date", yaxis_title="Net revenue", hovermode="x unified")
    r2.plotly_chart(fig_tr, use_container_width=True)

    st.divider()

    g1, g2 = st.columns(2)
    fig_dau = go.Figure()
    fig_dau.add_trace(go.Scatter(x=d["active_day"], y=d["dau"], mode="lines", name="DAU"))
    fig_dau.update_layout(title="Daily active users (trend)", xaxis_title="Date", yaxis_title="Users", hovermode="x unified")
    g1.plotly_chart(fig_dau, use_container_width=True)

    fig_sess = go.Figure()
    fig_sess.add_trace(go.Scatter(x=d["active_day"], y=d["sessions"], mode="lines", name="Sessions"))
    fig_sess.update_layout(title="Sessions (trend)", xaxis_title="Date", yaxis_title="Sessions", hovermode="x unified")
    g2.plotly_chart(fig_sess, use_container_width=True)

    fig_rates = go.Figure()
    fig_rates.add_trace(go.Scatter(x=d["active_day"], y=d["rev_per_dau"], mode="lines", name="Net revenue / active user"))
    fig_rates.add_trace(go.Scatter(x=d["active_day"], y=d["sessions_per_dau"], mode="lines", name="Sessions / active user"))
    fig_rates.update_layout(title="Per-user rates (trend)", xaxis_title="Date", yaxis_title="Value", hovermode="x unified")
    st.plotly_chart(fig_rates, use_container_width=True)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["rv_rev"], stackgroup="one", name="RV"))
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["fs_rev"], stackgroup="one", name="FS"))
    fig_comp.add_trace(go.Scatter(x=d["active_day"], y=d["iap_rev_net"], stackgroup="one", name="IAP (net)"))
    fig_comp.update_layout(title="Where revenue comes from (stacked)", xaxis_title="Date", yaxis_title="Revenue", hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    dim_map = {
        "Country": "country",
        "Platform": "platform",
        "Campaign": "campaign",
        "Payer segment": "payer_segment",
    }

    dim_label = st.selectbox(
        "Break down active users by…",
        list(dim_map.keys()),
        index=0,
        help="Shows how active users are distributed across a category (others grouped into 'Other')."
    )
    dim = dim_map[dim_label]

    max_available = int(bd[dim].nunique()) if (not bd.empty and dim in bd.columns) else Config.MAX_TOPN
    topn = st.number_input(
        f"Top N categories (max {max_available})",
        min_value=1,
        max_value=max(1, max_available),
        value=min(Config.HOME_TOPN_DEFAULT, max_available),
        step=1,
        help="Keeps charts readable by grouping the rest into 'Other'."
    )

    if bd.empty:
        st.info("No breakdown data for the selected dates.")
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
            title=f"Active user mix over time ({dim_label})",
            hover_data={"active_users": True, "day_total": True, "share": ":.2%"}
        )
        fig_stack.update_layout(barmode="stack", hovermode="x unified")
        fig_stack.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_stack, use_container_width=True)

        comp = tmp2.groupby(dim).agg(active_users=("active_users", "sum")).reset_index().sort_values("active_users", ascending=False)
        fig_top = px.bar(comp, x=dim, y="active_users", title=f"Total active users by {dim_label}", hover_data={"active_users": True})
        fig_top.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_top, use_container_width=True)

        st.dataframe(
            comp.assign(share=(comp["active_users"] / comp["active_users"].sum()).map(lambda x: f"{x*100:.2f}%")),
            use_container_width=True
        )

# =============================================================================
# Monetization
# =============================================================================

with tabs[1]:
    st.subheader("🪙 Monetization trends")

    if not PLOTLY_OK:
        st.warning("Plotly is required. Install it: `pip install plotly`.")
        st.stop()

    users_n = int(u["user_id"].nunique())
    mean_total = float(u["total_rev_net"].mean())
    med_total = float(u["total_rev_net"].median())
    mean_ad = float(u["ad_rev"].mean())
    mean_iap = float(u["iap_rev_net"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users in view", f"{users_n:,}", help="Distinct users in the filtered user table.")
    c2.metric("Net revenue / user (mean)", _money(mean_total, 6), help="Net revenue = (RV+FS) + 0.7×IAP.")
    c3.metric("Net revenue / user (median)", _money(med_total, 6), help="Median is robust to heavy tails.")
    c4.metric("Ad revenue / user (mean)", _money(mean_ad, 6), help="Ad revenue = RV + FS.")
    c5.metric("IAP net / user (mean)", _money(mean_iap, 6), help="IAP net = 0.7 × IAP gross.")

    st.divider()

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
    fig.add_trace(go.Scatter(x=daily_rev["active_day"], y=daily_rev["ad_rev"], mode="lines", name="Ad (RV+FS)"))
    fig.add_trace(go.Scatter(x=daily_rev["active_day"], y=daily_rev["iap_rev_net"], mode="lines", name="IAP (net)"))
    fig.add_trace(go.Scatter(x=daily_rev["active_day"], y=daily_rev["total_rev_net"], mode="lines", name="Total (net)"))
    fig.update_layout(title="Revenue over time", xaxis_title="Date", yaxis_title="Net revenue", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    mix = pd.DataFrame({
        "component": ["RV", "FS", "IAP (Net)"],
        "value": [float(u["rv_rev"].sum()), float(u["fs_rev"].sum()), float(u["iap_rev_net"].sum())]
    })
    fig_mix = px.pie(mix, values="value", names="component", title="Revenue mix (selected view)")
    left.plotly_chart(fig_mix, use_container_width=True)

    payer_ts = build_daily_active_payer_share(sessions=sessions_g, universe=universe, country_f=country_f, platform_f=platform_f, campaign_f=campaign_f)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=payer_ts["active_day"], y=payer_ts["payer_share_active"], mode="lines", name="Payer share"))
    fig_p.update_layout(title="Payer share over time (active users)", xaxis_title="Date", yaxis_title="Share", hovermode="x unified")
    fig_p.update_yaxes(tickformat=".0%")
    right.plotly_chart(fig_p, use_container_width=True)

# =============================================================================
# Engagement
# =============================================================================

with tabs[2]:
    st.subheader("🎮 Engagement")

    if not PLOTLY_OK:
        st.warning("Plotly is required. Install it: `pip install plotly`.")
        st.stop()

    mean_sessions = float(u["sessions"].mean())
    med_sessions = float(u["sessions"].median())
    active_share = float(u["is_active"].mean())
    mean_len = float(u["total_session_length"].mean())
    mean_games = float(u["total_game_count"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active share (windowed)", _pct(active_share), help="Share of users with ≥1 session in the assignment window.")
    c2.metric("Sessions / user (mean)", _num(mean_sessions, 2))
    c3.metric("Sessions / user (median)", _num(med_sessions, 2))
    c4.metric("Playtime / user (mean)", _num(mean_len, 2), help="Sum of session_length per user within the window.")
    c5.metric("Games / user (mean)", _num(mean_games, 2), help="Sum of game_count per user within the window.")

    colL, colR = st.columns(2)
    colL.plotly_chart(px.histogram(u, x="sessions", nbins=60, title="Sessions per user (distribution)"), use_container_width=True)
    colR.plotly_chart(px.histogram(u, x="total_session_length", nbins=60, title="Playtime per user (distribution)"), use_container_width=True)

    st.divider()

    s = sessions_g[sessions_g["active_day"].notna()][["active_day", "user_id", "session_id", "game_count"]].copy()
    s = s.merge(universe[["user_id", "country", "platform", "campaign", "payer_segment"]], on="user_id", how="left")
    s = s[
        s["country"].astype(str).isin(country_f) &
        s["platform"].astype(str).isin(platform_f) &
        s["campaign"].astype(str).isin(campaign_f) &
        s["payer_segment"].astype(str).isin(payer_f)
    ].copy()

    daily_eng = (
        s.groupby("active_day")
        .agg(dau=("user_id", "nunique"), sessions=("session_id", "count"), total_game_count=("game_count", "sum"))
        .reset_index()
        .sort_values("active_day")
    )
    daily_eng["sessions_per_user"] = daily_eng["sessions"] / daily_eng["dau"].replace(0, np.nan)
    daily_eng["gamecount_per_user"] = daily_eng["total_game_count"] / daily_eng["dau"].replace(0, np.nan)

    t1, t2 = st.columns(2)

    fig_su = go.Figure()
    fig_su.add_trace(go.Scatter(x=daily_eng["active_day"], y=daily_eng["sessions_per_user"], mode="lines", name="Sessions / user"))
    fig_su.update_layout(title="Sessions per user over time", xaxis_title="Date", yaxis_title="Sessions / user", hovermode="x unified")
    t1.plotly_chart(fig_su, use_container_width=True)

    fig_gu = go.Figure()
    fig_gu.add_trace(go.Scatter(x=daily_eng["active_day"], y=daily_eng["gamecount_per_user"], mode="lines", name="Games / user"))
    fig_gu.update_layout(title="Games per user over time", xaxis_title="Date", yaxis_title="Games / user", hovermode="x unified")
    t2.plotly_chart(fig_gu, use_container_width=True)

# =============================================================================
# Segments
# =============================================================================

with tabs[3]:
    st.subheader("🔍 Segment view")

    dim = st.selectbox(
        "Segment by",
        ["country", "platform", "campaign", "payer_segment", "manufacturer", "model"],
        index=0,
        help="Segments are based on user attributes (from the assigned-user universe)."
    )

    metric = st.selectbox(
        "Metric to compare",
        ["total_rev_net", "ad_rev", "rv_rev", "fs_rev", "iap_rev_net",
         "sessions", "total_session_length", "total_game_count"],
        index=0,
        help="Metrics are user-level within the selected assignment window."
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
        f"Top N for charts (max {max_available})",
        min_value=1,
        max_value=max(1, max_available),
        value=min(Config.SEG_TOPN_DEFAULT, max_available),
        step=1,
        help="Shows the top segments ranked by mean(metric)."
    )

    seg_top = seg.sort_values("mean", ascending=False).head(int(topn))

    fig = px.bar(seg_top, x=dim, y="mean", title=f"Top segments by average {metric}", hover_data=["users", "median", "zero_share", "sum"])
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        seg.head(300),
        x="users", y="mean", size="sum",
        hover_name=dim, hover_data=["median", "zero_share", "sum"],
        title="Segment map (size = total metric sum)"
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# A/B Compare
# =============================================================================

with tabs[4]:
    st.subheader("👩🏻‍🔬 A/B Compare (control vs test)")

    kpi = st.selectbox(
        "KPI",
        ["total_rev_net", "ad_rev", "rv_rev", "fs_rev", "iap_rev_net",
         "sessions", "total_session_length", "total_game_count", "iap_transactions"],
        index=0,
        help="Lift is computed as mean(test) − mean(control) on the current filtered user table."
    )

    colA, colB, colC = st.columns([1, 1, 2])
    use_w = colA.checkbox("Winsorize outliers", value=True, help="Clips extremes at a high quantile to reduce heavy-tail impact.")
    winsor_q = colB.slider("Winsor quantile", 0.95, 0.9999, Config.DEFAULT_WINSOR_Q, step=0.0001, help="Quantile used as an upper cap.")
    n_boot = colC.slider("Bootstrap iterations", 500, 5000, Config.DEFAULT_BOOTSTRAP_N, step=500, help="Used to form a 95% CI for mean lift.")

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

        st.metric("Mean lift (test − control)", f"{diff:+.6f}", delta=f"{rel:+.2f}%" if np.isfinite(rel) else "NA",
                  help="Mean lift computed on the current filtered user table.")
        st.write(f"Bootstrap 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}]")

        fig = px.histogram(df_ab, x=metric_col, color="cohort", barmode="overlay", nbins=60, title=f"{kpi}: distribution by cohort")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Both control and test are not present after filtering.")

# =============================================================================
# New Users & Retention (cohort comparison)
# =============================================================================

with tabs[5]:
    st.subheader("🆕 New users & retention (install-day cohort)")

    if not PLOTLY_OK:
        st.warning("Plotly is required. Install it: `pip install plotly`.")
        st.stop()

    install_min = pd.to_datetime(universe["install_day"].min()).date() if universe["install_day"].notna().any() else g_start
    install_max = pd.to_datetime(universe["install_day"].max()).date() if universe["install_day"].notna().any() else g_end

    cohort_range = st.date_input(
        "Install cohort range",
        value=(max(install_min, g_start), min(install_max, g_end)),
        min_value=install_min,
        max_value=install_max,
        help="Cohort users are defined by install_day within this range (then filters are applied)."
    )
    if isinstance(cohort_range, tuple) and len(cohort_range) == 2:
        c_start, c_end = cohort_range
    else:
        c_start, c_end = max(install_min, g_start), min(install_max, g_end)

    # Filter cohort users by the SAME global filters (except cohort — we compare both)
    cohort_users_filtered = universe[
        (universe["install_day"].dt.date >= c_start) &
        (universe["install_day"].dt.date <= c_end) &
        universe["country"].astype(str).isin(country_f) &
        universe["platform"].astype(str).isin(platform_f) &
        universe["campaign"].astype(str).isin(campaign_f) &
        universe["payer_segment"].astype(str).isin(payer_f)
    ].copy()

    result = build_install_cohort_kernel_by_cohort(
        sessions=sessions_g,
        universe=cohort_users_filtered,
        cohort_start=pd.to_datetime(c_start).tz_localize("UTC"),
        cohort_end=pd.to_datetime(c_end).tz_localize("UTC"),
        max_days=Config.RETENTION_MAX_DAYS
    )

    retention = result["retention"]
    kernel = result["kernel"]
    ltv = result["ltv_summary"]

    if ltv.empty or retention.empty or kernel.empty:
        st.info("No cohort users found for the selected install range + filters.")
    else:
        # KPI cards per cohort
        cA, cB = st.columns(2)

        def _cohort_row(coh: str) -> Dict:
            r = ltv[ltv["cohort"] == coh]
            if r.empty:
                return {"cohort_users": 0, "d30_ltv_net": np.nan, "d30_iap_contribution": np.nan}
            return r.iloc[0].to_dict()

        ctrl = _cohort_row(Config.CONTROL)
        test = _cohort_row(Config.TEST)

        cA.metric(
            "Control: cohort size / D30 LTV / IAP share",
            f"{int(ctrl['cohort_users']):,} | {_money(float(ctrl['d30_ltv_net']), 6)} | {_pct(float(ctrl['d30_iap_contribution']) if pd.notna(ctrl['d30_iap_contribution']) else 0.0)}",
            help="D30 LTV = net revenue per cohort user over day_since_install 0–29. IAP share = IAP(net)/Total(net)."
        )
        cB.metric(
            "Test: cohort size / D30 LTV / IAP share",
            f"{int(test['cohort_users']):,} | {_money(float(test['d30_ltv_net']), 6)} | {_pct(float(test['d30_iap_contribution']) if pd.notna(test['d30_iap_contribution']) else 0.0)}",
            help="Same definition as control, computed on the test cohort."
        )

        st.divider()

        # Rolling retention (compare cohorts)
        fig_ret = go.Figure()
        for coh, label in [(Config.CONTROL, "Control"), (Config.TEST, "Test")]:
            r = retention[retention["cohort"] == coh].sort_values("day_since_install")
            fig_ret.add_trace(go.Scatter(
                x=r["day_since_install"],
                y=r["rolling_retention"],
                mode="lines",
                name=label
            ))
        fig_ret.update_layout(
            title="Rolling retention (did they come back by day K?)",
            xaxis_title="Days since install (K)",
            yaxis_title="Retention",
            hovermode="x unified"
        )
        fig_ret.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ret, use_container_width=True)

        with st.expander("Optional: strict retention (active exactly on day K)"):
            fig_strict = go.Figure()
            for coh, label in [(Config.CONTROL, "Control"), (Config.TEST, "Test")]:
                r = retention[retention["cohort"] == coh].sort_values("day_since_install")
                fig_strict.add_trace(go.Scatter(
                    x=r["day_since_install"],
                    y=r["strict_retention"],
                    mode="lines",
                    name=label
                ))
            fig_strict.update_layout(
                title="Strict retention (active on day K)",
                xaxis_title="Days since install (K)",
                yaxis_title="Retention",
                hovermode="x unified"
            )
            fig_strict.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_strict, use_container_width=True)

        st.divider()

        # Kernel curves: sessions/playtime/revenue per user (compare cohorts)
        k1, k2 = st.columns(2)

        fig_s = go.Figure()
        for coh, label in [(Config.CONTROL, "Control"), (Config.TEST, "Test")]:
            r = kernel[kernel["cohort"] == coh].sort_values("day_since_install")
            fig_s.add_trace(go.Scatter(x=r["day_since_install"], y=r["sessions_per_user"], mode="lines", name=label))
        fig_s.update_layout(title="Sessions per user after install", xaxis_title="Days since install", yaxis_title="Sessions / user", hovermode="x unified")
        k1.plotly_chart(fig_s, use_container_width=True)

        fig_p = go.Figure()
        for coh, label in [(Config.CONTROL, "Control"), (Config.TEST, "Test")]:
            r = kernel[kernel["cohort"] == coh].sort_values("day_since_install")
            fig_p.add_trace(go.Scatter(x=r["day_since_install"], y=r["playtime_per_user"], mode="lines", name=label))
        fig_p.update_layout(title="Playtime per user after install", xaxis_title="Days since install", yaxis_title="Playtime / user", hovermode="x unified")
        k2.plotly_chart(fig_p, use_container_width=True)

        fig_r = go.Figure()
        for coh, label in [(Config.CONTROL, "Control"), (Config.TEST, "Test")]:
            r = kernel[kernel["cohort"] == coh].sort_values("day_since_install")
            fig_r.add_trace(go.Scatter(x=r["day_since_install"], y=r["rev_per_user"], mode="lines", name=label))
        fig_r.update_layout(title="Net revenue per user after install", xaxis_title="Days since install", yaxis_title="Net revenue / user", hovermode="x unified")
        st.plotly_chart(fig_r, use_container_width=True)

# =============================================================================
# Footer
# =============================================================================

st.divider()
with st.expander("📌 Definitions (quick reference)"):
    st.markdown(
        """
- **Net revenue** = (RV revenue + FS revenue) + 0.7 × IAP revenue  
- **Rolling retention (install-based)**: by day K, did the user return at least once since install?  
- **Strict retention (install-based)**: active specifically on day K  
        """
    )
st.caption("Note: session-log dataset → retention is observable only for users who appear in logs.")
