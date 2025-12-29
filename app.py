import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Overexposure Checker", layout="wide")
st.title("Overexposure Checker (Fills + Trades)")

st.write(
    """
    Upload a **fills CSV** and a **trades CSV** for an account.  
    The app will reconstruct positions from fills, detect overexposed intervals,  
    and highlight trades that overlap those intervals.
    """
)

# ---------------------------------------------------------
# Helpers & constants
# ---------------------------------------------------------

MINI_ROOTS = [
    "6A", "6B", "6C", "6E", "6J", "6N", "6S",
    "NQ", "RTY", "ES", "NKD",
    "HE", "LE",
    "HG", "GC", "PL", "SI",
    "ZC", "ZS", "ZM", "ZL", "ZW", "YM",
    "CL", "QM", "HO", "NG", "RB", "QG",
    "MBT", "MET",  # treat as mini even though named Micro
]

MICRO_ROOTS = [
    "M6A", "M6E",
    "MNQ", "M2K", "MES",
    "MYM",
    "MGC", "SIL",
    "MCL",
]

ROOT_TYPES = {root: "mini" for root in MINI_ROOTS}
ROOT_TYPES.update({root: "micro" for root in MICRO_ROOTS})
KNOWN_ROOTS = sorted(ROOT_TYPES.keys(), key=len, reverse=True)


def get_root_code(asset: str) -> str:
    code = str(asset).strip().upper()
    for root in KNOWN_ROOTS:
        if code.startswith(root):
            return root
    return code


def ratio_to_micro_weight(ratio: str) -> float:
    if ratio == "1:1":
        return 1.0
    if ratio == "1:5":
        return 1.0 / 5.0
    if ratio == "1:10":
        return 1.0 / 10.0
    return 1.0


def get_instrument_weight(asset: str, micro_weight: float) -> float:
    root = get_root_code(asset)
    t = ROOT_TYPES.get(root, "mini")
    if t == "mini":
        return 1.0
    if t == "micro":
        return micro_weight
    return 1.0


def parse_money_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("[$,]", "", regex=True),
        errors="coerce",
    )


def normalize_col_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def get_column(df: pd.DataFrame, candidates, required=True):
    norm_map = {normalize_col_name(col): col for col in df.columns}
    for cand in candidates:
        key = normalize_col_name(cand)
        if key in norm_map:
            return norm_map[key]
    if required:
        raise ValueError(f"Missing required column. Tried: {', '.join(candidates)}")
    return None


def parse_tradovate_datetime(series: pd.Series) -> pd.Series:
    """
    Handles strings like 'Feb 11 2025, 07:00:40 PM EST'.
    1) Strip trailing timezone, 2) parse with fixed format,
    3) fallback to generic to_datetime if needed.
    """
    s = series.astype(str).str.strip()
    s_clean = s.str.replace(r"\s+[A-Z]{3}$", "", regex=True)
    dt = pd.to_datetime(s_clean, format="%b %d %Y, %I:%M:%S %p", errors="coerce")
    if dt.notna().sum() < len(dt) * 0.5:
        dt = pd.to_datetime(series, errors="coerce")
    return dt


def detect_overexposed_intervals_from_fills(
    fills_df: pd.DataFrame,
    micro_weight: float,
    max_mini_equiv: float,
    min_duration_sec: float,
):
    """
    Using fills:
      - maintain net position per symbol over time
      - for each interval between fills, compute mini-equivalent exposure
      - if exposure > max_mini_equiv AND interval length >= min_duration_sec, record an overexposed interval
    """
    df = fills_df.dropna(subset=["timestamp_dt", "quantity"]).copy()
    df = df.sort_values("timestamp_dt").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    positions = {}  # symbol -> net contracts (signed)
    intervals = []

    gid = 0
    for i, row in df.iterrows():
        if i > 0:
            start = df.loc[i - 1, "timestamp_dt"]
            end = row["timestamp_dt"]
            duration = (end - start).total_seconds()
            if duration > 0:
                snapshot = {}
                for asset, pos in positions.items():
                    if pos == 0:
                        continue
                    contracts = abs(pos)
                    weight = get_instrument_weight(asset, micro_weight)
                    exposure = contracts * weight
                    if contracts != 0:
                        snapshot[asset] = {
                            "contracts": contracts,
                            "weight": weight,
                            "exposure": exposure,
                            "root": get_root_code(asset),
                        }
                if snapshot:
                    total_exposure = sum(v["exposure"] for v in snapshot.values())
                    if total_exposure > max_mini_equiv and duration >= min_duration_sec:
                        gid += 1
                        root_exposure = {}
                        for info in snapshot.values():
                            r = info["root"]
                            root_exposure[r] = root_exposure.get(r, 0.0) + info["exposure"]
                        intervals.append(
                            {
                                "GroupID": gid,
                                "IntervalStart": start,
                                "IntervalEnd": end,
                                "Duration_sec": duration,
                                "TotalExposureMiniEq": total_exposure,
                                "ExposureByRoot": ", ".join(
                                    f"{r}: {v:.2f}" for r, v in root_exposure.items()
                                ),
                            }
                        )

        qty = row["quantity"]
        action = str(row["action"]).strip().lower()
        asset = row["asset"]

        if action == "buy":
            delta = qty
        elif action == "sell":
            delta = -qty
        else:
            continue
        positions[asset] = positions.get(asset, 0.0) + delta

    return pd.DataFrame(intervals)


def compute_trade_overexposure(trades_df, intervals_df):
    """
    For each trade, check overlap with each overexposed interval.
    Mark Overexposed = True if overlap > 0.
    Also compute total seconds spent in overexposed intervals.
    """
    trades = trades_df.copy()
    trades["Overexposed"] = False
    trades["Overexp_Seconds"] = 0.0
    trades["Overexp_Groups"] = ""

    if intervals_df is None or intervals_df.empty:
        return trades

    group_ids_per_trade = [[] for _ in range(len(trades))]
    overexp_seconds = np.zeros(len(trades))

    for _, iv in intervals_df.iterrows():
        iv_start = iv["IntervalStart"]
        iv_end = iv["IntervalEnd"]
        gid = iv["GroupID"]

        # trades that could overlap
        mask = (trades["Close_dt"] > iv_start) & (trades["Open_dt"] < iv_end)
        idx = np.where(mask)[0]
        for i in idx:
            t_open = trades.iloc[i]["Open_dt"]
            t_close = trades.iloc[i]["Close_dt"]
            overlap_start = max(t_open, iv_start)
            overlap_end = min(t_close, iv_end)
            overlap = (overlap_end - overlap_start).total_seconds()
            if overlap > 0:
                overexp_seconds[i] += overlap
                group_ids_per_trade[i].append(int(gid))

    trades["Overexp_Seconds"] = overexp_seconds
    trades["Overexposed"] = trades["Overexp_Seconds"] > 0
    trades["Overexp_Groups"] = [
        ",".join(str(g) for g in sorted(set(gs))) if gs else "" for gs in group_ids_per_trade
    ]
    return trades


def highlight_overexposed(row):
    if row.get("Overexposed", False):
        return ["background-color: #ffcccc"] * len(row)  # light red
    else:
        return [""] * len(row)


# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
fills_file = st.file_uploader("Upload **Fills CSV**", type=["csv"], key="fills")
trades_file = st.file_uploader("Upload **Trades CSV**", type=["csv"], key="trades")

st.sidebar.header("Settings")

ratio_option = st.sidebar.selectbox(
    "Mini : Micro ratio",
    ["1:1", "1:5", "1:10"],
    index=1,
)

max_mini_equiv = st.sidebar.number_input(
    "Max allowed exposure (mini-equivalents)",
    min_value=0.0,
    value=5.0,
    step=0.5,
)

min_overexp_duration = st.sidebar.number_input(
    "Min overexposure duration (seconds) to flag",
    min_value=0.0,
    value=0.0,
    step=1.0,
)

if fills_file is None or trades_file is None:
    st.info("Upload both **fills** and **trades** CSV files to run the analysis.")
    st.stop()

micro_weight = ratio_to_micro_weight(ratio_option)

# ---------------------------------------------------------
# Parse Fills
# ---------------------------------------------------------
fills_raw = pd.read_csv(fills_file)
fills = fills_raw.copy()

# flexible column resolution
f_action_col = get_column(fills, ["action", "Action", "side", "Side"])
f_symbol_col = get_column(fills, ["asset", "Symbol", "Instrument"])
f_qty_col = get_column(fills, ["quantity", "Qty", "qty", "Contracts"])
f_ts_col = get_column(fills, ["timestamp", "Time", "time", "DateTime"])

fills["action"] = fills[f_action_col].astype(str)
fills["asset"] = fills[f_symbol_col].astype(str)
fills["quantity"] = pd.to_numeric(fills[f_qty_col], errors="coerce")
fills["timestamp_dt"] = parse_tradovate_datetime(fills[f_ts_col])

fills = fills.dropna(subset=["timestamp_dt", "quantity"]).copy()

# ---------------------------------------------------------
# Parse Trades
# ---------------------------------------------------------
trades_raw = pd.read_csv(trades_file)
trades = trades_raw.copy()

t_sym_col = get_column(trades, ["Symbol", "asset", "Instrument"])
t_open_col = get_column(trades, ["Open Time", "Open", "OpenTime"])
t_close_col = get_column(trades, ["Close Time", "Close", "CloseTime"])
t_pnl_col = get_column(trades, ["Net Profit", "NetProfit", "Net PnL", "P&L"], required=False)

trades["Symbol"] = trades[t_sym_col].astype(str)
trades["RootCode"] = trades["Symbol"].apply(get_root_code)

trades["Open_dt"] = parse_tradovate_datetime(trades[t_open_col])
trades["Close_dt"] = parse_tradovate_datetime(trades[t_close_col])
trades["Duration_sec"] = (trades["Close_dt"] - trades["Open_dt"]).dt.total_seconds()

if t_pnl_col is not None:
    trades["NetPnL"] = parse_money_to_float(trades[t_pnl_col])
else:
    trades["NetPnL"] = np.nan

# optionally account info if present
acc_id_col = get_column(trades, ["Account ID", "AccountId", "Account"], required=False)
acc_name_col = get_column(trades, ["Account Name", "AccountName"], required=False)

if acc_id_col is not None:
    trades["AccountID"] = trades[acc_id_col].astype(str)
else:
    trades["AccountID"] = ""

if acc_name_col is not None:
    trades["AccountName"] = trades[acc_name_col].astype(str)
else:
    trades["AccountName"] = ""

# ---------------------------------------------------------
# Detect overexposed intervals from fills
# ---------------------------------------------------------
intervals_df = detect_overexposed_intervals_from_fills(
    fills_df=fills,
    micro_weight=micro_weight,
    max_mini_equiv=max_mini_equiv,
    min_duration_sec=min_overexp_duration,
)

# ---------------------------------------------------------
# Flag trades
# ---------------------------------------------------------
trades_flagged = compute_trade_overexposure(trades, intervals_df)

# ---------------------------------------------------------
# Output
# ---------------------------------------------------------
st.subheader("Summary")

total_trades = len(trades_flagged)
overexp_trades = trades_flagged["Overexposed"].sum()
pct_overexp = (overexp_trades / total_trades * 100) if total_trades > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total trades", f"{total_trades}")
c2.metric("Overexposed trades", f"{int(overexp_trades)}")
c3.metric("Overexposed trades (%)", f"{pct_overexp:.1f}%")

st.write(
    f"Mini:Micro ratio = **{ratio_option}** "
    f"→ Micro weight = **{micro_weight:.3f} mini-equivalents per micro**."
)
st.write(
    f"Overexposure = total mini-equivalent exposure **> {max_mini_equiv}** "
    f"for at least **{min_overexp_duration} seconds**."
)

if intervals_df.empty:
    st.warning("No overexposed intervals detected with the current settings.")
else:
    st.markdown("**Detected overexposed intervals (from fills):**")
    st.dataframe(
        intervals_df.sort_values("IntervalStart"),
        use_container_width=True,
    )

st.markdown("### Trades (overexposed trades highlighted)")

display_cols = [
    "AccountID",
    "AccountName",
    "Symbol",
    "RootCode",
    "Open_dt",
    "Close_dt",
    "Duration_sec",
    "NetPnL",
    "Overexposed",
    "Overexp_Seconds",
    "Overexp_Groups",
]
display_cols = [c for c in display_cols if c in trades_flagged.columns]

styled = trades_flagged[display_cols].sort_values("Open_dt").style.apply(
    highlight_overexposed, axis=1
)
st.dataframe(styled, use_container_width=True)
