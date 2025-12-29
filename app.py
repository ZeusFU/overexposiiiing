import streamlit as st
import pandas as pd

st.title("Overexposure & Net PnL Analyzer (Fills + Trades)")

st.write(
    """
    Upload **both** a fills CSV and a trades CSV (like Tradovate exports), then:

    1. Choose the **Mini:Micro contract ratio** for this account (1:1, 1:5, 1:10).
    2. Enter the **max allowed simultaneous exposure in mini-equivalents**.
    3. Set a **minimum overexposure duration (seconds)** for the
       "Overexposure ≥ Xs" view.

    The app will:

    - Use the **root Code** (e.g. NQ/MNQ, ES/MES, MGC, SIL, MBT, MET).
    - Classify each product as **mini** or **micro** (MBT & MET treated as minis).
    - Apply mini-equivalent weights:
        - Minis (and MBT/MET): 1.0 mini-equivalent per contract.
        - Micros: (1 / chosen ratio) mini-equivalent per contract.
    - Reconstruct positions over time from fills and detect intervals where
      **total mini-equivalent exposure exceeds the max**.
    - Compute:
        - All overexposed intervals (no duration filter).
        - Overexposed intervals lasting **≥ X seconds**.
    - Cross-link those intervals to **trades** by time overlap.
    - Use **Net PnL** from the trades export to summarize:
        - Total Net PnL (all trades)
        - Net PnL from all overexposure
        - Net PnL from overexposure ≥ X seconds
        - Overexposure ≥ Xs as % of total Net PnL
    - For each overexposed interval (≥ Xs), show:
        - Exposure snapshot (per symbol)
        - Overlapping trades with **Bias** and **Mini/Micro** classification.
    - At the end, prepare a **top-3 overexposure narrative copy** ready to paste.
    """
)

# --- File uploaders ---------------------------------------------------------
fills_file = st.file_uploader("Upload fills CSV", type=["csv"], key="fills")
trades_file = st.file_uploader("Upload trades CSV", type=["csv"], key="trades")

# --- User-selected Mini:Micro ratio ----------------------------------------
ratio_option = st.selectbox(
    "Mini : Micro contract ratio",
    ["1:1", "1:5", "1:10"],
    index=1,  # default to 1:5
)

if ratio_option == "1:1":
    micro_weight_global = 1.0
elif ratio_option == "1:5":
    micro_weight_global = 1.0 / 5.0
elif ratio_option == "1:10":
    micro_weight_global = 1.0 / 10.0
else:
    micro_weight_global = 1.0  # fallback

max_mini_equiv = st.number_input(
    "Max allowed simultaneous exposure (in mini-equivalents)",
    min_value=0.0,
    value=5.0,
    step=0.5,
)

min_duration_seconds = st.number_input(
    "Minimum overexposure duration for 'Overexposure ≥ Xs' (seconds)",
    min_value=0,
    value=0,
    step=1,
)

# --------------------------------------------------------------------
# Root Code classification using instrument list logic
# --------------------------------------------------------------------
MINI_ROOTS = [
    # CME FX & index minis
    "6A", "6B", "6C", "6E", "6J", "6N", "6S",
    "NQ", "RTY", "ES", "NKD",
    "HE", "LE",
    # Metals (full/mini)
    "HG", "GC", "PL", "SI",
    # Grains & equity
    "ZC", "ZS", "ZM", "ZL", "ZW", "YM",
    # Energies
    "CL", "QM", "HO", "NG", "RB", "QG",
    # Cryptos treated as minis:
    "MBT", "MET",
]

MICRO_ROOTS = [
    # FX & index micros
    "M6A", "M6E",
    "MNQ", "M2K", "MES",
    "MYM",
    # Metals micros
    "MGC", "SIL",
    # Energy micros
    "MCL",
]

ROOT_TYPES = {root: "mini" for root in MINI_ROOTS}
ROOT_TYPES.update({root: "micro" for root in MICRO_ROOTS})

# Known roots list for prefix matching (longest-first)
KNOWN_ROOTS = sorted(ROOT_TYPES.keys(), key=len, reverse=True)


def get_root_code(asset: str) -> str:
    """Return the ROOT Code (e.g. 'MNQ', 'ES') by prefix match."""
    code = str(asset).strip().upper()
    for root in KNOWN_ROOTS:
        if code.startswith(root):
            return root
    return code  # unknown root, treated as mini by default


def get_instrument_weight(asset: str, micro_weight: float) -> float:
    """
    Return the mini-equivalent weight for a given symbol based on its
    root Code and the user-selected mini:micro ratio.

    - Minis: 1.0
    - Micros: micro_weight (1/ratio)
    - Unknown roots: default to 1.0
    """
    root = get_root_code(asset)
    type_ = ROOT_TYPES.get(root, "mini")
    if type_ == "mini":
        return 1.0
    elif type_ == "micro":
        return micro_weight
    else:
        return 1.0


def detect_overexposure_intervals(
    df: pd.DataFrame,
    max_mini_equiv: float,
    micro_weight: float,
):
    """
    Detect ALL overexposed intervals (without duration filtering).

    From a fills DataFrame with columns:
        - action (Buy/Sell)
        - asset  (symbol, e.g. MNQZ5, NQZ5, SILH4, etc.)
        - quantity
        - timestamp

    Returns:
        intervals: list of dicts (one row per overexposed interval)
        detail_rows: list of dicts (one row per symbol per interval)
        cleaned_df: cleaned & time-ordered fills used in the computation
    """
    required_cols = {"action", "asset", "quantity", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fills CSV is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["quantity", "timestamp_dt"])
    df = df.sort_values("timestamp_dt").reset_index(drop=True)

    if df.empty:
        return [], [], df

    positions = {}  # asset (full symbol) -> net position (signed)
    intervals = []
    detail_rows = []
    group_id = 0

    for i, row in df.iterrows():
        # Interval between fill i-1 and i uses positions BEFORE applying fill i
        if i > 0:
            start = df.loc[i - 1, "timestamp_dt"]
            end = row["timestamp_dt"]
            duration = (end - start).total_seconds()

            # Snapshot mini-equivalent exposure per symbol at this interval
            snapshot = {}  # asset -> info dict
            for asset, pos in positions.items():
                if pos == 0:
                    continue
                contracts = abs(pos)
                if contracts == 0:
                    continue

                root = get_root_code(asset)
                weight = get_instrument_weight(asset, micro_weight)
                exposure = contracts * weight  # mini-equivalent exposure

                snapshot[asset] = {
                    "root": root,
                    "contracts": contracts,
                    "weight": weight,
                    "exposure": exposure,
                }

            if snapshot:
                total_exposure = sum(info["exposure"] for info in snapshot.values())

                # Overexposure condition: total exposure > max
                if total_exposure > max_mini_equiv:
                    group_id += 1

                    # Aggregate exposure by root for the summary
                    root_exposure = {}
                    for info in snapshot.values():
                        r = info["root"]
                        root_exposure[r] = root_exposure.get(r, 0.0) + info["exposure"]

                    intervals.append(
                        {
                            "Group ID": group_id,
                            "Interval Start": start,
                            "Interval End": end,
                            "Duration_sec": duration,
                            "Total Mini-Equiv Exposure": total_exposure,
                            "Exposure by Root (mini-equiv)": ", ".join(
                                f"{r}: {root_exposure[r]:.2f}"
                                for r in sorted(root_exposure)
                            ),
                        }
                    )

                    # Detailed rows (one per symbol per interval)
                    for asset, info in snapshot.items():
                        detail_rows.append(
                            {
                                "Group ID": group_id,
                                "Interval Start": start,
                                "Interval End": end,
                                "Duration_sec": duration,
                                "Symbol": asset,
                                "Root Code": info["root"],
                                "Position Contracts": info["contracts"],
                                "Mini-Equiv Weight": info["weight"],
                                "Mini-Equiv Exposure": info["exposure"],
                            }
                        )

        # Apply current fill to update positions
        qty = df.loc[i, "quantity"]
        action = str(row["action"]).strip().lower()
        asset = row["asset"]

        if action == "buy":
            delta = qty
        elif action == "sell":
            delta = -qty
        else:
            # Unknown action, ignore
            continue

        positions[asset] = positions.get(asset, 0.0) + delta

    return intervals, detail_rows, df


def parse_money_to_float(series: pd.Series) -> pd.Series:
    """Convert a Series of money strings like '$190.50' to floats 190.50."""
    return pd.to_numeric(series.astype(str).str.replace("[$,]", "", regex=True), errors="coerce")


def infer_bias_column(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort inference of LONG/SHORT bias per trade.
    Looks for columns like 'Side', 'Direction', or 'Bias'.
    """
    bias_source_col = None
    for col in df.columns:
        lc = col.lower()
        if "side" in lc or "direction" in lc or "bias" in lc:
            bias_source_col = col
            break

    def map_bias(val):
        v = str(val).strip().lower()
        if v in ["long", "buy", "b"]:
            return "LONG"
        if v in ["short", "sell", "s"]:
            return "SHORT"
        return "UNKNOWN"

    if bias_source_col is not None:
        return df[bias_source_col].map(map_bias)
    else:
        # Fallback: unknown if we can't find any directional column
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)


def get_account_id(trades_df: pd.DataFrame) -> str:
    """
    Try to infer an account identifier from the trades CSV.
    Looks for any column containing 'account' in its name.
    """
    account_id = "XXXXX"
    for col in trades_df.columns:
        if "account" in col.lower():
            val = trades_df[col].dropna().astype(str)
            if not val.empty:
                account_id = val.iloc[0]
                break
    return account_id


if fills_file is not None and trades_file is not None:
    try:
        fills_df = pd.read_csv(fills_file)
        trades_df = pd.read_csv(trades_file)

        # --- Overexposure detection from fills --------------------------------
        intervals_all, detail_all, cleaned_fills = detect_overexposure_intervals(
            fills_df,
            max_mini_equiv=max_mini_equiv,
            micro_weight=micro_weight_global,
        )

        # --- Trades preprocessing ---------------------------------------------
        trades_df = trades_df.copy()
        trades_df["Open_dt"] = pd.to_datetime(trades_df["Open Time"], errors="coerce")
        trades_df["Close_dt"] = pd.to_datetime(trades_df["Close Time"], errors="coerce")
        trades_df["Duration_sec"] = (
            trades_df["Close_dt"] - trades_df["Open_dt"]
        ).dt.total_seconds()

        # Net PnL (Net Profit)
        if "Net Profit" in trades_df.columns:
            trades_df["NetProfit_val"] = parse_money_to_float(trades_df["Net Profit"])
        else:
            trades_df["NetProfit_val"] = pd.NA

        # Root & mini/micro classification per trade
        trades_df["Root Code"] = trades_df["Symbol"].apply(get_root_code)
        trades_df["Size Type"] = trades_df["Root Code"].apply(
            lambda r: "Micro" if ROOT_TYPES.get(r, "mini") == "micro" else "Mini"
        )

        # Bias (LONG / SHORT / UNKNOWN)
        trades_df["Bias"] = infer_bias_column(trades_df)

        # --- Partition intervals by duration threshold ------------------------
        intervals_df_all = pd.DataFrame(intervals_all)
        detail_df_all = pd.DataFrame(detail_all)

        if not intervals_df_all.empty:
            if min_duration_seconds > 0:
                intervals_df_filtered = intervals_df_all[
                    intervals_df_all["Duration_sec"] >= min_duration_seconds
                ].copy()
            else:
                intervals_df_filtered = intervals_df_all.copy()
        else:
            intervals_df_filtered = pd.DataFrame(columns=["Group ID"])

        filtered_group_ids = set(intervals_df_filtered["Group ID"].unique())
        detail_df_filtered = (
            detail_df_all[detail_df_all["Group ID"].isin(filtered_group_ids)].copy()
            if not detail_df_all.empty
            else pd.DataFrame(columns=["Group ID"])
        )

        # --- Compute trade index sets for PnL aggregation ----------------------
        overexposed_trade_indices_all = set()
        overexposed_trade_indices_filtered = set()

        # All overexposed intervals (no duration filter)
        for _, inter_row in intervals_df_all.iterrows():
            start_all = inter_row["Interval Start"]
            end_all = inter_row["Interval End"]
            mask_all = (trades_df["Open_dt"] < end_all) & (trades_df["Close_dt"] > start_all)
            overexposed_trade_indices_all.update(trades_df[mask_all].index.tolist())

        # Filtered intervals (>= duration threshold)
        for _, inter_row in intervals_df_filtered.iterrows():
            start_f = inter_row["Interval Start"]
            end_f = inter_row["Interval End"]
            mask_f = (trades_df["Open_dt"] < end_f) & (trades_df["Close_dt"] > start_f)
            overexposed_trade_indices_filtered.update(trades_df[mask_f].index.tolist())

        # --- Overexposed intervals with matching trades (only filtered ones) ---
        st.subheader("Overexposed intervals with matching trades")

        if intervals_df_filtered.empty:
            if min_duration_seconds > 0:
                st.info(
                    f"No intervals found where total exposure exceeded {max_mini_equiv} "
                    f"mini-equivalents for at least {min_duration_seconds} seconds "
                    f"under a {ratio_option} Mini:Micro ratio."
                )
            else:
                st.info(
                    f"No intervals found where total exposure exceeded {max_mini_equiv} "
                    f"mini-equivalents under a {ratio_option} Mini:Micro ratio."
                )
        else:
            for gid in sorted(intervals_df_filtered["Group ID"].unique()):
                inter_row = intervals_df_filtered[intervals_df_filtered["Group ID"] == gid].iloc[0]
                start = inter_row["Interval Start"]
                end = inter_row["Interval End"]

                # Trades whose lifespan overlaps this interval (already duration-filtered set)
                mask = (trades_df["Open_dt"] < end) & (trades_df["Close_dt"] > start)
                trades_for_interval = trades_df[mask].copy()

                with st.expander(
                    f"Interval {gid}: {start} → {end} "
                    f"(Exposure: {inter_row['Total Mini-Equiv Exposure']:.2f}, "
                    f"Duration: {inter_row['Duration_sec']:.2f}s)"
                ):
                    st.markdown("**Exposure snapshot (per symbol)**")
                    exp_rows = detail_df_filtered[detail_df_filtered["Group ID"] == gid].copy()
                    exp_cols = [
                        "Symbol",
                        "Root Code",
                        "Position Contracts",
                        "Mini-Equiv Weight",
                        "Mini-Equiv Exposure",
                    ]
                    st.dataframe(exp_rows[exp_cols], use_container_width=True)

                    st.markdown("**Trades overlapping this interval**")
                    if trades_for_interval.empty:
                        st.info("No trades from the trades export overlap this interval.")
                    else:
                        trade_cols = [
                            "Symbol",
                            "Root Code",
                            "Size Type",
                            "Bias",
                            "Volume",
                            "Open Time",
                            "Close Time",
                            "Duration_sec",
                            "Net Profit",
                            "NetProfit_val",
                            "Open_dt",  # used for sorting
                        ]
                        existing_trade_cols = [c for c in trade_cols if c in trades_for_interval.columns]

                        display_df = trades_for_interval.sort_values("Open_dt")
                        st.dataframe(
                            display_df[existing_trade_cols],
                            use_container_width=True,
                        )
                        st.caption(
                            "Each trade shows its **Bias** (LONG/SHORT) and whether it is "
                            "**Mini or Micro** (Size Type), based on the root Code."
                        )

        # --- Net PnL summary from trades --------------------------------------
        st.subheader("Net PnL Summary (from trades export)")

        total_net_pnl = None
        pnl_overexposure_all = None
        pnl_overexposure_filtered = None
        perc_overexposure = None

        if "NetProfit_val" in trades_df.columns and trades_df["NetProfit_val"].notna().any():
            total_net_pnl = trades_df["NetProfit_val"].sum()

            # PnL from trades involved in ANY overexposure interval (ignoring duration threshold)
            pnl_overexposure_all = (
                trades_df.loc[list(overexposed_trade_indices_all), "NetProfit_val"].sum()
                if overexposed_trade_indices_all
                else 0.0
            )

            # PnL from trades involved in overexposure intervals that meet duration threshold
            pnl_overexposure_filtered = (
                trades_df.loc[list(overexposed_trade_indices_filtered), "NetProfit_val"].sum()
                if overexposed_trade_indices_filtered
                else 0.0
            )

            # Overexposure as % of total (based on duration-filtered PnL)
            if total_net_pnl != 0:
                perc_overexposure = 100.0 * (pnl_overexposure_filtered / total_net_pnl)
            else:
                perc_overexposure = 0.0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Total Net PnL (all trades)",
                f"{total_net_pnl:,.2f}",
                help="Sum of Net Profit from the trades export.",
            )
            col2.metric(
                "Net PnL from all overexposure",
                f"{pnl_overexposure_all:,.2f}",
                help=(
                    "Sum of Net Profit from trades that overlapped at least one "
                    "overexposed interval (regardless of duration)."
                ),
            )
            col3.metric(
                f"Net PnL from overexposure ≥ {min_duration_seconds}s",
                f"{pnl_overexposure_filtered:,.2f}",
                help=(
                    "Sum of Net Profit from trades that overlapped at least one "
                    f"overexposed interval lasting ≥ {min_duration_seconds} seconds."
                ),
            )
            col4.metric(
                f"Overexposure ≥ {min_duration_seconds}s as % of total PnL",
                f"{perc_overexposure:,.2f}%",
                help=(
                    "Net PnL from overexposure intervals (≥ duration threshold) "
                    "as a percentage of total Net PnL."
                ),
            )
        else:
            st.warning(
                "No valid 'Net Profit' column found in the trades CSV, so Net PnL "
                "could not be computed. Make sure your trades export includes "
                "a 'Net Profit' column."
            )

        # --- Prepared copy for top-3 overexposures ----------------------------
        st.subheader("Prepared overexposure summary copy")

        if intervals_df_all.empty:
            st.info("No overexposure intervals detected; no copy can be prepared.")
        else:
            # Choose intervals for the copy:
            # Prefer duration-filtered set; if empty, fall back to all
            if not intervals_df_filtered.empty:
                base_df = intervals_df_filtered.copy()
            else:
                base_df = intervals_df_all.copy()

            # Sort by total exposure (largest first) and take top 3
            base_df = base_df.sort_values(
                "Total Mini-Equiv Exposure", ascending=False
            ).head(3)

            # Account ID & reference date
            account_id = get_account_id(trades_df)
            first_date = base_df.iloc[0]["Interval Start"].date()
            ref_date_str = first_date.strftime("%B %d, %Y")  # e.g. December 17, 2025

            # Prepare narrative copy
            copy_lines = []

            copy_lines.append(
                f"As detailed below from your trading history for your Sim Funded account {account_id} on {ref_date_str},"
            )
            copy_lines.append("")

            # Helper: format time ranges
            def fmt_interval_time(dt):
                # Example: 06:56:37 AM ET
                return dt.strftime("%I:%M:%S %p ET").lstrip("0")

            def fmt_trade_time(dt):
                # Example: Dec 17, 03:32:17 AM EST
                return dt.strftime("%b %d, %I:%M:%S %p EST")

            # For each selected overexposure instance
            for idx, (_, inter_row) in enumerate(base_df.iterrows()):
                start = inter_row["Interval Start"]
                end = inter_row["Interval End"]
                start_str = fmt_interval_time(start)
                end_str = fmt_interval_time(end)

                if idx == 0:
                    copy_lines.append(
                        f"Between {start_str} and {end_str}, your account held the following positions simultaneously:"
                    )
                else:
                    copy_lines.append("")
                    copy_lines.append(
                        f"In another instance, between {start_str} and {end_str}, your account again held positions simultaneously exceeding the maximum allowed contract exposure:"
                    )

                copy_lines.append("")

                # Trades overlapping this interval
                mask = (trades_df["Open_dt"] < end) & (trades_df["Close_dt"] > start)
                trades_for_interval = trades_df[mask].copy()

                if trades_for_interval.empty:
                    copy_lines.append("(No matching trades found in the trades export.)")
                    copy_lines.append("")
                else:
                    # For each trade, list contracts & open/close times
                    trades_for_interval = trades_for_interval.sort_values("Open_dt")
                    for _, trow in trades_for_interval.iterrows():
                        symbol = trow.get("Symbol", "")
                        size_type = trow.get("Size Type", "Mini")
                        size_word = "Mini" if str(size_type).lower().startswith("mini") else "Micro"

                        volume = trow.get("Volume", "")
                        try:
                            if pd.isna(volume):
                                volume_str = "X"
                            else:
                                volume_str = str(int(float(volume)))
                        except Exception:
                            volume_str = str(volume) if volume not in ("", None) else "X"

                        copy_lines.append(f"{volume_str} {size_word} contracts – {symbol}")
                        open_dt = trow.get("Open_dt")
                        close_dt = trow.get("Close_dt")

                        if pd.notna(open_dt):
                            copy_lines.append(f"")
                            copy_lines.append(f"Opened: {fmt_trade_time(open_dt)}")
                        if pd.notna(close_dt):
                            copy_lines.append(f"Closed: {fmt_trade_time(close_dt)}")
                        copy_lines.append("")

            # Concluding lines
            # Use max_mini_equiv as the limit in the narrative
            limit_val = int(max_mini_equiv) if max_mini_equiv.is_integer() else max_mini_equiv
            copy_lines.append(
                f"This exceeds the maximum allowed limit of {limit_val} contracts under the Cross-Instrument Policy based on your account plan limit."
            )
            copy_lines.append("")
            num_occ = int(intervals_df_all["Group ID"].nunique())
            copy_lines.append(
                f"Please note your account has been found to exceed the max allowed contracts on more than {num_occ} different occasions."
            )

            final_copy = "\n".join(copy_lines)

            st.text_area(
                "Copy (ready to paste):",
                value=final_copy,
                height=400,
            )

    except Exception as e:
        st.error(f"Error processing files: {e}")
else:
    st.info("Upload **both** a fills CSV and a trades CSV to begin.")
