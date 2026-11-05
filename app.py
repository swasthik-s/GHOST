import io
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GHOST ASBE | DPAM", page_icon="◈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1600px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px; border-radius: 9px;}
</style>
""", unsafe_allow_html=True)

st.title("◈ GHOST ASBE")
st.caption("Adaptive Session Behaviour Engine · DPAM Research Workbench")
st.info("Research mode only. Historical CSV analysis; no MT5 order connection or live execution.", icon="🧪")

def load_data(file):
    d = pd.read_csv(io.BytesIO(file.getvalue()), sep=None, engine="python")
    rename = {}
    for col in d.columns:
        key = str(col).strip().lower().replace("<", "").replace(">", "").replace(" ", "").replace("_", "")
        if key in {"open", "high", "low", "close"}:
            rename[col] = key
        elif key in {"datetime", "timestamp"}:
            rename[col] = "timestamp"
        elif key == "date":
            rename[col] = "date"
        elif key == "time":
            rename[col] = "time"
    d = d.rename(columns=rename)
    if "timestamp" not in d:
        if "date" in d and "time" in d:
            d["timestamp"] = d["date"].astype(str) + " " + d["time"].astype(str)
        elif "date" in d:
            d["timestamp"] = d["date"]
    missing = {"open", "high", "low", "close"} - set(d.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    for col in ["open", "high", "low", "close"]:
        d[col] = pd.to_numeric(d[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if "timestamp" in d:
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    d = d.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    valid = ((d.high >= d.low) & (d.high >= d.open) & (d.high >= d.close)
             & (d.low <= d.open) & (d.low <= d.close))
    d = d.loc[valid].reset_index(drop=True)
    if d.empty:
        raise ValueError("No valid OHLC rows remain.")
    return d

def build_features(d, expansion, body_threshold, max_age):
    x = d.copy()
    x["bar_range"] = (x.high - x.low).clip(lower=0)
    x["body"] = (x.close - x.open).abs()
    x["direction"] = np.sign(x.close - x.open).astype(int)
    x["body_ratio"] = np.where(x.bar_range > 0, x.body / x.bar_range, 0.0)
    x["prior_range_median20"] = x.bar_range.rolling(20, min_periods=10).median().shift(1)
    x["expansion_ratio"] = x.bar_range / x.prior_range_median20.replace(0, np.nan)
    x["impulse"] = ((x.expansion_ratio >= expansion) &
                    (x.body_ratio >= body_threshold) & (x.direction != 0))
    x["opposing_expansion"] = ((x.expansion_ratio >= expansion) &
                               (x.body_ratio >= body_threshold) & (x.direction != 0))

    # Only the most recent prior impulse is eligible, and only within max_age bars.
    positions = pd.Series(np.arange(len(x)), index=x.index, dtype=float)
    last_impulse = positions.where(x.impulse).ffill().shift(1)
    x["prior_impulse_age"] = positions - last_impulse
    x["prior_impulse_direction"] = x.direction.where(x.impulse).ffill().shift(1)
    x["prior_impulse_body"] = x.body.where(x.impulse).ffill().shift(1)
    valid_prior = last_impulse.notna() & x.prior_impulse_age.between(1, max_age)
    x["opposing_after_impulse"] = (
        x.opposing_expansion & valid_prior & (x.direction != x.prior_impulse_direction)
    )
    x["opposing_to_impulse_body_ratio"] = x.body / x.prior_impulse_body.replace(0, np.nan)

    # Background volatility uses prior bars only; the event candle is excluded.
    v20 = x.bar_range.rolling(20, min_periods=15).median().shift(1)
    v100 = x.bar_range.rolling(100, min_periods=60).median().shift(1)
    x["background_vol_ratio"] = v20 / v100.replace(0, np.nan)
    x["background_volatility"] = np.select(
        [x.background_vol_ratio < 0.80, x.background_vol_ratio >= 1.20],
        ["low", "high"], default="normal"
    )
    x.loc[x.background_vol_ratio.isna(), "background_volatility"] = "unknown"
    return x

def run_event_study(x, horizons):
    rows = []
    n = len(x)
    for i in np.flatnonzero(x.opposing_after_impulse.to_numpy()):
        confirmation_idx, entry_idx = i + 1, i + 2
        if entry_idx >= n:
            continue
        direction = int(x.at[i, "direction"])
        confirmed = int(x.at[confirmation_idx, "direction"]) == direction and direction != 0
        base = {
            "event_index": int(i),
            "confirmation_index": confirmation_idx,
            "entry_index": entry_idx,
            "event_time": str(x.at[i, "timestamp"]) if "timestamp" in x else "",
            "direction": direction,
            "impulse_age_bars": int(x.at[i, "prior_impulse_age"]),
            "expansion_ratio": float(x.at[i, "expansion_ratio"]),
            "opposing_to_impulse_body_ratio": x.at[i, "opposing_to_impulse_body_ratio"],
            "background_volatility": x.at[i, "background_volatility"],
            "confirmed": bool(confirmed),
        }
        if not confirmed:
            rows.append(base)
            continue
        entry = float(x.at[entry_idx, "open"])
        for horizon in horizons:
            exit_idx = entry_idx + horizon - 1
            if exit_idx >= n:
                continue
            future = x.iloc[entry_idx:exit_idx + 1]
            exit_price = float(x.at[exit_idx, "close"])
            signed = (exit_price - entry) * direction
            if direction > 0:
                mfe = float(future.high.max() - entry)
                mae = float(entry - future.low.min())
            else:
                mfe = float(entry - future.low.min())
                mae = float(future.high.max() - entry)
            rows.append({
                **base, "horizon_bars": int(horizon), "entry_price": entry,
                "exit_price": exit_price, "signed_return_price": signed,
                "mfe_price": mfe, "mae_price": mae,
            })
    return pd.DataFrame(rows)

with st.sidebar:
    st.header("Historical data")
    upload = st.file_uploader("Upload MT5 OHLC CSV", type=["csv", "txt"])
    st.divider()
    st.header("DPAM parameters")
    expansion = st.slider("Expansion threshold", 1.0, 4.0, 1.5, 0.1)
    body_threshold = st.slider("Minimum body / range", 0.30, 0.95, 0.60, 0.05)
    max_age = st.slider("Maximum prior impulse age (bars)", 1, 10, 3)
    horizons = st.multiselect("Forward horizons", [1, 2, 3, 5, 10, 15, 20], [1, 3, 5, 10])
    cluster_gap = st.slider("Non-overlap cluster gap", 1, 30, 10)
    run = st.button("Run DPAM analysis", type="primary", use_container_width=True,
                    disabled=(upload is None or not horizons))

if upload is None:
    st.subheader("Start a research run")
    st.write("Upload an MT5 OHLC export. Required columns: Open, High, Low and Close. A timestamp column is recommended.")
    cols = st.columns(3)
    cols[0].metric("Signal sequence", "Impulse → Opposing")
    cols[1].metric("Confirmation", "Next candle close")
    cols[2].metric("Entry", "Following candle open")
    st.markdown("""
    **Workflow:** upload historical data → configure the event model → run the study → inspect forward outcomes and event records → export results.
    """)
    st.stop()

try:
    data = load_data(upload)
except Exception as exc:
    st.error(f"CSV parsing failed: {exc}")
    st.stop()

params = (upload.name, expansion, body_threshold, max_age, tuple(sorted(horizons)), cluster_gap)
if run:
    with st.spinner("Calculating candle features and DPAM events..."):
        feat = build_features(data, expansion, body_threshold, max_age)
        events = run_event_study(feat, sorted(horizons))
        candidates = events.drop_duplicates("event_index") if not events.empty else events
        confirmed = candidates[candidates.confirmed] if not candidates.empty else candidates
        tradable = events[events.confirmed & events.horizon_bars.notna()] if not events.empty else events
        summary = (tradable.groupby("horizon_bars").agg(
            n=("signed_return_price", "size"),
            win_rate=("signed_return_price", lambda s: (s > 0).mean()),
            mean_return=("signed_return_price", "mean"),
            median_return=("signed_return_price", "median"),
            mean_mfe=("mfe_price", "mean"),
            mean_mae=("mae_price", "mean"),
        ).reset_index() if not tradable.empty else pd.DataFrame())

        indices = sorted(candidates.event_index.astype(int).unique()) if not candidates.empty else []
        non_overlap, last = [], -10**9
        for idx in indices:
            if idx - last >= cluster_gap:
                non_overlap.append(idx)
                last = idx
        report = {
            "engine": "GHOST ASBE DPAM v2.1",
            "file": upload.name,
            "bars": len(feat),
            "first_timestamp": str(feat.timestamp.iloc[0]) if "timestamp" in feat else None,
            "last_timestamp": str(feat.timestamp.iloc[-1]) if "timestamp" in feat else None,
            "impulses": int(feat.impulse.sum()),
            "event_candidates": len(candidates),
            "confirmed_candidates": len(confirmed),
            "confirmation_rate": len(confirmed) / len(candidates) if len(candidates) else None,
            "greedy_non_overlapping_candidates": len(non_overlap),
            "parameters": {
                "expansion": expansion, "body_ratio": body_threshold,
                "max_impulse_age": max_age, "horizons": sorted(horizons),
                "cluster_gap": cluster_gap
            },
            "limitations": [
                "No spread, commission, slippage, swaps or execution latency.",
                "Entry approximated at the open after confirmation closes.",
                "Returns are raw price units, not account-currency P/L or R.",
                "Greedy non-overlap count is a clustering diagnostic, not proof of independence.",
                "Research only; live execution is disabled."
            ]
        }
        st.session_state["result"] = (feat, events, summary, report, params)

if "result" not in st.session_state:
    st.warning("CSV loaded. Click Run DPAM analysis in the sidebar.")
    st.stop()

feat, events, summary, report, used_params = st.session_state["result"]
if used_params != params:
    st.warning("Settings or file changed since the previous run. Click Run DPAM analysis to refresh results.")

st.subheader(f"Results · {report['file']}")
st.caption(f"{report['bars']:,} valid candles · {report['first_timestamp']} → {report['last_timestamp']}")
metrics = st.columns(5)
for col, label, value in zip(metrics,
    ["Candles", "Impulse bars", "Event candidates", "Confirmed", "Non-overlap clusters"],
    [f"{report['bars']:,}", f"{report['impulses']:,}", f"{report['event_candidates']:,}",
     f"{report['confirmed_candidates']:,}", f"{report['greedy_non_overlapping_candidates']:,}"]):
    col.metric(label, value)
if report["confirmation_rate"] is not None:
    st.caption(f"Confirmation rate: {report['confirmation_rate']:.1%}")

overview, ledger, candle_tab, export_tab = st.tabs(
    ["Overview", "Event ledger", "Candle features", "Export & methodology"]
)
with overview:
    st.subheader("Forward outcomes")
    if summary.empty:
        st.warning("No confirmed events have sufficient future candles for the selected horizons.")
    else:
        st.line_chart(summary.set_index("horizon_bars")["win_rate"])
        shown = summary.copy()
        shown["win_rate"] = (shown.win_rate * 100).round(2).astype(str) + "%"
        st.dataframe(shown.round(3), use_container_width=True, hide_index=True)
        st.caption("Returns, MFE and MAE are raw XAUUSD price units—not dollars, account P/L or R-multiples.")
    st.subheader("Background volatility at event bars")
    if not candidates.empty:
        counts = candidates.background_volatility.value_counts().rename_axis("regime").reset_index(name="events")
        left, right = st.columns(2)
        left.bar_chart(counts.set_index("regime"))
        right.dataframe(counts, use_container_width=True, hide_index=True)

with ledger:
    if events.empty:
        st.info("No events matched these parameters.")
    else:
        a, b, c = st.columns(3)
        confirmation = a.selectbox("Confirmation", ["All", "Confirmed", "Not confirmed"])
        regime = b.selectbox("Background volatility", ["All"] + sorted(events.background_volatility.astype(str).unique()))
        available_horizons = sorted(events.horizon_bars.dropna().astype(int).unique().tolist())
        horizon = c.selectbox("Horizon", ["All"] + available_horizons)
        view = events.copy()
        if confirmation == "Confirmed":
            view = view[view.confirmed]
        elif confirmation == "Not confirmed":
            view = view[~view.confirmed]
        if regime != "All":
            view = view[view.background_volatility == regime]
        if horizon != "All":
            view = view[view.horizon_bars == int(horizon)]
        st.dataframe(view, use_container_width=True, hide_index=True, height=480)
        st.caption(f"{len(view):,} rows. Confirmed events may appear once per selected horizon.")

with candle_tab:
    visible = [c for c in [
        "timestamp", "open", "high", "low", "close", "bar_range", "body", "direction",
        "body_ratio", "expansion_ratio", "impulse", "opposing_expansion",
        "prior_impulse_age", "opposing_after_impulse", "background_vol_ratio",
        "background_volatility"
    ] if c in feat.columns]
    st.dataframe(feat[visible].tail(500), use_container_width=True, hide_index=True, height=500)
    st.caption("Latest 500 candles. Export the complete feature set from the Export tab.")

with export_tab:
    st.download_button("Download event ledger CSV", events.to_csv(index=False), "dpam_events.csv", "text/csv")
    st.download_button("Download summary CSV", summary.to_csv(index=False), "dpam_summary.csv", "text/csv")
    st.download_button("Download candle features CSV", feat.to_csv(index=False), "dpam_features.csv", "text/csv")
    st.download_button("Download report JSON", json.dumps(report, indent=2), "dpam_report.json", "application/json")
    st.markdown("""
    **Anti-lookahead timing**
    - Event candle: opposing expansion linked to the latest prior impulse within the age limit.
    - Confirmation: next candle closes in the event direction.
    - Entry: open of the candle after confirmation.
    - Outcomes begin at entry; the confirmation candle is excluded.
    - Background volatility uses prior rolling ranges, shifted to exclude the event candle.
    """)
    st.warning("Not a complete backtest: spread, commissions, slippage, swaps, latency, intrabar execution and FTMO constraints are not modelled.")

st.divider()
st.caption("GHOST ASBE · Research workbench · Live execution disabled")
