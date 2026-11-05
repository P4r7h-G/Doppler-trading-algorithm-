"""Optional local upload UI; files remain in the running container."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from doppler_research.pipeline import Config, build_event_panel, walk_forward
from doppler_research.schema import validate_events, validate_prices
from doppler_research.scenarios import calibrated_price_intervals
from doppler_research.scoring import score_forecasts

st.set_page_config(page_title="Doppler vs GARCH research", layout="wide")
st.title("Doppler vs GARCH: event volatility research")
st.caption("Upload your own split-adjusted prices and point-in-time verified event calendar. Nothing is uploaded to this project's authors.")
with st.expander("Where the Doppler calculation enters the model"):
    st.markdown("A 60-session older return spectrum is compared with a disjoint 20-session recent spectrum. "
                "Jensen–Shannon divergence measures the change in spectral shape; the log ratio of their "
                "frequency centroids records its direction. First and second daily differences measure "
                "velocity and acceleration. The **doppler_only** model sees these features alone; "
                "**doppler_state** adds volatility and volume state; **event_state_doppler** tests "
                "whether they add value beyond event history and ordinary state. All windows end at the "
                "forecast origin. See the README for exact formulas and leakage rules.")
prices_file = st.file_uploader("Canonical prices CSV", type="csv", key="prices")
events_file = st.file_uploader("Canonical events CSV", type="csv", key="events")
target = st.selectbox("Target variance proxy", ["squared_return", "gap_parkinson"])
drift = st.text_input("Optional assumed log return per reaction session, for calibrated price intervals", value="")
if prices_file and events_file and st.button("Run walk-forward comparison"):
    try:
        prices = validate_prices(pd.read_csv(prices_file))
        events = validate_events(pd.read_csv(events_file))
        config = Config(target=target)
        panel, skipped = build_event_panel(prices, events, config)
        scored, scores, comparisons = score_forecasts(walk_forward(panel, config), config.bootstrap_replicates, config.seed)
        c1, c2, c3 = st.columns(3)
        c1.metric("Common out-of-sample events", scored.event_id.nunique())
        c2.metric("Tickers tested", scored.ticker.nunique())
        c3.metric("Excluded events", len(skipped))
        tabs = st.tabs(["Comparison", "Event forecasts", "Feature audit", "Exclusions & downloads"])
        with tabs[0]:
            st.caption("Lower QLIKE, MAE and RMSE are better. Scores within a target use the same events for all models.")
            pooled = scores[scores.panel == "pooled"].set_index("model")
            st.bar_chart(pooled["qlike"])
            st.dataframe(scores, use_container_width=True)
            st.subheader("Paired QLIKE differences")
            st.caption("Negative model A minus model B favors model A. Bootstrap intervals are exploratory.")
            st.dataframe(comparisons, use_container_width=True)
        with tabs[1]:
            st.dataframe(scored, use_container_width=True)
        with tabs[2]:
            st.caption("Each row contains the causal spectral/state values and fitted GARCH parameters at that event's origin.")
            st.dataframe(panel, use_container_width=True)
        with tabs[3]:
            st.dataframe(skipped, use_container_width=True)
            for name, frame in [("event_features.csv", panel), ("per_event_forecasts.csv", scored),
                                ("common_sample_scores.csv", scores), ("bootstrap_comparisons.csv", comparisons),
                                ("skipped_events.csv", skipped)]:
                st.download_button(f"Download {name}", frame.to_csv(index=False), file_name=name, mime="text/csv")
        if drift.strip():
            scenarios = calibrated_price_intervals(scored, float(drift))
            st.subheader("Historical price intervals")
            st.caption("Empirical intervals require 40 earlier out-of-sample events per model and use the drift assumption entered above.")
            st.dataframe(scenarios, use_container_width=True)
            st.download_button("Download price_scenarios.csv", scenarios.to_csv(index=False),
                               file_name="price_scenarios.csv", mime="text/csv")
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
