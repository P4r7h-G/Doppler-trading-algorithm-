"""Local reviewer dashboard: ``streamlit run src/doppler_research/sports/dashboard.py``.

Uploads stay inside this process; nothing is sent to an external endpoint.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

from doppler_research.sports.audit import FEATURE_REGISTRY, explain
from doppler_research.sports.config import SportsConfig
from doppler_research.sports.io import TABLES, blobs_from_zip, load_blobs
from doppler_research.sports.study import OUTPUT_FILES, run_study
from doppler_research.sports.synthetic import generate

st.set_page_config(page_title="Sports market volatility research", layout="wide")
st.title("Sports exchange-market volatility after news")
st.caption("Forecasts the realized variation of a market-implied probability path after a news origin, "
           "and compares it with what the market actually did. Runs locally: uploaded files never leave "
           "this process. Not a match-result model, an odds point forecast, or a betting service.")

with st.sidebar:
    st.header("1. Data")
    source = st.radio("Input", ["Upload canonical files", "Upload ZIP", "Synthetic demo"], index=2)
    blobs: dict[str, tuple[str, bytes]] = {}
    if source == "Upload canonical files":
        for table in TABLES:
            f = st.file_uploader(f"{table} (.csv or .parquet)", type=["csv", "parquet"], key=table)
            if f is not None:
                blobs[table] = (f.name, f.getvalue())
    elif source == "Upload ZIP":
        z = st.file_uploader("ZIP containing the four canonical files", type="zip")
        if z is not None:
            blobs = blobs_from_zip(z.getvalue())
    else:
        n_fix = st.slider("Synthetic fixtures", 16, 48, 32, step=4)
        st.caption("Planted, clearly synthetic data. Its scores mean nothing about real markets.")

    st.header("2. Experiment (freeze before looking)")
    mode = st.selectbox("Forecast origin", ["post_news", "pre_release"],
                        help="post_news: first receipt + latency. pre_release: before a scheduled release.")
    horizons = st.multiselect("Horizons (minutes)", [5, 15, 30, 60, 120], default=[5, 30, 120])
    basis = st.selectbox("Probability basis", ["raw_midpoint", "normalized_midpoint"])
    sampling = st.selectbox("Target sampling", ["tick", "grid"])
    latency = st.number_input("Post-news processing latency (s)", 0.0, 600.0, 30.0)
    min_train = st.number_input("Minimum fully observed training events", 5, 500, 15)
    replicates = st.number_input("Bootstrap replicates", 100, 5000, 500, step=100)
    placebo = st.checkbox("Placebo timestamps (origin shifted 240 min earlier)", True)
    shuffled = st.checkbox("Shuffled news links", True)
    st.header("3. Data declaration")
    provider = st.text_input("Provider", "UNDECLARED")
    licence = st.text_input("Licence owner", "UNDECLARED")
    use = st.text_input("Permitted use", "UNDECLARED")
    config_file = st.file_uploader("Or load a full JSON config", type="json")
    run = st.button("Run walk-forward study", type="primary")


def build_config() -> SportsConfig:
    if config_file is not None:
        return SportsConfig.from_dict(json.loads(config_file.getvalue()))
    return SportsConfig(mode=mode, horizons_minutes=tuple(sorted(horizons)) or (30,), probability_basis=basis,
                        target_sampling=sampling, post_news_latency_seconds=float(latency),
                        min_train_events=int(min_train), bootstrap_replicates=int(replicates),
                        placebo_shift_minutes=240.0 if placebo else None, shuffled_link_test=shuffled,
                        data_declaration={"provider": provider, "licence_owner": licence,
                                          "permitted_use": use, "capture_method": "dashboard_upload"})


if run:
    try:
        config = build_config()
        if source == "Synthetic demo":
            with st.spinner("Generating synthetic markets"):
                tables = generate(n_fixtures=n_fix)
                blobs = {k: (f"{k}.csv", v.to_csv(index=False).encode()) for k, v in tables.items()}
        frames, manifest = load_blobs(blobs, config.data_declaration)
        out_dir = Path(tempfile.mkdtemp(prefix="sports_study_"))
        status = st.status("Running study", expanded=False)
        result = run_study(frames, manifest, config, out_dir, progress=status.write)
        status.update(label="Study complete", state="complete")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out_dir))
        st.session_state["study"] = {"result": result, "config": config, "zip": buffer.getvalue()}
    except (ValueError, KeyError) as exc:
        st.error(f"{type(exc).__name__}: {exc}")

if "study" not in st.session_state:
    st.info("Choose data and settings in the sidebar, then run. See the README for the canonical file contracts.")
    st.stop()

study = st.session_state["study"]
res, config = study["result"], study["config"]
a = res["artifacts"]
traces = res["traces"]
counts = res["manifest"]["counts"]
scores = a["common_sample_scores"]
scored = a["per_event_forecasts"]

c = st.columns(5)
c[0].metric("News items", counts["news_ids"])
c[1].metric("Quotes accepted", f"{counts['quotes_accepted']:,}")
c[2].metric("Quotes rejected", counts["quotes_rejected"])
c[3].metric("Main-panel event x horizon rows", counts["panel_rows_by_panel"].get("main", 0))
c[4].metric("Main-panel exclusions", sum(counts["exclusions_by_reason"].values()))

tabs = st.tabs(["Scores", "Forecast vs realized", "Market path", "Audit trail", "Features & GARCH",
                "Exclusions", "Controls", "Downloads"])

with tabs[0]:
    if scores.empty:
        st.warning("No common-sample forecasts. Check exclusions and minimum training events.")
    else:
        st.caption("Lower QLIKE is better. Every model in a horizon is scored on the same events. "
                   "Units: squared change in logit(implied probability), summed over the horizon.")
        h = st.selectbox("Horizon", sorted(scores["horizon_minutes"].unique()), key="score_h")
        main = scores[(scores["panel"] == "main") & (scores["horizon_minutes"] == h)]
        st.altair_chart(alt.Chart(main).mark_bar().encode(
            x=alt.X("qlike:Q", title="Mean QLIKE (lower is better)"),
            y=alt.Y("model:N", sort="x"), tooltip=list(main.columns)), width="stretch")
        st.dataframe(main, width="stretch", hide_index=True)
        st.subheader("Acceptance gate")
        st.caption("A Doppler benefit counts only if event+state+Doppler beats BOTH event+state and "
                   "event-adjusted GARCH with a fixture-clustered bootstrap interval below zero.")
        st.dataframe(a["gate_checks"], width="stretch", hide_index=True)
        st.subheader("Paired comparisons")
        st.dataframe(a["bootstrap_comparisons"][a["bootstrap_comparisons"]["panel"] == "main"],
                     width="stretch", hide_index=True)
        st.subheader("Breakdowns")
        dim = st.selectbox("Group by", sorted(a["breakdown_scores"]["group_by"].unique()))
        b = a["breakdown_scores"]
        st.dataframe(b[(b["panel"] == "main") & (b["group_by"] == dim) & (b["horizon_minutes"] == h)],
                     width="stretch", hide_index=True)

with tabs[1]:
    if not scored.empty:
        main = scored[scored["panel"] == "main"]
        col1, col2 = st.columns(2)
        h = col1.selectbox("Horizon", sorted(main["horizon_minutes"].unique()), key="fv_h")
        models = sorted(main["model"].unique())
        chosen = col2.multiselect("Models", models, default=[m for m in ("event_state_doppler", "event_state",
                                                                         "garch_horizon") if m in models])
        sub = main[(main["horizon_minutes"] == h) & main["model"].isin(chosen)]
        st.caption("Each point is one event. The diagonal is a perfect forecast (log-log axes).")
        lo = float(min(sub["forecast"].min(), sub["realized"].clip(lower=1e-8).min()))
        hi = float(max(sub["forecast"].max(), sub["realized"].max()))
        diag = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(color="gray").encode(
            x="x:Q", y="y:Q")
        pts = alt.Chart(sub.assign(realized=sub["realized"].clip(lower=1e-8))).mark_circle(size=60).encode(
            x=alt.X("forecast:Q", scale=alt.Scale(type="log"), title="Forecast realized variation"),
            y=alt.Y("realized:Q", scale=alt.Scale(type="log"), title="Realized variation"),
            color="model:N", tooltip=["event_key", "event_type", "model", "forecast", "realized", "qlike_loss"])
        st.altair_chart(diag + pts, width="stretch")
        st.dataframe(sub.sort_values(["event_key", "model"]), width="stretch", hide_index=True)

main_keys = sorted(k for k in traces if "::" not in k)

with tabs[2]:
    if main_keys:
        key = st.selectbox("Event", main_keys, key="path_event")
        trace = traces[key].to_dict()
        lin = a["feature_grid_lineage"]
        tl = a["target_lineage"]
        pre = lin[lin["event_key"] == key].assign(time=lambda d: pd.to_datetime(d["grid_time_utc"]),
                                                  phase="pre-origin (features)")
        post = tl[tl["event_key"] == key].assign(time=lambda d: pd.to_datetime(d["time_utc"]))
        hs = sorted(post["horizon_minutes"].unique())
        h = st.selectbox("Target horizon", hs, index=len(hs) - 1, key="path_h") if hs else None
        post = post[post["horizon_minutes"] == h].assign(phase="post-origin (target)")
        line = pd.concat([pre[["time", "logit", "phase"]], post[["time", "logit", "phase"]]])
        origin_step = next(s for s in trace["steps"] if s["function"].endswith("forecast_origin"))
        marks = [{"time": pd.Timestamp(origin_step["outputs"]["origin_utc"]), "label": "forecast origin"}]
        for k in ("source_published_at_utc", "first_seen_at_utc", "release_scheduled_known_at_utc"):
            if origin_step["inputs"].get(k):
                marks.append({"time": pd.Timestamp(origin_step["inputs"][k]), "label": k})
        if not post.empty:
            marks.append({"time": post["time"].max(), "label": f"target end ({h}m)"})
        marks = pd.DataFrame(marks)
        chart = alt.Chart(line).mark_line().encode(
            x=alt.X("time:T", title="UTC"), y=alt.Y("logit:Q", title="logit(implied probability)"),
            color="phase:N")
        rules = alt.Chart(marks).mark_rule(strokeDash=[4, 4]).encode(x="time:T", color=alt.value("black"),
                                                                     tooltip=["label", "time"])
        text = alt.Chart(marks).mark_text(angle=270, dy=-6, align="left").encode(x="time:T", y=alt.value(10),
                                                                                  text="label")
        st.altair_chart(chart + rules + text, width="stretch")
        st.caption("Pre-origin values are the replayed one-minute grid used for features; post-origin values "
                   "are the quote path used for the realized target. Every point maps to a quote source row "
                   "in the lineage tables.")

with tabs[3]:
    keys = sorted(traces)
    if keys:
        col1, col2, col3 = st.columns(3)
        key = col1.selectbox("Event", keys, key="audit_event")
        trace = traces[key].to_dict()
        hs = sorted(trace["horizons"], key=int)
        h = col2.selectbox("Horizon", hs, key="audit_h") if hs else None
        model_names = sorted(trace["horizons"][h]["models"]) if h else []
        model = col3.selectbox("Model", model_names,
                               index=model_names.index("event_state_doppler") if "event_state_doppler" in model_names
                               else 0) if model_names else None
        m = trace["horizons"][h]["models"].get(model) if model else None
        if m and m.get("terms"):
            terms = pd.DataFrame(m["terms"])
            terms["source_function"] = terms["term"].map(lambda t: FEATURE_REGISTRY.get(t, {}).get("function", ""))
            st.caption("How each feature moved the log forecast: contribution = coefficient x standardized "
                       "value. Forecast = exp(intercept + sum of contributions).")
            st.altair_chart(alt.Chart(terms).mark_bar().encode(
                x=alt.X("contribution_log:Q", title="Contribution to log forecast"),
                y=alt.Y("term:N", sort="-x"),
                color=alt.condition(alt.datum.contribution_log > 0, alt.value("#c0392b"), alt.value("#2471a3")),
                tooltip=list(terms.columns)), width="stretch")
            st.dataframe(terms, width="stretch", hide_index=True)
        st.code(explain(trace, int(h) if h else None, model), language="text")
        with st.expander("Raw trace JSON"):
            st.json(trace)

with tabs[4]:
    st.caption("Every causal feature, GARCH parameter and target, frozen at each event's origin.")
    st.dataframe(a["event_panel"], width="stretch", hide_index=True)
    st.subheader("Feature registry")
    st.dataframe(pd.DataFrame(FEATURE_REGISTRY).T.reset_index(names="feature"), width="stretch",
                 hide_index=True)

with tabs[5]:
    st.subheader("Excluded events")
    ex = a["excluded_events"]
    if not ex.empty:
        st.bar_chart(ex[ex["panel"] == "main"]["reason"].value_counts())
    st.dataframe(ex, width="stretch", hide_index=True)
    st.subheader("Forecasts not produced")
    st.dataframe(a["forecast_skips"], width="stretch", hide_index=True)
    st.subheader("Rejected quote rows")
    st.dataframe(a["rejected_quotes"], width="stretch", hide_index=True)

with tabs[6]:
    st.caption("If a model looks as good at placebo timestamps or on shuffled news links as on real news, "
               "its apparent skill is not about the news (e.g. time-to-kickoff or market-state artefacts).")
    if not scores.empty:
        pivot = scores.pivot_table(index=["horizon_minutes", "model"], columns="panel", values="qlike")
        st.dataframe(pivot, width="stretch")
    st.dataframe(a["signed_reprice_report"], width="stretch", hide_index=True)
    st.dataframe(a["event_counts"], width="stretch", hide_index=True)

with tabs[7]:
    st.download_button("Download all outputs (ZIP)", study["zip"], file_name="sports_study_outputs.zip",
                       mime="application/zip", type="primary")
    st.json({"outputs": OUTPUT_FILES, "run_manifest": json.loads(json.dumps(res["manifest"], default=str))},
            expanded=False)
