"""PulseCam live demo. Run locally:  streamlit run app/app.py
Research prototype - NOT a medical device."""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import av
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from pulsecam.draw import overlay
from pulsecam.pipeline import HRTracker

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "results", "e5_live_log.csv")

st.set_page_config(page_title="PulseCam", layout="wide")
st.title("PulseCam - contactless heart rate")
st.warning("Research prototype for learning. Not a medical device; do not use for diagnosis.")


@st.cache_resource
def get_tscan():
    try:
        from pulsecam.tscan import TSCANPredictor
        return TSCANPredictor()
    except Exception as e:  # torch or weights missing -> POS only
        st.sidebar.info(f"TS-CAN unavailable ({type(e).__name__}); using POS.")
        return None


if "tracker" not in st.session_state:
    st.session_state.tracker = HRTracker()
tracker = st.session_state.tracker
tscan = get_tscan()
choice = st.sidebar.radio("Model", ["TS-CAN (deep)", "POS (classical)"] if tscan else ["POS (classical)"])
tracker.method, tracker.tscan = ("tscan", tscan) if choice.startswith("TS") else ("pos", None)
ref = st.sidebar.number_input("Reference device reading (BPM, optional)", 0, 220, 0)


def on_frame(frame):
    bgr = frame.to_ndarray(format="bgr24")
    state = tracker.update(np.ascontiguousarray(bgr[:, :, ::-1]), time.time())
    return av.VideoFrame.from_ndarray(overlay(bgr, state), format="bgr24")


tab_live, tab_res, tab_status = st.tabs(["Live", "Results", "Status"])

with tab_live:
    left, right = st.columns([3, 2])
    with left:
        ctx = webrtc_streamer(key="pulsecam", video_frame_callback=on_frame, async_processing=True,
                              media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 30}, "audio": False})
    with right:
        m1, m2 = st.columns(2)
        hr_box, snr_box = m1.empty(), m2.empty()
        warn_box, cmp_box, trend, chart = st.empty(), st.empty(), st.empty(), st.empty()
        if st.button("Log this reading (for E5)") and tracker.state["hr"]:
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            new = not os.path.exists(LOG)
            with open(LOG, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "method", "pulsecam_bpm", "reference_bpm"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), tracker.method, round(tracker.state["hr"], 1), ref or ""])
            st.success("Logged.")

with tab_res:
    T, P = os.path.join(ROOT, "results", "tables"), os.path.join(ROOT, "results", "plots")
    f = os.path.join(T, "E1_E2_ubfc.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        st.subheader("E1 + E2: UBFC-rPPG, error by method")
        lvl = st.radio("Scoring level", ["window", "video"], horizontal=True, format_func=lambda x: "Per 10 s window" if x == "window" else "Per video (median)")
        g = d[d.level == lvl].sort_values("MAE")
        best, runner = g.iloc[0], g.iloc[1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Lowest error", f"{best.MAE:.2f} BPM", best.method)
        c2.metric("Next best", f"{runner.MAE:.2f} BPM", runner.method)
        c3.metric("n", int(best.n))
        st.bar_chart(g.set_index("method")["MAE"], horizontal=True)
        st.dataframe(g[["method", "n", "MAE", "RMSE", "MAPE", "Pearson"]].reset_index(drop=True), width="stretch")
        plots = sorted(x for x in os.listdir(P) if x.startswith("BA_ubfc_")) if os.path.isdir(P) else []
        if plots:
            pick = st.selectbox("Bland-Altman plot", plots, format_func=lambda x: x[len("BA_ubfc_"):-4])
            st.image(os.path.join(P, pick), width=420)
        st.caption("10 UBFC subjects; deep models are the toolbox's released checkpoints. Details in docs/report_draft.md.")
    else:
        st.info("No E1/E2 table yet. Run: python eval/ubfc_eval.py --root data/ubfc")
    if os.path.exists(LOG):
        st.subheader("E5: live readings vs reference device")
        e5 = pd.read_csv(LOG)
        st.dataframe(e5, width="stretch")
        r = e5.dropna(subset=["reference_bpm"])
        if len(r):
            st.metric("MAE vs reference", f"{(r.pulsecam_bpm - r.reference_bpm).abs().mean():.1f} BPM", f"{len(r)} readings")

with tab_status:
    st.subheader("Project checklist")
    cl = os.path.join(ROOT, "docs", "CHECKLIST.md")
    if os.path.exists(cl):
        txt = open(cl, encoding="utf-8").read()
        done, total = txt.count("- [x]"), txt.count("- [x]") + txt.count("- [ ]")
        st.progress(done / max(total, 1), text=f"{done} of {total} items ticked")
        st.markdown(txt)
    st.caption("PulseCam is a research prototype and not a medical device.")

while ctx.state.playing:  # refresh the side panel once a second
    s = tracker.state
    hr_box.metric("Heart rate (BPM)", f"{s['hr']:.0f}" if s["hr"] else "--")
    snr_box.metric("Signal quality (dB)", f"{s['snr']:.1f}" if s["snr"] is not None else "--")
    if s["warn"]:
        warn_box.warning(s["warn"])
    else:
        warn_box.empty()
    if s["hr"] and ref:
        cmp_box.metric("Error vs reference", f"{s['hr'] - ref:+.0f} BPM")
    if s["hr"]:
        hist = st.session_state.setdefault("hist", [])
        hist.append(s["hr"])
        del hist[:-120]
        trend.line_chart(pd.DataFrame({"BPM (median of last 5)": hist}), height=160)
    if len(s["bvp"]):
        chart.line_chart(s["bvp"], height=160)
    time.sleep(1)
