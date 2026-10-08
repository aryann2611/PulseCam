"""PulseCam live demo. Run locally:  streamlit run app/app.py
Research prototype - NOT a medical device."""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from pulsecam.draw import overlay
from pulsecam.pipeline import HRTracker

LOG = os.path.join(os.path.dirname(__file__), "..", "results", "e5_live_log.csv")

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


left, right = st.columns([3, 2])
with left:
    ctx = webrtc_streamer(key="pulsecam", video_frame_callback=on_frame, async_processing=True,
                          media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 30}, "audio": False})
with right:
    hr_box, warn_box, cmp_box, chart = st.empty(), st.empty(), st.empty(), st.empty()
    if st.button("Log this reading (for E5)") and tracker.state["hr"]:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        new = not os.path.exists(LOG)
        with open(LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "method", "pulsecam_bpm", "reference_bpm"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), tracker.method, round(tracker.state["hr"], 1), ref or ""])
        st.success("Logged.")

while ctx.state.playing:  # refresh the side panel once a second
    s = tracker.state
    hr_box.metric("Heart rate (BPM)", f"{s['hr']:.0f}" if s["hr"] else "--")
    if s["warn"]:
        warn_box.warning(s["warn"])
    else:
        warn_box.empty()
    if s["hr"] and ref:
        cmp_box.metric("Error vs reference", f"{s['hr'] - ref:+.0f} BPM")
    if len(s["bvp"]):
        chart.line_chart(s["bvp"], height=220)
    time.sleep(1)
