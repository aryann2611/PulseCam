"""PulseCam live demo. Run locally:  streamlit run app/app.py
Research prototype - NOT a medical device."""
import csv
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "results", "e5_live_log.csv")

st.set_page_config(page_title="PulseCam", page_icon="\U0001FAC0", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2.2rem; max-width: 1100px; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
div[data-testid="stMetric"] { background: var(--secondary-background-color); border-radius: 10px; padding: 0.75rem 1rem; }
div[data-testid="stTabs"] button[role="tab"] { font-size: 1.02rem; padding: 0.5rem 1.1rem; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)
st.title("PulseCam - contactless heart rate")
st.caption("rPPG from face video - estimates BPM and the pulse waveform from a webcam or an uploaded clip.")
st.warning("Research prototype for learning. Not a medical device; do not use for diagnosis.")


tab_live, tab_up, tab_res, tab_status = st.tabs(["Go live (webcam)", "Upload video", "Results", "Status"])

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
        for col, label, value, delta in (
            (c1, "Lowest error", f"{best.MAE:.2f} BPM", best.method),
            (c2, "Next best", f"{runner.MAE:.2f} BPM", runner.method),
            (c3, "n", int(best.n), None),
        ):
            with col.container(border=True):
                st.metric(label, value, delta)
        st.bar_chart(g.set_index("method")["MAE"], horizontal=True)
        st.dataframe(g[["method", "n", "MAE", "RMSE", "MAPE", "Pearson"]].reset_index(drop=True), width="stretch")
        plots = sorted(x for x in os.listdir(P) if x.startswith("BA_ubfc_")) if os.path.isdir(P) else []
        if plots:
            pick = st.selectbox("Bland-Altman plot", plots, format_func=lambda x: x[len("BA_ubfc_"):-4])
            st.image(os.path.join(P, pick), width=420)
        st.caption("All 42 UBFC subjects (the Bland-Altman plots are still from the first 10); deep models are the toolbox's released checkpoints. Details in docs/report_draft.md.")
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



@st.cache_resource
def get_tscan():
    try:
        from pulsecam.tscan import TSCANPredictor
        return TSCANPredictor()
    except Exception as e:  # torch or weights missing -> POS only
        st.sidebar.info(f"TS-CAN unavailable ({type(e).__name__}); using POS.")
        return None



# Slow part (scipy, torch, MediaPipe) comes last so the title and Results/Status render at once.
import av
from streamlit_webrtc import webrtc_streamer

from pulsecam.draw import overlay
from pulsecam.pipeline import HRTracker, analyze_video

if "tracker" not in st.session_state:
    st.session_state.tracker = HRTracker()
tracker = st.session_state.tracker
tscan = get_tscan()
st.sidebar.header("Settings")
choice = st.sidebar.radio("Model", ["TS-CAN (deep)", "POS (classical)"] if tscan else ["POS (classical)"])
tracker.method, tracker.tscan = ("tscan", tscan) if choice.startswith("TS") else ("pos", None)
ref = st.sidebar.number_input("Reference device reading (BPM, optional)", 0, 220, 0)
st.sidebar.caption("Set a reference reading to compare live estimates against it on the Live tab.")

with tab_up:
    st.subheader("Analyze a recorded video")
    st.caption("Upload the original file from the camera: at least 10 s, one face, still and well lit. "
               "Re-encoded or compressed clips can lose the pulse. The file is deleted right after analysis.")
    up = st.file_uploader("Face video", type=["mp4", "avi", "mov", "mkv"])
    key = (up.name, up.size, choice) if up else None
    if up and st.button("Analyze video", type="primary"):
        bar = st.progress(0.0, text="Analyzing video...")
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(up.name)[1], delete=False) as tmp:
            tmp.write(up.getbuffer())
        try:
            ests, final = analyze_video(tmp.name, tracker.method, tracker.tscan,
                                        lambda p: bar.progress(p, text=f"Analyzing video... {p:.0%}"))
        finally:
            os.remove(tmp.name)
        bar.empty()
        st.session_state.upload_result = (key, ests, final)
    res = st.session_state.get("upload_result")
    if key and res and res[0] == key:
        _, ests, final = res
        if not ests:
            st.error("No heart rate found: the clip is shorter than 10 s or no face was detected.")
        else:
            bpm = np.array([b for _, b in ests])
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Heart rate (BPM)", f"{np.median(bpm):.0f}")
                c2.metric("Range across windows", f"{bpm.min():.0f}-{bpm.max():.0f}")
                c3.metric("Windows scored", len(bpm))
            st.caption("BPM per 10 s window over the video")
            st.line_chart(pd.DataFrame({"BPM": bpm}, index=pd.Index([t for t, _ in ests], name="time (s)")), height=200)
            st.caption("Pulse waveform, last 10 s")
            st.line_chart(final["bvp"], height=160)



def on_frame(frame):
    bgr = frame.to_ndarray(format="bgr24")
    state = tracker.update(np.ascontiguousarray(bgr[:, :, ::-1]), time.time())
    return av.VideoFrame.from_ndarray(overlay(bgr, state), format="bgr24")


with tab_live:
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            ctx = webrtc_streamer(key="pulsecam", video_frame_callback=on_frame, async_processing=True,
                                  media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 30}, "audio": False})
    with right:
        with st.container(border=True):
            progress_box = st.empty()
            m1, m2 = st.columns(2)
            hr_box, snr_box = m1.empty(), m2.empty()
        warn_box, cmp_box, trend, chart = st.empty(), st.empty(), st.empty(), st.empty()
        hr_box.metric("Heart rate (BPM)", "--")
        snr_box.metric("Signal quality (dB)", "--")
        warn_box.info("Click START, then hold still facing the camera.")
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
    if not s["hr"] and s["progress"] < 1.0:
        progress_box.progress(s["progress"], text=f"Analyzing... {s['progress'] * 10:.0f}/10 s")
    else:
        progress_box.empty()
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
