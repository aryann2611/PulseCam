"""E1 / E2 on UBFC-rPPG with our own pipeline (face -> ROI/crop -> method -> bandpass -> FFT peak).

Why not the toolbox's main.py: its cached preprocessing gave 2-subject / "MAE 0.0" results on Colab; this
script scores every video the same way for every method, so the numbers are comparable.
Every 10 s window (step 5 s) is scored against the HR of the ground-truth PPG in the same window.

E1 = GREEN, ICA, CHROM, POS (classical)  vs  TS-CAN trained on PURE (toolbox release)
E2 = TS-CAN trained on PURE  vs  TS-CAN trained on SCAMPS, both tested on UBFC (we use the toolbox's released weights)
Usage: python eval/ubfc_eval.py --root data/ubfc [--max 10]     (needs subject*/vid.avi + ground_truth.txt)
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rPPG-Toolbox"))

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulsecam.face import FaceTracker, face_crop, roi_mean
from pulsecam.metrics import bland_altman, summary
from pulsecam.signal import fft_bpm, pos_bvp
from pulsecam.tscan import TOOLBOX, TSCANPredictor
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.ICA_POH import ICA_POH

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/ubfc")
ap.add_argument("--max", type=int, default=0, help="only the first N subjects (by number)")
ap.add_argument("--win", type=float, default=10, help="window length, s")
ap.add_argument("--step", type=float, default=5, help="window step, s")
ap.add_argument("--from-csv", help="skip the videos: rebuild tables/plots from an earlier E1_E2_ubfc_windows.csv")
ap.add_argument("--partial", help="CSV rewritten after every subject; if it exists the run resumes and skips those subjects")
a = ap.parse_args()
T, P = "results/tables", "results/plots"
os.makedirs(T, exist_ok=True)
os.makedirs(P, exist_ok=True)

TS = {"TSCAN_PURE": "PURE_TSCAN.pth", "TSCAN_SCAMPS": "SCAMPS_TSCAN.pth"}
tscan = {} if a.from_csv else {k: TSCANPredictor(os.path.join(TOOLBOX, "final_model_release", w)) for k, w in TS.items()}


def classical(name, rgb, fs):
    f = rgb[:, None, None, :]  # (T,1,1,3): the toolbox methods only average each frame, so pass the ROI mean
    return {"GREEN": lambda: GREEN(f), "ICA": lambda: ICA_POH(f, fs), "CHROM": lambda: CHROME_DEHAAN(f, fs),
            "POS": lambda: pos_bvp(rgb, fs)}[name]()


def read_video(path):
    cap, face, rgbs, crops, last = cv2.VideoCapture(path), FaceTracker(), [], [], None
    fs = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, bgr = cap.read()
    while ok:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = face.detect(rgb)
        m = roi_mean(rgb, box) if box else None
        if m is not None:
            last = (m, face_crop(rgb, box))
        if last is not None:  # no face in this frame -> repeat the last good one
            rgbs.append(last[0])
            crops.append(last[1])
        ok, bgr = cap.read()
    return fs, np.array(rgbs), np.array(crops)


subs = sorted(glob.glob(os.path.join(a.root, "subject*")), key=lambda s: int(re.findall(r"\d+", s)[-1]))
subs = [s for s in subs if os.path.exists(s + "/vid.avi") and os.path.exists(s + "/ground_truth.txt")]
subs = [] if a.from_csv else subs[:a.max] if a.max else subs
rows = []
if a.partial and os.path.exists(a.partial):
    rows = pd.read_csv(a.partial).to_dict("records")
done = {r["subject"] for r in rows}
for s in subs:
    sid = os.path.basename(s)
    if sid in done:
        print(f"{sid}: already in {a.partial}, skipped", flush=True)
        continue
    fs, rgb, crops = read_video(s + "/vid.avi")
    ppg = np.loadtxt(s + "/ground_truth.txt")[0]
    n, w, st = min(len(rgb), len(ppg)), int(a.win * fs), int(a.step * fs)
    print(f"{sid}: {n} frames @ {fs:.1f} fps", flush=True)
    for t0 in range(0, n - w + 1, st):
        sl = slice(t0, t0 + w)
        gt = fft_bpm(ppg[sl], fs)[0]
        sig = {m: (lambda m=m: classical(m, rgb[sl], fs)) for m in ["GREEN", "ICA", "CHROM", "POS"]}
        sig.update({k: (lambda k=k: tscan[k].predict(crops[sl])) for k in TS})
        for m, f in sig.items():
            try:
                bpm, snr = fft_bpm(f(), fs)
            except Exception as e:  # e.g. ICA can fail on a flat window; skip that window
                print("  skip", sid, m, t0, type(e).__name__)
                continue
            rows.append(dict(subject=sid, method=m, t0_s=round(t0 / fs, 1), pred_hr=bpm, ref_hr=gt, snr=snr))
    if a.partial:
        pd.DataFrame(rows).to_csv(a.partial, index=False)
        print(f"{sid}: saved ({len(rows)} rows)", flush=True)

d = pd.read_csv(a.from_csv) if a.from_csv else pd.DataFrame(rows)
d.to_csv(f"{T}/E1_E2_ubfc_windows.csv", index=False)
out = []
for m, g in d.groupby("method", sort=False):
    v = g.groupby("subject")[["pred_hr", "ref_hr"]].median()  # one number per video, like the toolbox's per-video score
    out.append(dict(method=m, level="window", **summary(g.pred_hr, g.ref_hr), SNR=g.snr.mean()))
    out.append(dict(method=m, level="video", **summary(v.pred_hr, v.ref_hr)))
tab = pd.DataFrame(out).round(2)
tab["experiment"] = tab.method.map(lambda m: "E1+E2" if m == "TSCAN_PURE" else "E2" if m == "TSCAN_SCAMPS" else "E1")
tab.to_csv(f"{T}/E1_E2_ubfc.csv", index=False)
print(tab.to_string(index=False))

for m, g in d.groupby("method"):
    fig, ax = plt.subplots(figsize=(5, 4))
    bland_altman(ax, g.pred_hr, g.ref_hr, f"UBFC {m}")
    fig.tight_layout()
    fig.savefig(f"{P}/BA_ubfc_{m}.png", dpi=150)
    plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 4))
tab[tab.level == "window"].set_index("method").MAE.plot.bar(ax=ax)
ax.set_ylabel("MAE (BPM), 10 s windows")
fig.tight_layout()
fig.savefig(f"{P}/E1_E2_mae.png", dpi=150)
