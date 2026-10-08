"""Run PulseCam on the own recordings and score against the reference heart rate (E3, E4, E5 input).

Files:  data/own/videos/P01_still.mp4  +  P01_still_hr.csv  (columns time_s,hr_bpm; one row is fine for a manual count)
Usage:  python eval/predict_own.py --method pos
        python eval/predict_own.py --method tscan --weights models/best_tscan.pth --stage finetuned
Appends rows to results/own_predictions.csv.  Fine-tuning must use different volunteers than these tests.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pandas as pd

from pulsecam.pipeline import HRTracker

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["pos", "tscan"], default="pos")
ap.add_argument("--weights", help="TS-CAN weights (default: models/best_tscan.pth or the toolbox's PURE_TSCAN.pth)")
ap.add_argument("--stage", default="pretrained", help="label for this run, e.g. pretrained / finetuned")
ap.add_argument("--videos", default="data/own/videos")
ap.add_argument("--meta", default="data/own/metadata.csv")
ap.add_argument("--out", default="results/own_predictions.csv")
ap.add_argument("--only", help="comma list of person IDs to score (use the held-out test volunteers)")
a = ap.parse_args()

tscan = None
if a.method == "tscan":
    from pulsecam.tscan import TSCANPredictor
    tscan = TSCANPredictor(a.weights)
meta = pd.read_csv(a.meta, dtype=str) if os.path.exists(a.meta) else pd.DataFrame(columns=["id", "monk_shade"])
monk = meta.drop_duplicates("id").set_index("id")["monk_shade"].to_dict()
only = set(a.only.split(",")) if a.only else None

rows = []
for path in sorted(glob.glob(os.path.join(a.videos, "*.mp4"))):
    stem = os.path.basename(path)[:-4]
    pid, cond = stem.split("_", 1)
    if only and pid not in only:
        continue
    hr_file = path[:-4] + "_hr.csv"
    if not os.path.exists(hr_file):
        print("skip (no reference):", stem)
        continue
    ref = float(pd.read_csv(hr_file)["hr_bpm"].mean())
    cap, tracker, ests, n = cv2.VideoCapture(path), HRTracker(a.method, tscan), [], 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, bgr = cap.read()
    while ok:
        n += 1
        tracker.update(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), n / fps)
        if tracker.last_est == n / fps:  # a new 10 s window was scored on this frame
            ests.append(tracker.state["raw"])
        ok, bgr = cap.read()
    if not ests:
        print("skip (clip shorter than 10 s or no face):", stem)
        continue
    rows.append(dict(id=pid, condition=cond, monk=monk.get(pid), method=a.method, stage=a.stage,
                     pred_hr=float(np.median(ests)), ref_hr=ref, n_windows=len(ests)))
    print(f"{stem}: pred {rows[-1]['pred_hr']:.1f}  ref {ref:.1f}")

if rows:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    print("saved", len(rows), "rows ->", a.out)
