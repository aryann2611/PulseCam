"""Plain OpenCV fallback demo (no browser). Run: python app/cv_app.py [--model tscan]   (q quits)"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from pulsecam.draw import overlay
from pulsecam.pipeline import HRTracker

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["pos", "tscan"], default="pos")
ap.add_argument("--source", default="0", help="camera index or a video file (pre-recorded backup)")
a = ap.parse_args()

tscan = None
if a.model == "tscan":
    from pulsecam.tscan import TSCANPredictor
    tscan = TSCANPredictor()
tracker = HRTracker(a.model, tscan)
cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
t0, n = time.time(), 0
while cap.isOpened():
    ok, bgr = cap.read()
    if not ok:
        break
    n += 1
    t = time.time() if a.source.isdigit() else n / (cap.get(cv2.CAP_PROP_FPS) or 30)  # file: video time
    state = tracker.update(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)
    cv2.imshow("PulseCam (not a medical device)", overlay(bgr, state))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
