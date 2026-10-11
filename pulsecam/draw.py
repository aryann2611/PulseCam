"""Overlay drawing shared by the Streamlit demo and the OpenCV fallback (BGR frames)."""
import cv2
import numpy as np

from .face import rois


def overlay(bgr, state):
    box = state["box"]
    if box is not None:
        x, y, w, h = map(int, box)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for x0, y0, x1, y1 in rois(box):
            cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)), (255, 200, 0), 1)
    hr = state["hr"]
    cv2.putText(bgr, f"{hr:.0f} BPM" if hr else "-- BPM", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    if state["warn"]:
        cv2.putText(bgr, state["warn"], (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    b = state["bvp"]
    if len(b) > 1:  # pulse strip along the bottom
        H, W = bgr.shape[:2]
        v = (b - b.mean()) / (b.std() + 1e-9)
        pts = np.stack([np.linspace(0, W - 1, len(v)), H - 40 - 25 * np.clip(v, -2, 2)], 1).astype(np.int32)
        cv2.polylines(bgr, [pts], False, (0, 255, 0), 2)
    cv2.putText(bgr, "Research prototype - not a medical device", (10, bgr.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return bgr
