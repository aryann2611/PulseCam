"""MediaPipe face detection, skin ROIs (forehead + cheeks) and face crops."""
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "blaze_face_short_range.tflite")
URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
       "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")


class FaceTracker:
    """Largest face per frame, box smoothed to stop ROI jitter. Returns (x, y, w, h) or None."""

    def __init__(self, smooth=0.6):
        if not os.path.exists(MODEL):
            urllib.request.urlretrieve(URL, MODEL)
        opts = vision.FaceDetectorOptions(base_options=mpp.BaseOptions(model_asset_path=MODEL))
        self.det, self.smooth, self.box = vision.FaceDetector.create_from_options(opts), smooth, None

    def detect(self, rgb):
        res = self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.detections:
            return None
        b = max(res.detections, key=lambda d: d.bounding_box.width).bounding_box
        new = np.array([b.origin_x, b.origin_y, b.width, b.height], float)
        self.box = new if self.box is None else self.smooth * self.box + (1 - self.smooth) * new
        return tuple(self.box)


def rois(box):
    """Forehead, left cheek, right cheek as (x0, y0, x1, y1)."""
    x, y, w, h = box
    return [(x + .3 * w, y + .02 * h, x + .7 * w, y + .2 * h),
            (x + .12 * w, y + .52 * h, x + .38 * w, y + .78 * h),
            (x + .62 * w, y + .52 * h, x + .88 * w, y + .78 * h)]


def roi_mean(rgb, box):
    """Mean RGB over the skin ROIs, or None if they fall outside the frame."""
    H, W = rgb.shape[:2]
    px = []
    for x0, y0, x1, y1 in rois(box):
        r = rgb[max(int(y0), 0):min(int(y1), H), max(int(x0), 0):min(int(x1), W)]
        if r.size:
            px.append(r.reshape(-1, 3))
    return np.concatenate(px).mean(0) if px else None


def face_crop(rgb, box, size=72, coef=1.5):
    """Square crop around the face, enlarged by `coef` like the toolbox's USE_LARGE_FACE_BOX."""
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(int(max(w, h) * coef), 8)
    x0, y0 = int(cx - s / 2), int(cy - s / 2)
    big = cv2.copyMakeBorder(rgb, s, s, s, s, cv2.BORDER_REPLICATE)
    c = big[y0 + s:y0 + 2 * s, x0 + s:x0 + 2 * s]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
