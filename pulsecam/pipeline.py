"""Live/offline heart-rate tracker: rolling 10 s buffer, estimate every second, median of last 5."""
import threading
from collections import deque

import cv2
import numpy as np

from .face import FaceTracker, face_crop, roi_mean
from .signal import fft_bpm, pos_bvp

FS, WIN_S, STEP_S, MEDIAN_N = 30, 10, 1.0, 5
MAX_MOTION = 0.12   # face-centre travel in the last 2 s, as a fraction of face width
MIN_BRIGHT = 60     # mean face brightness (0-255)
MIN_SNR = -3.0      # dB


class HRTracker:
    def __init__(self, method="pos", tscan=None, background=False):
        """background=True (live use): estimate in a thread so slow TS-CAN on CPU never stalls frame capture;
        an estimate due while one is running is skipped. Offline runs keep the default for repeatable results."""
        self.method, self.tscan, self.background, self.busy = method, tscan, background, False
        self.face, self.lock = FaceTracker(), threading.RLock()
        self.buf = deque()  # (t, rgb_mean, crop72, centre_x, centre_y, width, brightness)
        self.hist, self.last_est = deque(maxlen=MEDIAN_N), -1e9
        self.state = dict(hr=None, raw=None, bvp=np.zeros(0), snr=None, warn="Looking for a face...", box=None, progress=0.0)

    def update(self, rgb, t):
        """Feed one RGB frame (uint8) with timestamp t (s). Returns the current state dict."""
        box = self.face.detect(rgb)
        with self.lock:
            m = roi_mean(rgb, box) if box else None
            if m is None:
                self.state.update(warn="No face found", box=None)
                return self.state
            crop = face_crop(rgb, box)
            x, y, w, h = box
            self.buf.append((t, m, crop, x + w / 2, y + h / 2, w, float(crop.mean())))
            while t - self.buf[0][0] > WIN_S:
                self.buf.popleft()
            self.state["box"] = box
            span = self.buf[-1][0] - self.buf[0][0]
            self.state["progress"] = min(span / WIN_S, 1.0)
            if span < WIN_S - 0.5:
                self.state["warn"] = f"Analyzing... {span:.0f}/{WIN_S} s"
            elif t - self.last_est >= STEP_S and not self.busy:
                self.last_est = t
                if self.background:
                    self.busy = True
                    threading.Thread(target=self._estimate, args=(list(self.buf),), daemon=True).start()
                else:
                    self._estimate(list(self.buf))
            return self.state

    def _estimate(self, buf):
        try:
            ts = np.array([b[0] for b in buf])
            idx = np.searchsorted(ts, np.linspace(ts[0], ts[-1], WIN_S * FS)).clip(0, len(ts) - 1)  # resample to 30 fps
            win = [buf[i] for i in idx]
            if self.method == "tscan" and self.tscan:
                bvp = self.tscan.predict(np.stack([b[2] for b in win]))
            else:
                bvp = pos_bvp(np.stack([b[1] for b in win]), FS)
            bpm, snr = fft_bpm(bvp, FS)
            last = [b for b in buf if b[0] > ts[-1] - 2]
            move = np.hypot(np.ptp([b[3] for b in last]), np.ptp([b[4] for b in last])) / last[-1][5]
            warn = ("Hold still" if move > MAX_MOTION else "Too dark" if np.mean([b[6] for b in win]) < MIN_BRIGHT
                    else "Weak signal - hold still, face the light" if snr < MIN_SNR else "")
            with self.lock:
                self.hist.append(bpm)
                self.state.update(hr=float(np.median(self.hist)), raw=bpm, bvp=bvp, snr=snr, warn=warn)
        finally:
            self.busy = False


def analyze_video(path, method="pos", tscan=None, on_progress=None):
    """Run the tracker over a video file. Returns ([(t_s, bpm) per scored window], final state)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker, ests, n = HRTracker(method, tscan), [], 0
    ok, bgr = cap.read()
    while ok:
        n += 1
        tracker.update(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), n / fps)
        if tracker.last_est == n / fps:  # a new 10 s window was scored on this frame
            ests.append((n / fps, tracker.state["raw"]))
        if on_progress and total and n % 15 == 0:
            on_progress(min(n / total, 1.0))
        ok, bgr = cap.read()
    cap.release()
    return ests, tracker.state
