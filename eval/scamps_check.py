"""Why does TS-CAN trained on SCAMPS score ~14 BPM on UBFC with our scorer? Re-score the cached face crops several ways.

A0 old10       : per-window normalisation, cumsum only, our band-pass + FFT (what E1/E2 reported on 21 Oct, before the fix)
A  new10       : the same plus the toolbox's smooth detrend (pulsecam.signal.smooth_detrend), i.e. predict(detrend=True)
B  ours+tbpost : the same predictions, scored with the toolbox's own post-processing (cumsum -> detrend lambda=100 -> band-pass -> FFT)
C  tb10        : toolbox normalisation (whole video, then chunks of 180) + toolbox post-processing, 10 s windows
D  tbvideo     : like C but one heart rate per whole video (the toolbox's default metric)
Usage: python eval/scamps_check.py --cache <dir with subject*.npz from finetune_tscan.py prep> [--max 42]
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rPPG-Toolbox"))

import numpy as np
import pandas as pd
import torch

import importlib.util

from pulsecam.signal import fft_bpm
from pulsecam.tscan import TOOLBOX, TSCANPredictor

_spec = importlib.util.spec_from_file_location("tb_post", os.path.join(TOOLBOX, "evaluation", "post_process.py"))
_post = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_post)  # loaded by path: importing the toolbox's packages needs h5py and more
calculate_metric_per_video = _post.calculate_metric_per_video


class BaseLoader:  # verbatim copies of the toolbox's BaseLoader.diff_normalize_data / _label / standardized_data
    @staticmethod
    def diff_normalize_data(data):
        n, h, w, c = data.shape
        out = np.zeros((n - 1, h, w, c), dtype=np.float32)
        for j in range(n - 1):
            out[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
        out = out / np.std(out)
        out = np.append(out, np.zeros((1, h, w, c), dtype=np.float32), axis=0)
        out[np.isnan(out)] = 0
        return out

    @staticmethod
    def diff_normalize_label(label):
        d = np.diff(label, axis=0)
        out = np.append(d / np.std(d), np.zeros(1), axis=0)
        out[np.isnan(out)] = 0
        return out

    @staticmethod
    def standardized_data(data):
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--max", type=int, default=0)
a = ap.parse_args()
num = lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1])
files = sorted(glob.glob(os.path.join(a.cache, "subject*.npz")), key=num)
files = files[:a.max] if a.max else files
MODELS = {"PURE": "PURE_TSCAN.pth", "SCAMPS": "SCAMPS_TSCAN.pth"}
nets = {k: TSCANPredictor(os.path.join(TOOLBOX, "final_model_release", w)) for k, w in MODELS.items()}
dev = "cuda" if torch.cuda.is_available() else "cpu"
for p in nets.values():
    p.net.to(dev)


def forward(p, inp):  # inp (N,6,72,72) float32 -> diff-BVP (N,)
    with torch.no_grad():
        return p.net(torch.from_numpy(inp).to(dev)).cpu().numpy().ravel()


def tb_inputs(crops):  # toolbox: normalise over the whole video, then cut chunks of 180 frames
    x = crops.astype(np.float64)
    d = np.concatenate([BaseLoader.diff_normalize_data(x.copy()), BaseLoader.standardized_data(x.copy())], -1)
    n = len(d) // 180 * 180
    return np.ascontiguousarray(d[:n].transpose(0, 3, 1, 2), dtype=np.float32)


def tb_hr(pred_diff, label_diff, fs):  # toolbox post-processing; returns (hr_pred, hr_label)
    hl, hp, _, _ = calculate_metric_per_video(pred_diff, label_diff, fs=fs)
    return hp, hl


rows = []
for f in files:
    sid = os.path.basename(f)[:-4]
    z = np.load(f)
    crops, ppg, fs = z["crops"], z["ppg"], float(z["fs"])
    lab = BaseLoader.diff_normalize_label(ppg)
    w, st = int(10 * fs), int(5 * fs)
    inp = tb_inputs(crops)
    for m, p in nets.items():
        pred_tb = np.concatenate([forward(p, inp[i:i + 180]) for i in range(0, len(inp), 180)])  # toolbox-style predictions
        n = len(pred_tb)
        for t0 in range(0, min(len(crops), len(ppg)) - w + 1, st):
            ref = fft_bpm(ppg[t0:t0 + w], fs)[0]
            old = p.predict(crops[t0:t0 + w], detrend=False)  # per-window normalisation, cumsum only
            new = p.predict(crops[t0:t0 + w], detrend=True)
            rows.append(dict(subject=sid, model=m, variant="A0 old10", pred=fft_bpm(old, fs)[0], ref=ref))
            bpm, snr = fft_bpm(new, fs)
            rows.append(dict(subject=sid, model=m, variant="A new10", pred=bpm, ref=ref, snr=snr))
            hp, hl = tb_hr(np.concatenate([[old[0]], np.diff(old)]), lab[t0:t0 + w], fs)  # undo the cumsum exactly
            rows.append(dict(subject=sid, model=m, variant="B ours+tbpost", pred=hp, ref=hl))
            if t0 + w <= n:
                hp, hl = tb_hr(pred_tb[t0:t0 + w], lab[t0:t0 + w], fs)
                rows.append(dict(subject=sid, model=m, variant="C tb10", pred=hp, ref=hl))
        hp, hl = tb_hr(pred_tb, lab[:n], fs)
        rows.append(dict(subject=sid, model=m, variant="D tbvideo", pred=hp, ref=hl))
    print(sid, flush=True)

d = pd.DataFrame(rows)
d["err"] = (d.pred - d.ref).abs()
os.makedirs("results/tables", exist_ok=True)
d.to_csv("results/tables/scamps_check_rows.csv", index=False)
t = d.groupby(["variant", "model"]).err.agg(MAE="mean", median="median", n="count").round(2)
t.to_csv("results/tables/scamps_check.csv")
print(t.to_string())
# the TS-CAN rows of results/tables/E1_E2_ubfc.csv recomputed with the new post-processing (variant A)
from pulsecam.metrics import summary
out = []
for m, name in (("PURE", "TSCAN_PURE"), ("SCAMPS", "TSCAN_SCAMPS")):
    g = d[(d.variant == "A new10") & (d.model == m)]
    v = g.groupby("subject")[["pred", "ref"]].median()
    out.append(dict(method=name, level="window", **summary(g.pred, g.ref), SNR=g.snr.mean()))
    out.append(dict(method=name, level="video", **summary(v.pred, v.ref)))
pd.DataFrame(out).round(2).to_csv("results/tables/E1_E2_tscan_rescored.csv", index=False)
print("RESCORED_CSV_BEGIN", pd.DataFrame(out).round(2).to_csv(index=False), "RESCORED_CSV_END", sep="\n")
