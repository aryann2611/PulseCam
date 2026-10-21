"""Fine-tune TS-CAN on UBFC-rPPG and score it on people it never saw (split by person, never by clip).

prep : face crops (72x72) + PPG label of every subject -> one .npz per subject in --cache
train: start from the toolbox's PURE_TSCAN.pth, fine-tune on the train people, keep the epoch with the lowest
       MAE on the valid people, then score test people with the original and the fine-tuned weights.
Usage: python eval/finetune_tscan.py prep  --root /content/drive/MyDrive/DATASET_2 --cache /content/cache
       python eval/finetune_tscan.py train --cache /content/cache [--epochs 5]
Split: people sorted by number; index % 7 == 0 -> test, == 3 -> valid, the rest -> train (6 / 6 / 30 of 42).
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

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["prep", "train"])
ap.add_argument("--root", default="data/ubfc", help="prep: folder with subject*/vid.avi + ground_truth.txt")
ap.add_argument("--cache", default="data/ubfc_cache")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=2, help="chunks of 180 frames per step")
ap.add_argument("--out", default="models/finetuned_tscan_ubfc.pth")
a = ap.parse_args()
CHUNK, WIN_S, STEP_S = 180, 10, 5
num = lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1])


def read_crops(path):
    import cv2
    from pulsecam.face import FaceTracker, face_crop
    cap, face, crops, last = cv2.VideoCapture(path), FaceTracker(), [], None
    fs = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, bgr = cap.read()
    while ok:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        box = face.detect(rgb)
        if box:
            last = face_crop(rgb, box)
        if last is not None:  # no face in this frame -> repeat the last good crop
            crops.append(last)
        ok, bgr = cap.read()
    return fs, np.array(crops)


if a.stage == "prep":
    os.makedirs(a.cache, exist_ok=True)
    for s in sorted(glob.glob(os.path.join(a.root, "subject*")), key=num):
        sid, out = os.path.basename(s), os.path.join(a.cache, os.path.basename(s) + ".npz")
        if os.path.exists(out) or not (os.path.exists(s + "/vid.avi") and os.path.exists(s + "/ground_truth.txt")):
            continue
        fs, crops = read_crops(s + "/vid.avi")
        ppg = np.loadtxt(s + "/ground_truth.txt")[0]
        n = min(len(crops), len(ppg))
        np.savez(out + ".tmp.npz", crops=crops[:n], ppg=ppg[:n], fs=fs)
        os.replace(out + ".tmp.npz", out)  # a half-written file is never mistaken for a finished one
        print(f"{sid}: {n} frames cached", flush=True)
    sys.exit()

import torch

from pulsecam.metrics import summary
from pulsecam.signal import fft_bpm
from pulsecam.tscan import TOOLBOX, TSCANPredictor

torch.manual_seed(0)
np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
files = sorted(glob.glob(os.path.join(a.cache, "subject*.npz")), key=num)
ids = [os.path.basename(f)[:-4] for f in files]
split = {i: "test" if k % 7 == 0 else "valid" if k % 7 == 3 else "train" for k, i in enumerate(ids)}
os.makedirs("results/tables", exist_ok=True)
pd.DataFrame(dict(subject=ids, split=[split[i] for i in ids])).to_csv("results/tables/finetune_split.csv", index=False)
data = {}
for f, i in zip(files, ids):
    z = np.load(f)
    data[i] = (z["crops"], z["ppg"], float(z["fs"]))
print({k: sum(v == k for v in split.values()) for k in ("train", "valid", "test")}, "people; device", dev, flush=True)


def inputs(face):  # same preprocessing as TSCANPredictor.predict
    x = face.astype(np.float32)
    std = np.nan_to_num((x - x.mean()) / (x.std() + 1e-7))
    return np.ascontiguousarray(np.concatenate([TSCANPredictor._diff_norm(x), std], -1).transpose(0, 3, 1, 2))


def label(ppg):  # first difference of the PPG, unit std, like the toolbox's DiffNormalized label
    d = np.diff(ppg)
    return np.nan_to_num(np.append(d / (d.std() + 1e-7), 0))


def make_net(weights=None):
    return TSCANPredictor(weights).net.to(dev)


def score(net, subjects, step_s=STEP_S):
    """10 s windows -> predicted vs ground-truth HR (FFT peak), exactly like eval/ubfc_eval.py."""
    net.eval()
    rows = []
    for sid in subjects:
        crops, ppg, fs = data[sid]
        w, st = int(WIN_S * fs), int(step_s * fs)
        for t0 in range(0, len(crops) - w + 1, st):
            x = torch.from_numpy(inputs(crops[t0:t0 + w // 10 * 10])).to(dev)
            with torch.no_grad():
                bvp = np.cumsum(net(x).cpu().numpy().ravel())
            rows.append(dict(subject=sid, t0_s=round(t0 / fs, 1), pred_hr=fft_bpm(bvp, fs)[0], ref_hr=fft_bpm(ppg[t0:t0 + w], fs)[0]))
    return pd.DataFrame(rows)


mae = lambda d: float((d.pred_hr - d.ref_hr).abs().mean())
pick = lambda s: [i for i in ids if split[i] == s]
chunks = [(i, t0) for i in pick("train") for t0 in range(0, len(data[i][0]) - CHUNK + 1, CHUNK // 2)]
net = make_net(os.path.join(TOOLBOX, "final_model_release", "PURE_TSCAN.pth"))
best, best_sd = mae(score(net, pick("valid"), 10)), {k: v.clone() for k, v in net.state_dict().items()}
print(f"epoch 0 (released PURE weights): valid MAE {best:.2f} BPM, {len(chunks)} train chunks", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
for ep in range(1, a.epochs + 1):
    net.train()
    order, losses = np.random.permutation(len(chunks)), []
    for b in range(0, len(order) - a.batch + 1, a.batch):
        sel = [chunks[j] for j in order[b:b + a.batch]]
        x = np.concatenate([inputs(data[i][0][t:t + CHUNK]) for i, t in sel])
        y = np.concatenate([label(data[i][1][t:t + CHUNK]) for i, t in sel])
        loss = torch.nn.functional.mse_loss(net(torch.from_numpy(x).to(dev)).view(-1), torch.from_numpy(y).float().to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    v = mae(score(net, pick("valid"), 10))
    print(f"epoch {ep}: train loss {np.mean(losses):.3f}, valid MAE {v:.2f} BPM", flush=True)
    if v < best:
        best, best_sd = v, {k: t.clone() for k, t in net.state_dict().items()}
        print("  new best", flush=True)

net.load_state_dict(best_sd)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
torch.save({k: v.cpu() for k, v in net.state_dict().items()}, a.out)
test = pick("test")
res = {"PURE_released": score(make_net(os.path.join(TOOLBOX, "final_model_release", "PURE_TSCAN.pth")), test),
       "UBFC_finetuned": score(net, test)}
tab = pd.DataFrame([dict(model=m, level=lv, **summary(g.pred_hr, g.ref_hr))
                    for m, d in res.items()
                    for lv, g in (("window", d), ("video", d.groupby("subject")[["pred_hr", "ref_hr"]].median()))]).round(2)
tab.to_csv("results/tables/finetune_ubfc_test.csv", index=False)
pd.concat([d.assign(model=m) for m, d in res.items()]).to_csv("results/tables/finetune_ubfc_test_windows.csv", index=False)
print(f"test people: {test}\n{tab.to_string(index=False)}\nsaved {a.out} (best valid MAE {best:.2f})")
