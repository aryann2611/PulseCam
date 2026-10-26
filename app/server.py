"""PulseCam web app. Run:  python app/server.py   (opens http://127.0.0.1:8000)
Research prototype - NOT a medical device."""
import asyncio
import csv
import json
import os
import sys
import tempfile
import threading
import time
import uuid
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from pulsecam.face import FaceTracker, rois
from pulsecam.pipeline import HRTracker, analyze_video

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOTS = os.path.join(ROOT, "results", "plots")
LOG = os.path.join(ROOT, "results", "e5_live_log.csv")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
JOBS = {}


def load_tscan():
    try:
        from pulsecam.tscan import TSCANPredictor
        p = TSCANPredictor()
        # measured on this laptop: 2 threads 4.2 s per estimate, 4 threads 9.1 s and starves frame capture
        p.torch.set_num_threads(2)
        p.predict(np.random.rand(300, 72, 72, 3) * 255)  # warm-up: the first call takes seconds and would freeze the live view
        return p
    except Exception as e:  # torch or weights missing -> POS only
        print(f"TS-CAN unavailable ({type(e).__name__}); POS only.")
        return None


print("Loading models...")
TSCAN = load_tscan()
FaceTracker()  # the first detector in a process takes ~3.5 s to create, later ones ~0.3 s


def model(method):
    return ("tscan", TSCAN) if method == "tscan" and TSCAN else ("pos", None)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def public(s):
    box = s["box"]
    return dict(hr=s["hr"], snr=None if s["snr"] is None else float(s["snr"]), warn=s["warn"],
                progress=s["progress"], box=box and [float(v) for v in box],
                rois=box and [[float(v) for v in r] for r in rois(box)],
                bvp=np.round(s["bvp"], 4).tolist())


async def index(request):
    return FileResponse(os.path.join(HERE, "web", "index.html"), headers={"Cache-Control": "no-cache"})


async def info(request):
    return JSONResponse({"tscan": TSCAN is not None})


async def results(request):
    plots = sorted(p for p in os.listdir(PLOTS) if p.startswith("BA_ubfc_")) if os.path.isdir(PLOTS) else []
    return JSONResponse({"e1e2": read_csv(os.path.join(ROOT, "results", "tables", "E1_E2_ubfc.csv")),
                         "plots": plots, "e5": read_csv(LOG)})


async def status(request):
    with open(os.path.join(ROOT, "docs", "CHECKLIST.md"), encoding="utf-8") as f:
        return JSONResponse({"text": f.read()})


async def log_reading(request):
    d = await request.json()
    ref = d.get("ref")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "method", "pulsecam_bpm", "reference_bpm"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), model(d.get("method"))[0],
                    round(float(d["bpm"]), 1), round(float(ref), 1) if ref else ""])
    return JSONResponse({"ok": True})


def run_job(job, path, method, tscan):
    try:
        ests, final = analyze_video(path, method, tscan, lambda p: job.update(progress=p))
        bpm = [float(b) for _, b in ests]
        job["result"] = dict(bpm=float(np.median(bpm)), lo=min(bpm), hi=max(bpm), n=len(bpm), method=method,
                             series=[[round(t, 2), round(b, 1)] for (t, _), b in zip(ests, bpm)],
                             bvp=np.round(final["bvp"], 4).tolist()) if ests else None
    except Exception as e:
        job["error"] = f"Could not read this video ({type(e).__name__})."
    finally:
        os.remove(path)  # uploaded faces are never kept
        job.update(progress=1.0, done=True)


async def upload(request):
    ext = os.path.splitext(request.query_params.get("name", ""))[1].lower()
    if ext not in VIDEO_EXT:
        return JSONResponse({"error": "Use an mp4, avi, mov or mkv file."}, 400)
    method, tscan = model(request.query_params.get("method"))
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    jid = uuid.uuid4().hex
    JOBS[jid] = {"progress": 0.0, "done": False}
    threading.Thread(target=run_job, args=(JOBS[jid], path, method, tscan), daemon=True).start()
    return JSONResponse({"id": jid})


async def job(request):
    j = JOBS.get(request.path_params["jid"])
    return JSONResponse(dict(j)) if j else JSONResponse({"error": "unknown job"}, 404)


async def live(ws):
    """Browser sends a JSON config, then raw frames: float64 timestamp (s) + RGBA pixels. Replies with the state."""
    await ws.accept()
    tracker, w, h = await asyncio.to_thread(HRTracker, "pos", None, True), 0, 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                cfg = json.loads(msg["text"])
                tracker.method, tracker.tscan = model(cfg.get("method"))
                w, h = int(cfg.get("width", w)), int(cfg.get("height", h))
                continue
            data = msg.get("bytes") or b""
            if len(data) != 8 + w * h * 4:
                continue
            t = float(np.frombuffer(data, np.float64, 1)[0])
            rgb = np.ascontiguousarray(np.frombuffer(data, np.uint8, offset=8).reshape(h, w, 4)[:, :, :3])
            state = await asyncio.to_thread(tracker.update, rgb, t)
            await ws.send_json(public(state))
    except (WebSocketDisconnect, RuntimeError):
        pass


app = Starlette(routes=[
    Route("/", index),
    Route("/api/info", info),
    Route("/api/results", results),
    Route("/api/status", status),
    Route("/api/log", log_reading, methods=["POST"]),
    Route("/api/upload", upload, methods=["POST"]),
    Route("/api/job/{jid}", job),
    WebSocketRoute("/ws/live", live),
    Mount("/plots", StaticFiles(directory=PLOTS, check_dir=False)),
])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"PulseCam running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    threading.Timer(1.5, webbrowser.open, [f"http://127.0.0.1:{port}"]).start()
    # compressing 1.2 MB raw frames cost ~100 ms each and capped live capture at ~8 fps
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", ws_per_message_deflate=False)
