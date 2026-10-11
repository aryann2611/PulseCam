# PulseCam – context for Claude

Student deep-learning project: estimate resting heart rate (BPM) + pulse waveform from webcam face video (rPPG), live.
**Research prototype. NOT a medical device – never present it as diagnostic.** Source of truth: the project spec PDF (`PulseCam – Project Specification`, 7 Oct 2026). Budget ₹0: free tools only.

## Hard rules
- **Privacy:** never commit/upload faces, consent forms or the ID→name list. `data/` is gitignored; keep it that way. Use IDs (P01…), never names, in files.
- **Splits are by person:** one person's videos live in exactly one of train/valid/test. `OwnLoader.split_raw_data` and `eval/predict_own.py --only` enforce/serve this.
- Prefer existing rPPG-Toolbox components over new code. Explain simply – the team are students.
- Don't use the toolbox's own score on own recordings for E3/E4: its labels there are POS pseudo-labels (circular). Score against the reference HR with `eval/predict_own.py`.

## Pipeline
video 30 fps → MediaPipe face box + skin ROIs → crop 72×72 / normalise → model (POS, or TS-CAN from the toolbox) → bandpass 0.7–3.0 Hz → FFT peak → `HR = 60·f_peak`.
Demo: 10 s rolling buffer (300 frames), estimate every 1 s, show median of last 5, "Hold still" warning on motion / darkness / low SNR.

## Layout
| Path | What |
|---|---|
| `pulsecam/` | `signal.py` (bandpass, FFT BPM, POS), `metrics.py`, `face.py` (MediaPipe Tasks detector), `tscan.py` (TS-CAN inference using the toolbox model class), `pipeline.py` (`HRTracker`), `draw.py` |
| `app/app.py` | Streamlit + streamlit-webrtc dashboard (`streamlit run app/app.py`): Live tab (BPM, signal quality, trend, E5 logging), Results tab (E1/E2 from `results/tables`), Status tab (docs/CHECKLIST.md); `app/cv_app.py` = OpenCV fallback / plays a backup video |
| `eval/` | `ubfc_eval.py` (E1/E2 on UBFC, our own scoring), `predict_own.py` (E3/E4/E5 predictions), `report.py` (tables + Bland-Altman), `parse_logs.py` (toolbox logs → table; see gotcha on toolbox cache) |
| `toolbox_patch/` | `OwnLoader.py` + `install.py` (idempotent patch of `rPPG-Toolbox/`: dataset `PulseCam`, `PULSECAM_INIT` fine-tune hook) |
| `configs/train/` | `PulseCam_finetune_TSCAN.yaml` (POS pseudo-label fine-tune) |
| `notebooks/pulsecam_colab.ipynb` | Colab runbook for E1, E2, fine-tune |
| `docs/` | consent form, recording protocol, report outline, `CHECKLIST.md` |
| `data/`, `results/`, `models/` | private data; tables/plots/logs; weights (`best_tscan.pth` is picked up automatically by demo + eval) |
| `rPPG-Toolbox/` | cloned, gitignored; re-run `python toolbox_patch/install.py rPPG-Toolbox` after a fresh clone |

## Commands
- Tests: `.venv\Scripts\python -m pytest tests -q` (use the venv; global Python lacks deps)
- Demo: `streamlit run app/app.py` · fallback `python app/cv_app.py [--model tscan] [--source video.mp4]`
- Own recordings → results: `python eval/predict_own.py --method pos` / `--method tscan --stage pretrained|finetuned --only P17,P18,P19,P20`, then `python eval/report.py`
- E1/E2 (UBFC): `python eval/ubfc_eval.py --root data/ubfc [--max 10]` → `results/tables/E1_E2_ubfc*.csv`, `results/plots/`; `--from-csv results/tables/E1_E2_ubfc_windows.csv` rebuilds tables/plots without videos. Backup demo video: `python app/cv_app.py --source <video> --save results/backup_demo.mp4`
- Toolbox (Linux/Colab/WSL only; Windows here has no WSL2): `cd rPPG-Toolbox && python main.py --config_file <yaml>`; keep logs with `2>&1 | tee results/logs/E1_<name>.log`, then `python eval/parse_logs.py`
- The toolbox YAML dataset keyword for own data is `PulseCam`; fine-tune needs `PULSECAM_INIT=<weights.pth>`.

## Experiments (answer RQ1–RQ4)
E1 baselines (GREEN/ICA/CHROM/POS) vs deep on UBFC-rPPG · E2 train PURE/SCAMPS → test UBFC · E3 robustness per condition (still/move/talk/dim) · E4 MAE per Monk skin-tone band before/after fine-tune · E5 live demo vs reference device (`results/e5_live_log.csv`). Metrics: MAE (main), RMSE, MAPE, Pearson, SNR, Bland-Altman. Team target: beat every classical baseline on E1; MAE < 5 BPM on still, well-lit clips.

## Gotchas
- mediapipe ≥1.0 has no `mp.solutions`; the detector uses the Tasks API with `models/blaze_face_short_range.tflite` (auto-downloads if missing).
- Toolbox pre-trained weights are `DataParallel` checkpoints (`module.` prefix); `TSCANPredictor` strips it. TS-CAN: 72×72 input, frame depth 10, outputs the *derivative* of the BVP (we `cumsum`).
- `pulsecam/signal.py` is named like the stdlib module – run scripts from the repo root, never from inside `pulsecam/`.
- **E1/E2 done 12 Oct 2026** (Colab, 10 UBFC subjects: 1,3,4,5,8,9,10,11,12,13; 109 ten-second windows per method, scored against the HR of the ground-truth PPG; `eval/ubfc_eval.py`): window MAE GREEN 24.05, ICA 4.16, POS 2.84, CHROM 1.08, **TS-CAN (PURE weights) 0.70**, TS-CAN (SCAMPS weights) 21.99 BPM. Video-level (median per video): CHROM 0.35, TS-CAN-PURE 0.41, POS 0.87, ICA 1.73, GREEN 17.8, TS-CAN-SCAMPS 17.2. Caveats: the deep models are the toolbox's *released* PURE/SCAMPS checkpoints (we trained nothing); the SCAMPS result looks too poor (toolbox paper reports a few BPM) – possibly a preprocessing mismatch in our face crop, unverified; only 10 of 42 subjects.
- **Toolbox `main.py` on Colab gave bogus E1 numbers** (PhysNet/TS-CAN "MAE 0.0, Pearson 1.0" from 2 scored samples; its cached preprocessing/file list was inconsistent), so those logs were discarded; `ubfc_eval.py` replaces them. Classical toolbox run also crashed on `np.mat` in `POS_WANG.py` (now patched by `install.py`).
- Colab: all four OpenCV wheels clash – after `pip install mediapipe` run `pip uninstall -y opencv-contrib-python opencv-python opencv-python-headless && pip install opencv-python-headless==4.10.0.84` or Python segfaults. Drive mount needs a manual OAuth click each new session. Browser downloads are blocked in the automation session; results were read back through Drive's text preview.
- **Re-encoding kills the pulse**: UBFC subject1 re-saved as default mp4 → POS read 62 BPM (true 107); original file → 107.2. Never compress own recordings.
- Verified 12 Oct 2026: `predict_own.py` + `report.py` run end to end on a UBFC clip renamed `P99_still.mp4` (pred 107.2 vs ref 106.7); Streamlit server starts and `/_stcore/health` = ok; `cv_app.py --save` renders the annotated backup video. **Still unverified:** live webcam end-to-end, `OwnLoader` inside the toolbox. **Not done (need people/hardware):** own recordings, consent forms, E3/E4/E5, fine-tune, `models/best_tscan.pth` (the best model is the released `PURE_TSCAN.pth`, which the demo already uses by default).
- UBFC videos: only subject1's `vid.avi` is local (gdown fails: Drive quota). On Colab, the Drive shortcut `MyDrive/DATASET_2` mounted via `drive.mount` copies fine (10 subjects took ~7 min); raise `MAX_SUBJECTS` in the notebook for more. Ground-truth files for all 42 are in `data/ubfc/DATASET_2/`.
