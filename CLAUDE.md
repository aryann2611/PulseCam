# PulseCam – context for Claude

Student deep-learning project: estimate resting heart rate (BPM) + pulse waveform from webcam face video (rPPG), live.
**Research prototype. NOT a medical device – never present it as diagnostic.** Source of truth: the project spec PDF (`PulseCam – Project Specification`, 7 Oct 2026). Budget ₹0: free tools only.

## When the user says "continue" - read this first (updated 21 Oct 2026)
Do not re-explain the project or ask what to do. Run `git status` and `git log -3`, take the first item below that is possible without people/hardware, finish it end to end (tests, commit, push), then report in a few lines: what is done, what is still blocked on people. **Never fake people/hardware work** (recordings, consent forms, reference device, E3-E5, live webcam). Report honestly: TS-CAN only ties CHROM on 42 UBFC subjects, and UBFC fine-tuning gave no gain.

**Where we are:** E1/E2 done on all 42 UBFC subjects (`results/tables/E1_E2_ubfc.csv`); UBFC fine-tune done, no gain, released `PURE_TSCAN.pth` stays the default model; local dashboard fixed; all pushed. Full numbers in Gotchas below.

**Next actions, in order:**
1. **Refresh the 42-subject Bland-Altman plots.** `results/plots/BA_ubfc_*.png` are still from the first 10 subjects. The 42-subject window data is `MyDrive/PulseCam/ubfc42_partial.csv` (plots + tables also in `MyDrive/PulseCam/results_42subjects/`). Downloads are blocked for Claude, so ask the user to put the file in the repo folder (any path, e.g. `colab_upload/`), then `.venv\Scripts\python eval/ubfc_eval.py --from-csv <file>`; check that `results/tables/E1_E2_ubfc.csv` still shows CHROM 2.0 / TS-CAN-PURE 2.01 window MAE, then commit plots.
2. **Refresh the online dashboard** (artifact URL in Session notes): upload that CSV as the new `windows` file, replace `e1e2` and `status` data, and patch the hard-coded headline in `files/index.html` ("TS-CAN trained on PURE has the lowest heart-rate error ... CHROM") because CHROM and TS-CAN now tie. Use the ArtifactData tool; read the artifact type's instructions first.
3. **SCAMPS puzzle - IN PROGRESS (22 Oct).** Finding so far, from subject 1 only: the cause is our SCORING, not input preparation. Normalising per window vs whole video and chunk 180 vs 300 give identical predictions (`iso2` test); the toolbox's scorer (cumsum -> smoothness detrend lambda=100 -> order-1 band-pass 0.6-3.3 Hz -> periodogram) gives SCAMPS 7.2 vs our 11.0 BPM on the same predictions, and a few of our windows lock onto 45-57 BPM (slow drift leaking into the low band). Caveat: the toolbox's FFT bins are 3.5 BPM wide, so its errors look flatteringly small (PURE shows 0.00). Added `pulsecam.signal.smooth_detrend` (identical to the toolbox's `_detrend`, tested) and `TSCANPredictor.predict(detrend=False)` (opt-in, default unchanged so the E1/E2 table still matches the code) and `eval/scamps_check.py`. **Next:** the Chrome connection to Claude was down, so this was NOT yet run on all 42 subjects. On Colab (Drive mount, then terminal): paste `pulsecam/signal.py`, `pulsecam/tscan.py`, `eval/scamps_check.py` (heredoc / small python patch; compare md5), then `python eval/scamps_check.py --cache /content/drive/MyDrive/PulseCam/ubfc_cache` (cache has all 42 subjects, GPU, minutes). It prints old vs new vs toolbox-scorer MAE for PURE and SCAMPS and a `RESCORED_CSV` block. Report ALL scorers side by side. Only if `detrend=True` clearly helps on all 42 (and does not hurt PURE), flip the default, re-score TS-CAN rows of `E1_E2_ubfc.csv`, and update report/CLAUDE.md; never present a scorer chosen on the test data as if it were pre-registered.
4. **Verify `OwnLoader` inside the toolbox** on Colab (the Windows venv lacks h5py and other toolbox deps).
5. **Final report + slides** from `docs/report_draft.md` and `docs/slides_outline.md`: fill the TODO sections, answer RQ1-RQ2 from E1/E2, mark RQ3-RQ4 as not done.
6. **Only when the user says volunteer data exists:** follow `docs/recording_protocol.md`, `eval/predict_own.py`, E3/E4/E5, fine-tune on own recordings (`PULSECAM_INIT`, person-wise split), and install `models/best_tscan.pth` only if it beats the released weights on held-out people.

**Working rules learned the hard way:**
- Use `.venv\Scripts\python`; run `.venv\Scripts\python -m pytest tests -q` before committing. Commit messages end with the attribution line from the session's system reminder.
- Colab: the user must click the Drive permission popup. Click inside the terminal pane before every `type` (otherwise keystrokes hit the notebook and insert cells). Use `nohup` for long jobs, and a background `sleep N` Bash command to wait. Long `browser_batch` wait lists time out. Details in Gotchas and in memory `pulsecam-colab-gotchas`.
- Do not start a localhost receiver to move data out of the browser (denied by the safety check); results go to Drive and the user downloads them.
- If the user asks for a train/fine-tune, keep the person-wise split and never claim an improvement that is within noise.

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
- **E1/E2 done on all 42 UBFC subjects, 21 Oct 2026** (Colab, 490 ten-second windows per method, scored against the HR of the ground-truth PPG; `eval/ubfc_eval.py`; table `results/tables/E1_E2_ubfc.csv`): window MAE GREEN 19.38, ICA 6.73, POS 4.50, **CHROM 2.00, TS-CAN (PURE weights) 2.01**, TS-CAN (SCAMPS weights) 14.05 BPM. Per video (median): TS-CAN-PURE 0.67, CHROM 0.72, POS 2.57, ICA 3.65, SCAMPS 10.27, GREEN 16.77. **TS-CAN ties CHROM, it does not beat it** - the team target 'beat every classical baseline' is NOT met; the first 10-subject run (0.70 vs 1.08, kept in `*_10subjects.csv`) was a lucky sample. Caveats: deep models are the toolbox's *released* PURE/SCAMPS checkpoints; the SCAMPS result looks too poor (toolbox paper reports a few BPM) - possibly a preprocessing mismatch in our face crop, unverified. The BA plots in `results/plots/` are still from the 10-subject run; the full 42-subject window data is `MyDrive/PulseCam/ubfc42_partial.csv` (same format as `E1_E2_ubfc_windows.csv`) - download it and run `python eval/ubfc_eval.py --from-csv <it>` to refresh tables + plots.
- **UBFC fine-tune, 21 Oct 2026 (`eval/finetune_tscan.py prep|train`)**: from the released PURE weights, person-wise split 30 train / 6 valid / 6 test (`results/tables/finetune_split.csv`; test = subjects 1, 11, 18, 27, 36, 43). lr 5e-5 x 6 epochs overfits (valid 1.58 -> 2.04-2.97 BPM, released weights kept); lr 5e-6 x 3 epochs: test 1.58 vs 1.60 BPM per window (released) = no meaningful gain, so **no `models/best_tscan.pth` was installed** and the demo still uses `PURE_TSCAN.pth`. Weights are on Drive (`finetuned_tscan_ubfc_lr5e-6.pth`). Crops cache `MyDrive/PulseCam/ubfc_cache` (1.2 GB, 42 npz). A batch of 4 chunks silently dies on a CPU laptop (memory) - default batch is 2; run training on the Colab GPU.
- **Toolbox `main.py` on Colab gave bogus E1 numbers** (PhysNet/TS-CAN "MAE 0.0, Pearson 1.0" from 2 scored samples; its cached preprocessing/file list was inconsistent), so those logs were discarded; `ubfc_eval.py` replaces them. Classical toolbox run also crashed on `np.mat` in `POS_WANG.py` (now patched by `install.py`).
- Colab: all four OpenCV wheels clash – after `pip install mediapipe` run `pip uninstall -y opencv-contrib-python opencv-python opencv-python-headless && pip install opencv-python-headless==4.10.0.84` or Python segfaults. Drive mount needs a manual OAuth click each new session. Browser downloads are blocked in the automation session; results were read back through Drive's text preview.
- **Re-encoding kills the pulse**: UBFC subject1 re-saved as default mp4 → POS read 62 BPM (true 107); original file → 107.2. Never compress own recordings.
- Verified 12 Oct 2026: `predict_own.py` + `report.py` run end to end on a UBFC clip renamed `P99_still.mp4` (pred 107.2 vs ref 106.7); Streamlit server starts and `/_stcore/health` = ok; `cv_app.py --save` renders the annotated backup video. **Still unverified:** live webcam end-to-end, `OwnLoader` inside the toolbox. **Not done (need people/hardware):** own recordings, consent forms, E3/E4/E5, fine-tune, `models/best_tscan.pth` (the best model is the released `PURE_TSCAN.pth`, which the demo already uses by default).
- UBFC videos: only subject1's `vid.avi` is local (gdown fails: Drive quota). On Colab, `MyDrive/DATASET_2` (a Drive shortcut) mounted via `drive.mount` holds all 42; `ubfc_eval.py --root /content/drive/MyDrive/DATASET_2 --partial <csv on Drive>` reads it directly (~2 min per subject on CPU, saves after every subject and resumes). Ground-truth files for all 42 are in `data/ubfc/DATASET_2/`.
- Colab recipe used 21 Oct: the Drive `PulseCam_code.zip` is stale (no `ubfc_eval.py`, no toolbox) - paste the script via a terminal heredoc (the Windows copy has CRLF line endings, so strip the carriage returns before comparing md5 sums), `git clone` the toolbox for `final_model_release`, `sed -i 's/np\.mat(/np.asmatrix(/g'` over the toolbox (else ICA fails), run with `nohup`. Click inside the Colab terminal before typing or the keys go to the notebook (single letters are shortcuts there).

## Session notes – updated 21 Oct 2026 (resume here)
**State:** E1/E2 done on all 42 UBFC subjects, UBFC fine-tune done (no gain). Committed and pushed to the private GitHub repo. Two dashboards exist:
- Online results dashboard (Claude artifact, private until shared): https://claude.ai/artifact/GE52vCEUQkPr5pPVXBbWmv – reads 3 CSVs (E1_E2_ubfc, E1_E2_ubfc_windows, status) stored inside it; page opened fine in Chrome (headline numbers there are the OLD 10-subject run until its CSVs are replaced). To change it: edit its store with the ArtifactData tool (`files/index.html`, `datasets/*`), upload a new CSV for new numbers; status rows are `status.csv` (item,state,note).
- Local tabbed Streamlit dashboard `app/app.py` (Live / Results / Status). Headless test passes for Results + Status tabs (WebRTC widget stubbed).

**Dashboard load issue - fixed 21 Oct 2026.** The local app showed a blank page for ~13 s, then only the title, because scipy/torch/MediaPipe imports and model loading ran before the tabs. `app/app.py` now draws the title and tabs and fills Results/Status first, and imports/loads the heavy parts last (Live tab fills in ~15-20 s). Checked in Chrome: Results opens in ~3 s, Live tab shows the START button, no traceback; 4 tests pass. The live webcam stream itself is still untested.

**Still to do (needs people/hardware):** volunteer recordings + signed consents + reference HR device → E3/E4/E5, own-recordings fine-tune (the only way to beat the released model), `models/best_tscan.pth`; live webcam test of the demo; 2–3 min backup demo video; show the consent form to the guide. **Doable without people:** refresh BA plots from `ubfc42_partial.csv`; check why the SCAMPS-trained TS-CAN scores 14 BPM per window; verify `OwnLoader` inside the toolbox on Colab; final report + slides from `docs/`.
**Colab:** notebook https://colab.research.google.com/drive/1HQ4nJm8BuggK9Sc5iyE2493Xv_C9y1Pa (Drive folder `MyDrive/PulseCam`, results zip + CSV copies in it). The runtime may be gone; rerun cells 1–3, then the E1/E2 cell (it installs mediapipe and fixes the OpenCV wheels itself).
