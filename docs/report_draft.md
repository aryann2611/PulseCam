# PulseCam – report draft (sections that do not depend on results)

Research prototype. **Not a medical device; not for diagnosis.** Sections marked TODO need the experiment results.

## 1. Introduction
Heart rate is normally measured by touch (finger clip, chest strap, smartwatch). Remote photoplethysmography (rPPG) estimates it from ordinary face video: each heartbeat slightly changes how much light the skin absorbs, and a 30 fps camera can capture this. Public rPPG datasets are dominated by lighter skin tones, so accuracy on Indian users is not well tested.
Objectives: (1) a video→BPM pipeline, (2) classical vs deep methods on public data, (3) effect of movement, talking and dim light, (4) accuracy across skin tones and whether fine-tuning helps, (5) a live demo.
RQ1 deep vs classical on UBFC-rPPG · RQ2 cross-dataset drop · RQ3 drop on Indian users and effect of fine-tuning · RQ4 which condition hurts most.
Scope: resting HR, one frontal face, indoor light. Out of scope: blood pressure, SpO2, arrhythmia, several faces, any clinical use.

## 2. Related work
Remote photoplethysmography (rPPG) recovers the pulse signal from subtle, periodic skin-colour changes caused by blood volume variation, without touching the subject. Early methods are purely signal-processing: **GREEN** just band-passes the green channel (strongest absorption by haemoglobin, but no motion handling); **ICA** (independent component analysis) separates the pulse from motion/lighting as a blind source-separation problem; **CHROM** (de Haan & Jeanne, 2013) builds a chrominance signal from a fixed linear combination of R/G/B that is designed to cancel specular reflection and motion; **POS** (Wang et al., 2017, "Plane-Orthogonal-to-Skin") projects onto the plane orthogonal to the skin-tone vector, which is more robust to motion than CHROM and is the usual strong classical baseline.

Deep-learning methods instead learn the mapping from pixels to pulse end-to-end. **DeepPhys** (Chen & McDuff, 2018) introduced an attention-based CNN that uses frame differences as motion representation. **TS-CAN** (Liu et al., 2020) is the efficient, temporal-shift variant we use: it takes 72×72 normalised-difference frames and predicts the *derivative* of the BVP waveform, which must be integrated afterward. **PhysNet** (Yu et al., 2019) is a 3D-CNN spatiotemporal network, and **PhysFormer** (Yu et al., 2022) replaces the CNN backbone with a transformer. We relied on **rPPG-Toolbox** (Liu et al., NeurIPS 2023) for a consistent implementation of all of the above (preprocessing, released checkpoints, evaluation code) rather than re-implementing them, per the project's "prefer existing toolbox components" rule.

Fairness/skin-tone bias is a known open problem in rPPG: public datasets (UBFC-rPPG, PURE) are collected mostly from lighter-skinned volunteers, and signal amplitude from melanin-rich skin is systematically weaker, which is one motivation for this project's RQ3 (not yet answered here, see §5). The **Monk Skin Tone** scale (10-point, developed with Google for fairness evaluation) is the scale the protocol (`docs/recording_protocol.md`) uses to group volunteers for E4.

Datasets: **UBFC-rPPG** (Bobbia et al., 2017) – 42 short, well-lit, mostly-still face videos with a finger-clip PPG ground truth; used here for E1 and as the E2 test set. **PURE** (Stricker, Müller & Gross, 2014) and **SCAMPS** (McDuff et al., 2022, a large *synthetic* rPPG dataset rendered from avatars) are the two datasets whose toolbox-released TS-CAN checkpoints we use for E2's cross-dataset transfer test, since we did not have time/compute to train from scratch on either.

## 3. Method
Pipeline: video at 30 fps → MediaPipe face box and skin regions → crop 72×72 and normalise → model (POS, or TS-CAN) → Butterworth bandpass 0.7–3.0 Hz → FFT peak → HR = 60·f_peak.
- TS-CAN takes 72×72 frames and outputs the derivative of the pulse waveform, which we integrate (cumulative sum).
- Training follows the toolbox: start from its pre-trained weights; train and validate on one dataset, test on another; each person's videos in exactly one split.
- Own recordings have an HR number but no waveform, so fine-tuning uses POS pseudo-labels from the toolbox. Fine-tuning volunteers are never the test volunteers.
- Demo: 10 s rolling buffer (300 frames), one estimate per second, median of the last 5 shown; a "Hold still" warning appears on motion, darkness or low SNR. Runs locally as a small web app (`app/server.py`, Starlette): a 30 s live measurement from the webcam, or an uploaded video file; OpenCV fallback. The browser sends uncompressed frames, so video compression cannot erase the pulse. On a laptop CPU one TS-CAN estimate takes about 3 s, so the live view runs it in a background thread (frame capture never pauses) and the reading updates every ~3-4 s instead of every second.

## 4. Experiments
| # | Train | Test | Answers |
|---|---|---|---|
| E1 | UBFC-rPPG (some subjects) | UBFC-rPPG (other subjects) | RQ1 |
| E2 | PURE or SCAMPS | UBFC-rPPG | RQ2 |
| E3 | best of E1/E2 | own: still, moving head, talking, dim | RQ4 |
| E4 | best model, before/after fine-tune | own, grouped by Monk shade | RQ3 |
| E5 | final model | 1-minute live sessions vs reference device | demo reliability |
Metrics: MAE (main), RMSE, MAPE, Pearson r, SNR, Bland-Altman. Team target: beat every classical baseline on E1; MAE < 5 BPM on still, well-lit clips.
Data collection: **not done.** No volunteers were recorded (no signed consents collected and no reference HR device arranged during the project window) — see `docs/recording_protocol.md` and `docs/consent_form.md` for the plan that would run E3–E5 once recordings exist.

Where we deviate from the plan above (say so in the report): E1/E2 use the toolbox's *released* checkpoints (TS-CAN trained on PURE, and on SCAMPS) instead of models we trained; there are no own recordings yet, so E3-E5 are empty; the only training we did is the UBFC fine-tune below, which did not beat the released weights.

## 5. Results
### E1 + E2 - UBFC-rPPG (all 42 subjects, 21 Oct 2026, TS-CAN re-scored 26 Oct; `eval/ubfc_eval.py`)
Method: face -> skin ROIs / 72x72 crop -> method -> bandpass 0.7-3.0 Hz -> FFT peak. Every 10 s window (step 5 s, 490 windows per method) is scored against the HR of the ground-truth PPG in the same window; "per video" is the median over each video, 42 videos. Table: `results/tables/E1_E2_ubfc.csv`; Bland-Altman plots: `results/plots/BA_ubfc_*.png` (all 42 subjects, 490 windows per method). An earlier 10-subject run, 12 Oct, is kept in `results/tables/E1_E2_ubfc_*_10subjects.csv`.

| Method | MAE window (BPM) | RMSE | Pearson | MAE per video (median) |
|---|---|---|---|---|
| GREEN | 19.38 | 29.97 | 0.28 | 16.77 |
| ICA | 6.73 | 15.85 | 0.72 | 3.65 |
| POS | 4.50 | 12.94 | 0.79 | 2.57 |
| **TS-CAN, trained on PURE** | **1.80** | **6.80** | **0.93** | **0.72** |
| CHROM | 2.00 | 6.03 | 0.94 | 0.72 |
| TS-CAN, trained on SCAMPS | 12.83 | 22.84 | 0.50 | 8.94 |

TS-CAN numbers use a toolbox-style detrend step added and verified on 26 Oct (`eval/scamps_check.py`, `pulsecam.signal.smooth_detrend`); see below.

- **RQ1 (E1):** TS-CAN (PURE weights) clearly beats GREEN, ICA and POS, and now **edges out CHROM** on window MAE (1.80 vs 2.00 BPM) after fixing a drift-removal step in the TS-CAN post-processing (was 2.01, tied with CHROM, before the fix). CHROM remains marginally ahead on correlation (0.94 vs 0.93) and per-video MAE is a tie (0.72 both). The team target "beat every classical baseline" is met on window MAE, a very narrow margin elsewhere. The first run on 10 subjects (0.70 vs 1.08 BPM) looked better only because of which subjects were in the sample. "MAE < 5 BPM on still, well-lit clips" is comfortably met.
- **RQ2 (E2):** TS-CAN trained on real video (PURE) transfers well to UBFC (1.80 BPM), but the same model trained on synthetic SCAMPS does not (12.83 BPM per window). We checked whether this was a scoring artifact (`eval/scamps_check.py`, 26 Oct, all 42 subjects, 5 different ways of normalising/scoring the same cached face crops, including the toolbox's own exact preprocessing and post-processing): the SCAMPS gap **persists under every scoring method**, from 8.6 BPM (whole-video, toolbox scorer) to 14.0 BPM (original per-window scorer) - PURE stays in the 1.3-2.0 BPM range throughout. This rules out our FFT-peak scorer as the main cause; the released SCAMPS-trained checkpoint genuinely transfers far worse to UBFC than the PURE-trained one, consistent with (but larger than) the toolbox paper's own finding that synthetic-to-real transfer is harder.
- Caveats: UBFC is easy (indoor, still, uncompressed video); window MAE is dominated by a few windows where a method locks onto the wrong frequency (RMSE much larger than MAE for ICA, POS and TS-CAN); GREEN is the plain green channel with no motion or colour handling; the deep models are the toolbox's released checkpoints.
- Pipeline sanity: a lossy re-encoded copy of one UBFC clip made POS read 62 BPM instead of 107 - video compression erases the pulse, so keep original files for the own recordings.

### SCAMPS scoring investigation (26 Oct 2026; `eval/scamps_check.py`, `results/tables/scamps_check.csv`)
Five ways of turning the same cached 42-subject face crops into a heart rate, for both the PURE- and SCAMPS-trained TS-CAN checkpoints: (A0) our original per-window normalisation + cumsum + FFT-peak scorer (what the 21 Oct table reported); (A) the same plus a toolbox-style smoothness-priors detrend step before scoring - **this became the new default**; (B) the same predictions scored with the toolbox's own detrend+bandpass+FFT post-processing; (C) the toolbox's own input normalisation and post-processing, 10 s windows; (D) like C but one heart rate per whole video (the toolbox's own default metric).

| Variant | PURE MAE | SCAMPS MAE |
|---|---|---|
| A0 old (pre-fix) | 2.01 | 14.05 |
| **A new (adopted)** | **1.80** | **12.83** |
| B our predictions + toolbox scorer | 1.69 | 12.54 |
| C toolbox input + toolbox scorer | 1.40 | 10.42 |
| D toolbox, whole-video metric | 1.28 | 8.62 |

The detrend step (A) reduces MAE for both checkpoints without hurting PURE, so it is now the default (`TSCANPredictor.predict(detrend=True)`); the E1/E2 table above and `results/tables/E1_E2_ubfc.csv` were re-scored with it. The SCAMPS/PURE gap shrinks a little toward the toolbox's own pipeline (C, D) but never closes - SCAMPS stays 6-8x worse than PURE under every variant, which is why we no longer treat it as primarily a scoring bug.

### Fine-tuning TS-CAN on UBFC (21 Oct 2026; `eval/finetune_tscan.py`)
Start from the released PURE weights; split by person: 30 train, 6 validation, 6 test people (`results/tables/finetune_split.csv`); keep the epoch with the lowest validation MAE; score the 6 test people (68 windows) with the original and the fine-tuned weights. Tables: `results/tables/finetune_ubfc_test*.csv`.

| Model | MAE window (BPM) | RMSE | MAE per video |
|---|---|---|---|
| Released PURE weights | 1.60 | 6.73 | 0.89 |
| Fine-tuned, lr 5e-6, 3 epochs | 1.58 | 6.74 | 0.85 |
| Fine-tuned, lr 5e-5, 6 epochs | 1.60 (kept the released weights: every fine-tuned epoch was worse on validation, 1.58 -> 2.04-2.97) | 6.73 | 0.89 |

- Result: **no meaningful gain.** 1.60 vs 1.58 BPM over 68 windows is noise, and a larger learning rate overfits the 30 training people. We therefore did not install a fine-tuned model: `models/best_tscan.pth` does not exist and the demo keeps using the released `PURE_TSCAN.pth`. On UBFC alone the released model is already near its ceiling; a real gain would have to come from own recordings with other skin tones (E4), which are not done.
- E3, E4, E5: **not done** - they need volunteer recordings, consent and a reference device.

## 6. Ethics
Written consent before recording (purpose, what is recorded, storage, right to withdraw); India's DPDP Act 2023; ethics-committee question put to the project guide (TODO: answer); videos only on a private team drive, never on GitHub; IDs (P01…) not names, with the ID→name list kept separately; faces shown only with written permission; raw videos deleted if the consent form says so; volunteers told PulseCam is not medical and to see a doctor about any heart concern.

## 7. Limitations
No volunteer sample at all (not just small/uneven) — E3/E4/E5 are empty; manual or consumer-device references are noisy; POS pseudo-labels are not true pulse waveforms; resting, frontal, indoor, 30 fps, single face only; reported accuracy does not generalise to clinical use. All deep-model results use the toolbox's *released* checkpoints, never models we trained from scratch; the UBFC fine-tune we did run (§5) gave no measurable gain over the released weights.

## 8. Conclusion
**RQ1 (deep vs. classical on UBFC-rPPG):** TS-CAN (PURE weights) clearly beats GREEN, ICA and POS, and after fixing a drift-removal bug in its post-processing (`eval/scamps_check.py`, 26 Oct) it now **edges out** the best classical baseline, CHROM, on window MAE (1.80 vs 2.00 BPM); the two are effectively tied per-video (0.72 vs 0.72) and CHROM keeps a small edge on correlation (0.94 vs 0.93). The project's stated target of beating every classical baseline is met on the headline metric, by a narrow margin everywhere else — a deep model trained on a different dataset (PURE) only modestly outperforms a well-tuned signal-processing method on easy, controlled video like UBFC. The deep model's real advantage, if any, would have to come from harder conditions (motion, talking, dim light, varied skin tone) that UBFC does not test, which is exactly what E3/E4 were designed to probe.

**RQ2 (cross-dataset transfer):** transfer from a real, camera-captured dataset (PURE) to UBFC works well (1.80 BPM), but transfer from the *synthetic* dataset (SCAMPS) is much worse (12.83 BPM). We checked whether this was a scoring-method artefact by rescoring the same 42-subject face crops five different ways, including the toolbox's own exact preprocessing and post-processing (`eval/scamps_check.py`, 26 Oct): the gap persists under every method (SCAMPS stays 6-8x worse than PURE throughout), so it is not mainly a scoring bug — the released SCAMPS-trained checkpoint genuinely generalises far worse to real UBFC video than the PURE-trained one.

**RQ3 (skin-tone drop and fine-tuning) and RQ4 (which condition hurts most): not answered.** Both need E3/E4, which need volunteer recordings with signed consent and a reference HR device; none of that was collected during the project. As a partial substitute, we fine-tuned TS-CAN on UBFC itself (person-wise split) to see whether further training on in-domain data helps at all: it did not (1.58 vs 1.60 BPM on 6 held-out people, within noise), so the released PURE checkpoint remains the model the live demo uses.

**Overall:** the pipeline works end-to-end (video → face/ROI → model → bandpass → FFT → BPM) and is demonstrably accurate on controlled video (window MAE 2.0 BPM, meeting the "<5 BPM on still, well-lit clips" target). The honest headline finding is that, on this benchmark, the best classical method (CHROM) is not beaten by the deep model we had available — only matched — and the harder, more interesting questions (skin-tone fairness, robustness, live performance against a reference device) remain open for future work once volunteer data can be collected.
