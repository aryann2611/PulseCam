# PulseCam – report draft (sections that do not depend on results)

Research prototype. **Not a medical device; not for diagnosis.** Sections marked TODO need the experiment results.

## 1. Introduction
Heart rate is normally measured by touch (finger clip, chest strap, smartwatch). Remote photoplethysmography (rPPG) estimates it from ordinary face video: each heartbeat slightly changes how much light the skin absorbs, and a 30 fps camera can capture this. Public rPPG datasets are dominated by lighter skin tones, so accuracy on Indian users is not well tested.
Objectives: (1) a video→BPM pipeline, (2) classical vs deep methods on public data, (3) effect of movement, talking and dim light, (4) accuracy across skin tones and whether fine-tuning helps, (5) a live demo.
RQ1 deep vs classical on UBFC-rPPG · RQ2 cross-dataset drop · RQ3 drop on Indian users and effect of fine-tuning · RQ4 which condition hurts most.
Scope: resting HR, one frontal face, indoor light. Out of scope: blood pressure, SpO2, arrhythmia, several faces, any clinical use.

## 2. Related work – TODO
Cover: rPPG basics; GREEN, ICA, CHROM (de Haan & Jeanne, 2013), POS (Wang et al., 2017); DeepPhys (Chen & McDuff, 2018), TS-CAN (Liu et al., 2020), PhysNet (Yu et al., 2019), PhysFormer (Yu et al., 2022); rPPG-Toolbox (Liu et al., NeurIPS 2023); skin-tone fairness in rPPG; Monk Skin Tone scale. Datasets: UBFC-rPPG (Bobbia et al., 2017), PURE (Stricker, Müller & Gross, 2014), SCAMPS (McDuff et al., 2022).

## 3. Method
Pipeline: video at 30 fps → MediaPipe face box and skin regions → crop 72×72 and normalise → model (POS, or TS-CAN) → Butterworth bandpass 0.7–3.0 Hz → FFT peak → HR = 60·f_peak.
- TS-CAN takes 72×72 frames and outputs the derivative of the pulse waveform, which we integrate (cumulative sum).
- Training follows the toolbox: start from its pre-trained weights; train and validate on one dataset, test on another; each person's videos in exactly one split.
- Own recordings have an HR number but no waveform, so fine-tuning uses POS pseudo-labels from the toolbox. Fine-tuning volunteers are never the test volunteers.
- Demo: 10 s rolling buffer (300 frames), one estimate per second, median of the last 5 shown; a "Hold still" warning appears on motion, darkness or low SNR. Runs locally with Streamlit; OpenCV fallback.

## 4. Experiments
| # | Train | Test | Answers |
|---|---|---|---|
| E1 | UBFC-rPPG (some subjects) | UBFC-rPPG (other subjects) | RQ1 |
| E2 | PURE or SCAMPS | UBFC-rPPG | RQ2 |
| E3 | best of E1/E2 | own: still, moving head, talking, dim | RQ4 |
| E4 | best model, before/after fine-tune | own, grouped by Monk shade | RQ3 |
| E5 | final model | 1-minute live sessions vs reference device | demo reliability |
Metrics: MAE (main), RMSE, MAPE, Pearson r, SNR, Bland-Altman. Team target: beat every classical baseline on E1; MAE < 5 BPM on still, well-lit clips.
Data collection: TODO – N volunteers, Monk shades covered, reference device(s) used, camera settings, four 1-minute conditions (see `docs/recording_protocol.md`).

Where we deviate from the plan above (say so in the report): E1/E2 use the toolbox's *released* checkpoints (TS-CAN trained on PURE, and on SCAMPS) instead of models we trained; there are no own recordings yet, so E3-E5 are empty.

## 5. Results
### E1 + E2 - UBFC-rPPG (10 of 42 subjects, 12 Oct 2026; `eval/ubfc_eval.py`)
Method: face -> skin ROIs / 72x72 crop -> method -> bandpass 0.7-3.0 Hz -> FFT peak. Every 10 s window (step 5 s, 109 windows per method) is scored against the HR of the ground-truth PPG in the same window. Subjects 1, 3, 4, 5, 8, 9, 10, 11, 12, 13. Table: `results/tables/E1_E2_ubfc.csv`; plots: `results/plots/E1_E2_mae.png`, `BA_ubfc_*.png`.

| Method | MAE window (BPM) | RMSE | Pearson | MAE per video (median) |
|---|---|---|---|---|
| GREEN | 24.05 | 34.70 | 0.07 | 17.76 |
| ICA | 4.16 | 12.68 | 0.75 | 1.73 |
| POS | 2.84 | 10.55 | 0.82 | 0.87 |
| CHROM | 1.08 | 3.56 | 0.98 | 0.35 |
| **TS-CAN, trained on PURE** | **0.70** | 1.73 | 1.00 | 0.41 |
| TS-CAN, trained on SCAMPS | 21.99 | 32.95 | 0.14 | 17.17 |

- **RQ1 (E1):** the deep model (TS-CAN) has the lowest window MAE and beats every classical baseline (GREEN, ICA, POS, CHROM). CHROM is close (1.08 vs 0.70) and is slightly better per video (0.35 vs 0.41), so on these ten clean, still, well-lit subjects the margin is small. The team target (beat every baseline; MAE < 5 BPM on still, well-lit clips) is met on this subset.
- **RQ2 (E2):** TS-CAN trained on real video (PURE) transfers to UBFC almost perfectly (0.70 BPM), but the same model trained on synthetic SCAMPS fails (21.99 BPM). Part of this gap may be our preprocessing (the toolbox paper reports a much smaller SCAMPS gap) - check before claiming it.
- Caveats: ten subjects; UBFC is easy (indoor, still, uncompressed video); window MAE is dominated by a few windows where a method locks onto the wrong frequency (RMSE much larger than MAE for ICA and POS); GREEN is the plain green channel with no motion or colour handling.
- Not trained by us: no own training run was done, so "train on UBFC subjects" in E1 was not carried out.
- Pipeline sanity: a lossy re-encoded copy of one UBFC clip made POS read 62 BPM instead of 107 - video compression erases the pulse, so keep original files for the own recordings.
- E3, E4, E5: **not done** - they need volunteer recordings, consent and a reference device.

## 6. Ethics
Written consent before recording (purpose, what is recorded, storage, right to withdraw); India's DPDP Act 2023; ethics-committee question put to the project guide (TODO: answer); videos only on a private team drive, never on GitHub; IDs (P01…) not names, with the ID→name list kept separately; faces shown only with written permission; raw videos deleted if the consent form says so; volunteers told PulseCam is not medical and to see a doctor about any heart concern.

## 7. Limitations
Small, uneven volunteer sample; manual or consumer-device references are noisy; POS pseudo-labels are not true pulse waveforms; resting, frontal, indoor, 30 fps, single face only; reported accuracy does not generalise to clinical use.

## 8. Conclusion – TODO
Answer RQ1–RQ4 from the results. If the deep model does not beat the baselines, say so and explain why; that is still a valid finding.
