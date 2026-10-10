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

### Pipeline check (not a result)
On the first 43 s of one UBFC-rPPG subject (34 overlapping 10 s windows), POS gave MAE 2.6 BPM and TS-CAN (PURE weights) 3.4 BPM against the oximeter HR. One subject only; this shows the code runs on a real face and nothing more. TODO: the partial video used here was deleted – re-run on the full UBFC videos before quoting any number, or drop this paragraph.

## 5. Results – TODO (E1–E5 from `results/tables/`, `results/plots/`)

## 6. Ethics
Written consent before recording (purpose, what is recorded, storage, right to withdraw); India's DPDP Act 2023; ethics-committee question put to the project guide (TODO: answer); videos only on a private team drive, never on GitHub; IDs (P01…) not names, with the ID→name list kept separately; faces shown only with written permission; raw videos deleted if the consent form says so; volunteers told PulseCam is not medical and to see a doctor about any heart concern.

## 7. Limitations
Small, uneven volunteer sample; manual or consumer-device references are noisy; POS pseudo-labels are not true pulse waveforms; resting, frontal, indoor, 30 fps, single face only; reported accuracy does not generalise to clinical use.

## 8. Conclusion – TODO
Answer RQ1–RQ4 from the results. If the deep model does not beat the baselines, say so and explain why; that is still a valid finding.
