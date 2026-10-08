# Final report outline (cite every dataset/model paper – see References)

1. **Introduction** – problem (contact sensors), rPPG idea, gap (lighter-skin-dominated datasets; few tests on Indian users), objectives, RQ1–RQ4, scope/out of scope, "not a medical device".
2. **Related work** – rPPG basics; classical methods (GREEN, ICA, CHROM, POS); deep models (DeepPhys, TS-CAN, PhysNet, PhysFormer); datasets; skin-tone fairness in rPPG; rPPG-Toolbox.
3. **Method** – pipeline (video → face/ROI → crop/normalise/clips → model → bandpass 0.7–3.0 Hz → FFT → HR = 60·f_peak); training setup (pre-trained start, person-level splits); pseudo-label fine-tuning; demo design (10 s buffer, 1 s update, median of 5).
4. **Experiments** – E1–E5 table (train/test sets, what each answers), metrics (MAE, RMSE, MAPE, Pearson, SNR, Bland-Altman), data collection protocol, Monk grouping.
5. **Results** – `results/tables/*.csv`, `results/plots/*.png`: E1 (vs baselines), E2 (cross-dataset drop), E3 (which condition hurts most), E4 (MAE per skin-tone band, before/after fine-tuning), E5 (live vs reference).
6. **Ethics** – consent, DPDP Act 2023, private storage, IDs not names, deletion, no diagnostic use.
7. **Limitations** – small/uneven volunteer sample, noisy manual references, POS pseudo-labels, indoor/frontal/resting only, 30 fps assumption, single face.
8. **Conclusion** – answers to RQ1–RQ4; honest if the deep model did not beat the baselines.
