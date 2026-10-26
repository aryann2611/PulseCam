# Viva slides outline (≈12 slides; everyone presents)
1. Title, team, guide – "research prototype, not a medical device"
2. Problem: contact sensors; rPPG gap for Indian skin tones
3. How rPPG works (heartbeat → colour change → FFT, HR = 60·f)
4. Objectives and RQ1–RQ4
5. Pipeline diagram (video → face/ROI → model → bandpass → FFT)
6. Models: POS baseline, TS-CAN; pseudo-label fine-tuning
7. Data: UBFC-rPPG, PURE/SCAMPS, own recordings (no faces shown without written permission)
8. E1 + E2 results (all 42 UBFC subjects, released weights): TS-CAN (PURE) 1.80 BPM window MAE edges out CHROM 2.00, better than POS 4.50, ICA 6.73, GREEN 19.38; TS-CAN (SCAMPS) 12.83. Honest point: the deep model only narrowly beats the best classical method; per-video they tie exactly (0.72 both). SCAMPS gap checked 5 ways (`eval/scamps_check.py`) and is not a scoring artifact. UBFC fine-tune: no gain (1.60 vs 1.58 BPM on 6 unseen people) (`results/tables/E1_E2_ubfc.csv`, `finetune_ubfc_test.csv`)
9. E3 robustness + E4 skin-tone MAE before/after fine-tune – **not done**: no volunteer recordings, consent forms or reference device were collected during the project; UBFC-only fine-tune attempted instead (no gain: 1.58 vs 1.60 BPM on 6 held-out people)
10. Live demo – 52 s backup render of a UBFC clip ready (`results/backup_demo_ubfc_subject1.mp4`); live webcam run and a longer volunteer backup video not yet tried by a person; no reference device available to show side by side
11. Ethics and limitations
12. Conclusions per RQ, future work, thanks
