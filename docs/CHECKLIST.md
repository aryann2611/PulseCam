# Week-1 must-dos
- [ ] Email PURE authors for dataset access
- [ ] Download UBFC-rPPG and SCAMPS into `data/` (UBFC: all 42 videos used via Drive mount on Colab, 1 local; PURE/SCAMPS data not downloaded - released weights used instead)
- [x] Install rPPG-Toolbox on Colab; run one pre-trained model (done 12 Oct; table in `results/tables/E1_E2_ubfc.csv`)
- [ ] Arrange a free heart-rate reference (oximeter / smartwatch)
- [ ] Draft consent form and show it to the project guide (`docs/consent_form.md`)

# Final deliverables (all nine due at submission)
- [x] Code repository on GitHub (private until submission) with README on setup + running each experiment
- [ ] Trained weights for the best baseline-beating model (`models/best_tscan.pth`, shared via Drive) - best model is still the toolbox's released `PURE_TSCAN.pth` (used by default); we fine-tuned on UBFC (person-wise split, 21 Oct) and it did NOT beat the released weights (1.58 vs 1.60 BPM on 6 unseen people), so no `best_tscan.pth` yet - needs own recordings
- [ ] Own-recordings dataset + `data/own/metadata.csv` (stored privately)
- [x] Results tables E1–E2 and Bland-Altman plots (`results/`) - **E1 and E2 done on all 42 UBFC subjects**, BA plots refreshed to the full 42-subject data (26 Oct); E3-E5 need own recordings (not done)
- [x] Live demo that runs locally from one command: `python app/server.py` (or `run.bat`) - live path tested with UBFC frames over the WebSocket and in Chrome with a simulated camera (72 BPM in, 72 BPM out); upload tested with UBFC subject1 (107.2 BPM, true 106.7); a real webcam run still has to be tried by a person
- [ ] Backup demo video, 2–3 min screen recording (`python app/cv_app.py --source backup.mp4` also works) - a 52 s annotated render of UBFC subject1 exists (`results/backup_demo_ubfc_subject1.mp4`, gitignored); a longer one with a consenting volunteer is not done
- [x] Final report (`docs/report_draft.md`) - all sections filled; RQ1-RQ2 answered, RQ3-RQ4 marked not done
- [x] Presentation slides for the viva (`docs/slides_outline.md`) - filled, same honest status
- [ ] Signed consent forms, stored privately

# Demo day
- [ ] Videos only on private team drive; never on GitHub; ID→name list separate
- [ ] Show a face only with written permission; delete raw videos if the consent form says so
- [ ] Say PulseCam is not medical; run locally (not cloud); daylight, plain background, stay still 15 s
- [ ] Pre-recorded video ready; reference device reading shown next to PulseCam's
