# Week-1 must-dos
- [ ] Email PURE authors for dataset access
- [ ] Download UBFC-rPPG and SCAMPS into `data/` (UBFC: 10 of 42 videos used via Drive mount on Colab, 1 local; PURE/SCAMPS data not downloaded - released weights used instead)
- [x] Install rPPG-Toolbox on Colab; run one pre-trained model (done 12 Oct; table in `results/tables/E1_E2_ubfc.csv`)
- [ ] Arrange a free heart-rate reference (oximeter / smartwatch)
- [ ] Draft consent form and show it to the project guide (`docs/consent_form.md`)

# Final deliverables (all nine due at submission)
- [ ] Code repository on GitHub (private until submission) with README on setup + running each experiment
- [ ] Trained weights for the best baseline-beating model (`models/best_tscan.pth`, shared via Drive) - best model so far is the toolbox's released `PURE_TSCAN.pth` (used by default); nothing trained or fine-tuned by us yet
- [ ] Own-recordings dataset + `data/own/metadata.csv` (stored privately)
- [ ] Results tables E1–E5 and Bland-Altman plots (`results/`) - **E1 and E2 done**; E3-E5 need own recordings
- [ ] Live demo that runs locally from one command: `streamlit run app/app.py` - server starts and passes the health check; the webcam run still has to be tried by a person
- [ ] Backup demo video, 2–3 min screen recording (`python app/cv_app.py --source backup.mp4` also works) - a 52 s annotated render of UBFC subject1 exists (`results/backup_demo_ubfc_subject1.mp4`, gitignored); record a longer one with a consenting volunteer
- [ ] Final report (`docs/report_outline.md`)
- [ ] Presentation slides for the viva
- [ ] Signed consent forms, stored privately

# Demo day
- [ ] Videos only on private team drive; never on GitHub; ID→name list separate
- [ ] Show a face only with written permission; delete raw videos if the consent form says so
- [ ] Say PulseCam is not medical; run locally (not cloud); daylight, plain background, stay still 15 s
- [ ] Pre-recorded video ready; reference device reading shown next to PulseCam's
