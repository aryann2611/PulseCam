# PulseCam
Contactless heart rate from webcam face video (rPPG). Research prototype, **not a medical device**.

## Structure
```
PulseCam/
├── pulsecam/        # signal (bandpass/FFT/POS), metrics, face (MediaPipe), tscan, pipeline (HRTracker)
├── app/             # app.py = Streamlit demo, cv_app.py = OpenCV fallback / backup-video player
├── eval/            # predict_own.py, report.py (E3/E4/E5), parse_logs.py (E1/E2)
├── toolbox_patch/   # OwnLoader + install.py for rPPG-Toolbox
├── configs/train/   # PulseCam_finetune_TSCAN.yaml
├── notebooks/       # pulsecam_colab.ipynb (training runbook)
├── tests/           # python -m pytest tests
├── docs/            # consent_form, recording_protocol, report_outline, CHECKLIST
├── data/            # PRIVATE, gitignored: ubfc, pure, scamps, own (videos, consent, metadata.csv)
├── models/          # best_tscan.pth (gitignored), face detector
├── results/         # tables/, plots/, logs/
└── rPPG-Toolbox/    # cloned (gitignored)
```

## Commands

### Setup (Colab / Linux / WSL — toolbox does not run natively on Windows)
```bash
git clone https://github.com/ubicomplab/rPPG-Toolbox.git
cd rPPG-Toolbox && bash setup.sh conda     # or: bash setup.sh uv
conda activate rppg-toolbox
```

### Demo env (local laptop, Windows OK)
```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt      # torch is optional: without it the demo uses POS only
python toolbox_patch/install.py rPPG-Toolbox
streamlit run app/app.py             # fallback: python app/cv_app.py  (or --source backup.mp4)
python -m pytest tests -q
```

### E1 — classical baselines + deep models on UBFC-rPPG (set DATA_PATH in each yaml first)
```bash
cd rPPG-Toolbox
# classical: GREEN/ICA/CHROM/POS (edit METHOD list in the yaml)
python main.py --config_file configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml
# pre-trained deep models, test on UBFC (infer_configs)
python main.py --config_file configs/infer_configs/PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
python main.py --config_file configs/infer_configs/PURE_UBFC-rPPG_TSCAN_BASIC.yaml
python main.py --config_file configs/infer_configs/PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
# train on UBFC subjects: copy a train_config, set TRAIN/VALID/TEST DATASET: UBFC-rPPG with BEGIN/END subject splits
cp configs/train_configs/PURE_UBFC-rPPG_TSCAN_BASIC.yaml ../configs/train/UBFC_UBFC_TSCAN.yaml
```

### E2 — cross-dataset (train PURE or SCAMPS, test UBFC-rPPG)
```bash
python main.py --config_file configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
python main.py --config_file configs/train_configs/SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml   # fallback
```

### Fine-tune on own recordings (custom loader is in `toolbox_patch/`, uses POS pseudo-labels)
```bash
python toolbox_patch/install.py rPPG-Toolbox     # once
cd rPPG-Toolbox
PULSECAM_INIT=final_model_release/PURE_TSCAN.pth python main.py --config_file ../configs/train/PulseCam_finetune_TSCAN.yaml
```
Checkpoints (Colab): `runs/` is symlinked to Google Drive in the notebook so disconnects lose nothing.
Tee each toolbox run to `results/logs/E1_<name>.log`, then `python eval/parse_logs.py`.

### Own recordings -> E3 / E4 / E5
```bash
python eval/predict_own.py --method pos                                  # classical
python eval/predict_own.py --method tscan --stage pretrained --only P17,P18,P19,P20   # held-out people
# fine-tune on the other people (Colab): see notebooks/pulsecam_colab.ipynb, then
python eval/predict_own.py --method tscan --weights models/best_tscan.pth --stage finetuned --only P17,P18,P19,P20
python eval/report.py                    # tables -> results/tables, Bland-Altman -> results/plots
python eval/parse_logs.py                # E1/E2 table from toolbox logs in results/logs/*.log
```
See `docs/recording_protocol.md`, `docs/consent_form.md`, `docs/CHECKLIST.md`.
