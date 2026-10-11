"""Install the own-recordings loader into a cloned rPPG-Toolbox (safe to run twice).

    python toolbox_patch/install.py [path/to/rPPG-Toolbox]

Adds: dataset name "PulseCam" in main.py (train/valid/test), OwnLoader.py, and an optional
PULSECAM_INIT env var so TS-CAN fine-tuning starts from pre-trained weights:
    PULSECAM_INIT=final_model_release/PURE_TSCAN.pth python main.py --config_file ../configs/train/PulseCam_finetune_TSCAN.yaml
"""
import os
import re
import shutil
import sys

tb = sys.argv[1] if len(sys.argv) > 1 else "rPPG-Toolbox"
here = os.path.dirname(os.path.abspath(__file__))


def patch(rel, fn):
    p = os.path.join(tb, rel)
    s = open(p, encoding="utf-8").read()
    n = fn(s)
    if n != s:
        open(p, "w", encoding="utf-8").write(n)
        print("patched", rel)


shutil.copy(os.path.join(here, "OwnLoader.py"), os.path.join(tb, "dataset", "data_loader", "OwnLoader.py"))

patch("dataset/data_loader/__init__.py",
      lambda s: s if "OwnLoader" in s else s.rstrip("\n") + "\nimport dataset.data_loader.OwnLoader\n")


def main_py(s):
    for split in ("TRAIN", "VALID", "TEST"):
        if f'config.{split}.DATA.DATASET == "PulseCam"' in s:
            continue
        s = re.sub(rf'^( *)if config\.{split}\.DATA\.DATASET == "UBFC-rPPG":\n( *)(\w+) = data_loader\.UBFCrPPGLoader\.UBFCrPPGLoader\n',
                   lambda m: m.group(0) + f'{m.group(1)}elif config.{split}.DATA.DATASET == "PulseCam":\n'
                                          f'{m.group(2)}{m.group(3)} = data_loader.OwnLoader.OwnLoader\n',
                   s, count=1, flags=re.M)
    return s


patch("main.py", main_py)

line = "            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))\n"
hook = ('            if os.environ.get("PULSECAM_INIT"):  # fine-tune from pre-trained weights\n'
        '                self.model.load_state_dict(torch.load(os.environ["PULSECAM_INIT"], map_location=self.device))\n')
patch("neural_methods/trainer/TscanTrainer.py", lambda s: s if "PULSECAM_INIT" in s else s.replace(line, line + hook, 1))

# Colab has NumPy 2 / new SciPy: np.mat is gone, and gaussian_kde raises on tiny/degenerate samples (e.g. 2 subjects)
for f in ("dataset/data_loader/BaseLoader.py", "dataset/data_loader/BP4DPlusBigSmallLoader.py",
          "unsupervised_methods/methods/ICA_POH.py", "unsupervised_methods/methods/POS_WANG.py"):
    patch(f, lambda s: s.replace("np.mat(", "np.asmatrix("))
patch("evaluation/BlandAltmanPy.py", lambda s: s if "_safe_kde" in s else s.replace("gaussian_kde(xy)(xy)", "_safe_kde(xy)").replace(
    "from scipy.stats import gaussian_kde\n",
    "from scipy.stats import gaussian_kde\n\n\ndef _safe_kde(xy):\n    try:\n        return gaussian_kde(xy)(xy)\n"
    "    except Exception:  # too few / identical points: plot without density colour\n        return np.ones(xy.shape[1])\n", 1))
