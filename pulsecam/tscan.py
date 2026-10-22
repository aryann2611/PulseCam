"""TS-CAN inference using rPPG-Toolbox's model class and pre-trained / own weights."""
import importlib.util
import os

import numpy as np

from pulsecam.signal import smooth_detrend

ROOT = os.path.join(os.path.dirname(__file__), "..")
TOOLBOX = os.path.join(ROOT, "rPPG-Toolbox")


def default_weights():
    own = os.path.join(ROOT, "models", "best_tscan.pth")  # your trained / fine-tuned model wins
    return own if os.path.exists(own) else os.path.join(TOOLBOX, "final_model_release", "PURE_TSCAN.pth")


class TSCANPredictor:
    """predict(face): face (N, 72, 72, 3) uint8 RGB, N a multiple of 10 -> BVP (N,)."""

    def __init__(self, weights=None, depth=10, size=72):
        import torch
        spec = importlib.util.spec_from_file_location(
            "ts_can", os.path.join(TOOLBOX, "neural_methods", "model", "TS_CAN.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.torch, self.depth = torch, depth
        self.net = mod.TSCAN(frame_depth=depth, img_size=size)
        sd = torch.load(weights or default_weights(), map_location="cpu")
        self.net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        self.net.eval()

    @staticmethod
    def _diff_norm(x):
        d = (x[1:] - x[:-1]) / (x[1:] + x[:-1] + 1e-7)
        d = np.concatenate([d / d.std(), np.zeros_like(x[:1])])  # same layout as the toolbox
        return np.nan_to_num(d)

    def predict(self, face, detrend=False):
        x = face.astype(np.float32)
        x = x[: len(x) // self.depth * self.depth]
        std = np.nan_to_num((x - x.mean()) / (x.std() + 1e-7))
        inp = np.concatenate([self._diff_norm(x), std], -1).transpose(0, 3, 1, 2)
        with self.torch.no_grad():
            out = self.net(self.torch.from_numpy(np.ascontiguousarray(inp)))
        bvp = np.cumsum(out.numpy().ravel())  # model predicts the first difference of the BVP
        return smooth_detrend(bvp) if detrend else bvp  # detrend=True: toolbox-style drift removal, still being validated on all 42 UBFC subjects (eval/scamps_check.py); default stays the behaviour behind the E1/E2 table
