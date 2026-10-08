"""Turn rPPG-Toolbox console logs (E1, E2) into one table: results/tables/E1_E2_toolbox.csv

Save each run's output with tee, using the experiment in the name:
    python main.py --config_file <yaml> 2>&1 | tee results/logs/E1_physnet.log
Usage: python eval/parse_logs.py
"""
import glob
import os
import re

import pandas as pd

KEYS = {"MAE": "MAE", "RMSE": "RMSE", "MAPE": "MAPE", "Pearson": "Pearson", "SNR": "SNR"}
rows = []
for log in sorted(glob.glob("results/logs/*.log")):
    cur, row = os.path.basename(log)[:-4], {}
    for line in open(log, encoding="utf-8", errors="ignore"):
        m = re.match(r"Used Unsupervised Method: (\w+)", line)
        if m:  # unsupervised logs hold several methods; start a new row for each
            cur, row = f"{os.path.basename(log)[:-4]}:{m.group(1)}", {}
        m = re.match(r"FFT (\w+) \(FFT Label\): ([-\d.eE+]+) \+/- ([-\d.eE+]+)", line)
        if m and m.group(1) in KEYS:
            if not row:
                row = {"run": cur}
                rows.append(row)
            row[m.group(1)] = round(float(m.group(2)), 3)
out = pd.DataFrame(rows)
os.makedirs("results/tables", exist_ok=True)
out.to_csv("results/tables/E1_E2_toolbox.csv", index=False)
print(out.to_string(index=False) if len(out) else "no logs found in results/logs/*.log")
