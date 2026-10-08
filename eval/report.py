"""Build the E3 / E4 / E5 tables and Bland-Altman plots from results/own_predictions.csv.

E3: MAE per condition (still / move / talk / dim)      -> results/tables/E3_conditions.csv
E4: MAE per Monk skin-tone band, per stage             -> results/tables/E4_skintone.csv  (+ per-shade counts)
E5: live demo log vs reference device                  -> results/tables/E5_live.csv
Usage: python eval/report.py [--e4-condition still]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pulsecam.metrics import bland_altman, summary

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="results/own_predictions.csv")
ap.add_argument("--live", default="results/e5_live_log.csv")
ap.add_argument("--e4-condition", default=None, help="restrict E4 to one condition, e.g. still")
a = ap.parse_args()
T, P = "results/tables", "results/plots"
os.makedirs(T, exist_ok=True)
os.makedirs(P, exist_ok=True)


def table(df, by):
    return pd.DataFrame([dict(zip(by, k if isinstance(k, tuple) else (k,)), **summary(g.pred_hr, g.ref_hr))
                         for k, g in df.groupby(by)]).round(2)


def band(m):  # Monk shades 1-10 -> three bands so each group has enough people
    return None if pd.isna(m) else "light (1-3)" if int(m) <= 3 else "medium (4-6)" if int(m) <= 6 else "dark (7-10)"


if os.path.exists(a.pred):
    d = pd.read_csv(a.pred)
    e3 = table(d, ["method", "stage", "condition"])
    e3.to_csv(f"{T}/E3_conditions.csv", index=False)
    print("E3 - MAE per condition (the highest MAE is the condition that hurts most)\n", e3, "\n")

    d4 = d if not a.e4_condition else d[d.condition == a.e4_condition]
    d4 = d4.assign(skin_band=d4.monk.map(band)).dropna(subset=["skin_band"])
    if len(d4):
        e4 = table(d4, ["method", "stage", "skin_band"])
        e4["n_people"] = [d4[(d4.method == r.method) & (d4.stage == r.stage) & (d4.skin_band == r.skin_band)].id.nunique()
                          for r in e4.itertuples()]
        e4.to_csv(f"{T}/E4_skintone.csv", index=False)
        print("E4 - MAE per skin-tone band (compare stages: pretrained vs finetuned)\n", e4, "\n")
        print("People per Monk shade:\n", d4.drop_duplicates("id").monk.value_counts().sort_index().to_string(), "\n")

    for (m, s), g in d.groupby(["method", "stage"]):
        fig, ax = plt.subplots(figsize=(5, 4))
        bland_altman(ax, g.pred_hr, g.ref_hr, f"{m}/{s}")
        fig.tight_layout()
        fig.savefig(f"{P}/BA_{m}_{s}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    e3.pivot_table(index="condition", columns=["method", "stage"], values="MAE").plot.bar(ax=ax)
    ax.set_ylabel("MAE (BPM)")
    fig.tight_layout()
    fig.savefig(f"{P}/E3_mae_by_condition.png", dpi=150)

if os.path.exists(a.live):
    L = pd.read_csv(a.live).dropna(subset=["reference_bpm"])
    if len(L):
        e5 = table(L.rename(columns={"pulsecam_bpm": "pred_hr", "reference_bpm": "ref_hr"}), ["method"])
        e5.to_csv(f"{T}/E5_live.csv", index=False)
        print("E5 - live demo vs reference device\n", e5)
