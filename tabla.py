"""Junta results/runs/*/metricas.json en la tabla de ablación (media ± desv. sobre semillas).
Uso: python tabla.py  ->  results/tabla_ablacion.md (y tabla_ablacion_runs.csv con cada corrida)"""
import glob
import json

import numpy as np
import pandas as pd

NOMBRES = {"baseline_avg": "ResNet-50 + promedio temporal (línea base)", "resnet_rnn": "ResNet-50 + RNN",
           "resnet_lstm": "ResNet-50 + LSTM", "resnet_bilstm": "ResNet-50 + BiLSTM (sin atención)",
           "resnet_bilstm_attn": "ResNet-50 + BiLSTM + atención (propuesto)"}


def main():
    filas = []
    for f in sorted(glob.glob("results/runs/*/metricas.json")):
        m = json.load(open(f))
        fila = {"modelo": NOMBRES[m["modo"]] + (" + aumento CCTV" if m["robusto"] else ""), "semilla": m["semilla"],
                "acc": m["test"]["accuracy"] * 100, "f1": m["test"]["f1"] * 100, "auc": m["test"]["auc"] * 100,
                "f1_cctv": m["test_cctv"]["f1"] * 100, "min": m["minutos"], "vram": m["vram_gb"]}
        fila |= {f"acc_{k}": v["accuracy"] * 100 for k, v in m["test"]["por_fuente"].items()}
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv("results/tabla_ablacion_runs.csv", index=False)
    cols = [c for c in df.columns if c not in ("modelo", "semilla")]
    g = df.groupby("modelo", sort=False)
    res = g[cols].agg(["mean", "std"])
    res["n"] = g.size()
    orden = [NOMBRES[k] for k in NOMBRES] + [NOMBRES["resnet_bilstm_attn"] + " + aumento CCTV"]
    lin = ["| Modelo | n | Acc (%) | F1 (%) | AUC (%) | F1 CCTV sint. (%) | Acc RLVS | Acc Hockey | Acc Movies | min/run |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pm = lambda r, c: f"{r[(c, 'mean')]:.2f} ± {0 if np.isnan(r[(c, 'std')]) else r[(c, 'std')]:.2f}"
    for k in orden:
        if k in res.index:
            r = res.loc[k]
            lin.append(f"| {k} | {int(r['n'].iloc[0]) if hasattr(r['n'], 'iloc') else int(r['n'])} | {pm(r, 'acc')} | {pm(r, 'f1')} | {pm(r, 'auc')} | {pm(r, 'f1_cctv')} | "
                       f"{pm(r, 'acc_RLVS')} | {pm(r, 'acc_hockey-fight-vidoes')} | {pm(r, 'acc_movies-fight-detection-dataset')} | {r[('min', 'mean')]:.1f} |")
    lin.append(f"\nHoras de GPU totales (entrenamiento): {df['min'].sum() / 60:.2f} h en {json.load(open(f))['gpu']}")
    open("results/tabla_ablacion.md", "w").write("\n".join(lin) + "\n")
    print("\n".join(lin))


if __name__ == "__main__":
    main()
