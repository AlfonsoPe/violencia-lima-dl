"""Entrena una variante con una semilla (mismo protocolo que el notebook, celdas 9 y 15).
La partición es fija (semilla 42); la semilla de entrenamiento cambia la inicialización, el orden y los aumentos.

Uso: python entrenar.py <modo> <semilla> [robusto]
Salida: results/runs/<modo>[_robusto]_s<semilla>/ (metricas.json, historial.csv, predicciones_test.csv, modelo.pt)
"""
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from datos import IMG_SIZE
from model import SpatioTemporalViolenceModel

DEVICE = torch.device("cuda")
CFG = yaml.safe_load(open("resnet_bilstm_att.yaml"))


def fijar_semilla(s):
    random.seed(s), np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def degradar_cctv(frames):
    """Degradación sintética tipo cámara de serenazgo (idéntica al notebook): 80 px, JPEG q=35, desaturación, ruido."""
    out = []
    for img in frames:
        up = cv2.resize(cv2.resize(img, (80, 80), interpolation=cv2.INTER_LINEAR), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(up, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        dec = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
        gris = np.stack([cv2.cvtColor(dec, cv2.COLOR_RGB2GRAY)] * 3, axis=-1)
        mezcla = cv2.addWeighted(dec, 0.45, gris, 0.55, -15)
        out.append(np.clip(mezcla.astype(np.int16) + np.random.normal(0, 8, mezcla.shape).astype(np.int16), 0, 255).astype(np.uint8))
    return np.stack(out)


class Videos(Dataset):
    def __init__(self, df, train=False, cctv=False, robusto=False):
        self.df, self.train, self.cctv, self.robusto = df.reset_index(drop=True), train, cctv, robusto
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        fila = self.df.iloc[i]
        fr = np.load(fila["npy_path"])
        if self.cctv:
            np.random.seed(1000 + i)  # test CCTV determinista: misma degradación para todos los modelos
            fr = degradar_cctv(fr)
        elif self.train and self.robusto and random.random() < 0.4:
            fr = degradar_cctv(fr)
        if self.train and random.random() < 0.5:
            fr = np.flip(fr, axis=2).copy()
        x = (torch.from_numpy(fr).permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        return x, torch.tensor(float(fila["label"])), i


def semilla_worker(wid):
    s = torch.initial_seed() % 2**32
    np.random.seed(s), random.seed(s)


def cargador(df, semilla, **kw):
    g = torch.Generator().manual_seed(semilla)
    return DataLoader(Videos(df, **kw), batch_size=CFG["batch_size"], shuffle=kw.get("train", False), num_workers=int(os.environ.get("WORKERS", 4)),
                      pin_memory=True, worker_init_fn=semilla_worker, generator=g, persistent_workers=False)


@torch.no_grad()
def evaluar(modelo, ldr, df):
    modelo.eval()
    probs = np.zeros(len(df))
    for x, _, idx in ldr:
        with torch.autocast("cuda", dtype=torch.float16):
            probs[idx.numpy()] = torch.sigmoid(modelo(x.to(DEVICE)).float()).cpu().numpy()
    y, pred = df["label"].values, (probs >= 0.5).astype(int)
    m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
         "auc": roc_auc_score(y, probs), "confusion": confusion_matrix(y, pred).tolist()}
    m["por_fuente"] = {s: {"n": int((df.source == s).sum()), "accuracy": accuracy_score(y[df.source == s], pred[df.source == s])}
                       for s in sorted(df.source.unique())}
    return m, probs


def main(modo, semilla, robusto=""):
    semilla, robusto = int(semilla), robusto == "robusto"
    nombre = f"{modo}{'_robusto' if robusto else ''}_s{semilla}"
    salida = f"results/runs/{nombre}"
    os.makedirs(salida, exist_ok=True)
    tr, va, te = (pd.read_csv(f"results/split_{s}.csv") for s in ("train", "val", "test"))
    fijar_semilla(semilla)
    ld_tr = cargador(tr, semilla, train=True, robusto=robusto)
    ld_va, ld_te, ld_cc = cargador(va, semilla), cargador(te, semilla), cargador(te, semilla, cctv=True)
    modelo = SpatioTemporalViolenceModel(mode=modo, embed_dim=CFG["embed_dim"], hidden_dim=CFG["hidden_dim"], dropout=CFG["dropout"], pretrained=True).to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0, mejor, hist = time.time(), -1, []
    for ep in range(1, CFG["epochs"] + 1):
        modelo.train()
        perd = 0.0
        for x, y, _ in ld_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(modelo(x), y * (1 - CFG["label_smoothing"]) + CFG["label_smoothing"] / 2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            perd += loss.item() * len(y)
        sched.step()
        mv, _ = evaluar(modelo, ld_va, va)
        if mv["f1"] >= mejor:
            mejor = mv["f1"]
            torch.save(modelo.state_dict(), f"{salida}/modelo.pt")
        hist.append({"epoch": ep, "train_loss": perd / len(tr), "val_acc": mv["accuracy"], "val_f1": mv["f1"], "val_auc": mv["auc"]})
        print(nombre, hist[-1], flush=True)
    minutos, vram = (time.time() - t0) / 60, torch.cuda.max_memory_allocated() / 2**30
    modelo.load_state_dict(torch.load(f"{salida}/modelo.pt", map_location=DEVICE))
    m_te, p_te = evaluar(modelo, ld_te, te)
    m_cc, p_cc = evaluar(modelo, ld_cc, te)
    pd.DataFrame(hist).to_csv(f"{salida}/historial.csv", index=False)
    te.assign(prob=p_te, prob_cctv=p_cc).drop(columns=["npy_path"]).to_csv(f"{salida}/predicciones_test.csv", index=False)
    json.dump({"modo": modo, "robusto": robusto, "semilla": semilla, "val_f1": mejor, "test": m_te, "test_cctv": m_cc,
               "minutos": minutos, "vram_gb": vram, "gpu": torch.cuda.get_device_name(0)}, open(f"{salida}/metricas.json", "w"), indent=1)
    print(nombre, "test acc", round(m_te["accuracy"] * 100, 2), "f1", round(m_te["f1"], 4), "| cctv f1", round(m_cc["f1"], 4), f"| {minutos:.1f} min")


if __name__ == "__main__":
    main(*sys.argv[1:])
