"""Datos: misma construcción y partición que el notebook (celda 4 y 8), en script reproducible.

Fuentes (Kaggle): RLVS (2000), Hockey Fight (1000), Movies Fight (201).
- Se reservan 12 videos (2 por clase y fuente, semilla 42) para pruebas manuales; nunca entran a train/val/test.
- Partición estratificada 70/15/15 a nivel de video con semilla 42 (fija para todas las semillas de entrenamiento).
- Se pre-extraen 16 fotogramas uniformes 224x224 por video en .npy.

Uso: python datos.py <carpeta_datos> <carpeta_cache>
"""
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SEQ_LEN = 16
IMG_SIZE = 224
EXT = (".mp4", ".avi", ".mpg", ".mkv")


def recorrer(raiz):
    for r, dirs, files in os.walk(raiz):
        dirs.sort()
        for f in sorted(files):
            yield r, f


def construir(carpeta):
    registros = []
    # El zip de Kaggle trae dos copias idénticas de RLVS: se usa la primera carpeta con Violence/ y NonViolence/ (como el notebook)
    for r, dirs, _ in sorted(os.walk(os.path.join(carpeta, "real-life-violence-situations-dataset"))):
        if "Violence" in dirs and "NonViolence" in dirs:
            for clase in ("Violence", "NonViolence"):
                for f in sorted(os.listdir(os.path.join(r, clase))):
                    registros.append({"video_path": os.path.join(r, clase, f), "label": int(clase == "Violence"),
                                      "class_name": clase, "source": "RLVS"})
            break
    for slug in ("hockey-fight-vidoes", "movies-fight-detection-dataset"):
        for r, f in recorrer(os.path.join(carpeta, slug)):
            if not f.lower().endswith(EXT):
                continue
            d, n = os.path.basename(r).lower(), f.lower()
            if "nofi" in d or "nofi" in n or "non" in d or n.startswith("no"):
                registros.append({"video_path": os.path.join(r, f), "label": 0, "class_name": "NonViolence", "source": slug})
            elif "fi" in d or "fi" in n or "ights" in d:
                registros.append({"video_path": os.path.join(r, f), "label": 1, "class_name": "Violence", "source": slug})
    df = pd.DataFrame(registros)
    reservados = []
    for fuente in ("RLVS", "hockey-fight-vidoes", "movies-fight-detection-dataset"):
        for lab in (1, 0):
            reservados += df[(df.source == fuente) & (df.label == lab)].sample(n=2, random_state=SEED).index.tolist()
    pruebas = df.loc[reservados]
    df = df.drop(index=reservados).reset_index(drop=True)
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True), "pruebas": pruebas.reset_index(drop=True)}


def extraer(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    frames = []
    for idx in np.linspace(0, max(total - 1, 0), SEQ_LEN, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok or fr is None:
            frames.append(frames[-1].copy() if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        else:
            frames.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames)


def _guardar(tarea):
    video, salida = tarea
    if not os.path.exists(salida):
        np.save(salida, extraer(video))
    return salida


def main(carpeta, cache):
    os.makedirs("results", exist_ok=True)
    splits = construir(carpeta)
    tareas = []
    for nombre, df in splits.items():
        os.makedirs(os.path.join(cache, nombre), exist_ok=True)
        df["npy_path"] = [os.path.join(cache, nombre, f"vid_{i:04d}_label_{l}.npy") for i, l in enumerate(df["label"])]
        tareas += list(zip(df["video_path"], df["npy_path"]))
        df.to_csv(f"results/split_{nombre}.csv", index=False)
        print(nombre, len(df), df.groupby(["source", "label"]).size().to_dict(), flush=True)
    with Pool(os.cpu_count()) as p:
        for k, _ in enumerate(p.imap_unordered(_guardar, tareas, chunksize=8)):
            if k % 500 == 0:
                print(f"fotogramas {k}/{len(tareas)}", flush=True)
    print("listo")


if __name__ == "__main__":
    main(*sys.argv[1:])
