"""
Sistema de monitoreo de vigilancia - Municipalidad de Lima
Modelo: ResNet-50 + Bi-LSTM + Atención temporal (MIA-203, UNI FIIS)
Ejecutar:  python app.py
"""
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import gradio as gr

# ------------------------------------------------------------------ Configuración
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache_mp4"
CACHE_DIR.mkdir(exist_ok=True)

CKPT_NAME = "best_resnet_bilstm_attn_lima_robust.pt"
SEQ_LEN, IMG_SIZE = 16, 224
THRESHOLD = 0.5                      # umbral de alerta
N_CAMARAS = 4
VIDEO_EXTS = {".mp4", ".avi", ".mpg", ".mkv", ".mov", ""}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ Modelo (igual al del notebook)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, rnn_outputs):
        scores = self.attn(rnn_outputs).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context, weights


class SpatioTemporalViolenceModel(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        resnet = models.resnet50(weights=None)       # los pesos salen del checkpoint
        self.backbone_features = nn.Sequential(*list(resnet.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_proj = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, 1))

    def forward(self, x, return_attention=False):
        B, T, C, H, W = x.shape
        feat = self.backbone_features(x.view(B * T, C, H, W))
        pooled = self.global_pool(feat).flatten(1)
        emb = self.spatial_proj(pooled).view(B, T, -1)
        rnn_out, _ = self.rnn(emb)
        context, attn = self.attention(rnn_out)
        logits = self.classifier(context).squeeze(-1)
        return (logits, attn) if return_attention else logits


def cargar_modelo():
    for carpeta in (BASE_DIR, BASE_DIR / "checkpoints"):
        ruta = carpeta / CKPT_NAME
        if ruta.exists():
            m = SpatioTemporalViolenceModel().to(DEVICE)
            m.load_state_dict(torch.load(ruta, map_location=DEVICE))
            m.eval()
            print(f"[OK] Modelo cargado: {ruta} | dispositivo: {DEVICE}")
            return m
    raise FileNotFoundError(f"No encuentro {CKPT_NAME} en {BASE_DIR}")


MODEL = cargar_modelo()


# ------------------------------------------------------------------ Videos
def extract_uniform_frames(video_path, seq_len=SEQ_LEN, img_size=IMG_SIZE):
    """Misma extracción que en el entrenamiento: 16 fotogramas uniformes a 224x224."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((seq_len, img_size, img_size, 3), dtype=np.uint8)
    frames = []
    for idx in np.linspace(0, max(total - 1, 0), seq_len, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(frames[-1].copy() if frames else
                          np.zeros((img_size, img_size, 3), dtype=np.uint8))
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames, axis=0)


def preparar_reproduccion(path: Path) -> Path:
    """El navegador no reproduce .avi (xvid): se convierte a mp4/H.264 una sola vez."""
    if path.suffix.lower() == ".mp4":
        return path
    salida = CACHE_DIR / f"{path.stem}.mp4"
    if salida.exists():
        return salida
    try:
        import imageio_ffmpeg
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
             "-movflags", "+faststart", str(salida)],
            check=True, capture_output=True)
        return salida
    except Exception as e:
        print(f"[AVISO] No se pudo convertir {path.name}: {e}")
        return path


def listar_videos():
    vids = {}
    for p in sorted(BASE_DIR.iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VIDEO_EXTS:
            vids[p.name] = p
    return vids


VIDEOS = listar_videos()                                   # nombre -> ruta original
PLAY = {n: preparar_reproduccion(p) for n, p in VIDEOS.items()}   # nombre -> ruta reproducible
NOMBRES = list(VIDEOS.keys())
print(f"[OK] Videos encontrados: {NOMBRES}")


# ------------------------------------------------------------------ Análisis
def tarjeta(color_borde, color_fondo, titulo, filas):
    items = "".join(f"<li>{f}</li>" for f in filas)
    return (f"<div style='border:2px solid {color_borde};background:{color_fondo};"
            f"border-radius:10px;padding:12px;color:#111'>"
            f"<div style='font-size:1.1em;font-weight:700'>{titulo}</div>"
            f"<ul style='margin:6px 0 0 18px'>{items}</ul></div>")


SIN_ANALIZAR = tarjeta("#9ca3af", "#f3f4f6", "⏺ Sin analizar",
                       ["Presiona «Analizar situación de violencia»."])


@torch.no_grad()
def analizar(nombre):
    if not nombre or nombre not in VIDEOS:
        return tarjeta("#f59e0b", "#fef3c7", "⚠ Selecciona un video", [])
    t0 = time.time()
    frames = extract_uniform_frames(VIDEOS[nombre])
    if frames.max() == 0:
        return tarjeta("#f59e0b", "#fef3c7", "⚠ No se pudo leer el video", [nombre])

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = ((x - MEAN) / STD).unsqueeze(0).to(DEVICE)
    logits, attn = MODEL(x, return_attention=True)
    prob = torch.sigmoid(logits).item()
    attn = attn.squeeze(0).cpu().numpy()
    t_pico = int(np.argmax(attn)) + 1
    seg = time.time() - t0

    filas = [f"Probabilidad de violencia: <b>{prob * 100:.2f}%</b> "
             f"(umbral de alerta: {THRESHOLD * 100:.0f}%)",
             f"Fotograma de mayor atención: t = {t_pico}/{SEQ_LEN} (α = {attn.max():.3f})",
             f"Tiempo de análisis: {seg:.2f} s"]
    if prob >= THRESHOLD:
        return tarjeta("#dc2626", "#fee2e2", "🚨 ALERTA: VIOLENCIA DETECTADA", filas)
    return tarjeta("#16a34a", "#dcfce7", "✅ NORMAL: SIN VIOLENCIA", filas)


def cambiar_video(nombre):
    return PLAY.get(nombre), SIN_ANALIZAR


# ------------------------------------------------------------------ Interfaz
def crear_camara(i):
    defecto = NOMBRES[i % len(NOMBRES)] if NOMBRES else None
    with gr.Column(min_width=380):
        gr.Markdown(f"### 📹 Cámara {i + 1:02d}")
        selector = gr.Dropdown(NOMBRES, value=defecto, label="Fuente de video (simulada)")
        video = gr.Video(value=PLAY.get(defecto), interactive=False, height=260,
                         label=f"Cámara {i + 1:02d}")
        boton = gr.Button("🔍 Analizar situación de violencia", variant="primary")
        resultado = gr.HTML(SIN_ANALIZAR)
    selector.change(cambiar_video, selector, [video, resultado])
    boton.click(analizar, selector, resultado)


with gr.Blocks(title="Monitoreo de Vigilancia - Municipalidad de Lima") as demo:

    # ---- Página 1: inicio
    with gr.Column(visible=True) as pagina_inicio:
        gr.HTML(
            "<div style='text-align:center;padding:70px 10px'>"
            "<div style='font-size:60px'>🛡️</div>"
            "<h1 style='margin:10px 0'>Sistema de Monitoreo de Vigilancia</h1>"
            "<h2 style='margin:0;font-weight:500'>Municipalidad de Lima</h2>"
            "<p style='color:#6b7280;margin-top:14px'>Detección de situaciones de violencia "
            "en cámaras de videovigilancia<br>Prototipo académico · MIA-203 · UNI FIIS</p></div>")
        btn_acceder = gr.Button("Acceder", variant="primary", size="lg")

    # ---- Página 2: 4 cámaras
    with gr.Column(visible=False) as pagina_monitoreo:
        gr.Markdown("## 🛡️ Centro de Monitoreo — Municipalidad de Lima")
        with gr.Row():
            crear_camara(0)
            crear_camara(1)
        with gr.Row():
            crear_camara(2)
            crear_camara(3)
        gr.Markdown("<small>Prototipo académico entrenado con videos públicos "
                    "(RLVS, Hockey Fights, Movies Fights). Un resultado automático debe ser "
                    "verificado por un operador humano.</small>")
        btn_salir = gr.Button("⬅ Salir", size="sm")

    btn_acceder.click(lambda: (gr.update(visible=False), gr.update(visible=True)),
                      None, [pagina_inicio, pagina_monitoreo])
    btn_salir.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
                    None, [pagina_inicio, pagina_monitoreo])


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(BASE_DIR)], inbrowser=True)