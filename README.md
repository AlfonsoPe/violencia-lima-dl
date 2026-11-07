# Detección Espacio-Temporal de Violencia Urbana
### ResNet-50 + Bi-LSTM + Atención Temporal con Transferencia de Dominio a CCTV de Lima

**Proyecto Final Integrador — MIA-203 Redes Neuronales y Aprendizaje Profundo**  
Universidad Nacional de Ingeniería (UNI) — FIIS — Unidad de Posgrado

**Docente:** Dr. Aldo Camargo Fernández-Baca, Ph.D.

### Integrantes
- Alvaro Alejandro Siesquén Abad  
- Kiara Isumi Alvarez Barrionuevo  
- Sergio Alonso Ortiz Pauca  
- Alfonso Pérez Ramos  

---

## Descripción

Sistema de detección de situaciones de violencia en video mediante una arquitectura híbrida espacio-temporal:

- **Extractor espacial:** ResNet-50 preentrenada en ImageNet
- **Modelado temporal:** Bi-LSTM + Mecanismo de Atención Temporal
- **Optimización:** AdamW + Cosine Annealing + Mixed Precision
- **Evaluación de dominio:** In-Domain (RLVS + Hockey Fight + Movies Fight) y Out-of-Domain (simulación CCTV Lima Metropolitana)

El prototipo incluye una interfaz tipo **Centro de Monitoreo de la Municipalidad de Lima** desarrollada con Gradio.

## Resultados principales

| Modelo | F1 (In-Domain) | F1 (CCTV Lima) | Caída Domain Shift |
|--------|----------------|----------------|--------------------|
| Baseline (ResNet-50 + Avg) | 0.9795 | 0.8145 | −0.165 |
| ResNet-50 + Vanilla RNN | 0.9615 | 0.8299 | −0.132 |
| ResNet-50 + LSTM | 0.9793 | 0.8048 | −0.175 |
| **ResNet-50 + Bi-LSTM + Atención** | **0.9816** | 0.8511 | −0.131 |
| **Propuesto + Aumento CCTV (robusto)** | 0.9793 | **0.9662** | **−0.013** |

- Accuracy del modelo propuesto: **98.13%**
- Recall (sensibilidad): **1.000** en el conjunto de prueba limpio
- Errores en Test (481 videos): solo **10** (7 FP + 3 FN)

## Estructura del repositorio

```
violencia-lima-dl/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── resnet_bilstm_att.yaml
├── src/
│   └── model.py
├── notebooks/
│   └── Proyecto_violencia_Lima.ipynb
├── app/
│   └── app.py
├── results/          # tablas y figuras del artículo
└── data/             # README con enlaces de descarga (no se suben videos pesados)
```

## Cómo reproducir los experimentos

### 1. Clonar el repositorio
```bash
git clone https://github.com/AlfonsoPe/violencia-lima-dl.git
cd violencia-lima-dl
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el notebook principal
Abre `notebooks/Proyecto_violencia_Lima.ipynb` en **Google Colab** (GPU T4 recomendada) o en un entorno local con GPU.

El notebook descarga automáticamente los datasets públicos, realiza la partición sin fuga de datos, entrena las 4 arquitecturas + la versión robusta y genera todas las tablas y figuras.

### 4. Demostración interactiva (Gradio)
```bash
cd app
python app.py
```
Se abrirá en el navegador una interfaz con 4 cámaras simuladas del Centro de Monitoreo.

## Cumplimiento de requisitos del curso (R1–R6)

| Requisito | Evidencia |
|-----------|-----------|
| **R1** Fundamento en el curso | CNN (ResNet-50) + RNN/LSTM/Bi-LSTM + Atención temporal + AdamW |
| **R2** Originalidad | Multi-dataset + degradación CCTV Lima + aplicación a serenazgo |
| **R3** Línea base | ResNet-50 Frame-Averaging + Vanilla RNN |
| **R4** Experimentación | Ablaciones controladas + partición 70/15/15 a nivel de video |
| **R5** Interpretabilidad | Grad-CAM espacial + pesos de atención temporal αₜ |
| **R6** Costo y responsabilidad | Horas GPU T4 reportadas + análisis FP/FN + limitaciones éticas |

## Datasets utilizados (públicos)
- [Real-Life Violence Situations (RLVS)](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- Hockey Fight Videos
- Movies Fight Detection Dataset

## Costo computacional (R6)
- Hardware: NVIDIA Tesla T4 (Google Colab) — 14.56 GB VRAM
- Tiempo de entrenamiento por modelo: ≈ 11–13 minutos (7 épocas)
- VRAM pico: 2.3 – 2.8 GB

## Licencia y uso responsable
Uso estrictamente académico. Los resultados automáticos de detección de violencia **deben ser verificados por un operador humano** antes de cualquier acción operativa. El sistema no debe usarse como única fuente de decisión en contextos de seguridad ciudadana.

---
*Proyecto desarrollado para el curso MIA-203 — Universidad Nacional de Ingeniería (UNI), 2026*
