# ROCA

Captura 3D multicámara de caballo y jinete y puntuación asistida de pruebas de doma
(Equitación de Trabajo, reglamento RFHE 2026).

```
Cámaras sincronizadas → calibración con las letras de la pista → keypoints 2D
→ triangulación robusta → métricas biomecánicas → puntuación por reglas → informe web
```

Este repositorio contiene la **base del sistema** y un **proof of concept** que funciona sin vídeo:
un recorrido sintético (caballo + jinete con errores conocidos) atraviesa el pipeline real
(cámaras simuladas con ruido, oclusiones y falsos positivos → calibración → triangulación →
análisis → nota) y se visualiza en una web con visor 3D.

## Uso rápido

```bash
pip install -r requirements.txt
python scripts/run_poc.py            # genera web/data/ (≈1 min, la triangulación es lo más lento)
# abre web/index.html en el navegador (funciona con doble clic, sin servidor)
python -m pytest -q                  # tests
```

Para enviar la demo en **un solo archivo** (incluye estilos, código, datos y Three.js; funciona sin conexión):

```bash
python scripts/build_standalone.py   # genera dist/ROCA.html (~4 MB)
```

Opciones: `--profile ideal` (ejecución de referencia, todo 9), `--cams 4..8`, `--seed N`.

## Vídeo real

El mismo pipeline acepta vídeos reales: sólo cambia el origen de las detecciones 2D.

```bash
pip install -r requirements-video.txt            # OpenCV + RTMPose (jinete)
# Caballo (DeepLabCut, Python 3.10-3.12, entorno aparte): pip install -r requirements-dlc.txt

# 1. Calibración intrínseca, una vez por cámara y óptica
python scripts/make_charuco.py --out charuco.png  # imprimir al 100% y medir
python scripts/calibrate_intrinsics.py --video cam1_tablero.mp4 --name CAM1 \
    --square-m 0.08 --marker-m 0.06 --out sesion/calib/cam1_intr.json

# 2. Calibración extrínseca con las letras, cada vez que se montan las cámaras
python scripts/calibrate_extrinsics.py --intrinsics sesion/calib/cam1_intr.json \
    --frame-from sesion/videos/cam1.mp4 --click --out sesion/calib/cam1.json

# 3. Sesión: describir cámaras, vídeos, sincronización y detectores (ver roca/session.py)
python scripts/run_video.py sesion/session.json --solo-3d       # reconstrucción y caché de detecciones
python scripts/preview_detections.py sesion/session.json --camera CAM1 --out cam1_preview.mp4
python scripts/run_video.py sesion/session.json                 # análisis, nota e informe web
```

Detectores: **jinete** RTMPose (COCO-17, vía rtmlib; funciona sin onnxruntime con el backend
`opencv`); **caballo** DeepLabCut SuperAnimal-Quadruped sin reentrenar (`"type": "superanimal"`;
no tiene menudillos, que se estiman por geometría, y el pecho es aproximado) o un modelo propio
con los 26 puntos de ROCA (`"type": "dlc_project"`) entrenado con PFERD y frames propios:

```bash
python training/pferd_to_dlc.py inspect --dataset /ruta/PFERD/dataset
python training/pferd_to_dlc.py convert --dataset /ruta/PFERD/dataset --mapping mapping.json --out pferd_dlc/labeled-data
python training/train_horse.py --project-dir modelos --labeled pferd_dlc/labeled-data
```

PFERD es CC BY-NC-SA (uso no comercial): revisar la licencia antes de usarlo en producto.

Sincronización: por hardware (genlock/timecode → `"offset_frames"` por cámara) o con un flash
visible por todas las cámaras al inicio (`"sync": {"method": "flash"}`).

## Estructura

| Ruta | Qué hace |
|---|---|
| `roca/arena.py` | Pista 40×20, letras y puntos de control topográficos |
| `roca/skeleton.py` | 26 keypoints de caballo (mapeables a hSMAL/PFERD) y 14 de jinete |
| `roca/camera.py` | Modelo pinhole, proyección, **calibración extrínseca con las letras** (homografía + LM) |
| `roca/triangulation.py` | **DLT + RANSAC por pares** vectorizado, relleno de huecos y filtro Butterworth |
| `roca/protocol.py` | Hoja de ejercicios (Anexo II, alevines): textos, directrices, coeficientes, controles medibles y ruta |
| `roca/segmentation.py` | Divide la trayectoria en los 13 ejercicios |
| `roca/metrics/` | Pisadas y aires (paso/trote/galope y mano), transiciones, paradas, figuras, nuca (HNP), jinete, eventos sancionables |
| `roca/analysis.py` | Aplica las métricas a cada ejercicio |
| `roca/scoring.py` | Notas 0-10 por reglas trazables (Anexo I, coeficientes, *firewall* FEI, Art. 19) |
| `roca/advice.py` | Oportunidades de mejora ordenadas por puntos recuperables + ejercicios de entrenamiento |
| `roca/synth.py` | Generador sintético: cinemática, rig de cámaras y detecciones 2D simuladas |
| `roca/pipeline.py` | Orquestación del PoC; `triangulate_session` es el mismo código para vídeo real |
| `roca/video.py` | Lectura de vídeo (OpenCV) y sincronización por flash |
| `roca/calibration.py` | Calibración intrínseca ChArUco, corrección de distorsión, extrínseca con letras, E/S de cámaras |
| `roca/detectors/` | RTMPose (jinete), DeepLabCut SuperAnimal / modelo propio (caballo), traducción de esqueletos |
| `roca/session.py` | Pipeline con vídeo real: `session.json` → detecciones (con caché) → 3D → nota |
| `training/` | Conversión PFERD → DeepLabCut y entrenamiento del detector del caballo |
| `roca/report.py` | Exporta `web/data/report.js` y `motion.js` |
| `web/` | Front: visor Three.js, hoja de puntuación, mejoras, análisis, calidad 3D |
| `docs/ARQUITECTURA.md` | Diseño del sistema real, hardware y hoja de ruta |
| `docs/ROCA_Propuesta_Tecnica.pdf` | Documento de propuesta con capturas (`scripts/capturas_web.py` + `scripts/build_doc.py` lo regeneran) |

## Resultados del PoC (6 cámaras 4K, 50 fps)

- Error 3D medio (MPJPE) **≈2.8 cm**, aire correcto en ≈95% de los frames, error del ángulo de la nuca ≈2°.
- La nota calculada desde la reconstrucción coincide con la calculada desde el movimiento real.
- Se detectan todos los errores sembrados: círculo de 10 m hecho de 11.8 m, transición tardía en M,
  ruptura del galope en el círculo de 20 m (nota limitada a 5), galope largo insuficiente,
  parada final corta y no cuadrada, nariz >10° detrás de la vertical en el galope, tronco del
  jinete inclinado, rebote al trote y una caricia por delante de las riendas (−5).

**Importante:** la nota es orientativa. Los umbrales son provisionales y se deben calibrar con
hojas de jueces reales. La posición de la nuca se penaliza por regla explícita y no se aprende de
las notas de los jueces (ver `hnp_elite_calentamiento_vs_concurso_preprint.pdf`).

## Documentación de referencia

Documentos de terceros usados como base. No se incluyen en el repositorio por derechos de autor;
para ejecutar el proyecto no hacen falta.

- `12-Reglamento-Equitacion-Trabajo-2026-def.pdf`: pruebas, tabla 0-10, hojas de ejercicios, penalizaciones.
- `fei_manual_jueces_doma_2025.pdf`, `usdf_glosario_juicio_2015.pdf`: criterios de juicio.
- `hiperflexion_revision_sistematica_2024.pdf`, `hnp_*.pdf`: posición cabeza-cuello y bienestar.
