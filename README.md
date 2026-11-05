# Cohnetic

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

Opciones: `--profile ideal` (ejecución de referencia, todo 9), `--cams 4..8`, `--seed N`.

## Estructura

| Ruta | Qué hace |
|---|---|
| `cohnetic/arena.py` | Pista 40×20, letras y puntos de control topográficos |
| `cohnetic/skeleton.py` | 26 keypoints de caballo (mapeables a hSMAL/PFERD) y 14 de jinete |
| `cohnetic/camera.py` | Modelo pinhole, proyección, **calibración extrínseca con las letras** (homografía + LM) |
| `cohnetic/triangulation.py` | **DLT + RANSAC por pares** vectorizado, relleno de huecos y filtro Butterworth |
| `cohnetic/protocol.py` | Hoja de ejercicios (Anexo II, alevines): textos, directrices, coeficientes, controles medibles y ruta |
| `cohnetic/segmentation.py` | Divide la trayectoria en los 13 ejercicios |
| `cohnetic/metrics/` | Pisadas y aires (paso/trote/galope y mano), transiciones, paradas, figuras, nuca (HNP), jinete, eventos sancionables |
| `cohnetic/analysis.py` | Aplica las métricas a cada ejercicio |
| `cohnetic/scoring.py` | Notas 0-10 por reglas trazables (Anexo I, coeficientes, *firewall* FEI, Art. 19) |
| `cohnetic/advice.py` | Oportunidades de mejora ordenadas por puntos recuperables + ejercicios de entrenamiento |
| `cohnetic/synth.py` | Generador sintético: cinemática, rig de cámaras y detecciones 2D simuladas |
| `cohnetic/pipeline.py` | Orquestación; `triangulate_session` es el mismo código para vídeo real |
| `cohnetic/report.py` | Exporta `web/data/report.js` y `motion.js` |
| `web/` | Front: visor Three.js, hoja de puntuación, mejoras, análisis, calidad 3D |
| `docs/ARQUITECTURA.md` | Diseño del sistema real, hardware y hoja de ruta |
| `docs/Cohnetic_Propuesta_Tecnica.pdf` | Documento de propuesta con capturas (`scripts/capturas_web.py` + `scripts/build_doc.py` lo regeneran) |

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
