# Arquitectura de ROCA

## 1. Objetivo

Reconstruir en 3D al caballo y al jinete durante una prueba completa, a partir de 4-8 cámaras
alrededor de la pista, y convertir esa reconstrucción en:

1. una **nota orientativa** por ejercicio, alineada con la hoja oficial (Anexo II RFHE),
2. **oportunidades de mejora** ordenadas por puntos recuperables, con evidencia medida,
3. **indicadores de bienestar** (posición cabeza-cuello) independientes de la nota.

## 2. Pipeline

```
 ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────┐   ┌─────────┐
 │ Captura    │ → │ Calibración  │ → │ Keypoints 2D │ → │ Triangulación │ → │ Métricas│ → │ Nota +  │
 │ sincroniz. │   │ int. + ext.  │   │ + tracking   │   │ + filtrado    │   │ por ej. │   │ informe │
 └────────────┘   └──────────────┘   └──────────────┘   └───────────────┘   └─────────┘   └─────────┘
   hardware        camera.py          (fase 2)           triangulation.py   metrics/,     scoring.py
                                                         pipeline.py        analysis.py   advice.py
```

| Etapa | PoC (hecho) | Sistema real |
|---|---|---|
| Captura | Simulada (`synth.py`) | **Implementado** (`video.py`, `session.py`): vídeos por cámara, desfases por hardware o sincronización por flash. Recomendado: cámaras global shutter 4K, 50-120 fps, genlock/PTP |
| Calibración intrínseca | Conocida | **Implementado** (`calibration.py`): tablero ChArUco por cámara (OpenCV) + corrección de distorsión de las detecciones |
| Calibración extrínseca | **Real**: letras + esquinas de la pista, homografía + Levenberg-Marquardt | Igual, con las letras medidas con estación total / GPS RTK. Opcional: ajuste de haces con varita |
| Keypoints 2D | Simulados con ruido 1.6 px, 5% pérdidas, 35% oclusión del lado lejano, 1.5% falsos positivos | **Implementado** (`detectors/`): jinete RTMPose (rtmlib). Caballo: SuperAnimal-Quadruped sin reentrenar o modelo propio; conversión de **PFERD** y entrenamiento en `training/`. **Pendiente:** entrenar el modelo y validarlo con vídeo de pista |
| Triangulación | **Real**: RANSAC por pares + DLT con inliers, Butterworth 8 Hz | Igual; añadir restricciones de longitud de huesos y ajuste de modelo (hSMAL / SMPL-X) |
| Métricas y nota | **Reales** | Calibrar umbrales con jueces |

## 3. Hardware recomendado

- **Doma (40×20 m):** 6 cámaras. 4 en esquinas a 5-6 m de altura y 2 laterales en B y E.
- **Manejabilidad/velocidad (≥70×30 m):** 8 cámaras.
- 4K a 50 fps como mínimo; 100-120 fps para pisadas finas. Con 6.5 m/s al galope, 1/60 s de desincronización ≈ 11 cm de error.
- Micrófono sincronizado para detectar voz/chasquidos (−2, Art. 19).
- Resolución efectiva: con f≈2300 px un caballo a 60 m ocupa ≈90 px; para precisión en cascos y manos conviene más cámaras o lentes más largas por zona.

## 4. Métricas implementadas

| Métrica | Módulo | Uso en la nota |
|---|---|---|
| Pisadas por casco (altura + velocidad suavizadas) | `metrics/gait.py` | Base de todo lo demás |
| Aire y mano por tranco (plantillas de desfase entre miembros) | `metrics/gait.py` | Ruptura de aire / mano contraria → *firewall* 5 |
| Transiciones (cruce de velocidad o hueco entre rachas) | `metrics/gait.py` | Distancia a la letra |
| Regularidad (CV robusto del tranco) | `analysis.py` | Aires, nota de conjunto A |
| Paradas: duración, cuadratura, inmovilidad | `analysis.py` | Ejercicios 1, 6 y 13 |
| Círculos (ajuste Kåsa: diámetro, centro, redondez) y rectitud | `metrics/geometry.py` | Trazado y geometría |
| Alargamiento (velocidad y tranco frente a galope de trabajo) | `analysis.py` | Ejercicio 12 y nota B |
| Plano nasal respecto a la vertical, elevación del cuello | `metrics/posture.py` | Ejercicios con "posición de la nuca" y nota C; alerta de bienestar |
| Tronco del jinete, rebote de la pelvis, estabilidad de la mano | `metrics/posture.py` | Nota D |
| Caricia por delante de las riendas, dos manos > 3 s | `metrics/events.py` | Penalización −5 / descalificación |

## 5. Decisiones de diseño

- **Reglas trazables, no caja negra.** Cada deducción guarda la métrica, el valor y el umbral para que un juez la pueda auditar. Un modelo aprendido (de métricas a nota) puede venir después, pero con la nuca y el bienestar fuera del aprendizaje: la literatura de la carpeta muestra que los jueces premian la hiperflexión.
- **Nota base 9.** La visión no acredita la "expresión" que justifica un 10.
- **Mismo código para el PoC y para el sistema real.** Sólo cambia el origen de las detecciones 2D.
- **Web sin servidor.** Los datos se exportan como `.js` para abrir `index.html` con doble clic. Para producción: API (FastAPI) y almacenamiento por sesión.

## 6. Hoja de ruta

1. **Captura piloto (2-4 semanas):** 4-6 cámaras sincronizadas en una pista de doma; grabar 10-20 recorridos con hoja de juez.
2. **Keypoints del caballo:** ajustar DeepLabCut SuperAnimal con PFERD (revisar licencia CC BY-NC-SA y términos de hSMAL si el uso es comercial) y con 500-1000 frames propios etiquetados.
3. ~~Sustituir `synth.observe` por el detector real~~ Hecho: `roca/session.py` + `scripts/run_video.py`. Falta probarlo con vídeo de pista real.
4. **Calibrar umbrales** comparando con las notas de los jueces (correlación por ejercicio, ±0.5 puntos).
5. **Ampliar el protocolo:** resto de categorías del Anexo II, manejabilidad (obstáculos, derribos) y velocidad (tiempos).
6. **Audio** para voz/chasquidos; **SMPL-X/hSMAL** para la malla 3D en el visor.

## 7. Limitaciones conocidas

- El movimiento sintético es esquemático (sin balanceo de cuello al galope, sin incurvación del tronco).
- La clasificación del aire en los trancos de transición es ambigua por naturaleza; se usa un estimador robusto del instante de cambio (±1.5 m en la referencia ideal).
- La estabilidad de la mano tiene un suelo de ruido de la reconstrucción (≈1.5 cm); su umbral es provisional.
