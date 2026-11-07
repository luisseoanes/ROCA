# Entrenar y validar el detector del caballo en un PC con GPU NVIDIA

Guía para terminar el detector del caballo (26 puntos de ROCA) con PFERD en Windows y una GPU NVIDIA.
Probada la cadena completa en macOS. Los tiempos son estimaciones para una **GTX 1660 Ti (6 GB)**.

| Paso | Qué hace | Tiempo estimado |
|---|---|---|
| 1. Instalar | Python 3.12, PyTorch con CUDA, DeepLabCut | 20–30 min |
| 2. Descargar PFERD | ID_1 de test (~7 GB) + ID_3 e ID_5 para entrenar (~56 GB) | según la conexión (~1–2 h a 100 Mb/s) |
| 3. Convertir | ~5000 frames etiquetados | 15–20 min |
| 4. Entrenar | detector de cajas (5 épocas) + pose (60 épocas) | 3–6 h |
| 5. Validar | SuperAnimal y modelo propio sobre el caballo de test | 15–20 min |

## 1. Instalación (PowerShell)

Requisitos: driver NVIDIA reciente, [Python 3.12](https://www.python.org/downloads/) (marcar "Add to PATH") y Git.

```powershell
git clone https://github.com/luisseoanes/ROCA.git
cd ROCA
py -3.12 -m venv .venv-dlc
.venv-dlc\Scripts\activate
python -m pip install --upgrade pip

# PyTorch con CUDA ANTES que DeepLabCut (si no, pip instala la versión sólo CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dlc.txt
pip uninstall -y onnxruntime; pip install onnxruntime-gpu   # RTMPose en GPU (sólo hace falta para el jinete)

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Debe imprimir: True NVIDIA GeForce GTX 1660 Ti
python -m pytest -q                                          # 18 tests
```

## 2. Datos

```powershell
python scripts\download_pferd.py --demo                    # ID_1: caballo de test, NO se usa para entrenar
python scripts\download_pferd.py --subjects ID_3 ID_5      # entrenamiento; reanuda si se corta
python training\pferd_to_dlc.py inspect --dataset data\PFERD\dataset
```

Si ya tienes los datos en otro equipo, basta con copiar `data/PFERD/dataset/` (~60 GB, disco en exFAT) a
`ROCA\data\PFERD\dataset\` en lugar de descargarlos. Copiando también `data/validacion/` te ahorras repetir
la validación de SuperAnimal.

`inspect` debe mostrar `ID_3: 5 pruebas, 50 pares prueba/cámara, 50 con vídeo` e `ID_5: 7 pruebas, 70 ...`.

Marcadores que faltan en algún caballo (esos puntos quedan sin etiquetar en ese caballo; DeepLabCut los
trata como desconocidos, no como error): ID_3 no tiene nariz ni garganta, ID_5 no tiene ojo ni, en el
70 % de los frames, cadera izquierda. La punta de la cola no tiene marcador en PFERD.
Los datos se quedan en `data\` (ignorado por git). Licencia: investigación no comercial, sin redistribución.

## 3. Comprobaciones de referencia (opcionales, 1 min)

```powershell
python scripts\validate_pferd.py --source gt --clip 450 750 --session-dir data\validacion\trote_450_750 --out data\validacion\gt_exacto_10c.json --save-3d data\validacion\gt_exacto_10c.npz
```
Esperado: error 3D medio ≈ 0,22 cm (sólo el filtro de suavizado). Si sale muy distinto, algo está mal en los datos.

## 4. Convertir y entrenar

```powershell
python training\pferd_to_dlc.py convert --dataset data\PFERD\dataset --subjects ID_3,ID_5 --every 40 --out data\pferd_dlc\labeled-data

python training\train_horse.py --project-dir modelos --labeled data\pferd_dlc\labeled-data --device cuda `
    --epochs 60 --batch-size 8 --detector-epochs 5 --detector-batch-size 2 --save-epochs 10
```

- Con 6 GB de VRAM: si sale `CUDA out of memory`, bajar `--batch-size` a 4 y `--detector-batch-size` a 1.
- Se puede cortar y reanudar desde el último snapshot (cada 10 épocas) con la API de DeepLabCut si hiciera falta.
- Al terminar imprime la ruta del `config.yaml` y el error en píxeles por punto en el conjunto de test interno.
  Ese error es optimista (frames del mismo caballo); la cifra que vale es la del paso 5, con un caballo no visto.
- Para una 5070 (12 GB): `--batch-size 16 --detector-batch-size 4`, unas 1–2 h.

## 5. Validar sobre el caballo no visto (ID_1) y publicar los resultados

```powershell
$C6 = "23350,23348,23416,23417,23603,21386"
$CFG = (Get-ChildItem modelos\*\config.yaml).FullName

python scripts\validate_pferd.py --source detector --horse superanimal --device cuda --clip 450 750 --cams $C6 `
    --session-dir data\validacion\trote_450_750 --out data\validacion\superanimal_6c.json --save-3d data\validacion\superanimal_6c.npz

python scripts\validate_pferd.py --source detector --horse dlc_project --config $CFG --device cuda --clip 450 750 --cams $C6 `
    --session-dir data\validacion\trote_450_750 --out data\validacion\roca_6c.json --save-3d data\validacion\roca_6c.npz

python scripts\pferd_overlay.py         # vídeos con el esqueleto (web\data\pferd_video\, local)
python scripts\pferd_report.py          # -> web\data\pferd.js (métricas) y pferd_local.js (visor 3D, local)
python scripts\build_standalone.py      # -> dist\ROCA.html para enviar
```

Si en `data\validacion\` ya están los JSON de las validaciones hechas en el Mac (no se suben a git), se
pueden repetir con los comandos de arriba; tardan poco.

Después:

```powershell
git add web\data\pferd.js
git commit -m "Resultados de validación con PFERD: detector ROCA ajustado"
git push
```

El modelo (`modelos\`) no se sube a git: pesa cientos de MB y está entrenado con datos de licencia no
comercial. Para usarlo en una sesión de pista: `"horse": {"type": "dlc_project", "config": "<ruta>\\config.yaml", "device": "cuda"}`.

## Qué esperar

| Configuración | Error 3D medio (ID_1, trote, 6 cámaras) |
|---|---|
| 2D exacto | 0,22 cm |
| Detector simulado 3 px | 2,2 cm |
| Detector simulado 8 px | 6,7 cm |
| SuperAnimal sin ajustar | 23 cm (medido en el Mac; 6–10 cm en cabeza y patas, 37–57 cm en cruz/dorso/caderas por sesgo de definición) |
| ROCA ajustado con PFERD | *(objetivo: < 5 cm)* |

En PFERD 1 px ≈ 0,9 cm (1920×1088, cámaras a ~15 m). En pista, con 4K y cámaras más cerca, el mismo
error del detector en píxeles se traduce en menos centímetros.
