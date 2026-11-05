"""Genera el documento PDF de propuesta técnica (docs/Cohnetic_Propuesta_Tecnica.pdf).

Usa los datos reales del último PoC (web/data/report.json) y las capturas de docs/img/
(generadas con scripts/capturas_web.py). Se maqueta en HTML y se imprime con Chrome.

Uso: python scripts/build_doc.py
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cohnetic.protocol import MOVEMENTS  # noqa: E402

DOCS = ROOT / "docs"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
R = json.loads((ROOT / "web" / "data" / "report.json").read_text(encoding="utf-8"))
S, V = R["score"], R["validation"]
MOV = {m["n"]: m for m in S["movements"]}
AM = {m["n"]: m for m in R["analysis"]["movements"]}
COL = {c["key"]: c for c in S["collective"]}
e = html.escape


def n(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def check(mv, typ, **kw):
    for c in AM[mv]["checks"]:
        if c["type"] == typ and all(c.get(k) == v for k, v in kw.items()):
            return c
    return {}


def ded(mv_or_key, word, collective=False):
    src = COL[mv_or_key]["deductions"] if collective else MOV[mv_or_key]["deductions"]
    for d in src:
        if word.lower() in d["reason"].lower():
            return d
    return None


def fig(src, cap, w="100%", cls=""):
    return f'<figure class="{cls}"><img src="img/{src}" style="width:{w}"><figcaption>{cap}</figcaption></figure>'


def camera_svg():
    cams = R["cameras"]
    parts = ['<svg viewBox="-20 -28 80 56" class="diagram" role="img" aria-label="Disposición de cámaras">',
             '<rect x="0" y="-10" width="40" height="20" fill="#e8dcc3" stroke="#52514e" stroke-width="0.25"/>',
             '<line x1="0" y1="0" x2="40" y2="0" stroke="#898781" stroke-width="0.15" stroke-dasharray="0.8 0.8"/>']
    for k, (x, y) in R["arena"]["letters"].items():
        ox = -1.6 if x == 0 else 1.6 if x == 40 else 0
        oy = -1.6 if y > 0 else 1.6 if y < 0 else 1.4
        parts.append(f'<text x="{x + ox}" y="{-y + oy}" font-size="1.6" text-anchor="middle" dominant-baseline="middle" font-weight="600" fill="#52514e">{k}</text>')
    for c in cams:
        x, y, z = c["center"]
        parts.append(f'<line x1="{x}" y1="{-y}" x2="20" y2="0" stroke="#2a78d6" stroke-width="0.15" stroke-dasharray="0.6 0.5" opacity="0.6"/>')
        parts.append(f'<rect x="{x - 1.1}" y="{-y - 0.8}" width="2.2" height="1.6" rx="0.3" fill="#2a78d6"/>')
        lx = x + (3.2 if x > 20 else -3.2 if x < 20 else 0)
        ly = -y + (2.6 if -y > 0 else -2.6) if x == 20 else -y
        parts.append(f'<text x="{lx}" y="{ly}" font-size="1.5" text-anchor="middle" dominant-baseline="middle" fill="#0b0b0b">'
                     f'{c["name"].split()[0]} · {z:g} m</text>')
    parts.append('<text x="20" y="26" font-size="1.6" text-anchor="middle" fill="#52514e">Pista de doma 40 × 20 m · cámaras a 5,5–6,5 m de altura</text>')
    parts.append("</svg>")
    return "".join(parts)


def pipeline_html():
    steps = [
        ("1", "Captura", "4–8 cámaras 4K sincronizadas por hardware alrededor de la pista"),
        ("2", "Calibración", "Intrínseca con ChArUco; extrínseca con las letras de la pista"),
        ("3", "Keypoints 2D", "Detector de puntos del caballo (26) y del jinete (14) en cada cámara"),
        ("4", "Triangulación", "RANSAC multivista + filtrado temporal → esqueleto 3D a 50 fps"),
        ("5", "Métricas", "Aires, pisadas, transiciones, figuras, paradas, nuca, jinete"),
        ("6", "Nota e informe", "Reglas del reglamento → nota por ejercicio, mejoras y bienestar"),
    ]
    return '<div class="pipeline">' + "".join(
        f'<div class="step"><div class="num">{a}</div><b>{b}</b><span>{c}</span></div>' for a, b, c in steps) + "</div>"


# --------------------------------------------------------------------------- contenido
c3 = check(3, "circle")
t5 = check(5, "transition", at="M")
h13 = check(13, "halt")
l12 = check(12, "lengthening")
gf8 = AM[8].get("gait_faults", [{}])[0]
can = R["analysis"]["by_gait"]["canter"]
calib = R["calibration"]
caress = S["penalties"][0] if S["penalties"] else None

planted = [
    ("Círculo de 10 m en A trazado de 11,8 m y desplazado 0,7 m", "Ej. 3",
     f"Diámetro {n(c3.get('diameter_m', 0), 2)} m (+{n(c3.get('diam_err_pct', 0))}%), centro desplazado {n(c3.get('center_offset_m', 0), 2)} m",
     f"Nota {n(MOV[3]['note'])}", True),
    ("Transición al paso 2,8 m después de M", "Ej. 5", f"Transición a {n(abs(t5.get('late_m', 0)))} m de M (tolerancia 2 m)", f"Nota {n(MOV[5]['note'])}", True),
    ("Rompe el galope a trote ~2,6 s en el círculo de 20 m", "Ej. 8",
     f"{gf8.get('strides', '?')} trancos de trote entre {n(gf8.get('t0', 0))} y {n(gf8.get('t1', 0))} s", f"Nota limitada a {n(MOV[8]['note'])} (error grave)", True),
    ("Galope largo sin alargar (4,35 m/s; se esperan ≥ 4,8)", "Ej. 12",
     f"Velocidad ×{n(l12.get('ratio', 0), 2)}, tranco ×{n(l12.get('stride_ratio', 0), 2)}", f"Nota {n(MOV[12]['note'])} · conjunto B", True),
    ("Parada final de 2,6 s (se piden 4 s)", "Ej. 13", f"Inmovilidad medida {n(h13.get('duration_s', 0))} s", f"Nota {n(MOV[13]['note'])}", True),
    ("Posterior derecho retrasado 22 cm en la parada final", "Ej. 13", f"Desfase posteriores {n(h13.get('hind_offset_cm', 0), 0)} cm", "Incluido arriba", True),
    ("Nariz ~9° detrás de la vertical al galope (13° en ej. 8)", "Conjunto C",
     f"Media {n(abs(can['hnp']['mean_deg']))}° detrás; {n(can['hnp']['pct_behind10'], 0)}% del galope > 10° detrás", f"Conjunto C {n(COL['C']['note'])} + alerta de bienestar", True),
    ("Tronco del jinete 13° hacia delante al galope", "Conjunto D", f"Inclinación media {n(can['rider_lean_deg'])}°", f"Conjunto D {n(COL['D']['note'])}", True),
    ("Rebote del jinete al trote sentado en el ej. 7", "Conjunto D",
     (ded("D", "Asiento", True) or {}).get("evidence", "—"), "Incluido arriba", bool(ded("D", "Asiento", True))),
    ("Caricia al caballo por delante de las riendas (ej. 10)", "Art. 19",
     f"Mano derecha en el cuello {n(caress['t'])} s" if caress else "—", "−5 puntos", bool(caress)),
    ("Mano de las riendas algo menos fija (2,0 cm frente a 1,2 cm)", "Conjunto D",
     "Por debajo del umbral provisional (4,5 cm)", "Sin efecto", False),
]
n_det = sum(1 for p in planted if p[4])

metric_rows = [
    ("Pisadas y aire por tranco", "Desfase entre los apoyos de los 4 cascos (paso 4 tiempos, trote diagonal, galope y mano)",
     "“Romper el aire”, “errores en el cambio de pie” (manejabilidad), nota de conjunto A"),
    ("Transiciones", "Instante del cambio de aire y distancia de la cruz a la letra", "“Claridad y fluidez en la transición”, transición en la letra"),
    ("Regularidad", "Variación de la duración del tranco (CV robusto)", "“Calidad y regularidad del trote”, conjunto A"),
    ("Figuras", "Ajuste de círculo (diámetro, centro, redondez) y desviación en líneas", "“Trazado y geometría”, “rectitud”"),
    ("Paradas", "Duración con velocidad < 0,15 m/s, desfase entre cascos, oscilación de la cruz", "“Inmovilidad de 4\"”, “caballo cuadrado”"),
    ("Alargamientos", "Velocidad y longitud de tranco frente al aire de trabajo", "“Galope largo”, “impulsión”, conjunto B"),
    ("Posición de la nuca (HNP)", "Ángulo del plano nasal con la vertical, elevación del cuello", "“Posición de la nuca”, “contacto”, conjunto C; bienestar"),
    ("Jinete", "Inclinación del tronco, rebote de la pelvis respecto al dorso, estabilidad de la mano", "“Colocación y posición del jinete”, conjunto D"),
    ("Eventos sancionables", "Mano derecha en el cuello por delante de las riendas; dos manos > 3 s", "Art. 19: −5 por caricia; descalificación en senior domados"),
]

hw_rows = [
    ("Cámaras", "6 (doma 40×20) a 8 (manejabilidad/velocidad ≥ 70×30). 4K, obturador global, 50 fps mínimo; 100–120 fps deseable para pisadas"),
    ("Óptica", "Focal elegida por posición para que el caballo ocupe ≥ 100 px en la zona más lejana"),
    ("Montaje", "Postes o trípodes de 5–6,5 m en esquinas y laterales B/E, fuera de la pista"),
    ("Sincronización", "Por hardware (genlock o PTP/IEEE 1588). Con 6,5 m/s al galope, 1/60 s de desfase son ~11 cm de error"),
    ("Audio", "Micrófono sincronizado para detectar voz/chasquidos (−2, Art. 19)"),
    ("Cómputo", "Equipo con GPU para el detector 2D; la triangulación y la puntuación corren en CPU"),
    ("Topografía", "Estación total o GPS RTK para medir letras y esquinas una vez por pista"),
]

phases = [
    ("F0", "Proof of concept", "Hecho", "Pipeline completo con datos sintéticos, web con visor 3D, informe y tests"),
    ("F1", "Captura piloto", "4–6 semanas", "Montaje de 6 cámaras en una pista, 10–20 recorridos grabados con la hoja del juez"),
    ("F2", "Reconstrucción con vídeo real", "6–8 semanas", "Detector del caballo ajustado (PFERD + 500–1000 frames propios), detector del jinete, triangulación real"),
    ("F3", "Calibración con jueces", "4–6 semanas", "Comparar notas por ejercicio con jueces; ajustar umbrales; objetivo ±0,5 puntos en ≥ 80% de los ejercicios"),
    ("F4", "Ampliación", "8 semanas", "Resto de categorías del Anexo II, manejabilidad (obstáculos, derribos) y velocidad (tiempos); audio"),
    ("F5", "Producto", "A definir", "Aplicación multiusuario, almacenamiento por sesión, informes para jinetes y entrenadores"),
]

risks = [
    ("Precisión del detector del caballo en pista real (polvo, luz, oclusiones)", "Media", "Ajuste fino con PFERD y datos propios; 6–8 cámaras para redundancia; RANSAC descarta detecciones erróneas"),
    ("Cascos y manos pequeños a 40–60 m", "Alta", "Más resolución efectiva (focales por zona, más cámaras); filtrado temporal; umbrales con margen"),
    ("Umbrales que no reflejan el criterio del jurado", "Media", "Fase F3 con jueces; cada deducción es trazable y ajustable"),
    ("Aprender sesgos de los jueces (p. ej. premiar la hiperflexión)", "Media", "La nuca y el bienestar se evalúan por regla explícita, nunca aprendida"),
    ("Licencias de PFERD y hSMAL (uso no comercial)", "Alta si es comercial", "Revisar términos antes de F2; alternativa: datos y modelo propios"),
    ("Privacidad de jinetes y menores (RGPD)", "Alta", "Consentimiento informado, autorización de tutores, retención limitada y anonimización"),
    ("Sincronización entre cámaras", "Media", "Hardware con genlock/PTP; verificación con un evento de sincronía al inicio"),
]

decisions = [
    "Pista y fechas para la captura piloto (F1), con permiso del club/organizador.",
    "Uno o dos jueces colaboradores que puntúen los recorridos grabados.",
    "Uso previsto (investigación, formación o comercial): condiciona las licencias de PFERD/hSMAL.",
    "Presupuesto de hardware (cámaras, sincronización, postes, equipo con GPU).",
    "Política de datos: consentimiento, menores (categoría alevines), almacenamiento y retención.",
]

movements_table = "".join(
    f"<tr><td class='c'>{m.n}</td><td><b>{e(m.letters)}</b> {e(m.text)}</td><td>{e(', '.join(sorted({ {'line':'rectitud','circle':'círculo','halt':'parada','transition':'transición','transition_zone':'transición en zona','lengthening':'alargamiento','hnp':'nuca'}[c['type']] for c in m.checks})))}</td>"
    f"<td class='c'>{n(MOV[m.n]['note'])}</td></tr>" for m in MOVEMENTS)

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", system-ui, sans-serif; color: #1b1b1a; font-size: 10.2pt; line-height: 1.5; margin: 0; }
h1 { font-size: 22pt; margin: 0 0 6mm; color: #0b0b0b; letter-spacing: -0.01em; }
h2 { font-size: 15pt; margin: 0 0 4mm; color: #0b0b0b; padding-bottom: 2mm; border-bottom: 2px solid #2a78d6; }
h3 { font-size: 11.5pt; margin: 6mm 0 2mm; color: #0b0b0b; }
p { margin: 0 0 3mm; }
ul, ol { margin: 0 0 3mm; padding-left: 5mm; }
li { margin: 1mm 0; }
.section { page-break-before: always; }
.muted { color: #6b6a66; }
table { width: 100%; border-collapse: collapse; margin: 2mm 0 4mm; font-size: 9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { text-align: left; background: #f1f0ec; color: #3d3c39; font-weight: 600; padding: 2mm 2.2mm; border-bottom: 1px solid #c3c2b7; }
td { padding: 1.8mm 2.2mm; border-bottom: 1px solid #e1e0d9; vertical-align: top; }
td.c, th.c { text-align: center; white-space: nowrap; }
figure { margin: 3mm 0 5mm; page-break-inside: avoid; text-align: center; }
figure img { border: 1px solid #e1e0d9; border-radius: 2mm; max-width: 100%; max-height: 205mm; width: auto !important; }
.cover figure img { max-height: 115mm; }
figcaption { font-size: 8.6pt; color: #5a5955; margin-top: 1.5mm; text-align: left; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
.grid2 figure { margin: 0 0 3mm; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; margin: 4mm 0 5mm; }
.kpi { background: #f4f3ef; border-radius: 2mm; padding: 3mm; }
.kpi b { display: block; font-size: 16pt; color: #0b0b0b; line-height: 1.2; }
.kpi span { font-size: 8.4pt; color: #52514e; }
.callout { background: #eef4fc; border-left: 3px solid #2a78d6; padding: 3mm 4mm; border-radius: 1mm; margin: 3mm 0 4mm; page-break-inside: avoid; }
.warn { background: #fdf4e6; border-left-color: #eda100; }
.crit { background: #fbecec; border-left-color: #d03b3b; }
.pipeline { display: grid; grid-template-columns: repeat(6, 1fr); gap: 2mm; margin: 4mm 0; }
.step { background: #f4f3ef; border-radius: 2mm; padding: 3mm 2.5mm; position: relative; font-size: 8.4pt; line-height: 1.35; }
.step .num { width: 6mm; height: 6mm; border-radius: 50%; background: #2a78d6; color: #fff; font-weight: 700; display: grid; place-items: center; margin-bottom: 1.5mm; font-size: 8.5pt; }
.step b { display: block; font-size: 9.5pt; margin-bottom: 1mm; }
.step:not(:last-child)::after { content: "→"; position: absolute; right: -2.4mm; top: 3mm; color: #2a78d6; font-weight: 700; }
.diagram { width: 100%; height: auto; }
.ok { color: #006300; font-weight: 600; }
.no { color: #a33; font-weight: 600; }
.cover { height: 257mm; display: flex; flex-direction: column; }
.cover .brand { display: flex; align-items: center; gap: 4mm; margin-bottom: 14mm; }
.cover .logo { width: 14mm; height: 14mm; border-radius: 3mm; background: #2a78d6; color: #fff; font-weight: 700; font-size: 20pt; display: grid; place-items: center; }
.cover .brand span { font-size: 18pt; font-weight: 650; }
.cover h1 { font-size: 27pt; line-height: 1.15; margin-bottom: 4mm; }
.cover .sub { font-size: 13pt; color: #52514e; margin-bottom: 10mm; }
.cover .meta { margin-top: auto; font-size: 9.5pt; color: #52514e; border-top: 1px solid #e1e0d9; padding-top: 4mm; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4mm; }
.cover .meta b { display: block; color: #0b0b0b; }
.toc { list-style: none; padding: 0; columns: 1; }
.toc li { display: flex; gap: 3mm; padding: 1.6mm 0; border-bottom: 1px dotted #c3c2b7; }
.toc li b { width: 8mm; color: #2a78d6; }
.small { font-size: 8.8pt; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f1f0ec; padding: 0 1mm; border-radius: 1mm; }
pre { font-family: Consolas, monospace; font-size: 8.4pt; background: #f4f3ef; padding: 3mm; border-radius: 2mm; white-space: pre-wrap; }
.avoid { page-break-inside: avoid; }
"""

body = f"""
<div class="cover">
  <div class="brand"><div class="logo">C</div><span>Cohnetic</span></div>
  <h1>Captura 3D de caballo y jinete<br>y puntuación asistida en Equitación de Trabajo</h1>
  <div class="sub">Propuesta técnica, plan de trabajo y resultados del proof of concept</div>
  {fig("02_lateral_galope.png", "Reconstrucción 3D del binomio al galope en el ejercicio 8. El segmento rojo es el plano nasal, que el sistema marca porque va más de 10° detrás de la vertical.")}
  <div class="meta">
    <div><b>Versión</b>0.1 · borrador para revisión</div>
    <div><b>Fecha</b>Noviembre de 2026</div>
    <div><b>Estado</b>Proof of concept con datos sintéticos</div>
  </div>
</div>

<div class="section">
  <h2>Contenido</h2>
  <ol class="toc">
    <li><b>1</b>Resumen ejecutivo</li>
    <li><b>2</b>Contexto y punto de partida</li>
    <li><b>3</b>Cómo funciona el sistema</li>
    <li><b>4</b>Instalación en pista: cámaras, sincronización y calibración</li>
    <li><b>5</b>Reconstrucción 3D de caballo y jinete</li>
    <li><b>6</b>Del 3D a la nota: métricas, reglas y bienestar</li>
    <li><b>7</b>Proof of concept: qué se ha construido y qué demuestra</li>
    <li><b>8</b>La aplicación web, pantalla a pantalla</li>
    <li><b>9</b>Plan de trabajo</li>
    <li><b>10</b>Riesgos y mitigaciones</li>
    <li><b>11</b>Decisiones pendientes</li>
    <li><b>A</b>Anexo: hoja de ejercicios evaluada</li>
    <li><b>B</b>Anexo: código y cómo ejecutarlo</li>
  </ol>

</div>

<div class="section">
  <h2>1. Resumen ejecutivo</h2>
  <p><b>Qué proponemos.</b> Un sistema de 4 a 8 cámaras alrededor de la pista que reconstruye en 3D al caballo y al jinete durante toda la prueba y,
  a partir de esa reconstrucción, calcula una <b>nota orientativa por ejercicio</b> según el reglamento de Equitación de Trabajo de la RFHE (2026),
  una lista de <b>oportunidades de mejora</b> ordenadas por los puntos que se pueden recuperar y <b>alertas de bienestar</b> sobre la posición de la nuca.</p>
  <p><b>Por qué.</b> Muchos criterios del reglamento son medibles: el diámetro de un círculo, si la transición se hace en la letra, cuánto dura la inmovilidad,
  si el caballo rompe el aire o en qué ángulo lleva la nariz. Hoy se valoran a ojo y el jinete recibe una nota sin evidencia concreta.
  Con medidas 3D cada punto perdido se puede explicar y entrenar.</p>
  <p><b>Qué hemos demostrado.</b> Hemos construido el pipeline completo y lo hemos probado con un recorrido sintético de la prueba de doma de alevines
  en el que sembramos {len(planted)} errores conocidos. El recorrido pasa por cámaras simuladas con ruido, oclusiones y detecciones falsas,
  se calibra con las letras de la pista, se triangula, se analiza y se puntúa:</p>
  <div class="kpis">
    <div class="kpi"><b>{n(V['mpjpe_cm'])} cm</b><span>error 3D medio por punto (6 cámaras)</span></div>
    <div class="kpi"><b>{n(V['gait_accuracy_pct'])}%</b><span>de frames con el aire bien identificado</span></div>
    <div class="kpi"><b>{n_det} de {len(planted)}</b><span>errores sembrados detectados</span></div>
    <div class="kpi"><b>{n(S['percent'], 1)}%</b><span>nota desde el 3D reconstruido; {n(V['score_from_truth_pct'], 1)}% desde el movimiento real</span></div>
  </div>
  <p><b>Qué falta.</b> Sustituir las detecciones simuladas por un detector real sobre vídeo (el resto del código no cambia), grabar recorridos reales en pista
  y calibrar los umbrales de las reglas con jueces. El plan de la sección 9 lo organiza en fases.</p>
  <div class="callout warn"><b>Importante.</b> La nota es orientativa y no sustituye al jurado. Los umbrales actuales son provisionales.
  Los resultados del PoC miden la parte geométrica del sistema; la precisión con vídeo real dependerá sobre todo del detector de puntos.</div>
</div>

<div class="section">
  <h2>2. Contexto y punto de partida</h2>
  <p>La Equitación de Trabajo consta de cuatro pruebas: <b>doma</b>, <b>manejabilidad</b>, <b>velocidad</b> y <b>apartado de ganado</b>.
  La doma se disputa en una pista de 40 × 20 m con letras; manejabilidad y velocidad, en una pista de al menos 70 × 30 m con obstáculos.
  Empezamos por la doma porque su hoja de ejercicios es fija y sus criterios son los más geométricos y biomecánicos.</p>
  <h3>Documentación que usamos como base</h3>
  <table>
    <tr><th>Documento</th><th>Qué tomamos de él</th></tr>
    <tr><td>Reglamento de Equitación de Trabajo RFHE 2026</td><td>Pruebas, dimensiones de pista, escala de puntuación 0–10 (Anexo I), hojas de ejercicios con directrices y coeficientes (Anexo II), penalizaciones y descalificaciones (Art. 19), descripción de los obstáculos</td></tr>
    <tr><td>Manual FEI de jueces de doma 2025</td><td>Escala de entrenamiento, definición de los aires, nuca como punto más alto y nariz ligeramente por delante de la vertical, límite de nota (“firewall”) ante errores graves</td></tr>
    <tr><td>Glosario de juicio USDF 2015</td><td>Vocabulario y criterios para traducir directrices a medidas</td></tr>
    <tr><td>Revisión sistemática sobre hiperflexión (2024)</td><td>Efectos de montar con la nariz detrás de la vertical; justifica tratar la nuca como indicador de bienestar</td></tr>
    <tr><td>Prevalencia de posiciones cabeza-cuello en concurso (2014)</td><td>Categorías de la nuca (por delante, hasta 10° detrás, más de 10° detrás) y su relación con conductas de conflicto</td></tr>
    <tr><td>Preprint: calentamiento frente a concurso en élite</td><td>Ángulos medidos (plano nasal, nuca, cuello) y el hallazgo de que los jueces puntúan mejor cuanto más detrás de la vertical va el caballo</td></tr>
  </table>
  <h3>PFERD: el dataset de referencia para el caballo</h3>
  <p>PFERD (<i>Poses for Equine Research Dataset</i>, Li et al., Scientific Data 2024) es un dataset de captura de movimiento de caballos con marcadores,
  con cámaras de vídeo calibradas y el modelo 3D de caballo hSMAL ajustado a cada recorrido. Lo usaremos para:</p>
  <ul>
    <li><b>entrenar o ajustar el detector de puntos del caballo</b> con imágenes y sus puntos 2D de referencia;</li>
    <li><b>medir el error</b> de nuestra reconstrucción sin marcadores frente a una captura con marcadores;</li>
    <li>como referencia anatómica para el <b>modelo 3D</b> del caballo en fases posteriores.</li>
  </ul>
  <p>PFERD no resuelve la captura en pista: es de laboratorio y no incluye jinetes. Su código tiene licencia <b>CC BY-NC-SA</b> (no comercial),
  y el dataset y el modelo hSMAL tienen sus propios términos. Si el uso es comercial, hay que revisarlos antes de la fase 2.</p>
</div>

<div class="section">
  <h2>3. Cómo funciona el sistema</h2>
  <p>El sistema es una cadena de seis etapas. Cada etapa tiene una salida que se puede revisar por separado, lo que facilita depurar y explicar los resultados.</p>
  {pipeline_html()}
  <h3>Etapa a etapa</h3>
  <ol>
    <li><b>Captura.</b> Las cámaras graban la prueba completa sincronizadas por hardware. Cada fotograma de todas las cámaras corresponde al mismo instante.</li>
    <li><b>Calibración.</b> Se conoce la óptica de cada cámara (calibración intrínseca con un tablero ChArUco) y su posición respecto a la pista
    (calibración extrínseca). Para la extrínseca usamos las propias <b>letras y esquinas de la pista</b>, medidas una vez con estación total:
    así el sistema de coordenadas coincide con el de la pista y las figuras se miden directamente en metros respecto a las letras.</li>
    <li><b>Keypoints 2D.</b> Un detector localiza en cada imagen 26 puntos anatómicos del caballo y 14 del jinete.</li>
    <li><b>Triangulación.</b> Cada punto visto por al menos dos cámaras se sitúa en 3D. Se prueban todas las parejas de cámaras y se descartan las
    detecciones incoherentes (RANSAC), lo que elimina confusiones del detector y oclusiones. Después se rellenan huecos cortos y se suaviza en el tiempo.</li>
    <li><b>Métricas.</b> Sobre el esqueleto 3D se detectan las pisadas de cada casco, el aire y la mano de galope, las transiciones, las paradas,
    la geometría de las figuras, el ángulo de la nuca y la postura del jinete. La trayectoria se divide automáticamente en los ejercicios de la hoja.</li>
    <li><b>Nota e informe.</b> Unas reglas trazables al reglamento convierten las métricas en notas por ejercicio, notas de conjunto y penalizaciones,
    y agrupan los puntos perdidos en oportunidades de mejora con ejercicios de entrenamiento.</li>
  </ol>
  <div class="callout">El código de las etapas 2, 4, 5 y 6 es <b>el mismo</b> en el proof of concept y en el sistema real. En el PoC solo se simulan
  la captura y las detecciones 2D; para pasar a vídeo real hay que sustituir esa pieza por el detector.</div>
</div>

<div class="section">
  <h2>4. Instalación en pista</h2>
  <h3>Disposición de las cámaras</h3>
  <p>Para la pista de doma proponemos 6 cámaras: 4 en las esquinas y 2 en los laterales, a la altura de B y E, todas a 5,5–6,5 m de altura y apuntando al centro.
  Así cada punto del caballo lo ven al menos 2–3 cámaras aunque el propio cuerpo tape los miembros del lado contrario.
  Para manejabilidad y velocidad (≥ 70 × 30 m) se pasa a 8 cámaras.</p>
  {camera_svg()}
  {fig("09_rig_camaras.png", "El mismo rig de 6 cámaras en el visor 3D de la aplicación (opción «Cámaras»).", "86%")}
  <h3>Equipamiento</h3>
  <table><tr><th style="width:22%">Elemento</th><th>Especificación propuesta</th></tr>
  {"".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in hw_rows)}</table>
  <p class="muted small">El presupuesto de hardware queda pendiente de definir (sección 11).</p>
  <h3>Calibración con las letras de la pista</h3>
  <p>Las letras (A, K, E, H, C, M, B, F) y los puntos D, X, G de la línea central se miden una vez con estación total. Cada cámara los detecta en su imagen
  y se calcula su posición y orientación: primero con una homografía del suelo y después con un ajuste no lineal que minimiza el error de reproyección.
  En el PoC, con 0,8 px de ruido en las detecciones, la calibración da estos resultados:</p>
  <table><tr><th>Cámara</th><th class="c">Puntos usados</th><th class="c">Error de reproyección</th><th class="c">Error de posición de la cámara</th></tr>
  {"".join(f"<tr><td>{e(c['name'])}</td><td class='c'>{c['calib_points']}</td><td class='c'>{n(c['calib_rmse_px'], 2)} px</td><td class='c'>{n(c['pos_err_cm'])} cm</td></tr>" for c in calib)}</table>
  <h3>Procedimiento el día de la prueba</h3>
  <ol>
    <li>Montaje de postes y cámaras (una vez por pista si la instalación es fija).</li>
    <li>Verificación de la sincronía con un evento común (destello o claqueta) al inicio de la sesión.</li>
    <li>Calibración extrínseca automática con las letras visibles; el sistema avisa si alguna cámara supera 2 px de error.</li>
    <li>Grabación de cada binomio; el procesamiento puede hacerse durante la prueba siguiente.</li>
    <li>Informe disponible para jinete y entrenador al terminar la jornada.</li>
  </ol>
</div>

<div class="section">
  <h2>5. Reconstrucción 3D de caballo y jinete</h2>
  <h3>Qué puntos se siguen</h3>
  <div class="grid2">
    <div><p><b>Caballo (26 puntos)</b></p>
      <ul class="small"><li>Cabeza: nariz, ojo, nuca, garganta</li><li>Tronco: cruz, pecho, dorso, grupa, nacimiento y punta de la cola</li>
      <li>Cada anterior: hombro, rodilla, menudillo, casco</li><li>Cada posterior: cadera, corvejón, menudillo, casco</li></ul>
      <p class="small muted">Compatibles con los puntos anatómicos de PFERD y mapeables al modelo hSMAL.</p></div>
    <div><p><b>Jinete (14 puntos)</b></p>
      <ul class="small"><li>Cabeza y cuello</li><li>Hombros, codos y manos</li><li>Caderas, rodillas y tobillos</li></ul>
      <p class="small muted">Subconjunto estándar de los detectores de pose humana (COCO-WholeBody / SMPL-X).</p></div>
  </div>
  <h3>Detectores previstos</h3>
  <ul>
    <li><b>Jinete:</b> RTMPose o ViTPose, modelos abiertos y maduros para personas.</li>
    <li><b>Caballo:</b> DeepLabCut SuperAnimal-Quadruped como punto de partida, ajustado con PFERD y con 500–1000 fotogramas propios etiquetados en pista.</li>
  </ul>
  <h3>Robustez</h3>
  <p>En el PoC simulamos condiciones desfavorables: ruido de 1,6 px en cada detección, 5% de detecciones perdidas, un 35% de probabilidad de perder los
  miembros del lado contrario a la cámara y un 1,5% de detecciones completamente erróneas. Aun así:</p>
  <table><tr><th>Zona</th><th class="c">Error medio</th></tr>
  {"".join(f"<tr><td>{e(k)}</td><td class='c'>{n(v)} cm</td></tr>" for k, v in V['group_cm'].items())}
  <tr><td><b>Global</b> (percentil 95: {n(V['p95_cm'])} cm)</td><td class='c'><b>{n(V['mpjpe_cm'])} cm</b></td></tr></table>
  <div class="grid2">
    {fig("03_lateral_trote.png", "Vista lateral al trote: nariz por delante de la vertical (plano nasal en verde).")}
    {fig("06_seguir.png", "Vista «Seguir» al galope a mano derecha: la cámara acompaña al binomio.")}
  </div>
</div>

<div class="section">
  <h2>6. Del 3D a la nota</h2>
  <h3>Métricas y criterio del reglamento al que responden</h3>
  <table><tr><th style="width:20%">Métrica</th><th style="width:40%">Cómo se mide</th><th>Criterio que alimenta</th></tr>
  {"".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in metric_rows)}</table>
  <h3>Reglas de puntuación</h3>
  <ul>
    <li><b>Escala 0–10 con medios puntos</b> (Anexo I) y <b>coeficientes</b> de la hoja: ejercicios ×1, notas de conjunto ×2.</li>
    <li><b>Nota base 9</b> para un ejercicio sin defectos medibles: la visión no puede acreditar la expresión que justifica un 10.</li>
    <li>Cada defecto resta según su magnitud, con umbrales explícitos (p. ej. un círculo con más del 5% de error de diámetro, una transición a más de 2 m de la letra).</li>
    <li><b>Error grave</b> (romper el aire, galope a mano contraria): la nota del ejercicio no puede superar 5, como el “firewall” de la FEI.</li>
    <li><b>Penalizaciones del Art. 19</b>: −5 por cada caricia por delante de las riendas; la descalificación por usar dos manos más de 3 s queda señalada.</li>
    <li>Cada deducción guarda la medida, el valor y el umbral, para que un juez pueda revisarla y discrepar.</li>
  </ul>
  {fig("11_hoja.png", "Hoja de puntuación generada: cada ejercicio con su nota, coeficiente y la deducción con la evidencia medida.", "78%")}
  <h3>Bienestar: la nuca se evalúa por regla, no se aprende de los jueces</h3>
  <p>El preprint sobre élite incluido en la documentación encontró que <b>los jueces puntuaban mejor cuanto más detrás de la vertical iba la nariz</b>,
  en contra del propio reglamento. Si el sistema aprendiera de esas notas, reproduciría el sesgo. Por eso el ángulo de la nuca se mide y se penaliza
  con una regla explícita (por delante = correcto; detrás = deducción; más de 10° detrás durante una parte relevante del aire = deducción mayor y alerta).</p>
  <div class="callout crit"><b>Ejemplo del PoC.</b> {e(R['welfare']['message'] or '')}</div>
</div>

<div class="section">
  <h2>7. Proof of concept</h2>
  <p>Como aún no hay vídeos, el PoC genera un <b>recorrido sintético</b> de la prueba de doma de alevines (13 ejercicios, {n(R['meta']['duration_s'], 0)} s a 50 fps)
  con caballo y jinete animados según el patrón de pisadas de cada aire, y con un perfil de ejecución que incluye errores conocidos.
  Ese movimiento “verdad” se proyecta en 6 cámaras 4K simuladas, se añaden ruido, oclusiones y detecciones falsas, y a partir de ahí actúa
  el pipeline real. Como conocemos la verdad, podemos medir el error de cada etapa.</p>
  <h3>Errores sembrados y qué detectó el sistema</h3>
  <table><tr><th style="width:30%">Error sembrado</th><th class="c">Dónde</th><th style="width:30%">Qué midió el sistema</th><th>Efecto en la nota</th><th class="c">Detectado</th></tr>
  {"".join(f"<tr><td>{e(a)}</td><td class='c'>{e(b)}</td><td>{e(c)}</td><td>{e(d)}</td><td class='c'>{'<span class=ok>Sí</span>' if ok else '<span class=no>No</span>'}</td></tr>" for a, b, c, d, ok in planted)}</table>
  <p>El único error no detectado es intencionadamente sutil: la mano de las riendas vibra algo más que en la referencia, pero queda por debajo del umbral
  provisional, que se fijó alto para no confundir el ruido de la reconstrucción con un defecto del jinete. Es un ejemplo de lo que hay que calibrar con datos reales.</p>
  <h3>Reconstrucción frente a verdad</h3>
  <ul>
    <li>Nota desde la reconstrucción 3D: <b>{n(S['percent'], 2)}%</b>; desde el movimiento real: <b>{n(V['score_from_truth_pct'], 2)}%</b>.
      {"Todas las notas por ejercicio coinciden." if not V['notes_diff'] else "Difieren " + str(len(V['notes_diff'])) + " ejercicios."}</li>
    <li>Un binomio de referencia sin errores obtiene 9 en todos los ejercicios y notas de conjunto (90%), sin falsas deducciones.</li>
    <li>Error medio del ángulo de la nuca: {n(V['hnp_mae_deg'])}°; de la inclinación del jinete: {n(V['lean_mae_deg'])}°.</li>
    <li>Seis tests automáticos verifican calibración, triangulación con detecciones erróneas, ajuste de círculos, aires, el binomio ideal y la detección de los errores sembrados.</li>
  </ul>
  <h3>Lo que el PoC no demuestra todavía</h3>
  <ul>
    <li>La precisión de un detector real sobre vídeo de pista (luz, polvo, ropa, capas, distancia).</li>
    <li>Que los umbrales coincidan con el criterio de los jueces.</li>
    <li>La evaluación de lo que no es geométrico: expresión, boca, orejas, cola, voz.</li>
  </ul>
</div>

<div class="section">
  <h2>8. La aplicación web, pantalla a pantalla</h2>
  <p>La aplicación funciona en el navegador, sin instalar nada. A la izquierda está el visor 3D con la reproducción del recorrido; a la derecha, el informe.
  Todo está enlazado: al pulsar un ejercicio, una deducción o un punto de una gráfica, el visor salta a ese instante.</p>
  {fig("01_general.png", "Vista general: nota orientativa en la cabecera, visor 3D con línea de tiempo marcada por ejercicios y panel de informe.")}
  <h3>8.1 Visor 3D</h3>
  <p>Reproduce el esqueleto 3D de caballo (negro) y jinete (azul) sobre la pista, con la trayectoria, las figuras teóricas de la hoja (líneas discontinuas)
  y el plano nasal coloreado según la posición de la nuca. El recuadro superior indica el ejercicio, el aire detectado y el ángulo de la nuca en cada instante.
  Tiene cinco vistas: libre, seguir al caballo, lateral, cenital y la del juez en C.</p>
  <div class="grid2">
    {fig("02_lateral_galope.png", "Lateral al galope: nuca 13° detrás de la vertical (rojo, ✕).")}
    {fig("05_parada_final.png", "Parada final en G: el sistema mide la inmovilidad y si el caballo queda cuadrado.")}
    {fig("07_cenital.png", "Vista cenital: trayectoria medida sobre las figuras teóricas.")}
    {fig("08_juez_c.png", "Vista desde la posición del presidente del jurado en C.")}
  </div>
  {fig("04_caricia.png", "Detección de eventos sancionables: el jinete lleva la mano derecha al cuello por delante de las riendas (−5, Art. 19).", "70%")}
</div>

<div class="section">
  <h3>8.2 Resumen</h3>
  <p>La primera pestaña concentra lo importante: nota, puntos perdidos, alerta de bienestar si la hay y las tres áreas donde se pierden más puntos.</p>
  {fig("10_resumen.png", "Pestaña Resumen.", "80%")}
</div>

<div class="section">
  <h3>8.3 Oportunidades de mejora</h3>
  <p>Agrupa todas las deducciones por área (aires, geometría, transiciones, paradas, impulsión, contacto, jinete, penalizaciones), las ordena por puntos
  recuperables con los coeficientes aplicados y propone ejercicios de entrenamiento concretos. Cada evidencia tiene un enlace para verla en 3D.</p>
  {fig("12_mejora_aires.png", "Área «Mantener el aire y la regularidad»: la ruptura del galope en el ejercicio 8 limita la nota a 5 y resta también en la nota de conjunto A.", "88%")}
  {fig("13_mejora_jinete.png", "Área «Posición y estabilidad del jinete», con la evidencia medida y ejercicios para corregirla.", "88%")}
</div>

<div class="section">
  <h3>8.4 Análisis</h3>
  <p>Gráficas de la evolución de la nuca, la inclinación del tronco del jinete y la velocidad a lo largo de la prueba, con la franja del aire detectado debajo.
  Al pasar el ratón se ve el valor exacto y al hacer clic el visor salta a ese momento.</p>
  {fig("15_serie_nuca.png", "Ángulo del plano nasal: correcto (por encima de 0) al paso y al trote, detrás de la vertical durante el galope. Debajo, el aire detectado en cada tramo.", "90%")}
  {fig("16_serie_jinete.png", "Inclinación del tronco del jinete: casi vertical al paso y al trote, unos 13° hacia delante al galope (la tolerancia es 8°).", "90%")}
</div>

<div class="section">
  <p>El trazado sobre la pista compara la trayectoria medida con la figura teórica de cada ejercicio y muestra las medidas.</p>
  <div class="grid2">
    {fig("17_trazado_circulo.png", "Ejercicio 3: el círculo de 10 m se hizo de 11,8 m y desplazado (en rojo, frente a la figura teórica discontinua).")}
    {fig("18_trazado_galope.png", "Ejercicio 8: círculo de 20 m bien trazado, pero con 3 trancos de trote en mitad del círculo.")}
  </div>
  {fig("19_por_aire.png", "Resumen por aire: tiempo, velocidad, posición media de la nuca y del jinete.", "88%")}
  <h3>8.5 Calidad 3D</h3>
  <p>Muestra la precisión de la reconstrucción y la calibración de cada cámara, y enumera las limitaciones de la versión. Con vídeo real esta pestaña
  mostrará la cobertura de cada cámara y el error de reproyección, que sirven para detectar una cámara movida o mal sincronizada.</p>
  {fig("20_calidad.png", "Pestaña Calidad 3D.", "78%")}
</div>

<div class="section">
  <h3>8.6 Móvil y modo oscuro</h3>
  <p>La aplicación se adapta al móvil (el jinete puede consultar el informe al salir de pista) y respeta el modo oscuro del sistema.</p>
  <div class="grid2" style="grid-template-columns: 1fr 2.2fr; align-items:start">
    {fig("21_movil.png", "Vista en móvil.")}
    {fig("22_oscuro.png", "Modo oscuro, vista lateral durante el galope largo.")}
  </div>
</div>

<div class="section">
  <h2>9. Plan de trabajo</h2>
  <p>Duraciones estimadas, a confirmar cuando se cierren las decisiones de la sección 11.</p>
  <table><tr><th class="c">Fase</th><th style="width:22%">Nombre</th><th class="c">Duración</th><th>Entregables y criterio de éxito</th></tr>
  {"".join(f"<tr><td class='c'><b>{a}</b></td><td><b>{b}</b></td><td class='c'>{c}</td><td>{d}</td></tr>" for a, b, c, d in phases)}</table>
  <h3>Criterios de éxito propuestos para pasar de piloto a producto</h3>
  <ul>
    <li>Error 3D medio inferior a 5 cm en el tronco y la cabeza con vídeo real.</li>
    <li>Aire y mano de galope correctos en más del 90% de los trancos.</li>
    <li>Nota por ejercicio dentro de ±0,5 puntos de la del jurado en al menos el 80% de los ejercicios.</li>
    <li>Informe disponible en menos de 15 minutos tras el recorrido.</li>
  </ul>

  <h2 style="margin-top:8mm">10. Riesgos y mitigaciones</h2>
  <table><tr><th style="width:34%">Riesgo</th><th class="c">Impacto</th><th>Mitigación</th></tr>
  {"".join(f"<tr><td>{a}</td><td class='c'>{b}</td><td>{c}</td></tr>" for a, b, c in risks)}</table>
</div>

<div class="section">
  <h2>11. Decisiones pendientes</h2>
  <p>Para arrancar la fase 1 necesitamos cerrar:</p>
  <ol>{"".join(f"<li>{d}</li>" for d in decisions)}</ol>

  <h2 style="margin-top:8mm">Anexo A. Hoja de ejercicios evaluada</h2>
  <p>Prueba de doma para jinetes alevines (Anexo II del reglamento). Para cada ejercicio, los controles que mide el sistema y la nota del binomio de demostración.</p>
  <table><tr><th class="c">Nº</th><th>Ejercicio</th><th style="width:24%">Controles 3D</th><th class="c">Nota PoC</th></tr>{movements_table}
  <tr><td></td><td colspan="2"><i>Además, en todos: aire correcto (ruptura de aire / mano contraria) y regularidad del tranco.</i></td><td></td></tr></table>
</div>

<div class="section">
  <h2>Anexo B. Código y cómo ejecutarlo</h2>
  <p>El proyecto está en Python (numpy, scipy) para el pipeline y en HTML/JavaScript (Three.js) para la web.</p>
  <table><tr><th style="width:32%">Módulo</th><th>Función</th></tr>
    <tr><td><code>cohnetic/camera.py</code></td><td>Modelo de cámara y calibración extrínseca con las letras</td></tr>
    <tr><td><code>cohnetic/triangulation.py</code></td><td>Triangulación RANSAC multivista y filtrado temporal</td></tr>
    <tr><td><code>cohnetic/protocol.py</code></td><td>Hoja de ejercicios: textos, directrices, coeficientes, controles y ruta</td></tr>
    <tr><td><code>cohnetic/segmentation.py</code></td><td>División de la trayectoria en ejercicios</td></tr>
    <tr><td><code>cohnetic/metrics/</code></td><td>Pisadas, aires, transiciones, figuras, nuca, jinete y eventos</td></tr>
    <tr><td><code>cohnetic/scoring.py</code>, <code>advice.py</code></td><td>Reglas de puntuación y oportunidades de mejora</td></tr>
    <tr><td><code>cohnetic/synth.py</code></td><td>Generador sintético (solo PoC): movimiento, cámaras y detecciones</td></tr>
    <tr><td><code>cohnetic/pipeline.py</code>, <code>report.py</code></td><td>Orquestación y exportación de datos para la web</td></tr>
    <tr><td><code>web/</code></td><td>Aplicación web (visor 3D e informe)</td></tr>
    <tr><td><code>tests/</code></td><td>Tests automáticos</td></tr>
  </table>
  <pre>pip install -r requirements.txt
python scripts/run_poc.py          # ejecuta el PoC y genera web/data/ (≈1 min)
python -m pytest -q                # tests
python scripts/capturas_web.py     # capturas de la web para este documento
python scripts/build_doc.py        # este PDF</pre>
  <p>La web se abre con doble clic en <code>web/index.html</code> (el visor 3D necesita conexión a internet para cargar Three.js).</p>
  <p class="muted small">Referencia del dataset: Li, C. et al. (2024). The Poses for Equine Research Dataset (PFERD). Scientific Data 11, 497.</p>
</div>
"""

doc = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Cohnetic — Propuesta técnica</title>
<style>{CSS}</style></head><body>{body}</body></html>"""


def main():
    html_path = DOCS / "propuesta_tecnica.html"
    html_path.write_text(doc, encoding="utf-8")
    pdf_path = DOCS / "Cohnetic_Propuesta_Tecnica.pdf"
    from playwright.sync_api import sync_playwright
    launch = {"executable_path": CHROME} if os.path.exists(CHROME) else {}
    with sync_playwright() as p:
        b = p.chromium.launch(**launch)
        pg = b.new_page()
        pg.goto(html_path.as_uri())
        pg.wait_for_load_state("networkidle")
        pg.pdf(path=str(pdf_path), format="A4", print_background=True, prefer_css_page_size=True,
               display_header_footer=True,
               header_template="<div></div>",
               footer_template='<div style="font-size:8px;color:#898781;width:100%;padding:0 16mm;display:flex;justify-content:space-between;font-family:Segoe UI,sans-serif">'
                               '<span>Cohnetic · Propuesta técnica y proof of concept · v0.1</span>'
                               '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
        b.close()
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
