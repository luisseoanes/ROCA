"""Hoja de ejercicios: Prueba de doma para jinetes alevines (Anexo II, RFHE 2026).

Contiene:
  * MOVEMENTS: filas de la hoja (texto, directrices, coeficiente) + los
    controles medibles en 3D asociados a cada ejercicio.
  * COLLECTIVE: notas de conjunto (coeficiente 2).
  * build_route(profile): ruta (polilínea + aire) que describe la prueba.
    Con el perfil ideal es la referencia; el generador sintético la usa con
    un perfil "real" que incluye errores.
  * BOUNDARIES: puntos de paso para segmentar la trayectoria en ejercicios.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRACK_INSET = 0.6  # la pisada va ~0.6 m dentro del cercado
T = TRACK_INSET
YT = 10.0 - T

# Letras proyectadas sobre la pista de trabajo (por donde pasa el caballo)
TP = {
    "A": (T, 0.0), "C": (40.0 - T, 0.0), "X": (20.0, 0.0), "G": (34.0, 0.0), "D": (6.0, 0.0),
    "K": (6.0, YT), "E": (20.0, YT), "H": (34.0, YT),
    "F": (6.0, -YT), "B": (20.0, -YT), "M": (34.0, -YT),
    "cAK": (T, YT), "cAF": (T, -YT), "cCH": (40.0 - T, YT), "cCM": (40.0 - T, -YT),
}

GAITS = ("halt", "walk", "trot", "canter_L", "canter_R")
GAIT_ES = {"halt": "parada", "walk": "paso", "trot": "trote", "canter_L": "galope a mano izquierda",
           "canter_R": "galope a mano derecha", "canter": "galope", "unknown": "indeterminado"}


@dataclass
class Movement:
    n: int
    letters: str
    text: str
    directives: str
    coef: int = 1
    checks: list[dict] = field(default_factory=list)
    gaits: tuple[str, ...] | None = None  # aires permitidos en el ejercicio (None = varios)


MOVEMENTS: list[Movement] = [
    Movement(1, "A·X·C", "Entrada a trote de trabajo. X parada, inmovilidad 4\" y saludo. Partida al trote. C pista a mano derecha.",
             "La calidad del trote, la rectitud. La parada y las transiciones, la inmovilidad, el contacto y la posición de la nuca.",
             checks=[{"type": "line", "from": "A", "to": "X"}, {"type": "halt", "at": "X", "min_s": 4.0}, {"type": "hnp"}],
             gaits=("trot", "halt")),
    Movement(2, "MXK", "Cambio de mano por diagonal.", "Calidad y regularidad del trote. Rectitud e impulsión.",
             checks=[{"type": "line", "from": "M", "to": "K"}], gaits=("trot",)),
    Movement(3, "A", "Círculo de 10 m de diámetro a la izquierda.",
             "La regularidad y el equilibrio. La incurvación y la calidad en el círculo. Trazado y geometría.",
             checks=[{"type": "circle", "center": (T + 5.0, 0.0), "diameter": 10.0, "dir": "L", "at": "A"}], gaits=("trot",)),
    Movement(4, "FXH", "Cambio de mano por diagonal.", "Calidad y regularidad del trote. Rectitud e impulsión.",
             checks=[{"type": "line", "from": "F", "to": "H"}], gaits=("trot",)),
    Movement(5, "C·M·B", "Círculo de 10 m a la derecha. M caer al paso medio. B doblar a lo ancho.",
             "La regularidad y el equilibrio. La incurvación y la calidad en el círculo. Trazado y geometría. La transición.",
             checks=[{"type": "circle", "center": (40.0 - T - 5.0, 0.0), "diameter": 10.0, "dir": "R", "at": "C"},
                     {"type": "transition", "at": "M", "to": "walk"}], gaits=("trot", "walk")),
    Movement(6, "X·E", "X parada y partir al paso. E pista a la izquierda.",
             "Inmovilidad, caballo cuadrado y contacto. Incurvación al doblar.",
             checks=[{"type": "halt", "at": "X", "min_s": 0.0}, {"type": "hnp"}], gaits=("walk", "halt")),
    Movement(7, "K·F·B·M", "K trote. F caer al paso. B trote. M partir a galope de trabajo a mano izquierda.",
             "Claridad y fluidez en las transiciones. Equilibrio y contacto.",
             checks=[{"type": "transition", "at": "K", "to": "trot"}, {"type": "transition", "at": "F", "to": "walk"},
                     {"type": "transition", "at": "B", "to": "trot"}, {"type": "transition", "at": "M", "to": "canter_L"},
                     {"type": "hnp"}]),
    Movement(8, "CXC", "Círculo de 20 m de diámetro a la izquierda.",
             "La regularidad, el equilibrio. La incurvación y la calidad en el círculo. Trazado y geometría.",
             checks=[{"type": "circle", "center": (40.0 - T - 10.0, 0.0), "diameter": 20.0, "dir": "L", "at": "C"}],
             gaits=("canter_L",)),
    Movement(9, "H-E", "Entre H y E caer al trote.", "Transición.",
             checks=[{"type": "transition_zone", "from": "H", "to_letter": "E", "to": "trot"}], gaits=("canter_L", "trot")),
    Movement(10, "E·B·F", "E doblar a lo ancho. B pista a la derecha. F partir a galope de trabajo a mano derecha.",
             "Incurvación al doblar. Claridad y fluidez en la transición.",
             checks=[{"type": "transition", "at": "F", "to": "canter_R"}], gaits=("trot", "canter_R")),
    Movement(11, "AXA", "Círculo de 20 m de diámetro a la derecha.",
             "La regularidad y el equilibrio. La incurvación y la calidad en el círculo. Trazado y geometría.",
             checks=[{"type": "circle", "center": (T + 10.0, 0.0), "diameter": 20.0, "dir": "R", "at": "A"}], gaits=("canter_R",)),
    Movement(12, "KEH·HCM·B", "KEH galope largo. HCM galope corto. B caer al trote.",
             "Impulsión, claridad y fluidez de las transiciones. Transición y calidad de los aires.",
             checks=[{"type": "lengthening", "from": "K", "to": "H", "ref_from": "H", "ref_to": "M", "min_ratio": 1.2},
                     {"type": "transition", "at": "B", "to": "trot"}], gaits=("canter_R", "trot")),
    Movement(13, "A·X·G", "A doblar por la línea central. X paso medio. G parada, inmovilidad 4\" y saludo.",
             "Transición y calidad de los aires. La rectitud, la parada, la inmovilidad, el contacto y la posición de la nuca.",
             checks=[{"type": "line", "from": "A", "to": "G"}, {"type": "transition", "at": "X", "to": "walk"},
                     {"type": "halt", "at": "G", "min_s": 4.0}, {"type": "hnp"}], gaits=("trot", "walk", "halt")),
]

COLLECTIVE = [
    {"key": "A", "text": "Aires, franqueza y regularidad", "coef": 2},
    {"key": "B", "text": "Impulsión, deseo de avanzar, elasticidad del dorso", "coef": 2},
    {"key": "C", "text": "Sumisión y aceptación de la embocadura. Atención y confianza", "coef": 2},
    {"key": "D", "text": "Colocación y posición del jinete. Empleo y corrección de las ayudas", "coef": 2},
]

# Puntos de paso (vía, punto) que marcan el inicio de cada ejercicio
BOUNDARIES: list[tuple[int, list[tuple[float, float]], tuple[float, float]]] = [
    (1, [], TP["A"]),
    (2, [TP["C"]], TP["M"]),
    (3, [TP["X"]], TP["K"]),
    (4, [(T + 10.0, 0.0)], TP["A"]),
    (5, [TP["X"]], TP["H"]),
    (6, [(40.0 - T - 10.0, 0.0), TP["B"]], TP["X"]),
    (7, [TP["E"]], TP["K"]),
    (8, [TP["B"], TP["M"]], TP["C"]),
    (9, [(40.0 - T - 20.0, 0.0)], TP["H"]),
    (10, [], TP["E"]),
    (11, [TP["F"]], TP["A"]),
    (12, [(T + 20.0, 0.0)], TP["K"]),
    (13, [TP["M"], TP["B"], TP["F"]], TP["A"]),
]


# ---------------------------------------------------------------------------
# Perfil del binomio: parámetros de ejecución. IDEAL = ejecución de referencia.
# ---------------------------------------------------------------------------
IDEAL_PROFILE = {
    "circle3_diam": 10.0, "circle3_dx": 0.0,
    "circle5_diam": 10.0,
    "circle8_diam": 20.0, "circle11_diam": 20.0,
    "walk_at_M_delay": 0.0,
    "canter8_break": None,       # (desde_grados, hasta_grados) en que rompe a trote
    "ext_canter_speed": 5.0,     # m/s en galope largo (trabajo = 4.0)
    "halt1_s": 4.5, "halt6_s": 2.0, "halt13_s": 4.5,
    "halt13_unsquare": 0.0,      # metros que queda retrasado el posterior derecho
    "hnp": {"walk": 6.0, "trot": 5.0, "canter": 4.0},    # grados por delante de la vertical
    "hnp_override": {},          # {mov: grados}
    "lean_canter": 2.0,          # inclinación del tronco del jinete (grados, + hacia delante)
    "lean_other": 1.0,
    "bounce_movs": {},           # {mov: amplitud extra en m} rebote del jinete
    "hand_jitter": 0.012,
    "caress": [],                # [(mov, fracción_inicio, duración_s)]
}

# Perfil de demostración: binomio con errores realistas para el PoC
DEMO_PROFILE = {
    **IDEAL_PROFILE,
    "circle3_diam": 11.8, "circle3_dx": 0.7,
    "walk_at_M_delay": 2.8,
    "canter8_break": (200.0, 255.0),
    "ext_canter_speed": 4.35,
    "halt13_s": 2.6,
    "halt13_unsquare": 0.22,
    "hnp": {"walk": 5.0, "trot": 3.0, "canter": -9.0},
    "hnp_override": {8: -13.0},
    "lean_canter": 13.0,
    "bounce_movs": {7: 0.045},
    "hand_jitter": 0.02,
    "caress": [(10, 0.25, 1.6)],
}


@dataclass
class Segment:
    mov: int
    gait: str
    pts: np.ndarray            # (n,2) polilínea; para 'halt' un único punto
    halt_s: float = 0.0
    speed: float | None = None  # velocidad objetivo específica (p.ej. galope largo)
    tag: str = ""


def _arc(center, r, a0_deg, sweep_deg, step_m=0.5):
    n = max(8, int(abs(np.radians(sweep_deg)) * r / step_m))
    a = np.radians(a0_deg + np.linspace(0, sweep_deg, n + 1))
    return np.column_stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)])


def _pts(*names_or_pts):
    return np.array([TP[p] if isinstance(p, str) else p for p in names_or_pts], dtype=float)


def build_route(profile: dict) -> list[Segment]:
    p = profile
    S = Segment
    segs: list[Segment] = []
    # 1. Entrada por A al trote, parada en X, C pista a mano derecha
    segs += [S(1, "trot", _pts((-8.0, 0.0), "A", "X")),
             S(1, "halt", _pts("X"), halt_s=p["halt1_s"]),
             S(1, "trot", _pts("X", "C", "cCM", "M"))]
    # 2. Diagonal MXK
    segs += [S(2, "trot", _pts("M", "X", "K", "cAK", "A"))]
    # 3. Círculo 10 m a la izquierda en A (sentido antihorario)
    r3 = p["circle3_diam"] / 2
    c3 = (T + r3 + p["circle3_dx"], 0.0)
    start3 = (c3[0] - r3, 0.0)
    segs += [S(3, "trot", np.vstack([_pts("A"), _arc(c3, r3, 180, 360)]), tag="circle3")]
    if np.hypot(start3[0] - T, 0) > 0.05:
        segs[-1].pts = np.vstack([_pts("A"), [start3], _arc(c3, r3, 180, 360)[1:], _pts("A")])
    # 4. A -> F -> diagonal FXH
    segs += [S(4, "trot", _pts("A", "cAF", "F", "X", "H", "cCH", "C"))]
    # 5. Círculo 10 m a la derecha en C; M paso; B doblar a lo ancho
    r5 = p["circle5_diam"] / 2
    segs += [S(5, "trot", _arc((40.0 - T - r5, 0.0), r5, 0, -360), tag="circle5")]
    mdelay = p["walk_at_M_delay"]
    walk_start = (34.0 - mdelay, -YT)
    segs += [S(5, "trot", _pts("C", "cCM", "M", walk_start) if mdelay > 0 else _pts("C", "cCM", "M")),
             S(5, "walk", _pts(walk_start, "B", "X") if mdelay > 0 else _pts("M", "B", "X"))]
    # 6. X parada y partir al paso; E pista a la izquierda
    segs += [S(6, "halt", _pts("X"), halt_s=p["halt6_s"]),
             S(6, "walk", _pts("X", "E", "K"))]
    # 7. K trote, F paso, B trote, M galope izquierda
    segs += [S(7, "trot", _pts("K", "cAK", "A", "cAF", "F")),
             S(7, "walk", _pts("F", "B")),
             S(7, "trot", _pts("B", "M")),
             S(7, "canter_L", _pts("M", "cCM", "C"))]
    # 8. Círculo 20 m a la izquierda en C (antihorario desde C, ángulo 0)
    r8 = p["circle8_diam"] / 2
    c8 = (40.0 - T - r8, 0.0)
    brk = p["canter8_break"]
    if brk is None:
        segs += [S(8, "canter_L", _arc(c8, r8, 0, 360), tag="circle8")]
    else:
        a, b = brk
        segs += [S(8, "canter_L", _arc(c8, r8, 0, a), tag="circle8"),
                 S(8, "trot", _arc(c8, r8, a, b - a), tag="circle8"),
                 S(8, "canter_L", _arc(c8, r8, b, 360 - b), tag="circle8")]
    # 9. Entre H y E caer al trote
    segs += [S(9, "canter_L", _pts("C", "cCH", "H", (27.0, YT))),
             S(9, "trot", _pts((27.0, YT), "E"))]
    # 10. E doblar a lo ancho, B pista a la derecha, F galope a la derecha
    segs += [S(10, "trot", _pts("E", "X", "B", "F")),
             S(10, "canter_R", _pts("F", "cAF", "A"))]
    # 11. Círculo 20 m a la derecha en A (horario desde ángulo 180)
    r11 = p["circle11_diam"] / 2
    segs += [S(11, "canter_R", _arc((T + r11, 0.0), r11, 180, -360), tag="circle11")]
    # 12. KEH galope largo, HCM galope corto, B trote
    segs += [S(12, "canter_R", _pts("A", "cAK", "K")),
             S(12, "canter_R", _pts("K", "E", "H"), speed=p["ext_canter_speed"], tag="ext"),
             S(12, "canter_R", _pts("H", "cCH", "C", "cCM", "M", "B"), speed=3.6, tag="coll"),
             S(12, "trot", _pts("B", "F", "cAF", (T, -3.0)))]
    # 13. Línea central: X paso, G parada
    segs += [S(13, "trot", _pts((T, -3.0), (2.5, -0.3), (5.0, 0.0), "X")),
             S(13, "walk", _pts("X", "G")),
             S(13, "halt", _pts("G"), halt_s=p["halt13_s"], tag="final"),
             S(13, "walk", _pts("G", (38.0, 0.0)), tag="exit")]  # salida al paso tras el saludo
    return segs
