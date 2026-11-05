"""Oportunidades de mejora: agrupa las deducciones por área, las ordena por
puntos recuperables y propone ejercicios de entrenamiento concretos."""
from __future__ import annotations

from collections import defaultdict

CATS = {
    "aires": {
        "title": "Mantener el aire y la regularidad",
        "why": "Romper el aire es un error grave: la FEI limita la nota del ejercicio a 5 y la hoja lo valora en 'Aires, franqueza y regularidad'.",
        "drills": [
            "Círculos de 20 m al galope manteniendo la pierna interior en la cincha; si el caballo se apaga, agrandar el círculo antes de que rompa.",
            "Contar en voz baja el ritmo del galope (1-2-3) y mantenerlo igual en recta y en círculo.",
            "Transiciones galope-trote-galope cada medio círculo para enseñar a esperar la ayuda en lugar de caerse al trote.",
        ],
    },
    "geometria": {
        "title": "Precisión en las figuras",
        "why": "Las directrices de los círculos piden 'trazado y geometría'. Un círculo de 10 m que mide 12 m se ve desde la mesa del jurado.",
        "drills": [
            "Marcar con 4 conos los puntos de tangencia del círculo (cada cuarto) y pasar por ellos.",
            "Contar los trancos de cada cuarto de círculo: si son iguales, el círculo es redondo.",
            "Mirar al siguiente cuarto del círculo, no al suelo, y mantener la incurvación con la pierna interior.",
        ],
    },
    "transiciones": {
        "title": "Transiciones en la letra",
        "why": "Las transiciones deben hacerse cuando los hombros del caballo pasan por la letra; las directrices piden 'claridad y fluidez en la transición'.",
        "drills": [
            "Preparar con media parada 2-3 trancos antes de la letra y pedir la transición con el hombro del caballo a su altura.",
            "Practicar transiciones en puntos fijos marcados (conos cada 10 m) hasta que salgan a menos de 1 m.",
            "Grabar las transiciones desde el lateral y comprobar dónde está el hombro en el primer tranco del nuevo aire.",
        ],
    },
    "paradas": {
        "title": "Paradas cuadradas e inmóviles",
        "why": "En X y G se juzgan 'la parada, la inmovilidad de 4\" y el caballo cuadrado'.",
        "drills": [
            "Paradas desde el paso junto a la valla, contando 5 segundos antes de saludar.",
            "Si queda un posterior retrasado, activarlo con la pierna de ese lado justo antes de cerrar la mano.",
            "Alternar parada - 3 pasos atrás - parada para mejorar el equilibrio sobre los posteriores.",
        ],
    },
    "impulsion": {
        "title": "Impulsión y diferenciación de los galopes",
        "why": "El galope largo KEH debe mostrar un alargamiento claro de tranco y de cuadro; la impulsión entra también en las notas de conjunto.",
        "drills": [
            "En la recta: 4 trancos de trabajo, 4 alargando y 4 reuniendo, buscando tranco más largo (no más rápido).",
            "Galopar sobre barras a 3.2-3.5 m para estirar el tranco con el dorso arqueado.",
            "Marcar el inicio y el final del alargamiento con conos para que la diferencia sea evidente.",
        ],
    },
    "contacto": {
        "title": "Posición de la nuca: nariz por delante de la vertical",
        "why": "El reglamento y la FEI piden la nuca como punto más alto y el plano nasal ligeramente por delante de la vertical. "
               "La literatura de la carpeta asocia montar detrás de la vertical con más conductas de conflicto y peor bienestar.",
        "drills": [
            "Estiramientos hacia delante y abajo con la nariz por delante de la vertical para soltar el dorso.",
            "Ceder la mano para comprobar el autoporte: el caballo debe mantener la actitud sin apoyarse.",
            "Revisar el ajuste de la embocadura y de la muserola, y el trabajo en calentamiento (evitar la hiperflexión).",
        ],
    },
    "jinete": {
        "title": "Posición y estabilidad del jinete",
        "why": "La nota de conjunto D valora la colocación y posición del jinete y la corrección de las ayudas.",
        "drills": [
            "Galope sin estribos o a la cuerda para encontrar el asiento con el tronco vertical (oreja-hombro-cadera-talón).",
            "Trote sentado acompañando con la cadera, no con la espalda; empezar en tramos cortos.",
            "Llevar la mano de las riendas fija sobre la cruz; trabajar con una fusta horizontal bajo los pulgares.",
        ],
    },
    "penalizacion": {
        "title": "Evitar penalizaciones",
        "why": "Cada caricia por delante de las riendas cuesta -5 puntos en la doma (Art. 19).",
        "drills": [
            "Acariciar sólo tras el saludo final; durante la prueba, premiar con la voz baja (sin chasquidos) o cediendo la mano.",
        ],
    },
    "precision": {"title": "Precisión en la letra", "why": "Las paradas y transiciones se valoran en la letra.",
                  "drills": ["Practicar paradas en un cono situado en la línea central."]},
    "recorrido": {"title": "Recorrido", "why": "Los errores de recorrido penalizan -5 (el tercero elimina).",
                  "drills": ["Repasar la prueba a pie y con el caballo antes del concurso."]},
}


def opportunities(score: dict) -> list[dict]:
    agg = defaultdict(lambda: {"points": 0.0, "items": []})
    for m in score["movements"]:
        lost_cap = 0.0
        for d in m["deductions"]:
            agg[d["cat"]]["points"] += d["pts"] * m["coef"]
            agg[d["cat"]]["items"].append({"where": f"Ejercicio {m['n']} ({m['letters']})", "mov": m["n"], "t": m["t0"],
                                           "reason": d["reason"], "evidence": d["evidence"], "pts": d["pts"] * m["coef"]})
        if m["capped"]:
            lost_cap = (m["base"] - sum(d["pts"] for d in m["deductions"])) - m["note"]
            if lost_cap > 0:
                agg["aires"]["points"] += lost_cap * m["coef"]
    for c in score["collective"]:
        for d in c["deductions"]:
            agg[d["cat"]]["points"] += d["pts"] * c["coef"]
            agg[d["cat"]]["items"].append({"where": f"Nota de conjunto {c['key']}", "mov": None, "t": None,
                                           "reason": d["reason"], "evidence": d["evidence"], "pts": d["pts"] * c["coef"]})
    for p in score["penalties"]:
        agg["penalizacion"]["points"] += p["pts"]
        agg["penalizacion"]["items"].append({"where": "Penalización", "mov": None, "t": p["t"], "reason": p["reason"],
                                             "evidence": p["rule"], "pts": p["pts"]})
    out = []
    for cat, v in agg.items():
        info = CATS.get(cat, {"title": cat, "why": "", "drills": []})
        out.append({"cat": cat, "title": info["title"], "points": round(v["points"], 1),
                    "percent": round(100 * v["points"] / score["max"], 2), "why": info["why"],
                    "drills": info["drills"], "items": sorted(v["items"], key=lambda i: -i["pts"])})
    return sorted(out, key=lambda o: -o["points"])
