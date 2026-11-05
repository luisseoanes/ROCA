"""Puntuación orientativa basada en reglas trazables a la documentación.

* Escala 0-10 con medios puntos (Anexo I, reglamento RFHE 2026).
* Coeficientes de la hoja de ejercicios (Anexo II): ejercicios x1, conjunto x2.
* "Firewall" FEI: un error grave (romper el aire, galope a mano contraria)
  limita la nota del ejercicio a 5 como máximo.
* Penalizaciones del Art. 19 (doma): caricia por delante de las riendas -5.

La nota base de un ejercicio sin defectos medibles es BASE (9.0): la visión
artificial no puede acreditar la "expresión" que justifica un 10. Cada
deducción indica la métrica y el umbral usados, para que un juez la revise.
"""
from __future__ import annotations

import numpy as np

from .protocol import COLLECTIVE, GAIT_ES, MOVEMENTS

BASE = 9.0
FIREWALL = 5.0


def _ded(reason: str, pts: float, cat: str, evidence: str) -> dict:
    return {"reason": reason, "pts": pts, "cat": cat, "evidence": evidence}


def _round_half(x: float) -> float:
    return float(np.clip(np.floor(x * 2 + 0.5) / 2, 0, 10))


def score_movement(mv, res: dict) -> dict:
    deds: list[dict] = []
    cap = 10.0
    for c in res["checks"]:
        t = c["type"]
        if c.get("missing"):
            deds.append(_ded(f"No se detectó: {t}", 2.0, "recorrido", "El sistema no encontró el elemento en la ventana del ejercicio."))
            continue
        if t == "circle":
            e = abs(c["diam_err_pct"])
            if e > 5:
                pts = 0.5 if e <= 10 else 1.0 if e <= 15 else 1.5
                deds.append(_ded(f"Círculo de {c['diameter_m']} m (debe ser {c['expected_m']:g} m)", pts, "geometria",
                                 f"Diámetro ajustado {c['diameter_m']} m, error {c['diam_err_pct']:+.1f}% (tolerancia ±5%)."))
            if c["center_offset_m"] > 0.75:
                deds.append(_ded("Círculo descentrado respecto a la línea central / letra", 0.5 if c["center_offset_m"] <= 1.5 else 1.0,
                                 "geometria", f"Centro desplazado {c['center_offset_m']} m del teórico."))
            if c["roundness_rms_m"] > 0.25:
                deds.append(_ded("Círculo poco redondo", 0.5 if c["roundness_rms_m"] <= 0.5 else 1.0, "geometria",
                                 f"Desviación radial RMS {c['roundness_rms_m']} m (tolerancia 0.25 m)."))
            if not c["direction_ok"]:
                deds.append(_ded("Círculo a la mano contraria (error de recorrido)", 3.0, "recorrido", "Sentido de giro opuesto al de la hoja."))
        elif t == "line":
            if c["rms_dev_m"] > 0.25:
                deds.append(_ded(f"Falta de rectitud {c['from']}→{c['to']}", 0.5 if c["rms_dev_m"] <= 0.5 else 1.0, "geometria",
                                 f"Desviación lateral RMS {c['rms_dev_m']} m, máx. {c['max_dev_m']} m."))
        elif t == "halt":
            if c["min_s"] and c["duration_s"] < c["min_s"]:
                deficit = c["min_s"] - c["duration_s"]
                deds.append(_ded(f"Inmovilidad insuficiente en {c['at']} ({c['duration_s']} s de {c['min_s']:g} s)",
                                 1.0 if deficit < 2 else 1.5, "paradas", f"Velocidad < 0.15 m/s durante {c['duration_s']} s."))
            sq = max(c["front_offset_cm"], c["hind_offset_cm"])
            if sq > 10:
                which = "posterior" if c["hind_offset_cm"] >= c["front_offset_cm"] else "anterior"
                deds.append(_ded(f"Parada no cuadrada (bípedo {which})", 0.5 if sq <= 18 else 1.0, "paradas",
                                 f"Desfase longitudinal entre cascos {which}s: {sq} cm (tolerancia 10 cm)."))
            if c["immobility_cm"] > 3:
                deds.append(_ded("Falta de inmovilidad", 0.5, "paradas", f"Oscilación de la cruz {c['immobility_cm']} cm."))
            if c["dist_letter_m"] > 1.5:
                deds.append(_ded(f"Parada fuera de la letra {c['at']}", 0.5, "precision", f"A {c['dist_letter_m']} m de {c['at']}."))
        elif t == "transition":
            late = c["late_m"]
            a = abs(late)
            if a > 2.0:
                pts = 1.0 if a <= 4 else 1.5 if a <= 6 else 2.0
                cuando = "tarde" if late > 0 else "pronto"
                deds.append(_ded(f"Transición a {GAIT_ES[c['to']]} {cuando} en {c['at']}", pts, "transiciones",
                                 f"Cambio de aire a {a:.1f} m de la letra {c['at']} (tolerancia 2 m)."))
        elif t == "transition_zone":
            if c["outside_m"] > 0:
                deds.append(_ded("Transición fuera de la zona indicada", 1.0 if c["outside_m"] < 3 else 2.0, "transiciones",
                                 f"A {c['outside_m']} m fuera del tramo {c['zone']}."))
        elif t == "lengthening":
            r = c["ratio"]
            if r < c["min_ratio"]:
                pts = 1.0 if r >= 1.1 else 2.0
                deds.append(_ded("Alargamiento insuficiente en el galope largo", pts, "impulsion",
                                 f"Velocidad {c['speed_ext']} m/s vs {c['speed_ref']} m/s de trabajo (x{r}); "
                                 f"tranco {c['stride_ext_m']} m vs {c['stride_ref_m']} m. Se espera x{c['min_ratio']} o más."))
        elif t == "hnp":
            if c.get("pct_behind10", 0) > 20:
                deds.append(_ded("Nariz claramente detrás de la vertical", 1.0, "contacto",
                                 f"{c['pct_behind10']}% del tiempo >10° detrás; media {c['mean_deg']:+.1f}°."))
            elif c.get("mean_deg", 0) < 0:
                deds.append(_ded("Nariz detrás de la vertical", 0.5, "contacto",
                                 f"Media {c['mean_deg']:+.1f}° ({c['pct_behind']}% del tiempo detrás)."))
    for gf in res.get("gait_faults", []):
        expected = "/".join(GAIT_ES[g] for g in mv.gaits if g != "halt")
        wrong_lead = gf["gait"].startswith("canter") and any(g.startswith("canter") for g in mv.gaits)
        what = "Galope a mano contraria / trocado" if wrong_lead else "Rompe el aire"
        deds.append(_ded(f"{what}: {gf['strides']} trancos de {GAIT_ES[gf['gait']]}", 1.0, "aires",
                         f"Entre {gf['t0']} s y {gf['t1']} s se detecta {GAIT_ES[gf['gait']]} (se pide {expected}). Error grave: nota máx. {FIREWALL:g}."))
        cap = min(cap, FIREWALL)
    cvs = [v["cv_pct"] for v in res["regularity"].values()]
    if cvs and max(cvs) > 6:
        g, v = max(res["regularity"].items(), key=lambda kv: kv[1]["cv_pct"])
        deds.append(_ded(f"Irregularidad del {GAIT_ES[g]}", 0.5 if v["cv_pct"] <= 9 else 1.0, "aires",
                         f"Variación de la duración del tranco {v['cv_pct']}% (tolerancia 6%)."))
    raw = BASE - sum(d["pts"] for d in deds)
    note = _round_half(min(raw, cap))
    return {"n": mv.n, "letters": mv.letters, "text": mv.text, "directives": mv.directives, "coef": mv.coef,
            "note": note, "base": BASE, "capped": cap < 10 and raw > cap, "deductions": deds,
            "t0": res["t0"], "t1": res["t1"]}


def score_collective(analysis: dict, movs_scored: list[dict]) -> list[dict]:
    an = analysis
    bg = an["by_gait"]
    movs = an["movements"]
    out = []
    for col in COLLECTIVE:
        deds = []
        k = col["key"]
        if k == "A":
            cv = [v["cv_pct"] for m in movs for g, v in m["regularity"].items() if g != "walk"]
            if cv and np.mean(cv) > 4.5:
                deds.append(_ded("Regularidad general mejorable", 0.5, "aires", f"CV medio del tranco {np.mean(cv):.1f}%."))
            nf = sum(len(m.get("gait_faults", [])) for m in movs)
            if nf:
                deds.append(_ded(f"{nf} ruptura(s) de aire en la prueba", min(2.0, 1.0 * nf), "aires", "Ver ejercicios señalados."))
        elif k == "B":
            for m in movs:
                for c in m["checks"]:
                    if c["type"] == "lengthening" and c["ratio"] < c["min_ratio"]:
                        deds.append(_ded("Poca diferencia entre galope de trabajo y largo", 1.0, "impulsion",
                                         f"Aumento de velocidad x{c['ratio']} (se espera ≥ x{c['min_ratio']})."))
            if "trot" in bg and bg["trot"]["speed_mean"] < 2.9:
                deds.append(_ded("Trote de trabajo con poca impulsión", 0.5, "impulsion", f"Velocidad media {bg['trot']['speed_mean']} m/s."))
        elif k == "C":
            allh = [g["hnp"] for g in bg.values() if g.get("hnp")]
            can = bg.get("canter", {}).get("hnp", {})
            if can and can.get("pct_behind10", 0) > 25:
                deds.append(_ded("Galope con la nariz >10° detrás de la vertical", 1.5, "contacto",
                                 f"{can['pct_behind10']}% del galope >10° detrás; media {can['mean_deg']:+.1f}°."))
            elif can and can.get("mean_deg", 0) < 0:
                deds.append(_ded("Galope con la nariz detrás de la vertical", 1.0, "contacto", f"Media {can['mean_deg']:+.1f}°."))
            tr = bg.get("trot", {}).get("hnp", {})
            if tr and tr.get("mean_deg", 0) < 0:
                deds.append(_ded("Trote con la nariz detrás de la vertical", 0.5, "contacto", f"Media {tr['mean_deg']:+.1f}°."))
            nev = len(an["events"]["caress"])
            if nev:
                deds.append(_ded("Atención/confianza: el jinete suelta la mano para acariciar", 0.5, "jinete", f"{nev} evento(s)."))
        elif k == "D":
            can = bg.get("canter", {})
            if can.get("rider_lean_deg", 0) > 8:
                deds.append(_ded("Tronco inclinado hacia delante en el galope", 1.0 if can["rider_lean_deg"] <= 12 else 1.5, "jinete",
                                 f"Inclinación media {can['rider_lean_deg']}° (vertical = 0°, tolerancia 8°)."))
            # Rebote: exceso sobre la línea base de la sesión (descuenta el ruido de medida)
            b = np.array([m["rider"]["bounce_std_cm"] or 0 for m in movs])
            base_b = float(np.median(b))
            excess = np.sqrt(np.clip(b ** 2 - base_b ** 2, 0, None))
            k_w = int(np.argmax(excess))
            if excess[k_w] > 0.6:
                deds.append(_ded(f"Asiento poco estable al trote (ejercicio {movs[k_w]['n']})", 0.5 if excess[k_w] <= 1.2 else 1.0, "jinete",
                                 f"Rebote de la pelvis respecto al dorso {b[k_w]:.1f} cm frente a {base_b:.1f} cm de base "
                                 f"(exceso {excess[k_w]:.1f} cm, tolerancia 0.6 cm)."))
            hj = np.mean([g["hand_jitter_cm"] for g in bg.values()])
            if hj > 4.5:  # umbral provisional: calibrar con jinetes reales
                deds.append(_ded("Mano de las riendas poco fija", 0.5, "jinete", f"Movimiento parásito medio {hj:.1f} cm (tolerancia 4.5 cm)."))
        raw = BASE - sum(d["pts"] for d in deds)
        out.append({"key": k, "text": col["text"], "coef": col["coef"], "note": _round_half(raw), "deductions": deds})
    return out


def score(analysis: dict) -> dict:
    res_by_n = {m["n"]: m for m in analysis["movements"]}
    movs = [score_movement(mv, res_by_n[mv.n]) for mv in MOVEMENTS]
    coll = score_collective(analysis, movs)
    penalties = []
    for i, ev in enumerate(analysis["events"]["caress"][:2]):
        penalties.append({"reason": f"Caricia por delante de las riendas ({i + 1}ª)", "pts": 5.0, "t": ev["t"],
                          "rule": "Art. 19 — Penalizaciones en la prueba de doma: -5 puntos cada una."})
    disq = [ev for ev in analysis["events"]["two_hands"] if ev["disqualifying"]]
    total_max = sum(10 * m["coef"] for m in movs) + sum(10 * c["coef"] for c in coll)
    gross = sum(m["note"] * m["coef"] for m in movs) + sum(c["note"] * c["coef"] for c in coll)
    pen = sum(p["pts"] for p in penalties)
    total = gross - pen
    return {
        "movements": movs, "collective": coll, "penalties": penalties,
        "not_evaluable": [{"reason": "Arrear con la voz o chasquidos (-2)", "why": "Requiere captura de audio sincronizada (no incluida en el PoC)."}],
        "disqualified": bool(disq), "disqualification_events": disq,
        "gross": round(gross, 1), "penalty_pts": pen, "total": round(total, 1), "max": total_max,
        "percent": round(100 * total / total_max, 2),
    }
