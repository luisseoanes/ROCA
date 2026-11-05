"""Genera las capturas de la web para la documentación (docs/img/).

Requiere: playwright (pip install playwright) y Chrome instalado (o `playwright install chromium`).
Uso: python scripts/capturas_web.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "index.html"
OUT = ROOT / "docs" / "img"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads((ROOT / "web" / "data" / "report.json").read_text(encoding="utf-8"))
    mov = {m["n"]: m for m in rep["score"]["movements"]}
    caress_t = rep["score"]["penalties"][0]["t"] if rep["score"]["penalties"] else mov[10]["t0"]
    halt13 = next(c for c in next(m for m in rep["analysis"]["movements"] if m["n"] == 13)["checks"] if c["type"] == "halt")

    launch = {"args": ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"]}
    if os.path.exists(CHROME):
        launch["executable_path"] = CHROME
    with sync_playwright() as p:
        b = p.chromium.launch(**launch)
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="light")
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(WEB.as_uri())
        pg.wait_for_timeout(2500)

        def set_time(t):
            pg.evaluate(f"""() => {{ const s = document.querySelector('#scrub');
                s.value = Math.round({t} / {rep['meta']['duration_s']} * 1000); s.dispatchEvent(new Event('input')); }}""")
            pg.wait_for_timeout(700)

        def view(v):
            pg.click(f'.seg button[data-view="{v}"]')
            pg.wait_for_timeout(300)

        def settle(n=12):  # dejar que las vistas que siguen al caballo converjan
            for _ in range(n):
                pg.wait_for_timeout(120)

        viewer = pg.locator(".viewer")

        # 1. Vista general de la aplicación
        set_time(mov[3]["t0"] + 7)
        pg.screenshot(path=str(OUT / "01_general.png"))
        # 2. Vista lateral al galope (nuca detrás de la vertical)
        set_time(mov[8]["t0"] + 6); view("side"); settle(); set_time(mov[8]["t0"] + 6.2); settle()
        viewer.screenshot(path=str(OUT / "02_lateral_galope.png"))
        # 3. Vista lateral al trote (nuca correcta)
        set_time(mov[2]["t0"] + 4); settle(); set_time(mov[2]["t0"] + 4.2); settle()
        viewer.screenshot(path=str(OUT / "03_lateral_trote.png"))
        # 4. Caricia por delante de las riendas
        set_time(caress_t + 0.5); settle(); set_time(caress_t + 0.6); settle()
        viewer.screenshot(path=str(OUT / "04_caricia.png"))
        # 5. Parada final
        set_time(halt13["t0"] + 1.2); settle(); set_time(halt13["t0"] + 1.3); settle()
        viewer.screenshot(path=str(OUT / "05_parada_final.png"))
        # 6. Seguimiento
        view("follow"); set_time(mov[11]["t0"] + 5); settle(); set_time(mov[11]["t0"] + 5.2); settle(20)
        viewer.screenshot(path=str(OUT / "06_seguir.png"))
        # 7. Cenital
        view("top"); set_time(mov[11]["t0"] + 8)
        viewer.screenshot(path=str(OUT / "07_cenital.png"))
        # 8. Juez en C
        view("judge"); set_time(mov[2]["t0"] + 3)
        viewer.screenshot(path=str(OUT / "08_juez_c.png"))
        # 9. Rig de cámaras
        view("orbit"); pg.check("#t-cams"); pg.wait_for_timeout(600)
        viewer.screenshot(path=str(OUT / "09_rig_camaras.png"))
        pg.uncheck("#t-cams"); view("orbit")

        rep_panel = pg.locator(".report")
        def clip_shot(loc, path, max_h):
            bb = loc.bounding_box()
            pg.screenshot(path=str(path), full_page=True,
                          clip={"x": bb["x"], "y": bb["y"], "width": bb["width"], "height": min(bb["height"], max_h)})

        # 10. Resumen (KPIs, alerta y primeras áreas)
        clip_shot(rep_panel, OUT / "10_resumen.png", 1000)
        # 11. Hoja de puntuación (primeros ejercicios, legible)
        pg.click('.tabs button[data-tab="sheet"]'); pg.wait_for_timeout(400)
        clip_shot(rep_panel, OUT / "11_hoja.png", 1150)
        # 12. Mejoras
        pg.click('.tabs button[data-tab="improve"]'); pg.wait_for_timeout(400)
        pg.locator("#pane-improve .opp").nth(0).screenshot(path=str(OUT / "12_mejora_aires.png"))
        pg.locator("#pane-improve .opp").nth(1).screenshot(path=str(OUT / "13_mejora_jinete.png"))
        pg.locator("#pane-improve .opp").nth(3).screenshot(path=str(OUT / "14_mejora_otra.png"))
        # 15-18. Análisis
        pg.click('.tabs button[data-tab="analysis"]'); pg.wait_for_timeout(600)
        sec = pg.locator("#pane-analysis .section")
        pg.hover("#ts-hit", position={"x": 520, "y": 120}); pg.wait_for_timeout(200)
        sec.nth(0).screenshot(path=str(OUT / "15_serie_nuca.png"))
        pg.mouse.move(5, 5)
        pg.click('.chart-tabs button[data-s="lean"]'); pg.wait_for_timeout(300)
        sec.nth(0).screenshot(path=str(OUT / "16_serie_jinete.png"))
        pg.click('.chart-tabs button[data-s="hnp"]')
        pg.select_option("#mov-sel", "3"); pg.wait_for_timeout(400)
        sec.nth(1).screenshot(path=str(OUT / "17_trazado_circulo.png"))
        pg.select_option("#mov-sel", "8"); pg.wait_for_timeout(400)
        sec.nth(1).screenshot(path=str(OUT / "18_trazado_galope.png"))
        sec.nth(2).screenshot(path=str(OUT / "19_por_aire.png"))
        # 20. Calidad 3D
        pg.click('.tabs button[data-tab="quality"]'); pg.wait_for_timeout(400)
        rep_panel.screenshot(path=str(OUT / "20_calidad.png"))
        # 21. Móvil
        pm = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, color_scheme="light")
        pm.goto(WEB.as_uri()); pm.wait_for_timeout(2000)
        pm.screenshot(path=str(OUT / "21_movil.png"))
        # 22. Modo oscuro
        pd = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="dark")
        pd.goto(WEB.as_uri()); pd.wait_for_timeout(2000)
        pd.click('.seg button[data-view="side"]')
        pd.evaluate(f"""() => {{ const s = document.querySelector('#scrub');
            s.value = Math.round({mov[12]['t0'] + 9} / {rep['meta']['duration_s']} * 1000); s.dispatchEvent(new Event('input')); }}""")
        pd.wait_for_timeout(2500)
        pd.screenshot(path=str(OUT / "22_oscuro.png"))
        b.close()
    print("Capturas en", OUT, "| errores JS:", errors or "ninguno")


if __name__ == "__main__":
    main()
