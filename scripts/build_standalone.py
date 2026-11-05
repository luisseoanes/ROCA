"""Genera un único HTML autónomo para enviar (dist/ROCA.html).

Incrusta en index.html: styles.css, app.js, los datos (data/report.js, data/motion.js)
y Three.js (descargado de jsdelivr). El resultado se abre con doble clic, sin
servidor y sin conexión a internet.

Uso: python scripts/build_standalone.py
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
OUT = ROOT / "dist" / "ROCA.html"
CDN_RE = re.compile(r'<script src="(https://cdn\.jsdelivr\.net/npm/three@[^"]+)"></script>')


def js_tag(code: str) -> str:
    # evita cerrar la etiqueta <script> antes de tiempo si el código contiene "</script"
    return "<script>\n" + code.replace("</script", "<\\/script") + "\n</script>"


def main():
    html = (WEB / "index.html").read_text(encoding="utf-8")

    css = (WEB / "styles.css").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="styles.css">', "<style>\n" + css + "\n</style>")

    for name in ("data/report.js", "data/motion.js", "app.js"):
        tag = f'<script src="{name}"></script>'
        assert tag in html, f"no se encontró {tag} en index.html"
        code = (WEB / name).read_text(encoding="utf-8")
        html = html.replace(tag, js_tag(code))

    offline = True
    for url in CDN_RE.findall(html):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                code = r.read().decode("utf-8")
            html = html.replace(f'<script src="{url}"></script>', js_tag(code))
        except Exception as exc:  # sin internet: se deja el enlace al CDN
            offline = False
            print(f"Aviso: no se pudo incrustar {url} ({exc}); el visor 3D necesitará conexión.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB) · funciona sin conexión: {'sí' if offline else 'no'}")


if __name__ == "__main__":
    main()
