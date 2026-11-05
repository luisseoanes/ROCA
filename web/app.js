/* Cohnetic — visor 3D e informe de la prueba (PoC).
 * Datos: window.COHNETIC_REPORT (report.js) y window.COHNETIC_MOTION (motion.js),
 * generados por scripts/run_poc.py.
 */
(function () {
  "use strict";
  const R = window.COHNETIC_REPORT;
  const M = window.COHNETIC_MOTION;
  if (!R || !M) {
    document.body.innerHTML = '<p style="padding:24px">Faltan los datos. Ejecuta <code>python scripts/run_poc.py</code> para generar web/data/.</p>';
    return;
  }

  // ------------------------------------------------------------------ utilidades
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : Number(v).toLocaleString("es-ES", { minimumFractionDigits: d, maximumFractionDigits: d }));
  const fmtT = (t) => { t = Math.max(0, t); const m = Math.floor(t / 60); const s = Math.floor(t % 60); return `${m}:${String(s).padStart(2, "0")}`; };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const GAITS = [
    { code: 0, key: "halt", name: "Parada", short: "Parada", var: "--g-halt" },
    { code: 1, key: "walk", name: "Paso", short: "Paso", var: "--g-walk" },
    { code: 2, key: "trot", name: "Trote", short: "Trote", var: "--g-trot" },
    { code: 3, key: "canter_L", name: "Galope a mano izquierda", short: "Gal. izq.", var: "--g-canterL" },
    { code: 4, key: "canter_R", name: "Galope a mano derecha", short: "Gal. dcha.", var: "--g-canterR", darkText: true },
    { code: 5, key: "unknown", name: "Transición / indeterminado", short: "", var: "--g-unknown" },
  ];
  const gaitBy = (code) => GAITS[code] ?? GAITS[5];
  const CAT_ES = { aires: "Aires", geometria: "Geometría", transiciones: "Transiciones", paradas: "Paradas", impulsion: "Impulsión",
    contacto: "Contacto / nuca", jinete: "Jinete", penalizacion: "Penalizaciones", precision: "Precisión", recorrido: "Recorrido" };

  const S = R.score;
  const movs = S.movements;
  const duration = M.n / M.fps;
  const NH = M.nh, NR = M.nr, NJ = NH + NR;
  const hk = Object.fromEntries(M.horse_names.map((n, i) => [n, i]));

  function hnpStatus(a) {
    if (a == null) return { cls: "", label: "—", icon: "" };
    if (a >= 0) return { cls: "good", label: "por delante de la vertical", icon: "✓", color: css("--good") };
    if (a >= -10) return { cls: "warning", label: "detrás de la vertical", icon: "!", color: css("--warning") };
    return { cls: "critical", label: "más de 10° detrás", icon: "✕", color: css("--critical") };
  }
  const movAt = (t) => movs.find((m) => t >= m.t0 && t < m.t1) || (t >= movs[movs.length - 1].t0 ? movs[movs.length - 1] : null);

  // ------------------------------------------------------------------ cabecera
  $("#test-name").textContent = R.meta.test;
  $("#rider-name").textContent = `${R.meta.rider} · ${R.meta.horse}`;
  $("#synthetic-badge").hidden = !R.meta.synthetic;
  $("#score-pct").textContent = fmt(S.percent, 2);
  $("#score-detail").innerHTML = `${fmt(S.total, 1)} de ${S.max} puntos` +
    (S.penalty_pts ? `<br>incluye −${fmt(S.penalty_pts, 0)} de penalización` : "");
  $("#gen-date").textContent = `Generado ${R.meta.generated}.`;

  // ------------------------------------------------------------------ estado de reproducción
  const state = { t: 0, playing: false, speed: 1, view: "orbit", listeners: [] };
  function setTime(t, { fromScrub = false } = {}) {
    const nt = Math.min(Math.max(0, t), duration - 1 / M.fps);
    if (Math.abs(nt - state.t) > 1 && typeof three !== "undefined" && three) three.snap(); // salto: colocar la cámara de golpe
    state.t = nt;
    if (!fromScrub) $("#scrub").value = Math.round((state.t / duration) * 1000);
    $("#time").textContent = fmtT(state.t);
    state.listeners.forEach((fn) => fn(state.t));
  }
  const onTime = (fn) => state.listeners.push(fn);

  // ------------------------------------------------------------------ escena 3D
  let three = null;
  try {
    three = initThree();
  } catch (err) {
    console.error(err);
    const msg = $("#stage-msg");
    msg.hidden = false;
    msg.textContent = "No se pudo iniciar el visor 3D (" + err.message + "). El informe sigue disponible.";
  }

  function initThree() {
    const host = $("#stage");
    if (!window.THREE || !window.THREE.OrbitControls) {
      const msg = $("#stage-msg");
      msg.hidden = false;
      msg.textContent = "No se pudo cargar Three.js (¿sin conexión a internet?). El informe sigue disponible.";
      return null;
    }
    const T = window.THREE;
    const renderer = new T.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    host.appendChild(renderer.domElement);
    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(45, 16 / 10, 0.1, 400);
    camera.position.set(-6, 22, 30);
    const controls = new T.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.8, 0);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.495;
    scene.add(new T.HemisphereLight(0xffffff, 0x888877, 0.9));
    const sun = new T.DirectionalLight(0xffffff, 0.5);
    sun.position.set(10, 30, 10);
    scene.add(sun);

    // Mundo pista (x, y, z arriba, m) -> three (x-20, z, -y)
    const W = (x, y, z = 0) => new T.Vector3(x - 20, z, -y);

    const groups = { arena: new T.Group(), figs: new T.Group(), cams: new T.Group(), trail: new T.Group() };
    Object.values(groups).forEach((g) => scene.add(g));

    const mats = {};
    function themeColors() {
      return {
        bg: css("--stage-bg"), sand: css("--sand"), fence: css("--text-secondary"), grid: css("--axis"),
        horse: css("--text-primary"), rider: css("--accent"), fig: css("--text-secondary"), trail: css("--accent"),
        letter: css("--text-primary"), cam: css("--text-muted"),
      };
    }

    // Suelo y cercado
    const ground = new T.Mesh(new T.PlaneGeometry(40, 20), new T.MeshLambertMaterial({ color: 0xd9c9a8 }));
    ground.rotation.x = -Math.PI / 2;
    groups.arena.add(ground);
    const outer = new T.Mesh(new T.PlaneGeometry(70, 46), new T.MeshLambertMaterial({ color: 0xcccccc }));
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.01;
    groups.arena.add(outer);
    const fenceGeo = new T.BufferGeometry().setFromPoints([W(0, -10, 0.3), W(40, -10, 0.3), W(40, 10, 0.3), W(0, 10, 0.3), W(0, -10, 0.3)]);
    const fence = new T.Line(fenceGeo, new T.LineBasicMaterial({ color: 0x52514e }));
    groups.arena.add(fence);
    const cl = new T.Line(new T.BufferGeometry().setFromPoints([W(0, 0, 0.005), W(40, 0, 0.005)]),
      new T.LineDashedMaterial({ color: 0x898781, dashSize: 0.6, gapSize: 0.6 }));
    cl.computeLineDistances();
    groups.arena.add(cl);

    function letterSprite(txt, color) {
      const c = document.createElement("canvas");
      c.width = c.height = 128;
      const g = c.getContext("2d");
      g.fillStyle = "rgba(255,255,255,0.92)";
      g.beginPath(); g.arc(64, 64, 52, 0, Math.PI * 2); g.fill();
      g.fillStyle = color; g.font = "bold 64px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(txt, 64, 68);
      const s = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(c), depthTest: false }));
      s.scale.set(1.3, 1.3, 1);
      return s;
    }
    for (const [name, [x, y]] of Object.entries(R.arena.letters)) {
      const onLine = ["D", "X", "G"].includes(name);
      const ox = x === 0 ? -1.2 : x === 40 ? 1.2 : 0;
      const oy = y > 0 ? 1.2 : y < 0 ? -1.2 : 0;
      const s = letterSprite(name, "#0b0b0b");
      s.position.copy(W(x + ox, y + oy, onLine ? 0.05 : 0.7));
      if (onLine) s.scale.set(0.8, 0.8, 1);
      groups.arena.add(s);
    }

    // Figuras teóricas
    const figMat = new T.LineDashedMaterial({ color: 0x52514e, dashSize: 0.4, gapSize: 0.3 });
    for (const f of R.expected) {
      let pts = [];
      if (f.type === "circle") {
        for (let i = 0; i <= 96; i++) { const a = (i / 96) * Math.PI * 2; pts.push(W(f.center[0] + f.r * Math.cos(a), f.center[1] + f.r * Math.sin(a), 0.02)); }
      } else { pts = [W(f.a[0], f.a[1], 0.02), W(f.b[0], f.b[1], 0.02)]; }
      const ln = new T.Line(new T.BufferGeometry().setFromPoints(pts), figMat);
      ln.computeLineDistances();
      groups.figs.add(ln);
    }

    // Cámaras del rig
    for (const c of R.cameras) {
      const [x, y, z] = c.center;
      const box = new T.Mesh(new T.BoxGeometry(0.6, 0.4, 0.4), new T.MeshLambertMaterial({ color: 0x52514e }));
      box.position.copy(W(x, y, z));
      box.lookAt(W(20, 0, 0.5));
      groups.cams.add(box);
      const ln = new T.Line(new T.BufferGeometry().setFromPoints([W(x, y, z), W(20, 0, 0.5)]),
        new T.LineDashedMaterial({ color: 0x898781, dashSize: 0.5, gapSize: 0.5, transparent: true, opacity: 0.5 }));
      ln.computeLineDistances();
      groups.cams.add(ln);
      const pole = new T.Line(new T.BufferGeometry().setFromPoints([W(x, y, 0), W(x, y, z)]), new T.LineBasicMaterial({ color: 0x898781 }));
      groups.cams.add(pole);
      const lbl = letterSprite(c.name.split(" ")[0].replace("CAM", ""), "#2a78d6");
      lbl.position.copy(W(x, y, z + 0.9));
      lbl.scale.set(0.9, 0.9, 1);
      groups.cams.add(lbl);
    }
    groups.cams.visible = false;

    // Trayectoria completa (tenue) + estela reciente
    const xy = R.series.xy;
    const full = new T.Line(new T.BufferGeometry().setFromPoints(xy.map(([x, y]) => W(x, y, 0.03))),
      new T.LineBasicMaterial({ color: 0x2a78d6, transparent: true, opacity: 0.25 }));
    groups.trail.add(full);
    const TRAIL_N = Math.round(M.fps * 8);
    const trailPos = new Float32Array(TRAIL_N * 3);
    const trailGeo = new T.BufferGeometry();
    trailGeo.setAttribute("position", new T.BufferAttribute(trailPos, 3));
    const trail = new T.Line(trailGeo, new T.LineBasicMaterial({ color: 0x2a78d6 }));
    groups.trail.add(trail);

    // Esqueletos
    function makeSkel(edges, color, size) {
      const pos = new Float32Array(edges.length * 2 * 3);
      const g = new T.BufferGeometry();
      g.setAttribute("position", new T.BufferAttribute(pos, 3));
      const lines = new T.LineSegments(g, new T.LineBasicMaterial({ color }));
      const n = edges.reduce((m, e) => Math.max(m, e[0], e[1]), 0) + 1;
      const jp = new Float32Array(n * 3);
      const jg = new T.BufferGeometry();
      jg.setAttribute("position", new T.BufferAttribute(jp, 3));
      const pts = new T.Points(jg, new T.PointsMaterial({ color, size, sizeAttenuation: true }));
      const grp = new T.Group();
      grp.add(lines, pts);
      scene.add(grp);
      return { grp, pos, g, jp, jg, edges, lines, pts };
    }
    const horse = makeSkel(M.horse_edges, 0x0b0b0b, 0.09);
    const rider = makeSkel(M.rider_edges, 0x2a78d6, 0.09);
    const nasalGeo = new T.BufferGeometry();
    nasalGeo.setAttribute("position", new T.BufferAttribute(new Float32Array(6), 3));
    const nasalMat = new T.LineBasicMaterial({ color: 0x0ca30c, linewidth: 3 });
    const nasal = new T.Line(nasalGeo, nasalMat);
    scene.add(nasal);
    // tubo para el plano nasal (linewidth no funciona en WebGL)
    const nasalTube = new T.Mesh(new T.CylinderGeometry(0.03, 0.03, 1, 8), new T.MeshBasicMaterial({ color: 0x0ca30c }));
    scene.add(nasalTube);

    function applyTheme() {
      const c = themeColors();
      scene.background = new T.Color(c.bg);
      ground.material.color.set(c.sand);
      outer.material.color.set(c.bg);
      fence.material.color.set(c.fence);
      figMat.color.set(c.fig);
      horse.lines.material.color.set(c.horse); horse.pts.material.color.set(c.horse);
      rider.lines.material.color.set(c.rider); rider.pts.material.color.set(c.rider);
      trail.material.color.set(c.trail); full.material.color.set(c.trail);
    }
    applyTheme();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { applyTheme(); renderCharts(); });

    const D = M.data;
    const P = (f, j, out) => { const i = (f * NJ + j) * 3; out.set(D[i] / 100 - 20, D[i + 2] / 100, -D[i + 1] / 100); return out; };
    const tmpA = new T.Vector3(), tmpB = new T.Vector3();

    function update(t) {
      const f = Math.min(M.n - 1, Math.floor(t * M.fps));
      for (const [sk, off] of [[horse, 0], [rider, NH]]) {
        let k = 0;
        for (const [a, b] of sk.edges) {
          P(f, a + off, tmpA); P(f, b + off, tmpB);
          sk.pos[k++] = tmpA.x; sk.pos[k++] = tmpA.y; sk.pos[k++] = tmpA.z;
          sk.pos[k++] = tmpB.x; sk.pos[k++] = tmpB.y; sk.pos[k++] = tmpB.z;
        }
        sk.g.attributes.position.needsUpdate = true;
        sk.g.computeBoundingSphere();
        const n = sk.jp.length / 3;
        for (let j = 0; j < n; j++) { P(f, j + off, tmpA); sk.jp[j * 3] = tmpA.x; sk.jp[j * 3 + 1] = tmpA.y; sk.jp[j * 3 + 2] = tmpA.z; }
        sk.jg.attributes.position.needsUpdate = true;
        sk.jg.computeBoundingSphere();
      }
      // plano nasal coloreado según la zona
      const poll = P(f, hk.poll, new T.Vector3()), nose = P(f, hk.nose, new T.Vector3());
      const st = hnpStatus(M.hnp[f]);
      nasalTube.material.color.set(st.color || "#888");
      const mid = poll.clone().add(nose).multiplyScalar(0.5);
      const dir = nose.clone().sub(poll);
      nasalTube.position.copy(mid);
      nasalTube.scale.set(1, dir.length() * 1.15, 1);
      nasalTube.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), dir.normalize());
      nasal.visible = false;
      // estela de 8 s
      for (let i = 0; i < TRAIL_N; i++) {
        const ff = Math.max(0, f - (TRAIL_N - 1 - i));
        const a = P(ff, hk.mid_back, tmpA);
        trailPos[i * 3] = a.x; trailPos[i * 3 + 1] = 0.04; trailPos[i * 3 + 2] = a.z;
      }
      trailGeo.attributes.position.needsUpdate = true;
      trailGeo.computeBoundingSphere();
      return f;
    }

    // Vistas que siguen al caballo: se recalculan en cada frame de render (también en pausa)
    let snapCam = false;
    function followCam(t) {
      if (state.view !== "follow" && state.view !== "side") return;
      const f = Math.min(M.n - 1, Math.floor(t * M.fps));
      const c = P(f, hk.mid_back, new T.Vector3());
      const wv = P(f, hk.withers, new T.Vector3()), cr = P(f, hk.croup, new T.Vector3());
      const fwd = wv.clone().sub(cr).setY(0).normalize();
      const side = new T.Vector3(-fwd.z, 0, fwd.x);
      const pos = state.view === "follow"
        ? c.clone().addScaledVector(fwd, -4.5).add(new T.Vector3(0, 3.0, 0)).addScaledVector(side, -6)
        : c.clone().addScaledVector(side, -8).setY(1.8);
      const tgt = c.clone().setY(1.3);
      const k = snapCam ? 1 : 0.15;
      camera.position.lerp(pos, k);
      controls.target.lerp(tgt, snapCam ? 1 : 0.25);
      snapCam = false;
    }

    function setView(v) {
      state.view = v;
      snapCam = true;
      controls.enabled = v === "orbit" || v === "top" || v === "judge";
      if (v === "top") { camera.position.set(0, 48, 0.01); controls.target.set(0, 0, 0); }
      if (v === "judge") { camera.position.set(23, 2.2, 0); controls.target.set(0, 1, 0); }
      if (v === "orbit") { camera.position.set(-6, 22, 30); controls.target.set(0, 0.8, 0); }
      if (v === "rig") { state.view = "orbit"; controls.enabled = true; camera.position.set(-30, 42, 58); controls.target.set(0, 0, 0); }
    }

    function resize() {
      const w = host.clientWidth, h = host.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    new ResizeObserver(resize).observe(host);
    resize();

    let last = performance.now();
    function loop(now) {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (state.playing) {
        let t = state.t + dt * state.speed;
        if (t >= duration - 1 / M.fps) { t = duration - 1 / M.fps; togglePlay(false); }
        setTime(t);
      }
      followCam(state.t);
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

    return { update, setView, groups, horse, rider, nasalTube, snap: () => (snapCam = true) };
  }

  // ------------------------------------------------------------------ controles de reproducción
  function togglePlay(force) {
    state.playing = force ?? !state.playing;
    $("#btn-play").textContent = state.playing ? "❚❚" : "▶";
    $("#btn-play").setAttribute("aria-label", state.playing ? "Pausar" : "Reproducir");
  }
  $("#btn-play").addEventListener("click", () => { if (state.t >= duration - 0.1) setTime(0); togglePlay(); });
  $("#scrub").addEventListener("input", (e) => setTime((e.target.value / 1000) * duration, { fromScrub: true }));
  $("#speed").addEventListener("change", (e) => (state.speed = parseFloat(e.target.value)));
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowRight") setTime(state.t + 2);
    if (e.code === "ArrowLeft") setTime(state.t - 2);
  });
  $$(".seg button").forEach((b) => b.addEventListener("click", () => {
    $$(".seg button").forEach((x) => x.classList.toggle("on", x === b));
    three && three.setView(b.dataset.view);
  }));
  const tog = (id, fn) => $(id).addEventListener("change", (e) => three && fn(e.target.checked));
  tog("#t-horse", (v) => (three.horse.grp.visible = v));
  tog("#t-rider", (v) => (three.rider.grp.visible = v));
  tog("#t-trail", (v) => (three.groups.trail.visible = v));
  tog("#t-figs", (v) => (three.groups.figs.visible = v));
  tog("#t-cams", (v) => {
    three.groups.cams.visible = v;
    if (v && state.view === "orbit") three.setView("rig"); // alejar para ver el rig completo
  });
  tog("#t-nasal", (v) => (three.nasalTube.visible = v));

  // marcas de ejercicios en la línea de tiempo
  const marks = $("#mov-marks");
  let prevPct = -99, lifted = false;
  movs.forEach((m) => {
    const b = document.createElement("button");
    b.textContent = m.n;
    b.title = `Ejercicio ${m.n}: ${m.text}`;
    const pct = (m.t0 / duration) * 100;
    // escalonar marcas muy próximas para que no se solapen
    lifted = pct - prevPct < 3 ? !lifted : false;
    prevPct = pct;
    b.style.left = `${pct}%`;
    if (lifted) b.style.top = "13px";
    b.addEventListener("click", () => jumpTo(m.t0, m.n));
    marks.appendChild(b);
  });
  function jumpTo(t, n) {
    setTime(t);
    if (n != null) highlightMov(n);
    $(".viewer").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // HUD y sincronización
  let curMov = null;
  onTime((t) => {
    const f = three ? three.update(t) : Math.min(M.n - 1, Math.floor(t * M.fps));
    const g = gaitBy(M.gait[f]);
    const hud = $("#hud-gait");
    hud.textContent = g.name;
    hud.style.setProperty("--chip", css(g.var));
    const a = M.hnp[f];
    const st = hnpStatus(a);
    $("#hud-hnp").innerHTML = `${st.icon} ${fmt(Math.abs(a), 0)}° ${esc(st.label)}`;
    const m = movAt(t);
    $("#hud-mov").textContent = m ? `${m.n} · ${m.letters}` : "Entrada";
    if (m && m.n !== curMov) {
      curMov = m.n;
      $$("#mov-marks button").forEach((b) => b.classList.toggle("cur", +b.textContent === m.n));
      $$("tr.row[data-mov]").forEach((r) => r.classList.toggle("cur", +r.dataset.mov === m.n));
    }
  });

  // ------------------------------------------------------------------ pestañas
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".pane").forEach((p) => p.classList.toggle("on", p.dataset.pane === b.dataset.tab));
    if (b.dataset.tab === "analysis") renderCharts();
  }));
  function showTab(name) { $(`.tabs button[data-tab="${name}"]`).click(); }

  // ------------------------------------------------------------------ Resumen
  const opp = R.opportunities;
  const lost = S.max - S.total;
  const low = movs.filter((m) => m.note < 6);
  const w = R.welfare;
  $("#pane-summary").innerHTML = `
    <div class="section kpis">
      <div class="kpi"><div class="v num">${fmt(S.percent, 1)}%</div><div class="l">Nota orientativa</div></div>
      <div class="kpi"><div class="v num">${fmt(lost, 1)}</div><div class="l">Puntos por debajo del máximo</div></div>
      <div class="kpi"><div class="v num">${low.length}</div><div class="l">Ejercicios por debajo de 6</div></div>
      <div class="kpi"><div class="v num">${S.penalties.length}</div><div class="l">Penalizaciones</div></div>
    </div>
    ${w.alert ? `<div class="section alert critical"><span class="ico" aria-hidden="true">✕</span>
      <div><b>Alerta de bienestar: posición de la nuca</b>${esc(w.message)}</div></div>` : `<div class="section alert good"><span class="ico" aria-hidden="true">✓</span><div><b>Posición de la nuca</b>Sin alertas de hiperflexión.</div></div>`}
    <div class="section">
      <h3>Dónde están los puntos</h3>
      <p class="secondary">Áreas ordenadas por los puntos que se pueden recuperar (con coeficientes aplicados).</p>
      ${opp.slice(0, 3).map((o) => oppCard(o, false)).join("")}
      <button class="link" id="see-all-opp">Ver todas las oportunidades de mejora →</button>
    </div>
    <div class="section">
      <h3>Lo mejor y lo peor de la prueba</h3>
      ${bestWorst()}
    </div>
    ${R.meta.synthetic ? `<div class="section alert warning"><span class="ico" aria-hidden="true">!</span><div><b>Proof of concept con datos sintéticos</b>
      El recorrido se ha generado por simulación con errores conocidos y ha pasado por el pipeline real (cámaras simuladas → triangulación → métricas → puntuación). Ver <button class="link" id="go-quality">Calidad 3D</button>.</div></div>` : ""}
  `;
  $("#see-all-opp").addEventListener("click", () => showTab("improve"));
  $("#go-quality")?.addEventListener("click", () => showTab("quality"));

  function bestWorst() {
    const sorted = [...movs].sort((a, b) => a.note - b.note);
    const worst = sorted.slice(0, 3), best = sorted.slice(-3).reverse();
    const li = (m) => `<div class="item"><div><b>${m.n}. ${esc(m.letters)}</b> <span class="secondary">${esc(m.text)}</span>
      ${m.deductions[0] ? `<div class="ev">${esc(m.deductions[0].reason)}</div>` : ""}</div>
      <div class="n"><span class="pts num">${fmt(m.note, 1)}</span> <button class="link" data-jump="${m.t0}" data-mov="${m.n}">ver</button></div></div>`;
    return `<div class="grid2"><div><div class="muted">A mejorar</div>${worst.map(li).join("")}</div>
      <div><div class="muted">Más logrados</div>${best.map(li).join("")}</div></div>`;
  }

  function oppCard(o, full) {
    const maxPts = opp[0]?.points || 1;
    const items = o.items.slice(0, full ? 99 : 2);
    return `<div class="opp">
      <div class="opp-head"><h4>${esc(o.title)}</h4><span class="pts">−${fmt(o.points, 1)} <small>pts · ${fmt(o.percent, 1)}%</small></span></div>
      <div class="bar" aria-hidden="true"><span style="width:${(100 * o.points) / maxPts}%"></span></div>
      ${full ? `<p class="secondary">${esc(o.why)}</p>` : ""}
      <div class="items">${items.map((i) => `<div class="item"><div>${esc(i.reason)}<div class="ev">${esc(i.where)} · ${esc(i.evidence)}</div></div>
        <div class="n">${i.t != null ? `<button class="link" data-jump="${i.t}" ${i.mov ? `data-mov="${i.mov}"` : ""}>ver en 3D</button>` : ""}</div></div>`).join("")}</div>
      ${full ? `<details open><summary>Cómo mejorarlo</summary><ul>${o.drills.map((d) => `<li>${esc(d)}</li>`).join("")}</ul></details>` : ""}
    </div>`;
  }

  // ------------------------------------------------------------------ Mejoras
  $("#pane-improve").innerHTML = `<div class="section"><p class="secondary">Cada área agrupa las deducciones detectadas, con la evidencia medida en 3D y ejercicios de entrenamiento.
    Los puntos incluyen los coeficientes de la hoja (conjunto ×2).</p>${opp.map((o) => oppCard(o, true)).join("")}</div>
    ${S.not_evaluable.length ? `<div class="section alert warning"><span class="ico" aria-hidden="true">!</span><div><b>No evaluable en esta versión</b>
      ${S.not_evaluable.map((n) => `${esc(n.reason)} — ${esc(n.why)}`).join("<br>")}</div></div>` : ""}`;

  // ------------------------------------------------------------------ Hoja de puntuación
  const sumMov = movs.reduce((s, m) => s + m.note * m.coef, 0);
  const sumCol = S.collective.reduce((s, c) => s + c.note * c.coef, 0);
  const noteCell = (v) => `<span class="note"><span class="num">${fmt(v, 1)}</span><span class="note-bar" aria-hidden="true"><span style="width:${v * 10}%"></span></span></span>`;
  const dedList = (ds) => ds.map((d) => `<div class="ded"><b>−${fmt(d.pts, 1)}</b> ${esc(d.reason)} <span class="muted">· ${esc(d.evidence)}</span></div>`).join("");
  $("#pane-sheet").innerHTML = `
    <p class="secondary">Haz clic en un ejercicio para verlo en 3D. Nota base sin defectos medibles: ${fmt(movs[0].base, 0)} (la visión no puede acreditar la expresión que justifica un 10).</p>
    <table>
      <thead><tr><th class="n">Nº</th><th>Ejercicio</th><th class="n">Nota</th><th class="n">Coef.</th><th class="n">Total</th></tr></thead>
      <tbody>
        ${movs.map((m) => `<tr class="row" data-mov="${m.n}" data-t="${m.t0}">
          <td class="n">${m.n}</td>
          <td><b>${esc(m.letters)}</b> ${esc(m.text)}
            <div class="ded muted">${esc(m.directives)}</div>${dedList(m.deductions)}
            ${m.capped ? `<span class="flag">Error grave · nota limitada a 5</span>` : ""}</td>
          <td class="n">${noteCell(m.note)}</td><td class="n">${m.coef}</td><td class="n num">${fmt(m.note * m.coef, 1)}</td></tr>`).join("")}
        <tr><td colspan="5" class="muted" style="padding-top:14px">Notas de conjunto</td></tr>
        ${S.collective.map((c) => `<tr><td class="n">${c.key}</td><td>${esc(c.text)}${dedList(c.deductions)}</td>
          <td class="n">${noteCell(c.note)}</td><td class="n">${c.coef}</td><td class="n num">${fmt(c.note * c.coef, 1)}</td></tr>`).join("")}
        ${S.penalties.map((p) => `<tr><td class="n">!</td><td>${esc(p.reason)}<div class="ded muted">${esc(p.rule)}</div>
          <button class="link" data-jump="${p.t}">ver en 3D</button></td><td></td><td></td><td class="n num">−${fmt(p.pts, 1)}</td></tr>`).join("")}
      </tbody>
      <tfoot><tr><td></td><td>Total (${fmt(sumMov, 1)} ejercicios + ${fmt(sumCol, 1)} conjunto${S.penalty_pts ? ` − ${fmt(S.penalty_pts, 0)} penalización` : ""})</td>
        <td></td><td></td><td class="n num">${fmt(S.total, 1)} / ${S.max}</td></tr>
        <tr><td></td><td>Porcentaje</td><td></td><td></td><td class="n num">${fmt(S.percent, 2)}%</td></tr></tfoot>
    </table>`;
  $$("#pane-sheet tr.row").forEach((r) => r.addEventListener("click", () => jumpTo(+r.dataset.t, +r.dataset.mov)));

  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-jump]");
    if (!b) return;
    jumpTo(+b.dataset.jump, b.dataset.mov ? +b.dataset.mov : null);
  });

  // ------------------------------------------------------------------ Análisis (gráficos)
  const ser = R.series;
  const SERIES = {
    hnp: { label: "Ángulo del plano nasal", unit: "°", data: ser.hnp, dom: [-20, 15], refs: [[0, "vertical"], [-10, "10° detrás"]],
      fmt: (v) => `${fmt(Math.abs(v), 1)}° ${v >= 0 ? "por delante" : "detrás"}`, note: "Positivo = nariz por delante de la vertical (lo correcto). Negativo = detrás." },
    lean: { label: "Inclinación del tronco del jinete", unit: "°", data: ser.lean, dom: [-5, 20], refs: [[0, "vertical"], [8, "tolerancia"]],
      fmt: (v) => `${fmt(v, 1)}° ${v >= 0 ? "hacia delante" : "hacia atrás"}`, note: "0° = tronco vertical. Positivo = inclinado hacia delante." },
    speed: { label: "Velocidad del caballo", unit: "m/s", data: ser.speed, dom: [0, 6], refs: [],
      fmt: (v) => `${fmt(v, 2)} m/s`, note: "Velocidad del centro del tronco." },
  };
  let curSeries = "hnp";
  let selMov = 0;

  $("#pane-analysis").innerHTML = `
    <div class="section">
      <h3>Evolución durante la prueba</h3>
      <div class="chart-tabs" role="group" aria-label="Serie">
        <button data-s="hnp" class="on">Nuca (HNP)</button><button data-s="lean">Tronco jinete</button><button data-s="speed">Velocidad</button>
      </div>
      <p class="muted" id="series-note"></p>
      <div id="ts"></div>
      <div class="legend" id="gait-legend" style="margin-top:8px"></div>
      <div id="strip"></div>
    </div>
    <div class="section">
      <h3>Trazado sobre la pista</h3>
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">
        <select id="mov-sel" aria-label="Ejercicio"><option value="0">Todo el recorrido</option>
          ${movs.map((m) => `<option value="${m.n}">${m.n}. ${esc(m.letters)} — ${esc(m.text.slice(0, 48))}</option>`).join("")}</select>
        <span class="legend" style="margin:0"><span><i style="background:var(--accent)"></i>Trayectoria medida</span>
        <span><i style="background:transparent;border:1px dashed var(--text-secondary)"></i>Figura teórica</span>
        <span><i style="background:var(--critical)"></i>Ejercicio seleccionado</span></span>
      </div>
      <div id="topview"></div>
      <div id="fig-detail" class="secondary" style="margin-top:6px"></div>
    </div>
    <div class="section">
      <h3>Resumen por aire</h3>
      <table><thead><tr><th>Aire</th><th class="n">Tiempo</th><th class="n">Velocidad</th><th class="n">Nuca media</th><th class="n">% detrás</th><th class="n">% >10° detrás</th><th class="n">Tronco jinete</th></tr></thead>
      <tbody>${Object.entries(R.analysis.by_gait).map(([g, v]) => `<tr><td>${{ walk: "Paso", trot: "Trote", canter: "Galope" }[g] || g}</td>
        <td class="n">${fmt(v.seconds, 0)} s</td><td class="n">${fmt(v.speed_mean, 2)} m/s</td>
        <td class="n">${fmt(Math.abs(v.hnp.mean_deg), 1)}° ${v.hnp.mean_deg >= 0 ? "delante" : "detrás"}</td>
        <td class="n">${fmt(v.hnp.pct_behind, 0)}%</td><td class="n">${fmt(v.hnp.pct_behind10, 0)}%</td><td class="n">${fmt(v.rider_lean_deg, 1)}°</td></tr>`).join("")}</tbody></table>
    </div>
    <div class="section">
      <h3>Transiciones detectadas</h3>
      <table><thead><tr><th class="n">Tiempo</th><th>De</th><th>A</th><th></th></tr></thead><tbody>
      ${R.analysis.transitions.map((t) => `<tr><td class="n">${fmtT(t.t)}</td><td>${gaitName(t.from)}</td><td>${gaitName(t.to)}</td>
        <td class="n"><button class="link" data-jump="${Math.max(0, t.t - 2)}">ver</button></td></tr>`).join("")}</tbody></table>
    </div>`;
  function gaitName(k) { return (GAITS.find((g) => g.key === k) || GAITS[5]).name; }
  $$(".chart-tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".chart-tabs button").forEach((x) => x.classList.toggle("on", x === b));
    curSeries = b.dataset.s;
    renderCharts();
  }));
  $("#mov-sel").addEventListener("change", (e) => { selMov = +e.target.value; renderTop(); if (selMov) setTime(movs[selMov - 1].t0); });
  $("#gait-legend").innerHTML = GAITS.slice(0, 5).map((g) => `<span><i style="background:var(${g.var})"></i>${g.name}</span>`).join("");

  function highlightMov(n) {
    selMov = n;
    const sel = $("#mov-sel");
    if (sel) sel.value = n;
    renderTop();
  }

  const tip = $("#tooltip");
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let lx = x + 14, ly = y + 14;
    if (lx + r.width > innerWidth - 8) lx = x - r.width - 14;
    if (ly + r.height > innerHeight - 8) ly = y - r.height - 14;
    tip.style.left = `${lx}px`; tip.style.top = `${ly}px`;
  }
  const hideTip = () => (tip.hidden = true);

  let tsCursor = null, topDot = null;
  function renderCharts() {
    if (!$("#ts")) return;
    renderTS();
    renderStrip();
    renderTop();
  }

  function renderTS() {
    const host = $("#ts");
    const W = Math.max(320, host.clientWidth || 600), H = 210;
    const m = { l: 40, r: 12, t: 10, b: 24 };
    const sp = SERIES[curSeries];
    $("#series-note").textContent = sp.note;
    const n = sp.data.length, sfps = ser.fps;
    const x = (t) => m.l + (t / duration) * (W - m.l - m.r);
    const [y0, y1] = sp.dom;
    const y = (v) => m.t + (1 - (Math.min(y1, Math.max(y0, v)) - y0) / (y1 - y0)) * (H - m.t - m.b);
    const ticks = [];
    const step = (y1 - y0) / 5 <= 1 ? 1 : (y1 - y0) / 5 <= 5 ? 5 : 10;
    for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) ticks.push(v);
    let d = "";
    for (let i = 0; i < n; i++) { const v = sp.data[i]; if (v == null) continue; d += `${d ? "L" : "M"}${x(i / sfps).toFixed(1)},${y(v).toFixed(1)}`; }
    host.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(sp.label)} a lo largo de la prueba">
      ${ticks.map((v) => `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("")}
      ${sp.refs.map(([v, l]) => `<line class="ref" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text class="lbl" x="${W - m.r}" y="${y(v) - 4}" text-anchor="end">${l}</text>`).join("")}
      <line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>
      ${movs.map((mv) => `<line class="grid" x1="${x(mv.t0)}" x2="${x(mv.t0)}" y1="${H - m.b}" y2="${H - m.b + 4}"/><text x="${x(mv.t0)}" y="${H - 6}" text-anchor="middle">${mv.n}</text>`).join("")}
      <text x="${m.l}" y="${H - 6}" text-anchor="start" style="display:none">ej.</text>
      <path class="series" d="${d}"/>
      <line class="cursor" id="ts-cursor" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}"/>
      <g id="ts-hover" style="display:none"><line class="hair" y1="${m.t}" y2="${H - m.b}"/><circle class="dot" r="4.5"/></g>
      <rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" id="ts-hit" style="cursor:crosshair"/>
    </svg>`;
    tsCursor = { el: $("#ts-cursor"), x };
    moveCursor(state.t);
    const hit = $("#ts-hit"), hov = $("#ts-hover");
    const svg = host.querySelector("svg");
    const tAt = (ev) => { const r = svg.getBoundingClientRect(); const px = ((ev.clientX - r.left) / r.width) * W; return Math.min(duration, Math.max(0, ((px - m.l) / (W - m.l - m.r)) * duration)); };
    hit.addEventListener("mousemove", (ev) => {
      const t = tAt(ev), i = Math.min(n - 1, Math.round(t * sfps)), v = sp.data[i];
      hov.style.display = "";
      hov.querySelector("line").setAttribute("x1", x(t)); hov.querySelector("line").setAttribute("x2", x(t));
      hov.querySelector("circle").setAttribute("cx", x(t)); hov.querySelector("circle").setAttribute("cy", y(v));
      const mv = movAt(t), g = gaitBy(ser.gait[i]);
      showTip(`<div class="t">${fmtT(t)} · ${mv ? `Ejercicio ${mv.n}` : "Entrada"} · ${g.name}</div><div class="v">${sp.fmt(v)}</div><div class="t">Clic para ir a este instante</div>`, ev.clientX, ev.clientY);
    });
    hit.addEventListener("mouseleave", () => { hov.style.display = "none"; hideTip(); });
    hit.addEventListener("click", (ev) => setTime(tAt(ev)));
  }

  function renderStrip() {
    const host = $("#strip");
    const W = Math.max(320, host.clientWidth || 600), H = 26, ml = 40, mr = 12;
    const x = (t) => ml + (t / duration) * (W - ml - mr);
    // bloques de aire a partir de las transiciones detectadas (coherente con la tabla)
    const sfps = 1, runs = [];
    const code = (k) => (GAITS.find((g) => g.key === k) || GAITS[5]).code;
    const tr = R.analysis.transitions;
    if (tr.length) {
      runs.push({ c: code(tr[0].from), s: 0, e: tr[0].t });
      tr.forEach((t, i) => runs.push({ c: code(t.to), s: t.t, e: i + 1 < tr.length ? tr[i + 1].t : duration }));
    }
    host.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Aire detectado a lo largo de la prueba">
      <text x="${ml - 6}" y="${H / 2 + 4}" text-anchor="end">Aire</text>
      ${runs.map((r) => {
        const gg = gaitBy(r.c), x0 = x(r.s / sfps), x1 = x(r.e / sfps), w = Math.max(0, x1 - x0 - 2);
        return `<rect x="${x0 + 1}" y="2" width="${w}" height="${H - 4}" rx="3" fill="var(${gg.var})" data-run="${r.s}"/>` +
          (w > 46 && gg.short ? `<text class="seg-lbl ${gg.darkText ? "dark" : ""}" x="${x0 + 1 + w / 2}" y="${H / 2 + 4}" text-anchor="middle">${gg.short}</text>` : "");
      }).join("")}
    </svg>`;
    host.querySelectorAll("rect[data-run]").forEach((el) => {
      const r = runs.find((q) => q.s === +el.dataset.run);
      el.addEventListener("mousemove", (ev) => showTip(`<div class="v">${gaitBy(r.c).name}</div><div class="t">${fmtT(r.s / sfps)} – ${fmtT(r.e / sfps)} · ${fmt((r.e - r.s) / sfps, 1)} s</div>`, ev.clientX, ev.clientY));
      el.addEventListener("mouseleave", hideTip);
      el.addEventListener("click", () => setTime(r.s / sfps));
      el.style.cursor = "pointer";
    });
  }

  function renderTop() {
    const host = $("#topview");
    if (!host) return;
    const xy = ser.xy, sfps = ser.fps;
    const vx = (x) => x, vy = (y) => -y;
    const pathOf = (a, b) => { let d = ""; for (let i = a; i < b; i++) d += `${d ? "L" : "M"}${vx(xy[i][0]).toFixed(2)},${vy(xy[i][1]).toFixed(2)}`; return d; };
    let sel = "";
    let detail = "";
    if (selMov) {
      const mv = movs[selMov - 1];
      sel = pathOf(Math.floor(mv.t0 * sfps), Math.min(xy.length, Math.ceil(mv.t1 * sfps)));
      const am = R.analysis.movements.find((q) => q.n === selMov);
      detail = am.checks.map((c) => {
        if (c.type === "circle") return `Círculo: ${fmt(c.diameter_m, 2)} m (teórico ${c.expected_m} m, ${c.diam_err_pct > 0 ? "+" : ""}${fmt(c.diam_err_pct, 1)}%), centro desplazado ${fmt(c.center_offset_m, 2)} m, redondez RMS ${fmt(c.roundness_rms_m, 2)} m.`;
        if (c.type === "line") return `Línea ${c.from}→${c.to}: desviación RMS ${fmt(c.rms_dev_m, 2)} m (máx. ${fmt(c.max_dev_m, 2)} m).`;
        if (c.type === "halt") return `Parada en ${c.at}: ${fmt(c.duration_s, 1)} s, a ${fmt(c.dist_letter_m, 2)} m de la letra; desfase cascos anteriores ${fmt(c.front_offset_cm, 0)} cm / posteriores ${fmt(c.hind_offset_cm, 0)} cm.`;
        if (c.type === "transition") return c.missing ? `Transición en ${c.at}: no detectada.` : `Transición a ${gaitName(c.to).toLowerCase()} en ${c.at}: a ${fmt(Math.abs(c.late_m), 1)} m (${c.late_m > 0 ? "tarde" : "pronto"}).`;
        if (c.type === "lengthening") return `Galope largo: ${fmt(c.speed_ext, 2)} m/s frente a ${fmt(c.speed_ref, 2)} m/s de trabajo (×${fmt(c.ratio, 2)}); tranco ${fmt(c.stride_ext_m, 2)} m vs ${fmt(c.stride_ref_m, 2)} m.`;
        if (c.type === "hnp") return `Nuca: media ${fmt(Math.abs(c.mean_deg), 1)}° ${c.mean_deg >= 0 ? "por delante" : "detrás"}, ${fmt(c.pct_behind, 0)}% del tiempo detrás de la vertical.`;
        if (c.type === "transition_zone") return `Transición al trote en x=${fmt(c.x_m, 1)} m (zona ${c.zone[0]}–${c.zone[1]} m).`;
        return "";
      }).filter(Boolean).join("<br>");
      if (am.gait_faults) detail += `<br>Ruptura de aire: ${am.gait_faults.map((f) => `${f.strides} trancos de ${gaitName(f.gait).toLowerCase()} (${fmtT(f.t0)}–${fmtT(f.t1)})`).join(", ")}.`;
    }
    const L = R.arena.letters;
    host.innerHTML = `<svg class="chart topview" viewBox="-3 -12.5 46 25" role="img" aria-label="Vista cenital de la trayectoria">
      <rect class="fence" x="0" y="-10" width="40" height="20" stroke-width="0.08"/>
      <line x1="0" x2="40" y1="0" y2="0" stroke="var(--axis)" stroke-width="0.05" stroke-dasharray="0.4 0.4"/>
      ${R.expected.map((f) => f.type === "circle"
        ? `<circle class="exp" cx="${vx(f.center[0])}" cy="${vy(f.center[1])}" r="${f.r}" style="stroke-width:0.07;stroke-dasharray:0.35 0.25"/>`
        : `<line class="exp" x1="${vx(f.a[0])}" y1="${vy(f.a[1])}" x2="${vx(f.b[0])}" y2="${vy(f.b[1])}" style="stroke-width:0.07;stroke-dasharray:0.35 0.25"/>`).join("")}
      <path class="path" d="${pathOf(0, xy.length)}" style="stroke-width:0.09"/>
      ${sel ? `<path class="path-sel" d="${sel}" style="stroke-width:0.16"/>` : ""}
      ${Object.entries(L).map(([k, [x, y]]) => {
        const ox = x === 0 ? -1.3 : x === 40 ? 1.3 : 0, oy = y > 0 ? 1.3 : y < 0 ? -1.3 : 0, line = ["D", "X", "G"].includes(k);
        return `<text class="letter" x="${vx(x + ox)}" y="${vy(y + (line ? 0.9 : oy))}" style="font-size:${line ? 0.8 : 1.1}px">${k}</text>`;
      }).join("")}
      <circle class="horse-dot" id="top-dot" r="0.45" style="stroke-width:0.12"/>
    </svg>`;
    $("#fig-detail").innerHTML = detail;
    topDot = $("#top-dot");
    moveCursor(state.t);
  }

  function moveCursor(t) {
    if (tsCursor && tsCursor.el.isConnected) { const px = tsCursor.x(t); tsCursor.el.setAttribute("x1", px); tsCursor.el.setAttribute("x2", px); }
    if (topDot && topDot.isConnected) {
      const i = Math.min(ser.xy.length - 1, Math.round(t * ser.fps));
      topDot.setAttribute("cx", ser.xy[i][0]); topDot.setAttribute("cy", -ser.xy[i][1]);
    }
  }
  onTime(moveCursor);
  let rz;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => $(".pane.on")?.dataset.pane === "analysis" && renderCharts(), 150); });

  // ------------------------------------------------------------------ Calidad 3D
  const V = R.validation;
  $("#pane-quality").innerHTML = `
    <div class="section">
      <h3>Precisión de la reconstrucción</h3>
      <p class="secondary">En el PoC se conoce la verdad del movimiento, así que se mide el error del pipeline completo
        (calibración con letras + detecciones 2D con ruido, oclusiones y falsos positivos + triangulación RANSAC + filtrado).</p>
      <div class="kpis">
        <div class="kpi"><div class="v num">${fmt(V.mpjpe_cm, 1)} cm</div><div class="l">Error medio por punto (MPJPE)</div></div>
        <div class="kpi"><div class="v num">${fmt(V.p95_cm, 1)} cm</div><div class="l">Percentil 95</div></div>
        <div class="kpi"><div class="v num">${fmt(V.gait_accuracy_pct, 1)}%</div><div class="l">Aire correcto por frame</div></div>
        <div class="kpi"><div class="v num">${fmt(V.hnp_mae_deg, 1)}°</div><div class="l">Error medio ángulo nuca</div></div>
      </div>
    </div>
    <div class="section">
      <h3>Puntuación: reconstrucción frente a verdad</h3>
      <p>Nota calculada desde la reconstrucción 3D: <b class="num">${fmt(S.percent, 2)}%</b> · desde el movimiento real: <b class="num">${fmt(V.score_from_truth_pct, 2)}%</b>.
      ${V.notes_diff.length ? `Ejercicios con nota distinta: ${V.notes_diff.map((d) => `${d.n} (${fmt(d.rec, 1)} vs ${fmt(d.truth, 1)})`).join(", ")}.` : "Todas las notas por ejercicio coinciden."}</p>
    </div>
    <div class="section grid2">
      <div><h3>Error por zona</h3><table><tbody>${Object.entries(V.group_cm).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="n">${fmt(v, 1)} cm</td></tr>`).join("")}
        <tr><td>Cobertura bruta (≥2 vistas)</td><td class="n">${fmt(V.raw_coverage_pct, 1)}%</td></tr>
        <tr><td>Error de reproyección</td><td class="n">${fmt(V.reproj_px, 2)} px</td></tr>
        <tr><td>Error inclinación del jinete</td><td class="n">${fmt(V.lean_mae_deg, 1)}°</td></tr></tbody></table></div>
      <div><h3>Calibración de cámaras (letras de la pista)</h3><table class="cam-table"><thead><tr><th>Cámara</th><th class="n">Puntos</th><th class="n">RMSE</th><th class="n">Error posición</th></tr></thead>
        <tbody>${R.calibration.map((c) => `<tr><td>${esc(c.name)}</td><td class="n">${c.calib_points}</td><td class="n">${fmt(c.calib_rmse_px, 2)} px</td><td class="n">${fmt(c.pos_err_cm, 1)} cm</td></tr>`).join("")}</tbody></table></div>
    </div>
    <div class="section">
      <h3>Limitaciones de esta versión</h3>
      <ul class="secondary">
        <li>El movimiento y las detecciones 2D son sintéticos; con vídeo real el error dependerá del detector de keypoints (RTMPose / DeepLabCut ajustado con PFERD).</li>
        <li>Los umbrales de las reglas son provisionales y deben calibrarse con hojas de jueces reales.</li>
        <li>No se evalúa la voz/chasquidos (necesita audio), la expresión, la boca, las orejas ni la cola.</li>
        <li>La posición de la nuca se penaliza por regla explícita, no se aprende de las notas de los jueces (la literatura muestra que estas premian la hiperflexión).</li>
      </ul>
    </div>`;

  // arranque
  setTime(movs[0].t0 > 1 ? movs[0].t0 - 1 : 0);
})();
