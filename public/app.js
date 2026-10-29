/* ColdStart GAN — app */
(function () {
  'use strict';
  const E = window.Engine, CATALOG = window.CATALOG;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const C = { real: css('--real'), synth: css('--synth'), disc: css('--disc'), pop: css('--pop'), text: css('--text'), muted: css('--muted'), faint: css('--faint'), line: 'rgba(174,180,214,.14)', ink: css('--ink') };
  const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
  const pct = (x) => Math.round(x * 100) + '%';
  const f2 = (x) => x.toFixed(2);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------- state ----------------
  const world = E.makeWorld(7);
  const S = {
    domain: 'movies', biased: false, lr: 0.003, speed: 8,
    training: true, stopAt: 1500, firstRun: true,
    view: 'overview', heroN: 5000, playN: 2000,
    liked: new Set(), probe: null, probeCount: 0,
    synthCache: new Map(), simAugCache: new Map(),
    lastHeroEval: -1, bench: null, benchEpoch: -1,
    msPerStep: 0, stage: 3, resampleSeed: 1,
  };
  function rebuild() {
    S.real = E.makeRealData(world, 20, { seed: 11, biased: S.biased });
    S.realRows = S.real.map((u) => u.row);
    S.gan = new E.GAN(world, { lr: S.lr });
    S.pop = E.popularity(S.realRows, world.nItems);
    S.simReal = E.itemSimilarity([{ rows: S.realRows, weight: 1 }], world.nItems);
    invalidate();
    initRealDots();
  }
  function invalidate() { S.synthCache.clear(); S.simAugCache.clear(); S.lastHeroEval = -1; }
  function getSynth(n) {
    const key = S.gan.epoch + ':' + n;
    if (!S.synthCache.has(key)) {
      if (S.synthCache.size > 8) S.synthCache.delete(S.synthCache.keys().next().value);
      S.synthCache.set(key, S.gan.sample(n, 99));
    }
    return S.synthCache.get(key);
  }
  function getSimAug(n) {
    const key = S.gan.epoch + ':' + n;
    if (!S.simAugCache.has(key)) {
      if (S.simAugCache.size > 8) S.simAugCache.delete(S.simAugCache.keys().next().value);
      S.simAugCache.set(key, E.itemSimilarity([{ rows: S.realRows, weight: 1 }, { rows: getSynth(n), weight: 1 }], world.nItems));
    }
    return S.simAugCache.get(key);
  }
  const dom = () => CATALOG[S.domain];
  const itemName = (i) => dom().items[world.items[i].cat][i % 6];
  const catName = (c) => dom().cats[c];

  // ---------------- canvas helpers ----------------
  function fit(cv) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return null;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  function ema(arr, a = 0.06) { const out = []; let v = arr[0]; for (const x of arr) { v = v + a * (x - v); out.push(v); } return out; }
  function downsample(arr, max) { if (arr.length <= max) return arr; const out = [], step = arr.length / max; for (let i = 0; i < max; i++) out.push(arr[Math.floor(i * step)]); out.push(arr[arr.length - 1]); return out; }

  function lineChart(cv, series, opt = {}) {
    const f = fit(cv); if (!f) return; const { ctx, w, h } = f;
    ctx.clearRect(0, 0, w, h);
    const padL = 34, padR = 10, padT = 8, padB = opt.xLabels ? 24 : 18;
    const iw = w - padL - padR, ih = h - padT - padB;
    let yMin = opt.yMin ?? 0, yMax = opt.yMax;
    if (yMax == null) { yMax = 0; series.forEach((s) => s.data.forEach((v) => (yMax = Math.max(yMax, v)))); yMax = Math.max(0.5, Math.ceil(yMax * 4) / 4); }
    const n = Math.max(...series.map((s) => s.data.length), 2);
    const X = (i) => padL + (opt.xs ? (opt.xs[i] - opt.xs[0]) / (opt.xs[opt.xs.length - 1] - opt.xs[0]) : i / (n - 1)) * iw;
    const Y = (v) => padT + ih - ((v - yMin) / (yMax - yMin)) * ih;
    ctx.font = '11px ' + css('--font-body'); ctx.fillStyle = C.faint; ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    const ticks = opt.yTicks || 4;
    for (let t = 0; t <= ticks; t++) {
      const v = yMin + ((yMax - yMin) * t) / ticks, y = Math.round(Y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(opt.yFmt ? opt.yFmt(v) : v.toFixed(v < 2 ? 2 : 1), padL - 6, y);
    }
    if (opt.ref != null) { ctx.setLineDash([3, 4]); ctx.strokeStyle = 'rgba(243,243,250,.35)'; ctx.beginPath(); ctx.moveTo(padL, Y(opt.ref)); ctx.lineTo(w - padR, Y(opt.ref)); ctx.stroke(); ctx.setLineDash([]); }
    if (opt.xLabels) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; opt.xLabels.forEach((l, i) => ctx.fillText(l, X(i), h - padB + 7)); }
    else if (opt.xMaxLabel) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('0', padL, h - padB + 4); ctx.textAlign = 'right'; ctx.fillText(opt.xMaxLabel, w - padR, h - padB + 4); }
    series.forEach((s) => {
      if (s.data.length < 2) return;
      if (s.fill) {
        const g = ctx.createLinearGradient(0, padT, 0, padT + ih); g.addColorStop(0, s.color + '33'); g.addColorStop(1, s.color + '00');
        ctx.beginPath(); s.data.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.lineTo(X(s.data.length - 1), padT + ih); ctx.lineTo(X(0), padT + ih); ctx.closePath(); ctx.fillStyle = g; ctx.fill();
      }
      ctx.beginPath(); s.data.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.lineJoin = 'round'; ctx.stroke();
      const li = s.data.length - 1;
      ctx.beginPath(); ctx.arc(X(li), Y(s.data[li]), s.dots ? 4 : 3.5, 0, Math.PI * 2); ctx.fillStyle = s.color; ctx.fill();
      if (s.dots) s.data.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(v), 3.2, 0, Math.PI * 2); ctx.fillStyle = s.color; ctx.fill(); });
    });
  }

  // ---------------- brand mark ----------------
  function drawBrand(t) {
    const f = fit($('#brandMark')); if (!f) return; const { ctx, w, h } = f;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    ctx.strokeStyle = 'rgba(174,180,214,.3)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, 15, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = C.real; ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
    for (let k = 0; k < 3; k++) {
      const a = t / 1400 + (k * Math.PI * 2) / 3;
      ctx.fillStyle = k === 2 ? C.disc : C.synth;
      ctx.beginPath(); ctx.arc(cx + Math.cos(a) * 15, cy + Math.sin(a) * 15, 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  // ---------------- domain words ----------------
  function applyDomain() {
    const d = dom();
    $$('[data-word="label-lower"]').forEach((el) => (el.textContent = d.label.toLowerCase()));
    $$('[data-word="unit"]').forEach((el) => (el.textContent = d.unit));
    renderItemGrid(); renderRecs(); renderCatBars(); renderPipe(); renderChallenges();
    if (S.bench) renderBench();
  }

  // ---------------- views ----------------
  function go(view) {
    S.view = view;
    $$('.view').forEach((v) => v.classList.toggle('is-active', v.dataset.view === view));
    $$('.rail-btn').forEach((b) => { const on = b.dataset.go === view; b.classList.toggle('is-active', on); if (on && window.innerWidth <= 860) b.scrollIntoView({ inline: 'center', block: 'nearest' }); });
    window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' });
    if (view === 'eval' && (!S.bench || S.benchEpoch !== S.gan.epoch) && !S.training) runBench();
    if (view === 'eval') updateBenchMeta();
    if (view === 'play') renderRecs();
    if (view === 'arch') { renderPipe(); renderChallenges(); }
    if (view === 'lab') { renderCatBars(); drawMatrices(); }
    try { history.replaceState(null, '', '#' + view); } catch (e) { /* ignore */ }
  }

  let toastT;
  function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 2600); }

  // ---------------- hero ----------------
  const hero = { shown: 0 };
  function drawHeroMatrix() {
    const f = fit($('#heroMatrix')); if (!f) return; const { ctx, w, h } = f;
    ctx.clearRect(0, 0, w, h);
    const nI = world.nItems, top = 18, gapCat = 6;
    const cw = (w - gapCat * 3) / nI;
    const colX = (i) => i * cw + world.items[i].cat * gapCat;
    ctx.font = '10.5px ' + css('--font-body'); ctx.textBaseline = 'top';
    for (let c = 0; c < 4; c++) { ctx.fillStyle = C.faint; ctx.textAlign = 'left'; ctx.fillText(catName(c), colX(c * 6) + 1, 0, cw * 6 - 2); }
    const realH = 4, realGap = 1.5;
    S.realRows.forEach((r, u) => {
      for (let i = 0; i < nI; i++) {
        const x = colX(i) + 0.75, y = top + u * (realH + realGap);
        if (world.items[i].isNew) { ctx.strokeStyle = 'rgba(174,180,214,.18)'; ctx.strokeRect(x + 0.5, y + 0.5, cw - 2.5, realH - 1); continue; }
        ctx.fillStyle = r[i] ? C.real : 'rgba(174,180,214,.07)';
        ctx.fillRect(x, y, cw - 1.5, realH);
      }
    });
    const sTop = top + 20 * (realH + realGap) + 16;
    const maxRows = Math.max(0, Math.floor(h - sTop - 2));
    const target = S.heroN === 0 ? 0 : Math.max(4, Math.round((S.heroN / 5000) * maxRows));
    hero.shown += (target - hero.shown) * (reduceMotion ? 1 : 0.12);
    const rows = Math.round(hero.shown);
    const syn = S.heroN ? getSynth(Math.max(S.heroN, maxRows)) : [];
    for (let u = 0; u < rows && u < syn.length; u++) {
      const r = syn[u];
      for (let i = 0; i < nI; i++) {
        if (!r[i]) continue;
        ctx.fillStyle = world.items[i].isNew ? 'rgba(94,234,212,.95)' : 'rgba(94,234,212,.62)';
        ctx.fillRect(colX(i) + 0.75, sTop + u, cw - 1.5, 1);
      }
    }
    ctx.fillStyle = C.faint; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    if (S.heroN) ctx.fillText(`showing ${rows} of ${fmtInt(S.heroN)} synthetic rows`, w, sTop - 3);
    else ctx.fillText('no synthetic rows', w, sTop - 3);
  }
  function updateHeroStats(force) {
    if (!force && S.lastHeroEval === S.gan.epoch) return;
    S.lastHeroEval = S.gan.epoch;
    const ev = E.evaluate(world, S.realRows, getSynth(S.heroN), { nTest: 300 });
    $('#hsRealP').textContent = f2(ev.real.precision); $('#hsAugP').textContent = f2(ev.augmented.precision);
    $('#hsRealN').textContent = pct(ev.real.newItemHit); $('#hsAugN').textContent = pct(ev.augmented.newItemHit);
    $('#hsRealC').textContent = pct(ev.real.coverage); $('#hsAugC').textContent = pct(ev.augmented.coverage);
  }

  // ---------------- lab: duel canvas ----------------
  const corners = [[0, 0], [1, 0], [1, 1], [0, 1]];
  function project(row, rng) {
    const c = [0, 0, 0, 0]; let t = 0;
    row.forEach((v, i) => { if (v) { c[world.items[i].cat] += v; t += v; } });
    let x = 0.5, y = 0.5;
    if (t) { x = 0; y = 0; for (let k = 0; k < 4; k++) { x += (c[k] / t) * corners[k][0]; y += (c[k] / t) * corners[k][1]; } }
    const j = rng ? 0.05 : 0;
    return [x + (rng ? (rng() - 0.5) * j : 0), y + (rng ? (rng() - 0.5) * j : 0)];
  }
  let realDots = [], parts = [], duelFrame = 0, partSeed = 1000;
  function initRealDots() {
    const r = E.mulberry32(42);
    realDots = S.realRows.map((row) => project(row, r));
    parts = [];
  }
  function refreshParticles() {
    const rows = S.gan.sample(180, partSeed++);
    const r = E.mulberry32(partSeed);
    rows.forEach((row, k) => {
      const [tx, ty] = project(row, r);
      const s = S.gan.score(row);
      if (!parts[k]) parts[k] = { x: 0.5, y: 0.5, tx, ty, s };
      Object.assign(parts[k], { tx, ty, s });
    });
  }
  function mixColor(s) {
    // teal when D is fooled (s high), violet when caught (s low)
    const a = [94, 234, 212], b = [167, 139, 250], t = Math.max(0, Math.min(1, s));
    return `rgb(${Math.round(b[0] + (a[0] - b[0]) * t)},${Math.round(b[1] + (a[1] - b[1]) * t)},${Math.round(b[2] + (a[2] - b[2]) * t)})`;
  }
  function drawDuel() {
    const f = fit($('#duel')); if (!f) return; const { ctx, w, h } = f;
    duelFrame++;
    if (duelFrame % (S.training ? 6 : 45) === 1 || !parts.length) refreshParticles();
    ctx.fillStyle = 'rgba(16,12,42,.34)'; ctx.fillRect(0, 0, w, h);
    const pad = 44, size = Math.min(w, h) - pad * 2, ox = (w - size) / 2, oy = (h - size) / 2;
    const P = (x, y) => [ox + x * size, oy + y * size];
    ctx.strokeStyle = 'rgba(174,180,214,.16)'; ctx.lineWidth = 1;
    ctx.strokeRect(ox + 0.5, oy + 0.5, size, size);
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + size, oy + size); ctx.moveTo(ox + size, oy); ctx.lineTo(ox, oy + size); ctx.strokeStyle = 'rgba(174,180,214,.06)'; ctx.stroke();
    ctx.font = '600 12px ' + css('--font-body'); ctx.fillStyle = C.muted;
    [[0, 'left', 'bottom', -6], [1, 'right', 'bottom', -6], [2, 'right', 'top', 6], [3, 'left', 'top', 6]].forEach(([k, al, bl, dy]) => {
      ctx.textAlign = al; ctx.textBaseline = bl; const [x, y] = P(corners[k][0], corners[k][1]); ctx.fillText(catName(k), x, y + dy);
    });
    parts.forEach((p) => {
      const e = reduceMotion ? 1 : 0.09; p.x += (p.tx - p.x) * e; p.y += (p.ty - p.y) * e;
      const [x, y] = P(p.x, p.y);
      ctx.fillStyle = mixColor(p.s); ctx.globalAlpha = 0.85;
      ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 1;
    realDots.forEach(([rx, ry]) => {
      const [x, y] = P(rx, ry);
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fillStyle = 'rgba(245,184,75,.10)'; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, 4.6, 0, Math.PI * 2); ctx.fillStyle = C.real; ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(13,10,34,.9)'; ctx.stroke();
    });
  }
  function drawLabCharts() {
    const H = S.gan.history; if (!H.length) return;
    const d = downsample(ema(H.map((r) => r.dLoss)), 260), g = downsample(ema(H.map((r) => r.gLoss)), 260);
    lineChart($('#lossChart'), [{ data: d, color: C.disc, fill: true }, { data: g, color: C.synth, fill: true }], { xMaxLabel: 'step ' + fmtInt(S.gan.epoch) });
    const dr = downsample(ema(H.map((r) => r.dReal)), 260), df = downsample(ema(H.map((r) => r.dFake)), 260);
    lineChart($('#dChart'), [{ data: dr, color: C.real }, { data: df, color: C.synth }], { yMin: 0, yMax: 1, ref: 0.5, xMaxLabel: 'step ' + fmtInt(S.gan.epoch) });
    const last = H[H.length - 1];
    $('#stEpoch').textContent = fmtInt(S.gan.epoch); $('#stD').textContent = f2(ema(H.slice(-60).map((r) => r.dLoss), 0.2).pop()); $('#stG').textContent = f2(ema(H.slice(-60).map((r) => r.gLoss), 0.2).pop());
    void last;
  }
  let realismCache = null;
  function getRealism() {
    if (!realismCache || realismCache.epoch !== S.gan.epoch) realismCache = { epoch: S.gan.epoch, r: E.realism(S.realRows, getSynth(1000), world) };
    return realismCache.r;
  }
  function renderCatBars() {
    const r = getRealism();
    $('#stMae').textContent = r.mae.toFixed(3);
    $('#catBars').innerHTML = [0, 1, 2, 3].map((c) => `<div class="cb-row"><span title="${catName(c)}">${catName(c)}</span><div class="cb-bars"><div class="cb-bar r" style="width:${(r.catReal[c] * 100).toFixed(1)}%"></div><div class="cb-bar s" style="width:${(r.catSynth[c] * 100).toFixed(1)}%"></div></div></div>`).join('');
  }
  function drawMx(cv, rows, color) {
    const f = fit(cv); if (!f) return; const { ctx, w, h } = f;
    ctx.clearRect(0, 0, w, h);
    const nI = world.nItems, gap = 5, top = 14, cw = (w - gap * 3) / nI, rh = (h - top) / rows.length;
    const colX = (i) => i * cw + world.items[i].cat * gap;
    ctx.font = '10px ' + css('--font-body'); ctx.fillStyle = C.faint; ctx.textBaseline = 'top';
    for (let i = 0; i < nI; i++) if (world.items[i].isNew) { ctx.textAlign = 'center'; ctx.fillStyle = C.synth; ctx.fillText('new', colX(i) + cw / 2, 0); }
    rows.forEach((r, u) => {
      for (let i = 0; i < nI; i++) {
        const x = colX(i) + 0.6, y = top + u * rh + 0.6;
        ctx.fillStyle = r[i] ? color : 'rgba(174,180,214,.06)';
        ctx.fillRect(x, y, cw - 1.2, rh - 1.2);
      }
    });
  }
  function drawMatrices() {
    drawMx($('#mxReal'), S.realRows, C.real);
    drawMx($('#mxSynth'), S.gan.sample(20, 5000 + S.resampleSeed), C.synth);
  }

  // ---------------- rail status ----------------
  function drawRail() {
    const H = S.gan.history;
    const dot = $('#rsDot'); dot.classList.toggle('on', S.training);
    $('#rsText').textContent = S.training ? `Training · step ${fmtInt(S.gan.epoch)}` : `Paused · step ${fmtInt(S.gan.epoch)}`;
    const f = fit($('#rsSpark')); if (!f || H.length < 2) return; const { ctx, w, h } = f;
    ctx.clearRect(0, 0, w, h);
    const d = downsample(ema(H.map((r) => r.dLoss)), 100), g = downsample(ema(H.map((r) => r.gLoss)), 100);
    const mx = Math.max(...d, ...g, 0.5);
    [[d, C.disc], [g, C.synth]].forEach(([s, col]) => {
      ctx.beginPath(); s.forEach((v, i) => { const x = (i / (s.length - 1)) * w, y = h - 2 - (v / mx) * (h - 4); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
    });
  }

  // ---------------- playground ----------------
  const coverHues = [262, 345, 205, 150];
  function cover(i) {
    const it = world.items[i], hgt = coverHues[it.cat] + (i % 6) * 9, k = i % 6;
    const px = [20, 80, 50, 15, 70, 40][k], py = [30, 20, 80, 70, 60, 10][k];
    return `radial-gradient(circle at ${px}% ${py}%, hsla(${hgt + 40},80%,70%,.55), transparent 45%), repeating-linear-gradient(${45 + k * 30}deg, hsla(${hgt},50%,60%,.10) 0 2px, transparent 2px 9px), linear-gradient(135deg, hsl(${hgt},48%,30%), hsl(${hgt + 30},55%,14%))`;
  }
  function renderItemGrid() {
    const g = $('#itemGrid'); const hidden = S.probe ? S.probe.hidden : new Set();
    g.innerHTML = [0, 1, 2, 3].map((c) => `<div class="cat-block"><h4>${catName(c)}</h4><div class="cat-items">${
      world.items.filter((it) => it.cat === c).map((it) => `<button class="item${hidden.has(it.id) ? ' hidden-like' : ''}" data-i="${it.id}" aria-pressed="${S.liked.has(it.id)}"><span class="cover" style="background:${cover(it.id)}"></span>${it.isNew ? '<span class="tag">New</span>' : ''}<span class="chk"></span><span class="nm">${itemName(it.id)}</span></button>`).join('')
    }</div></div>`).join('');
    const n = S.liked.size;
    $('#pickCount').textContent = n ? `${n} picked` : 'Nothing picked yet';
  }
  function renderProbeInfo() {
    const el = $('#probeInfo');
    if (!S.probe) { el.hidden = true; return; }
    const p = S.probe; el.hidden = false;
    el.innerHTML = `<b>Simulated user #${p.n}</b> mostly likes <b>${catName(p.primary)}</b>${p.hasSecondary ? `, and some <b>${catName(p.secondary)}</b>` : ''}. Two of their likes are picked for you. Dashed outlines show the <b>${p.hidden.size}</b> likes they kept hidden; a good recommender should find them.`;
  }
  function recList(method) {
    const liked = [...S.liked];
    if (method === 'popularity') return E.recommend(null, world.nItems, liked, 5, S.pop).map((r) => ({ ...r, score: S.pop[r.item] }));
    const sim = method === 'real' ? S.simReal : getSimAug(S.playN);
    return E.recommend(sim, world.nItems, liked, 5, S.pop);
  }
  function why(method, item) {
    const liked = [...S.liked];
    if (method === 'popularity') return `${S.pop[item]} of 20 real users ${dom().verb} it`;
    const sim = method === 'real' ? S.simReal : getSimAug(S.playN);
    let best = -1, bv = 0;
    liked.forEach((i) => { const v = sim[i * world.nItems + item]; if (v > bv) { bv = v; best = i; } });
    if (world.items[item].isNew && method === 'augmented') return best >= 0 ? `New release. Synthetic users pair it with ${itemName(best)}` : 'New release with synthetic history';
    if (best < 0 || bv === 0) return liked.length ? 'No overlap found, falls back to popularity' : 'Popularity fallback';
    return `Because you liked ${itemName(best)}`;
  }
  function renderRecs() {
    const map = { popularity: ['#resPop', '#footPop'], real: ['#resReal', '#footReal'], augmented: ['#resAug', '#footAug'] };
    const hidden = S.probe ? S.probe.hidden : null;
    $('#augSub').textContent = `20 real + ${fmtInt(S.playN)} synthetic users`;
    for (const m of Object.keys(map)) {
      const recs = recList(m);
      const max = Math.max(...recs.map((r) => r.score), 1e-9);
      let hits = 0, news = 0;
      $(map[m][0]).innerHTML = recs.map((r, idx) => {
        const isNew = world.items[r.item].isNew; if (isNew) news++;
        const hit = hidden && hidden.has(r.item); if (hit) hits++;
        const cls = hidden ? (hit ? ' hit' : ' miss') : '';
        const badge = hit ? '<span class="badge hitb">Hit</span>' : isNew ? '<span class="badge">New</span>' : '';
        return `<li class="rec${cls}" style="animation-delay:${idx * 45}ms"><div class="rt"><span>${itemName(r.item)}</span>${badge}</div><span class="why">${why(m, r.item)}</span><div class="bar" style="width:${Math.max(4, (r.score / max) * 100).toFixed(0)}%"></div></li>`;
      }).join('');
      const foot = $(map[m][1]);
      if (hidden) foot.innerHTML = `<b>${hits} of 5</b> are hidden likes`;
      else if (m === 'augmented') foot.innerHTML = news ? `<b>${news}</b> new release${news > 1 ? 's' : ''} in the list` : '';
      else if (m === 'real') foot.textContent = 'Can never show the 4 new releases';
      else foot.textContent = 'Ignores your picks';
    }
  }
  function loadProbe() {
    S.probeCount++;
    const rng = E.mulberry32(777 + S.probeCount * 13);
    let u;
    for (let k = 0; k < 50; k++) { u = E.sampleUser(world, rng); if (u.row.reduce((a, b) => a + b, 0) >= 5 && u.row.some((v, i) => v && world.items[i].isNew)) break; }
    const likes = u.row.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
    for (let i = likes.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [likes[i], likes[j]] = [likes[j], likes[i]]; }
    let shown = likes.filter((i) => !world.items[i].isNew).slice(0, 2);
    const hidden = new Set(likes.filter((i) => !shown.includes(i)));
    S.liked = new Set(shown);
    S.probe = { n: S.probeCount, primary: u.primary, secondary: u.secondary, hasSecondary: u.taste[u.secondary] > -1, hidden };
    renderProbeInfo(); renderItemGrid(); renderRecs();
  }

  // ---------------- evaluation ----------------
  const BENCH_N = [0, 250, 1000, 2500, 5000];
  function runBench() {
    const btn = $('#benchBtn'); btn.disabled = true; btn.textContent = 'Running…';
    const res = []; let k = 0;
    const next = () => {
      const n = BENCH_N[k];
      res.push({ n, ev: E.evaluate(world, S.realRows, getSynth(n), { nTest: 300 }) });
      k++;
      if (k < BENCH_N.length) setTimeout(next, 30);
      else { S.bench = res; S.benchEpoch = S.gan.epoch; btn.disabled = false; btn.textContent = 'Run again'; renderBench(); updateBenchMeta(); }
    };
    setTimeout(next, 30);
  }
  function updateBenchMeta() {
    const m = $('#benchMeta');
    if (!S.bench) { m.textContent = S.training ? 'The benchmark runs as soon as training pauses.' : 'Runs against the generator as it is right now.'; return; }
    const stale = S.benchEpoch !== S.gan.epoch;
    m.textContent = `Generator at step ${fmtInt(S.benchEpoch)}${S.biased ? ', trained on biased seed data' : ''}. 300 test users, top 5, 2 revealed likes each.` + (stale ? ' The generator has changed since this run.' : '');
  }
  function renderBench() {
    const last = S.bench[S.bench.length - 1].ev;
    const defs = [
      ['precision', 'Precision@5', 'Share of the top 5 that the user actually likes'],
      ['ndcg', 'NDCG@5', 'Like precision, but hits near the top count more'],
      ['coverage', 'Catalog coverage', 'Share of all 24 items that ever get recommended'],
      ['newItemHit', 'New releases found', 'For users who like a new release, how often one is surfaced'],
    ];
    const rows = [['popularity', 'Popularity', C.pop], ['real', 'Real only', C.real], ['augmented', 'Real + GAN', C.synth]];
    $('#metricGrid').innerHTML = defs.map(([key, title, desc]) => {
      const vals = rows.map(([m]) => last[m][key]);
      const mx = Math.max(...vals, 0.01);
      const lift = last.real[key] > 0 ? ((last.augmented[key] - last.real[key]) / last.real[key]) * 100 : null;
      const liftTxt = lift == null ? (last.augmented[key] > 0 ? `${pct(last.augmented[key])}<small>real only can't do this</small>` : '—') : `${lift >= 0 ? '+' : ''}${lift.toFixed(1)}%<small>vs real only</small>`;
      return `<div class="metric"><h3>${title}</h3><p>${desc}</p>${rows.map(([m, lbl, col], i) => `<div class="mbar"><span>${lbl}</span><div class="track"><div class="fill" style="background:${col}" data-w="${((vals[i] / mx) * 100).toFixed(1)}"></div></div><span class="val">${key === 'precision' || key === 'ndcg' ? f2(vals[i]) : pct(vals[i])}</span></div>`).join('')}<div class="lift">${liftTxt}</div></div>`;
    }).join('');
    requestAnimationFrame(() => $$('.mbar .fill').forEach((el) => (el.style.width = el.dataset.w + '%')));
    lineChart($('#curveChart'), [
      { data: S.bench.map((b) => b.ev.popularity.precision), color: C.pop, dots: true },
      { data: S.bench.map((b) => b.ev.real.precision), color: C.real, dots: true },
      { data: S.bench.map((b) => b.ev.augmented.precision), color: C.synth, dots: true, width: 2.5 },
    ], { xs: BENCH_N.map((_, i) => i), xLabels: BENCH_N.map((n) => fmtInt(n)), yMin: 0.2, yMax: 0.65, yTicks: 3 });
    const P = last;
    const dp = P.augmented.precision - P.real.precision;
    const items = [
      `With 5,000 synthetic users, precision@5 goes from <b>${f2(P.real.precision)}</b> to <b>${f2(P.augmented.precision)}</b> (${dp >= 0 ? '+' : ''}${(dp * 100).toFixed(1)} points). Popularity scores <b>${f2(P.popularity.precision)}</b>.`,
      `Real-only filtering never recommends a new release, because nobody has interacted with one. The GAN version surfaces one for <b>${pct(P.augmented.newItemHit)}</b> of the users who'd like it.`,
      `Coverage rises from <b>${pct(P.real.coverage)}</b> to <b>${pct(P.augmented.coverage)}</b> of the catalog, so more items get a chance to be seen.`,
      S.biased ? `The generator learned from <b>biased seed data</b>. Compare the category shares in step 2: the skew carries into the synthetic users.` : `Gains depend on the generator: reset it or train with <b>biased seed data</b> in step 2 and run this again to see how results move.`,
    ];
    if (dp < 0) items[0] = `Precision@5 here is <b>${f2(P.augmented.precision)}</b> with synthetic users versus <b>${f2(P.real.precision)}</b> without. This generator is hurting accuracy on items with history, which is why augmentation needs evaluation before it ships.`;
    $('#readList').innerHTML = items.map((t) => `<li>${t}</li>`).join('');
  }

  // ---------------- architecture ----------------
  const STAGES = [
    { kind: 'real', name: 'Users', d: () => `20 real users on a new ${dom().label.toLowerCase()} platform. That's the whole history the system starts with.`, st: () => [['Real users', '20'], ['Catalog', '24 ' + dom().unit], ['New releases', '4']] },
    { kind: 'real', name: 'Interaction dataset', d: () => `Who ${dom().verb} what, stored as a 20 × 24 binary matrix.`, st: () => { const n = S.realRows.flat().reduce((a, b) => a + b, 0); return [['Interactions', fmtInt(n)], ['Density', pct(n / 480)]]; } },
    { kind: 'real', name: 'Preprocessing', d: () => 'Interactions are binarised. Each item is tagged with its attributes (its category). New releases are hidden from the discriminator, since there is no real data to compare against.', st: () => [['Items D sees', '20'], ['Attributes per item', '4']] },
    { kind: 'gan', name: 'GAN training', d: () => 'The generator and discriminator train against each other with Adam. The discriminator uses one-sided label smoothing and input noise so it can\'t just memorise 20 rows.', st: () => { const H = S.gan.history; const l = H[H.length - 1] || {}; return [['Step', fmtInt(S.gan.epoch)], ['D loss', l.dLoss ? f2(l.dLoss) : '—'], ['G loss', l.gLoss ? f2(l.gLoss) : '—'], ['Per step', S.msPerStep ? S.msPerStep.toFixed(2) + ' ms' : '—']]; } },
    { kind: 'synth', name: 'Synthetic dataset', d: () => 'The trained generator samples as many users as you ask for. Each one is a binary row over all 24 items, new releases included.', st: () => { const r = getRealism(); return [['Synthetic users', '5,000'], ['Density', pct(r.densitySynth)], ['Item rate error', r.mae.toFixed(3)]]; } },
    { kind: 'synth', name: 'Recommendation engine', d: () => 'Item-to-item cosine similarity over 20 real rows plus the synthetic rows. Real rows are never dropped.', st: () => [['Rows', '20 + 5,000'], ['Similarity pairs', '276']] },
    { kind: 'out', name: 'Top-N list', d: () => 'For each user, items are scored by similarity to what they already liked, and the top 5 are shown.', st: () => { const ev = S.bench ? S.bench[S.bench.length - 1].ev : null; return [['Precision@5', ev ? f2(ev.augmented.precision) : 'Run step 4'], ['New releases found', ev ? pct(ev.augmented.newItemHit) : '—']]; } },
  ];
  function renderPipe() {
    $('#pipe').innerHTML = STAGES.map((s, i) => `<li><button class="stage" data-kind="${s.kind}" data-s="${i}" aria-pressed="${i === S.stage}"><span class="sn">Stage ${i + 1}</span><b>${s.name}</b><span class="sv">${s.st()[0][1]}</span></button></li>`).join('');
    const s = STAGES[S.stage];
    $('#stageDetail').innerHTML = `<h3>${s.name}</h3><p>${s.d()}</p><div class="sd-stats">${s.st().map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('')}</div>`;
  }
  function renderChallenges() {
    const r = getRealism();
    const items = [
      ['Training GANs is computationally expensive', `This toy GAN takes ${S.msPerStep ? S.msPerStep.toFixed(2) : '~1'} ms per step in your browser. Real catalogs with millions of items need GPUs and hours.`],
      ['Synthetic data must remain realistic', `Right now the average gap between real and synthetic like rates is ${r.mae.toFixed(3)}. The duel view in step 2 shows where they diverge.`],
      ['Biased training data creates biased synthetic data', 'Switch on biased seed data in step 2 and watch the category shares: the generator copies the skew.'],
      ['Requires careful evaluation', 'Step 4 scores every model against simulated users whose true taste is known, not against the synthetic data itself.'],
    ];
    $('#challenges').innerHTML = items.map(([t, s]) => `<li><b>${t}</b><span>${s}</span></li>`).join('');
  }
  let flowT = 0;
  function drawFlow() {
    const cv = $('#pipeFlow'); const f = fit(cv); if (!f) return; const { ctx, w, h } = f;
    ctx.clearRect(0, 0, w, h);
    const y = h / 2;
    ctx.strokeStyle = 'rgba(174,180,214,.18)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    flowT += S.training ? 1.6 : 0.6;
    const ganX = (3.5 / 7) * w;
    for (let k = 0; k < 26; k++) {
      const x = ((k * 53 + flowT) % (w + 40)) - 20;
      const after = x > ganX;
      const dens = after ? 1 : (k % 3 === 0 ? 1 : 0);
      if (!dens) continue;
      ctx.fillStyle = after ? C.synth : C.real;
      ctx.globalAlpha = 0.85; ctx.beginPath(); ctx.arc(x, y + (after ? Math.sin(k * 1.7) * 6 : 0), after ? 2.4 : 3.2, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = C.disc; ctx.beginPath(); ctx.arc(ganX, y, 5, 0, Math.PI * 2); ctx.fill();
  }

  // ---------------- loop ----------------
  let frame = 0;
  function loop(t) {
    frame++;
    if (S.training) {
      const t0 = performance.now();
      for (let i = 0; i < S.speed; i++) S.gan.step(S.realRows);
      const dt = (performance.now() - t0) / S.speed;
      S.msPerStep = S.msPerStep ? S.msPerStep * 0.95 + dt * 0.05 : dt;
      invalidate();
      if (S.gan.epoch >= S.stopAt) {
        S.training = false; syncTrainBtn();
        toast(S.firstRun ? `Generator trained for ${fmtInt(S.gan.epoch)} steps` : `Paused at step ${fmtInt(S.gan.epoch)}`);
        S.firstRun = false;
        realismCache = null;
        if (S.view === 'lab') { renderCatBars(); drawMatrices(); }
        if (S.view === 'play') renderRecs();
        if (S.view === 'eval') runBench();
        if (S.view === 'arch') { renderPipe(); renderChallenges(); }
      }
    }
    if (frame % 3 === 0) drawBrand(t);
    if (S.view === 'overview') {
      drawHeroMatrix();
      if (!S.training && S.lastHeroEval !== S.gan.epoch) updateHeroStats();
      else if (S.training && frame % 40 === 0) updateHeroStats(true);
    }
    if (S.view === 'lab') {
      drawDuel();
      if (frame % 3 === 0) drawLabCharts();
      if (S.training && frame % 30 === 0) { renderCatBars(); drawMatrices(); }
    }
    if (S.view === 'arch') { drawFlow(); if (S.training && frame % 20 === 0) renderPipe(); }
    if (frame % 8 === 0) drawRail();
    requestAnimationFrame(loop);
  }

  function syncTrainBtn() { $('#trainBtn').textContent = S.training ? 'Pause training' : 'Resume training'; }

  // ---------------- events ----------------
  function bind() {
    const sel = $('#domainSelect');
    sel.innerHTML = Object.entries(CATALOG).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
    sel.addEventListener('change', () => { S.domain = sel.value; applyDomain(); toast(`Showing a ${CATALOG[S.domain].label.toLowerCase()} catalog`); });
    document.addEventListener('click', (e) => { const g = e.target.closest('[data-go]'); if (g) go(g.dataset.go); });

    const hn = $('#heroN');
    hn.addEventListener('input', () => { S.heroN = +hn.value; $('#heroNOut').textContent = fmtInt(S.heroN); clearTimeout(hn._t); hn._t = setTimeout(() => updateHeroStats(true), 90); });

    $('#trainBtn').addEventListener('click', () => {
      S.training = !S.training;
      if (S.training && S.gan.epoch >= S.stopAt) S.stopAt = S.gan.epoch + 1500;
      syncTrainBtn();
    });
    $('#stepBtn').addEventListener('click', () => { for (let i = 0; i < 50; i++) S.gan.step(S.realRows); invalidate(); realismCache = null; renderCatBars(); drawMatrices(); drawLabCharts(); });
    $('#resetBtn').addEventListener('click', () => { rebuild(); S.training = true; S.stopAt = 1500; syncTrainBtn(); renderCatBars(); drawMatrices(); toast('Weights reset. Training from step 0'); });
    $('#lrSel').addEventListener('change', (e) => { S.lr = +e.target.value; S.gan.setLR(S.lr); toast(`Learning rate set to ${S.lr}`); });
    $('#speedSel').addEventListener('change', (e) => { S.speed = +e.target.value; });
    $('#biasTgl').addEventListener('change', (e) => {
      S.biased = e.target.checked; rebuild(); S.training = true; S.stopAt = 1500; syncTrainBtn();
      $('#biasNote').innerHTML = S.biased ? `80% of the real users now love <b>${catName(0)}</b>. Watch the synthetic share copy that skew as it trains.` : 'Turn on <b>Biased seed data</b> to see a known risk: if 80% of the real users share one taste, the synthetic users copy that skew.';
      S.liked.clear(); S.probe = null; renderProbeInfo(); renderItemGrid(); renderRecs(); renderCatBars(); drawMatrices();
      toast(S.biased ? 'Retraining on biased seed data' : 'Retraining on balanced seed data');
    });
    $('#resampleBtn').addEventListener('click', () => { S.resampleSeed++; drawMatrices(); });

    $('#itemGrid').addEventListener('click', (e) => {
      const b = e.target.closest('.item'); if (!b) return;
      const i = +b.dataset.i;
      S.liked.has(i) ? S.liked.delete(i) : S.liked.add(i);
      if (S.probe) S.probe.hidden.delete(i);
      renderItemGrid(); renderRecs();
      const nb = $(`.item[data-i="${i}"]`); if (nb) nb.classList.add('pop');
    });
    $('#probeBtn').addEventListener('click', loadProbe);
    $('#clearBtn').addEventListener('click', () => { S.liked.clear(); S.probe = null; renderProbeInfo(); renderItemGrid(); renderRecs(); });
    const pn = $('#playN');
    pn.addEventListener('input', () => { S.playN = +pn.value; $('#playNOut').textContent = fmtInt(S.playN); clearTimeout(pn._t); pn._t = setTimeout(renderRecs, 60); });

    $('#benchBtn').addEventListener('click', runBench);
    $('#pipe').addEventListener('click', (e) => { const b = e.target.closest('.stage'); if (!b) return; S.stage = +b.dataset.s; renderPipe(); });
    window.addEventListener('resize', () => { if (S.bench && S.view === 'eval') renderBench(); if (S.view === 'lab') drawMatrices(); });
  }

  // ---------------- boot ----------------
  rebuild();
  bind();
  applyDomain();
  loadProbe();
  const h = (location.hash || '').slice(1);
  if (['overview', 'lab', 'play', 'eval', 'arch'].includes(h)) go(h);
  requestAnimationFrame(loop);
})();
