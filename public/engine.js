/* ColdStart GAN — engine
 * A seeded "world" with known ground-truth tastes, a tiny GAN with hand-written
 * backprop + Adam, an item-item collaborative filter and an evaluation bench.
 * Runs in the browser (window.Engine) and in Node (module.exports).
 */
(function (root) {
  'use strict';

  // ---------- RNG ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  // ---------- World with ground truth ----------
  const N_ITEMS = 24, N_CATS = 4;

  function makeWorld(seed = 7) {
    const rng = mulberry32(seed);
    const items = [];
    for (let i = 0; i < N_ITEMS; i++) {
      const per = N_ITEMS / N_CATS;
      items.push({ id: i, cat: Math.floor(i / per), quality: gauss(rng) * 0.7, isNew: i % per === per - 1 });
    }
    return { seed, items, nItems: N_ITEMS, nCats: N_CATS };
  }

  // A user's taste: one primary category, often a secondary one.
  function sampleTaste(rng, opts = {}) {
    const primary = opts.forcePrimary != null ? opts.forcePrimary : Math.floor(rng() * N_CATS);
    let secondary = Math.floor(rng() * N_CATS);
    if (secondary === primary) secondary = (secondary + 1) % N_CATS;
    const taste = new Array(N_CATS).fill(-2.2);
    taste[primary] = 1.6 + gauss(rng) * 0.3;
    if (rng() < 0.6) taste[secondary] = 0.2 + gauss(rng) * 0.3;
    return { taste, primary, secondary };
  }

  function likeProbs(world, taste) {
    return world.items.map((it) => sigmoid(taste[it.cat] + 0.9 * it.quality));
  }

  function sampleUser(world, rng, opts) {
    const t = sampleTaste(rng, opts);
    const p = likeProbs(world, t.taste);
    const row = p.map((pi) => (rng() < pi ? 1 : 0));
    return { ...t, probs: p, row };
  }

  // The small real dataset: n users. `biased` makes most of them love category 0.
  function makeRealData(world, n = 20, { seed = 11, biased = false } = {}) {
    const rng = mulberry32(seed);
    const users = [];
    for (let u = 0; u < n; u++) {
      const force = biased && rng() < 0.8 ? 0 : null;
      const u = sampleUser(world, rng, { forcePrimary: force });
      u.fullRow = u.row.slice();
      u.row = u.row.map((v, i) => (world.items[i].isNew ? 0 : v));
      users.push(u);
    }
    return users;
  }

  // ---------- Tiny MLP pieces ----------
  function mat(rows, cols, rng, scale) {
    const m = new Float64Array(rows * cols);
    for (let i = 0; i < m.length; i++) m[i] = gauss(rng) * scale;
    return m;
  }
  function adamState(n) { return { m: new Float64Array(n), v: new Float64Array(n) }; }

  function Adam(params, lr, b1 = 0.5, b2 = 0.999) {
    this.params = params; this.lr = lr; this.b1 = b1; this.b2 = b2; this.t = 0;
    this.state = params.map((p) => adamState(p.length));
  }
  Adam.prototype.step = function (grads) {
    this.t++;
    const { b1, b2, lr } = this; const eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let k = 0; k < this.params.length; k++) {
      const p = this.params[k], g = grads[k], s = this.state[k];
      for (let i = 0; i < p.length; i++) {
        s.m[i] = b1 * s.m[i] + (1 - b1) * g[i];
        s.v[i] = b2 * s.v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= (lr * (s.m[i] / c1)) / (Math.sqrt(s.v[i] / c2) + eps);
      }
    }
  };

  // Two-layer net: in -> hidden (act) -> out (linear; caller applies sigmoid)
  function Net(nIn, nHid, nOut, rng, act) {
    this.nIn = nIn; this.nHid = nHid; this.nOut = nOut; this.act = act;
    this.W1 = mat(nIn, nHid, rng, Math.sqrt(2 / nIn)); this.b1 = new Float64Array(nHid);
    this.W2 = mat(nHid, nOut, rng, Math.sqrt(1 / nHid)); this.b2 = new Float64Array(nOut);
  }
  Net.prototype.params = function () { return [this.W1, this.b1, this.W2, this.b2]; };
  Net.prototype.forward = function (x) {
    const { nIn, nHid, nOut, W1, b1, W2, b2 } = this;
    const pre = new Float64Array(nHid), h = new Float64Array(nHid), out = new Float64Array(nOut);
    for (let j = 0; j < nHid; j++) {
      let s = b1[j];
      for (let i = 0; i < nIn; i++) s += x[i] * W1[i * nHid + j];
      pre[j] = s;
      h[j] = this.act === 'relu' ? Math.max(0, s) : (s > 0 ? s : 0.2 * s);
    }
    for (let k = 0; k < nOut; k++) {
      let s = b2[k];
      for (let j = 0; j < nHid; j++) s += h[j] * W2[j * nOut + k];
      out[k] = s;
    }
    return { x, pre, h, out };
  };
  // dOut: gradient wrt linear output. Accumulates into grads, returns dX.
  Net.prototype.backward = function (cache, dOut, grads, needDx) {
    const { nIn, nHid, nOut, W1, W2 } = this;
    const [gW1, gb1, gW2, gb2] = grads;
    const dh = new Float64Array(nHid);
    for (let k = 0; k < nOut; k++) {
      gb2[k] += dOut[k];
      for (let j = 0; j < nHid; j++) {
        gW2[j * nOut + k] += cache.h[j] * dOut[k];
        dh[j] += W2[j * nOut + k] * dOut[k];
      }
    }
    for (let j = 0; j < nHid; j++) {
      const d = cache.pre[j] > 0 ? 1 : (this.act === 'relu' ? 0 : 0.2);
      dh[j] *= d;
      gb1[j] += dh[j];
      for (let i = 0; i < nIn; i++) gW1[i * nHid + j] += cache.x[i] * dh[j];
    }
    if (!needDx) return null;
    const dx = new Float64Array(nIn);
    for (let i = 0; i < nIn; i++) {
      let s = 0;
      for (let j = 0; j < nHid; j++) s += W1[i * nHid + j] * dh[j];
      dx[i] = s;
    }
    return dx;
  };
  function zeroGrads(net) { return net.params().map((p) => new Float64Array(p.length)); }
  function scaleGrads(gs, s) { gs.forEach((g) => { for (let i = 0; i < g.length; i++) g[i] *= s; }); }

  // ---------- GAN ----------
  // Attribute-aware generator: noise -> taste vector over item attributes,
  // then logit(item) = taste · attributes(item) + bias(item). New items have no
  // bias of their own (they use the mean) but still get interactions through
  // their attributes. The discriminator only ever sees items with real history.
  function GAN(world, opts = {}) {
    this.world = world;
    this.nItems = world.nItems;
    this.known = opts.known || world.items.map((it) => it.id).filter((i) => !world.items[i].isNew);
    this.nz = opts.nz || 8;
    this.hidden = opts.hidden || 32;
    this.lr = opts.lr || 0.003;
    this.batch = opts.batch || 16;
    this.rng = mulberry32(opts.seed || 3);
    this.F = world.items.map((it) => { const f = new Array(world.nCats).fill(0); f[it.cat] = 1; return f; });
    this.nf = world.nCats;
    this.G = new Net(this.nz, this.hidden, this.nf, this.rng, 'relu');
    this.bias = new Float64Array(this.nItems);
    this.D = new Net(this.known.length, this.hidden, 1, this.rng, 'leaky');
    this.optG = new Adam(this.G.params().concat([this.bias]), this.lr);
    this.dLrScale = opts.dLrScale != null ? opts.dLrScale : 0.5;
    this.gSteps = opts.gSteps || 2;
    this.inputNoise = opts.inputNoise != null ? opts.inputNoise : 0.2;
    this.optD = new Adam(this.D.params(), this.lr * this.dLrScale);
    this.epoch = 0;
    this.history = [];
  }
  GAN.prototype.setLR = function (lr) { this.lr = lr; this.optG.lr = lr; this.optD.lr = lr * this.dLrScale; };
  GAN.prototype.noise = function (r) {
    const g = r || this.rng; const z = new Float64Array(this.nz);
    for (let i = 0; i < this.nz; i++) z[i] = gauss(g);
    return z;
  };
  GAN.prototype.meanKnownBias = function () {
    let s = 0; for (const i of this.known) s += this.bias[i]; return s / this.known.length;
  };
  GAN.prototype._gen = function (z) {
    const cache = this.G.forward(z); const e = cache.out;
    const mb = this.meanKnownBias();
    const p = new Float64Array(this.nItems);
    for (let i = 0; i < this.nItems; i++) {
      let l = this.world.items[i].isNew ? mb : this.bias[i];
      for (let c = 0; c < this.nf; c++) l += e[c] * this.F[i][c];
      p[i] = sigmoid(l);
    }
    return { cache, p };
  };
  GAN.prototype.generateProbs = function (z) { return this._gen(z || this.noise()).p; };
  GAN.prototype.taste = function (z) { return Array.from(this.G.forward(z || this.noise()).out); };
  GAN.prototype.knownOnly = function (p) { return Float64Array.from(this.known, (i) => p[i]); };
  GAN.prototype.score = function (fullRow) { return sigmoid(this.D.forward(this.knownOnly(fullRow)).out[0]); };

  // One adversarial step. realRows are full-width 0/1 rows.
  GAN.prototype.step = function (realRows) {
    const B = this.batch, rng = this.rng, K = this.known.length;
    const gD = zeroGrads(this.D);
    let dLoss = 0, dReal = 0, dFake = 0;
    for (let b = 0; b < B; b++) {
      const r = realRows[Math.floor(rng() * realRows.length)];
      const x = new Float64Array(K);
      for (let k = 0; k < K; k++) x[k] = r[this.known[k]] * 0.9 + 0.05 + (rng() - 0.5) * this.inputNoise;
      const c = this.D.forward(x); const s = sigmoid(c.out[0]);
      dLoss += -Math.log(s + 1e-9); dReal += s;
      this.D.backward(c, [s - 0.9], gD, false);
      const fp = this.knownOnly(this._gen(this.noise()).p);
      const cf = this.D.forward(fp); const sf = sigmoid(cf.out[0]);
      dLoss += -Math.log(1 - sf + 1e-9); dFake += sf;
      this.D.backward(cf, [sf], gD, false);
    }
    scaleGrads(gD, 1 / B); this.optD.step(gD);

    let gLoss = 0;
    for (let gs = 0; gs < this.gSteps; gs++) {
    const gG = zeroGrads(this.G); const gB = new Float64Array(this.nItems);
    for (let b = 0; b < B; b++) {
      const { cache, p } = this._gen(this.noise());
      const xk = this.knownOnly(p);
      const cd = this.D.forward(xk); const s = sigmoid(cd.out[0]);
      gLoss += -Math.log(s + 1e-9);
      const dx = this.D.backward(cd, [s - 1], zeroGrads(this.D), true);
      const de = new Float64Array(this.nf);
      for (let k = 0; k < K; k++) {
        const i = this.known[k];
        const dl = dx[k] * p[i] * (1 - p[i]);
        gB[i] += dl;
        for (let c = 0; c < this.nf; c++) de[c] += dl * this.F[i][c];
      }
      this.G.backward(cache, de, gG, false);
    }
    gG.push(gB); scaleGrads(gG, 1 / B); this.optG.step(gG);
    }
    gLoss /= this.gSteps;

    this.epoch++;
    const rec = { epoch: this.epoch, dLoss: dLoss / B, gLoss: gLoss / B, dReal: dReal / B, dFake: dFake / B };
    this.history.push(rec);
    return rec;
  };

  // Sample n synthetic users (binary rows) from the generator.
  GAN.prototype.sample = function (n, seed = 99) {
    const r = mulberry32(seed);
    const rows = [];
    for (let u = 0; u < n; u++) {
      const p = this._gen(this.noise(r)).p;
      const row = new Array(p.length);
      for (let i = 0; i < p.length; i++) row[i] = r() < p[i] ? 1 : 0;
      rows.push(row);
    }
    return rows;
  };

  // ---------- Recommenders ----------
  function popularity(rows, nItems) {
    const c = new Array(nItems).fill(0);
    rows.forEach((r) => r.forEach((v, i) => (c[i] += v)));
    return c;
  }

  // Item-item cosine similarity. `weights` lets real rows count more than synthetic.
  function itemSimilarity(rowSets, nItems) {
    const co = new Float64Array(nItems * nItems);
    const cnt = new Float64Array(nItems);
    rowSets.forEach(({ rows, weight }) => {
      rows.forEach((r) => {
        const on = [];
        for (let i = 0; i < nItems; i++) if (r[i]) on.push(i);
        for (const i of on) {
          cnt[i] += weight;
          for (const j of on) co[i * nItems + j] += weight;
        }
      });
    });
    const sim = new Float64Array(nItems * nItems);
    for (let i = 0; i < nItems; i++)
      for (let j = 0; j < nItems; j++)
        sim[i * nItems + j] = i === j || !cnt[i] || !cnt[j] ? 0 : co[i * nItems + j] / Math.sqrt(cnt[i] * cnt[j]);
    return sim;
  }

  // Returns [{item, score}] sorted, excluding liked.
  function recommend(sim, nItems, liked, k = 5, pop) {
    const L = new Set(liked);
    const scores = [];
    for (let j = 0; j < nItems; j++) {
      if (L.has(j)) continue;
      let s = 0;
      if (sim) for (const i of liked) s += sim[i * nItems + j];
      // tiny popularity tie-break so empty-history users still get a list
      if (pop) s += pop[j] * 1e-4;
      scores.push({ item: j, score: s });
    }
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, k);
  }

  // ---------- Evaluation bench ----------
  // Cold-start test users from the true world reveal `reveal` likes; we score
  // the top-k against the rest of what they actually like.
  function evaluate(world, realRows, synthRows, opts = {}) {
    const k = opts.k || 5, reveal = opts.reveal || 2, nTest = opts.nTest || 300;
    const synthWeight = opts.synthWeight != null ? opts.synthWeight : 1;
    const nI = world.nItems;
    const pop = popularity(realRows, nI);
    const simReal = itemSimilarity([{ rows: realRows, weight: 1 }], nI);
    const simAug = itemSimilarity([{ rows: realRows, weight: 1 }, { rows: synthRows, weight: synthWeight }], nI);
    const rng = mulberry32(opts.seed || 2024);
    const methods = { popularity: null, real: simReal, augmented: simAug };
    const acc = {}; Object.keys(methods).forEach((m) => (acc[m] = { p: 0, ndcg: 0, recall: 0, seen: new Set(), n: 0, newHit: 0, newN: 0 }));
    let done = 0, guard = 0;
    while (done < nTest && guard++ < nTest * 20) {
      const u = sampleUser(world, rng);
      const likes = u.row.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
      if (likes.length < reveal + 2) continue;
      // shuffle
      for (let i = likes.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [likes[i], likes[j]] = [likes[j], likes[i]]; }
      const shown = likes.slice(0, reveal), hidden = new Set(likes.slice(reveal));
      for (const m of Object.keys(methods)) {
        const recs = recommend(methods[m], nI, shown, k, pop);
        let hits = 0, dcg = 0, idcg = 0;
        recs.forEach((r, idx) => {
          if (hidden.has(r.item)) { hits++; dcg += 1 / Math.log2(idx + 2); }
          acc[m].seen.add(r.item);
        });
        for (let idx = 0; idx < Math.min(k, hidden.size); idx++) idcg += 1 / Math.log2(idx + 2);
        const wantsNew = [...hidden].some((i) => world.items[i].isNew);
        if (wantsNew) { acc[m].newN++; if (recs.some((r) => hidden.has(r.item) && world.items[r.item].isNew)) acc[m].newHit++; }
        acc[m].p += hits / k; acc[m].recall += hits / hidden.size; acc[m].ndcg += idcg ? dcg / idcg : 0; acc[m].n++;
      }
      done++;
    }
    const out = {};
    for (const m of Object.keys(acc)) {
      const a = acc[m];
      out[m] = { precision: a.p / a.n, recall: a.recall / a.n, ndcg: a.ndcg / a.n, coverage: a.seen.size / nI, newItemHit: a.newN ? a.newHit / a.newN : 0 };
    }
    return out;
  }

  // Simple realism stats between two row sets
  function realism(realRows, synthRows, world) {
    const nI = world.nItems;
    const mean = (rows) => { const m = new Array(nI).fill(0); rows.forEach((r) => r.forEach((v, i) => (m[i] += v / rows.length))); return m; };
    const mr = mean(realRows), ms = mean(synthRows);
    const catShare = (m) => { const c = new Array(world.nCats).fill(0); m.forEach((v, i) => (c[world.items[i].cat] += v)); const s = c.reduce((a, b) => a + b, 0) || 1; return c.map((v) => v / s); };
    const density = (rows) => rows.reduce((a, r) => a + r.reduce((x, y) => x + y, 0), 0) / (rows.length * nI || 1);
    let mae = 0; for (let i = 0; i < nI; i++) mae += Math.abs(mr[i] - ms[i]); mae /= nI;
    return { itemMeanReal: mr, itemMeanSynth: ms, catReal: catShare(mr), catSynth: catShare(ms), densityReal: density(realRows), densitySynth: density(synthRows), mae };
  }

  const api = { mulberry32, gauss, sigmoid, makeWorld, makeRealData, sampleUser, likeProbs, GAN, popularity, itemSimilarity, recommend, evaluate, realism, N_ITEMS, N_CATS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Engine = api;
})(typeof window !== 'undefined' ? window : globalThis);
