"""
GAN-augmented synthetic interaction generator for cold-start recommendation.
Reference implementation in PyTorch, matching the browser demo.

Protocol ("new platform" cold start)
  * A small set of seed users (default 20) is the only real interaction data.
  * A share of items are "new releases": they exist in the catalog with attributes
    but have no interactions in the seed data.
  * A GAN learns from the seed data. Its generator writes interactions through item
    attributes (genres), so it can also write history for new releases.
  * Item-to-item CF is trained on (a) seed data only and (b) seed + synthetic rows.
  * Held-out test users reveal 2 likes; we score top-K against their other likes.

Usage
  python gan_coldstart.py --demo                      # simulated world, no download
  python gan_coldstart.py --data ./ml-100k            # MovieLens-100K folder (u.data, u.item)
  python gan_coldstart.py --data ./ml-100k --n-real 50 --synthetic 5000 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- data
@dataclass
class Dataset:
    seed_rows: np.ndarray      # (n_real, n_items) binary, new-item columns are zero
    test_rows: np.ndarray      # (n_test, n_items) binary, full history
    attrs: np.ndarray          # (n_items, n_attrs) multi-hot item attributes
    is_new: np.ndarray         # (n_items,) bool
    item_names: list[str]
    attr_names: list[str]


def load_movielens(path: Path, n_real: int, n_test: int, top_items: int, new_frac: float, like_at: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    ratings = np.loadtxt(path / "u.data", dtype=int)  # user item rating ts
    genres = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    names, attr = {}, {}
    with open(path / "u.item", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            iid = int(p[0]); names[iid] = p[1]; attr[iid] = np.array(p[5:24], dtype=float)
    likes = ratings[ratings[:, 2] >= like_at]
    counts = np.bincount(likes[:, 1], minlength=max(names) + 1)
    items = np.argsort(-counts)[:top_items]
    col = {iid: j for j, iid in enumerate(items)}
    users = np.unique(likes[:, 0])
    mat = np.zeros((users.max() + 1, len(items)), dtype=np.float32)
    for u, i, _, _ in likes:
        if i in col:
            mat[u, col[i]] = 1
    active = np.where(mat.sum(1) >= 5)[0]
    rng.shuffle(active)
    seed_u, test_u = active[:n_real], active[n_real:n_real + n_test]
    is_new = np.zeros(len(items), dtype=bool)
    is_new[rng.choice(len(items), int(len(items) * new_frac), replace=False)] = True
    seed_rows = mat[seed_u].copy(); seed_rows[:, is_new] = 0
    return Dataset(seed_rows, mat[test_u], np.stack([attr[i] for i in items]).astype(np.float32), is_new,
                   [names[i] for i in items], genres)


def make_demo(n_real: int, n_test: int, seed: int) -> Dataset:
    """Same simulated world as the browser demo: 4 categories x 6 items, 1 new release each."""
    rng = np.random.default_rng(seed)
    n_items, n_cats = 24, 4
    cat = np.arange(n_items) // 6
    quality = rng.normal(0, 0.7, n_items)
    is_new = (np.arange(n_items) % 6) == 5

    def user():
        taste = np.full(n_cats, -2.2)
        p, s = rng.integers(n_cats), rng.integers(n_cats)
        s = (s + 1) % n_cats if s == p else s
        taste[p] = 1.6 + rng.normal(0, 0.3)
        if rng.random() < 0.6:
            taste[s] = 0.2 + rng.normal(0, 0.3)
        prob = 1 / (1 + np.exp(-(taste[cat] + 0.9 * quality)))
        return (rng.random(n_items) < prob).astype(np.float32)

    seed_rows = np.stack([user() for _ in range(n_real)]); seed_rows[:, is_new] = 0
    test_rows = np.stack([user() for _ in range(n_test)])
    attrs = np.eye(n_cats, dtype=np.float32)[cat]
    names = [f"item {i}" + (" (new)" if is_new[i] else "") for i in range(n_items)]
    return Dataset(seed_rows, test_rows, attrs, is_new, names, ["cat A", "cat B", "cat C", "cat D"])


# ----------------------------------------------------------------------------- GAN
class Generator(nn.Module):
    """noise -> taste over attributes -> logit(item) = taste . attrs(item) + bias(item).
    New releases get the mean bias of known items, so their scores come only from attributes."""

    def __init__(self, attrs: torch.Tensor, is_new: torch.Tensor, nz: int = 8, hidden: int = 32):
        super().__init__()
        self.register_buffer("attrs", attrs)
        self.register_buffer("is_new", is_new)
        self.net = nn.Sequential(nn.Linear(nz, hidden), nn.ReLU(), nn.Linear(hidden, attrs.shape[1]))
        self.bias = nn.Parameter(torch.zeros(attrs.shape[0]))
        self.nz = nz

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        taste = self.net(z)
        b = torch.where(self.is_new, self.bias[~self.is_new].mean(), self.bias)
        return torch.sigmoid(taste @ self.attrs.T + b)


class Discriminator(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_gan(ds: Dataset, steps: int, lr: float, batch: int, seed: int, log_every: int = 250):
    torch.manual_seed(seed)
    known = torch.tensor(~ds.is_new)
    G = Generator(torch.tensor(ds.attrs), torch.tensor(ds.is_new))
    D = Discriminator(int(known.sum()))
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    real_all = torch.tensor(ds.seed_rows)[:, known]
    history = []
    for step in range(1, steps + 1):
        # discriminator: one-sided label smoothing + input noise so it can't memorise a handful of rows
        idx = torch.randint(0, len(real_all), (batch,))
        real = real_all[idx] * 0.9 + 0.05 + (torch.rand(batch, real_all.shape[1]) - 0.5) * 0.2
        fake = G(torch.randn(batch, G.nz))[:, known].detach()
        d_loss = bce(D(real), torch.full((batch,), 0.9)) + bce(D(fake), torch.zeros(batch))
        optD.zero_grad(); d_loss.backward(); optD.step()
        # generator: two non-saturating updates per discriminator update
        for _ in range(2):
            g_loss = bce(D(G(torch.randn(batch, G.nz))[:, known]), torch.ones(batch))
            optG.zero_grad(); g_loss.backward(); optG.step()
        history.append((step, d_loss.item(), g_loss.item()))
        if log_every and step % log_every == 0:
            print(f"  step {step:5d}  D loss {d_loss.item():.3f}  G loss {g_loss.item():.3f}")
    return G, history


@torch.no_grad()
def sample(G: Generator, n: int, seed: int = 99) -> np.ndarray:
    if n == 0:
        return np.zeros((0, G.attrs.shape[0]), dtype=np.float32)
    g = torch.Generator().manual_seed(seed)
    p = G(torch.randn(n, G.nz, generator=g))
    return torch.bernoulli(p, generator=g).numpy()


# ----------------------------------------------------------------------------- recommenders
def item_similarity(rows: np.ndarray) -> np.ndarray:
    co = rows.T @ rows
    cnt = np.diag(co).copy()
    denom = np.sqrt(np.outer(cnt, cnt))
    sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
    np.fill_diagonal(sim, 0)
    return sim


def evaluate(ds: Dataset, synth: np.ndarray, k: int = 5, reveal: int = 2, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)
    pop = ds.seed_rows.sum(0)
    sims = {"popularity": None, "real_only": item_similarity(ds.seed_rows),
            "real_plus_gan": item_similarity(np.vstack([ds.seed_rows, synth]))}
    acc = {m: dict(p=0.0, ndcg=0.0, seen=set(), n=0, new_hit=0, new_n=0) for m in sims}
    for row in ds.test_rows:
        likes = np.flatnonzero(row)
        if len(likes) < reveal + 2:
            continue
        rng.shuffle(likes)
        shown, hidden = likes[:reveal], set(likes[reveal:].tolist())
        for m, sim in sims.items():
            score = pop * 1e-4 + (sim[shown].sum(0) if sim is not None else 0)
            score[shown] = -np.inf
            top = np.argsort(-score)[:k]
            hits = [i in hidden for i in top]
            dcg = sum(1 / math.log2(r + 2) for r, h in enumerate(hits) if h)
            idcg = sum(1 / math.log2(r + 2) for r in range(min(k, len(hidden))))
            a = acc[m]; a["p"] += sum(hits) / k; a["ndcg"] += dcg / idcg; a["n"] += 1; a["seen"].update(top.tolist())
            if any(ds.is_new[i] for i in hidden):
                a["new_n"] += 1
                a["new_hit"] += any(h and ds.is_new[i] for i, h in zip(top, hits))
    n_items = ds.seed_rows.shape[1]
    return {m: {"precision@k": a["p"] / a["n"], "ndcg@k": a["ndcg"] / a["n"], "coverage": len(a["seen"]) / n_items,
                "new_item_hit": a["new_hit"] / a["new_n"] if a["new_n"] else 0.0, "test_users": a["n"]} for m, a in acc.items()}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="MovieLens-100K folder containing u.data and u.item")
    ap.add_argument("--demo", action="store_true", help="use the simulated world from the browser demo")
    ap.add_argument("--n-real", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--top-items", type=int, default=300)
    ap.add_argument("--new-frac", type=float, default=0.15)
    ap.add_argument("--synthetic", type=int, nargs="+", default=[0, 250, 1000, 2500, 5000])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    a = ap.parse_args()

    if a.data and not a.demo:
        ds = load_movielens(a.data, a.n_real, a.n_test, a.top_items, a.new_frac, 4, a.seed)
        print(f"MovieLens: {a.n_real} seed users, {ds.seed_rows.shape[1]} items ({ds.is_new.sum()} new), {len(ds.test_rows)} test users")
    else:
        ds = make_demo(a.n_real, a.n_test, a.seed)
        print(f"Demo world: {a.n_real} seed users, 24 items (4 new), {a.n_test} test users")

    print(f"Training GAN for {a.steps} steps")
    G, history = train_gan(ds, a.steps, a.lr, a.batch, a.seed)

    results = {}
    print(f"\n{'synthetic':>9}  {'method':<14} {'P@'+str(a.k):>6} {'NDCG':>6} {'cover':>6} {'new hit':>8}")
    for n in a.synthetic:
        ev = evaluate(ds, sample(G, n), k=a.k)
        results[n] = ev
        for m, r in ev.items():
            print(f"{n:>9}  {m:<14} {r['precision@k']:6.3f} {r['ndcg@k']:6.3f} {r['coverage']:6.2f} {r['new_item_hit']:8.2f}")
    a.out.write_text(json.dumps({"config": {k: str(v) for k, v in vars(a).items()}, "results": results,
                                 "loss_history": history[:: max(1, len(history) // 300)]}, indent=2))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
