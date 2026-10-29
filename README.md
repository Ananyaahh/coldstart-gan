# ColdStart GAN

Demo video link : https://youtu.be/DN1MbMW94Tg 

**A GAN that writes synthetic user–item interactions so a recommender can work on day one.**

New platforms have too few users for a recommender to learn from. ColdStart GAN trains a generative adversarial network on a tiny set of real users (20), samples thousands of synthetic users from it, and merges them with the real data to train a recommender. Real data is never replaced, only added to.

> **Demo video:** [Watch the 5-minute walkthrough](PASTE-YOUR-VIDEO-LINK-HERE)

Recommender Systems course project, Vellore Institute of Technology.

---

## The problem

A recommender can only learn from interactions it has seen. Cold start comes in three forms:

| Kind | What is missing | What goes wrong |
|---|---|---|
| New user | No history | The first screen is a guess |
| New item | No ratings | Collaborative filtering never shows it |
| New platform | Very few users | Not enough data to train a model |

Popularity ranking gives everyone the same list, content-based filtering personalises only along known attributes, and hybrid systems still need enough interaction data to train.

## Our approach

```
Real seed data (20 users)
        │
   Pre-processing
        │
   GAN training  ◄── Generator vs Discriminator
        │
   Synthetic users (0 to 5,000)
        │
   Merge with real rows
        │
   Item-based collaborative filter
        │
   Top-5 recommendations
```

**Generator.** Takes 8 random numbers, turns them into a taste vector over item attributes (for example genre), and scores every item through those attributes:

```
z ~ N(0, I)  →  taste t  →  P(like item i) = σ( t · attributes(i) + bias(i) )
```

Because items are scored through their attributes, the generator can write interaction history for **new items that no real user has touched**. A new item gets the mean bias of known items, so its score comes only from its attributes.

**Discriminator.** Sees an interaction row (known items only) and predicts whether it came from a real user.

**Stable training with 20 rows.** One-sided label smoothing (0.9), input noise (±0.1), a discriminator that learns at half the generator's rate, two generator updates per discriminator update, Adam (β₁ = 0.5), batch size 16, 1,500 steps.

**Recommender.** Item-to-item cosine similarity over real plus synthetic rows. Top-5 items by similarity to what the user already liked.

## Results

Averaged over 8 random seed datasets, with 5,000 synthetic users (PyTorch implementation):

| Metric | Popularity | Real data only | Real + GAN |
|---|---|---|---|
| Precision@5 | 0.377 | 0.485 | **0.510** |
| NDCG@5 | 0.395 | 0.531 | **0.549** |
| Catalog coverage | 33% | 83% | **97%** |
| New items recommended | 0% | 0% | **about 50%** |

**Read these honestly:**

- The biggest and most reliable gain is for **new items**. A model trained on real data alone cannot recommend an item nobody has interacted with.
- For items that already have history, the improvement is small, and one of the eight datasets got slightly worse.
- A plain GAN that ignores item attributes gave **no** improvement. The attributes are what make the difference.
- Biased seed data produces biased synthetic data. With 80% of seed users preferring one category, that category's share went from 0.50 (real) to 0.55 (synthetic).
- All results come from a **simulated world** with known user tastes (24 items, 4 categories). Validation on real datasets such as MovieLens is the next step.

## Features of the web app

| Page | What you can do |
|---|---|
| **1. The problem** | Drag a slider to add 0 to 5,000 synthetic rows to the interaction matrix, with live accuracy numbers |
| **2. Train the GAN** | Watch the generator and discriminator compete live: taste-space plot, loss curves, discriminator verdict, real vs synthetic matrices, category shares. Pause, step, reset, change learning rate and speed, and switch on **Biased seed data** |
| **3. Be a new user** | Pick items or load a simulated user with hidden likes, and compare Popularity, Real only and Real + GAN side by side |
| **4. Evaluate** | Run a benchmark on 300 simulated cold-start users: Precision@5, NDCG@5, coverage, new-item hit rate, and a curve over synthetic volume |
| **5. Architecture** | The 7-stage pipeline with live stats per stage, open problems and roadmap |

A platform picker in the top bar relabels the catalog for movies, music, e-commerce, courses, jobs, food delivery or news.

The GAN trains **in your browser**. There is no backend.

## Getting started

You need [Node.js](https://nodejs.org) 18 or newer.

```bash
npm install
npm run dev
```

The app opens at http://localhost:5173.

### All commands

| Command | What it does |
|---|---|
| `npm run dev` | Start the development server |
| `npm run build` | Production build into `dist/` (can be uploaded to any static host) |
| `npm run preview` | Serve the production build locally |
| `npm run py:setup` | Install PyTorch and NumPy for the Python version |
| `npm run py:run` | Run the PyTorch version and print the results table |

### Python (PyTorch) version

An independent PyTorch implementation of the same method, which prints a results table. It is optional and not connected to the web app.

```bash
pip install -r ml/requirements.txt
python ml/gan_coldstart.py --demo
```

Useful options:

```bash
python ml/gan_coldstart.py --demo --seed 12                    # different random dataset
python ml/gan_coldstart.py --demo --steps 3000 --synthetic 0 1000 5000
```

It can also run on [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) (a rating of 4 or more counts as a like, and the 19 genre flags are the item attributes):

```bash
python ml/gan_coldstart.py --data path/to/ml-100k --n-real 50 --steps 3000
```

> The MovieLens path has only been tested for loading, not for results.

## Evaluation protocol

The world is simulated, so every test user's true taste is known and the recommendations are scored against the truth, not against the synthetic data itself.

- **World:** 24 items in 4 categories. The last item in each category is a new release with no real interactions (4 new items).
- **Seed data:** 20 real users, with new-item columns empty.
- **Test:** 300 fresh users. Each reveals 2 likes, the model recommends 5 items, and the other likes are the hidden targets.
- **Compared:** popularity, item CF on real data only, and item CF on real plus synthetic data (0, 250, 1,000, 2,500, 5,000 rows).
- **Metrics:** Precision@5, NDCG@5, catalog coverage, new-item hit rate.
- **Repeats:** 8 independent seed datasets, averaged.

## Project structure

```
coldstart-gan-npm/
├── index.html            Page markup
├── public/
│   ├── engine.js         GAN (hand-written backprop and Adam), item CF, evaluation
│   ├── app.js            UI, charts, training loop
│   ├── catalog.js        Item names for 7 platforms (all titles are invented)
│   └── styles.css
├── ml/
│   ├── gan_coldstart.py  PyTorch reference implementation
│   └── requirements.txt
├── package.json
└── vite.config.js
```

## Limitations

- Results come from a simulation. The attributes given to the generator match how tastes were generated, which favours the method.
- Gains for items that already have history are small and not guaranteed on every dataset.
- Synthetic data copies and can amplify bias in the seed data.
- We make no privacy claim for the synthetic users.
- Training here is tiny. A catalog with millions of items would need far larger networks and GPUs.

## Roadmap

Graph neural networks over user–item graphs, federated learning across platforms, privacy-preserving recommendation, real-time synthetic generation, multimodal recommendation, and evaluation on real datasets.

## Team

- Ananya B
- Arya Kelychankandy
- Tarun V

---

*We don't replace real user data. We help recommendation systems become useful before enough real data exists.*
