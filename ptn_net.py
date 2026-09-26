"""Ein Netz im Stil von N-BEATS in numpy: Blöcke mit Rückblick (Backcast) und Vorausschau (Forecast), Residuen zwischen den Blöcken, Vorwärts- und Rückwärtsrechnung von Hand, Adam.

Eingabe: das normierte Rückblick-Fenster w (L Werte) und Zusatzmerkmale c (Dc Werte). Block k bekommt den Rest r_k (r_0 = w) und die Zusatzmerkmale,
    h1 = relu(W1 [r_k, c] + b1),  h2 = relu(W2 h1 + b2),  back_k = Wb h2 + bb,  fc_k = Wf h2 + bf,  r_(k+1) = r_k - back_k,
die Prognose ist die Summe der Vorausschauen  yhat = sum_k fc_k  (H Werte). Jeder Block erklärt also, was die Blöcke davor vom Fenster übrig ließen.
Verlust: mittlerer absoluter Fehler (L1) auf der normierten Log-Skala; die Ableitung wird von Hand berechnet und in den Tests gegen Differenzenquotienten geprüft."""

from dataclasses import dataclass

import numpy as np

_NAMES = ("W1", "b1", "W2", "b2", "Wb", "bb", "Wf", "bf")


@dataclass(frozen=True)
class Shape:
    window: int
    covariates: int
    horizon: int
    blocks: int
    width: int


def init(shape, seed=0):
    """He-Initialisierung der versteckten Schichten, kleine Ausgangsschichten (die Prognose startet nahe null)."""
    rng = np.random.default_rng(seed)
    p = {}
    d_in = shape.window + shape.covariates
    for k in range(shape.blocks):
        p[f"{k}.W1"] = rng.normal(size=(d_in, shape.width)) * np.sqrt(2.0 / d_in)
        p[f"{k}.b1"] = np.zeros(shape.width)
        p[f"{k}.W2"] = rng.normal(size=(shape.width, shape.width)) * np.sqrt(2.0 / shape.width)
        p[f"{k}.b2"] = np.zeros(shape.width)
        p[f"{k}.Wb"] = rng.normal(size=(shape.width, shape.window)) * (0.1 / np.sqrt(shape.width))
        p[f"{k}.bb"] = np.zeros(shape.window)
        p[f"{k}.Wf"] = rng.normal(size=(shape.width, shape.horizon)) * (0.1 / np.sqrt(shape.width))
        p[f"{k}.bf"] = np.zeros(shape.horizon)
    return p


def shape_of(p):
    blocks = sum(1 for k in p if k.endswith(".W1"))
    d_in, width = p["0.W1"].shape
    window, horizon = p["0.Wb"].shape[1], p["0.Wf"].shape[1]
    return Shape(window, d_in - window, horizon, blocks, width)


def n_params(p):
    return int(sum(v.size for v in p.values()))


def copy(p):
    return {k: v.copy() for k, v in p.items()}


def forward(p, w, c, cache=False):
    """w: (B, L), c: (B, Dc) -> yhat (B, H); mit cache=True zusätzlich die Zwischenwerte für die Rückwärtsrechnung."""
    sh = shape_of(p)
    r = w
    yhat = np.zeros((w.shape[0], sh.horizon))
    store = []
    for k in range(sh.blocks):
        u = np.concatenate([r, c], axis=1)
        h1 = np.maximum(u @ p[f"{k}.W1"] + p[f"{k}.b1"], 0.0)
        h2 = np.maximum(h1 @ p[f"{k}.W2"] + p[f"{k}.b2"], 0.0)
        back = h2 @ p[f"{k}.Wb"] + p[f"{k}.bb"]
        yhat = yhat + h2 @ p[f"{k}.Wf"] + p[f"{k}.bf"]
        store.append((u, h1, h2))
        r = r - back
    return (yhat, store) if cache else yhat


def loss_and_grads(p, w, c, y, loss="l1"):
    """Mittlerer Fehler (über Zeilen und Horizonte) und seine Ableitung nach allen Parametern."""
    sh = shape_of(p)
    yhat, store = forward(p, w, c, cache=True)
    diff = yhat - y
    n = diff.size
    if loss == "l1":
        val = float(np.abs(diff).mean())
        g_out = np.sign(diff) / n
    else:
        val = float((diff ** 2).mean())
        g_out = 2.0 * diff / n
    grads = {}
    G = np.zeros_like(w)                                   # Ableitung nach dem Rest r_(k+1) des letzten Blocks: der Rest geht nicht in die Prognose ein
    for k in reversed(range(sh.blocks)):
        u, h1, h2 = store[k]
        g_back = -G
        grads[f"{k}.Wf"] = h2.T @ g_out
        grads[f"{k}.bf"] = g_out.sum(axis=0)
        grads[f"{k}.Wb"] = h2.T @ g_back
        grads[f"{k}.bb"] = g_back.sum(axis=0)
        d2 = (g_out @ p[f"{k}.Wf"].T + g_back @ p[f"{k}.Wb"].T) * (h2 > 0)
        grads[f"{k}.W2"] = h1.T @ d2
        grads[f"{k}.b2"] = d2.sum(axis=0)
        d1 = (d2 @ p[f"{k}.W2"].T) * (h1 > 0)
        grads[f"{k}.W1"] = u.T @ d1
        grads[f"{k}.b1"] = d1.sum(axis=0)
        G = G + (d1 @ p[f"{k}.W1"].T)[:, :sh.window]       # der Rest r_k wirkt direkt (Identität) und über den Block k
    return val, grads


def mae(p, w, c, y):
    return float(np.abs(forward(p, w, c) - y).mean())


def train(p, w, c, y, steps, batch, lr, seed=0, steps_per_epoch=None, val=None, snapshots=(), beta1=0.9, beta2=0.999):
    """Adam mit Mini-Batches (die Reihenfolge der Zeilen wird bei jedem Durchlauf neu gemischt). Rückgabe: Parameter, Verlauf {"train": ..., "val": ...} (MAE am Ende jeder Epoche; Epoche = steps_per_epoch Schritte) und
    Momentaufnahmen {Epoche: Parameter} für die angegebenen Epochen (0 = vor dem Training)."""
    p = copy(p)
    rng = np.random.default_rng(seed)
    n = len(w)
    batch = min(batch, n)
    spe = steps_per_epoch or max(1, int(np.ceil(n / batch)))
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    hist = {"train": [], "val": []}
    snaps = {0: copy(p)} if 0 in snapshots else {}
    order = rng.permutation(n)
    pos = 0
    for step in range(1, steps + 1):
        if pos + batch > n:
            order = rng.permutation(n)
            pos = 0
        idx = order[pos:pos + batch]
        pos += batch
        _, g = loss_and_grads(p, w[idx], c[idx], y[idx])
        for k in p:
            m[k] = beta1 * m[k] + (1 - beta1) * g[k]
            v2[k] = beta2 * v2[k] + (1 - beta2) * g[k] ** 2
            mh = m[k] / (1 - beta1 ** step)
            vh = v2[k] / (1 - beta2 ** step)
            p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if step % spe == 0:
            ep = step // spe
            sub = np.arange(0, n, max(1, n // 4000))
            hist["train"].append(mae(p, w[sub], c[sub], y[sub]))
            if val is not None:
                hist["val"].append(mae(p, *val))
            if ep in snapshots:
                snaps[ep] = copy(p)
    return p, hist, snaps
