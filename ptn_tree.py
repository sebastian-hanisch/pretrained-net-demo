"""Der LightGBM-Baumkern (Ke et al. 2017, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"): zwei Unterschiede zu xgboost-demo, beide um Rechenzeit zu sparen, nicht um die Mathematik zu ändern.

1. **Histogramm-Split-Suche statt exakter Suche:** jedes Merkmal wird VOR dem Wachsen einmal in `max_bin` Bins eingeteilt (Quantil-Grenzen über die ganze Trainingsmenge, einmal berechnet, für alle Runden
   und alle Knoten wiederverwendet). Ein Knoten braucht dann nur noch die Summen von Gradient und Hesse-Diagonale JE BIN (ein Histogramm), nicht mehr jede einzelne Schwelle - die Split-Suche kostet
   `O(Bins)` statt `O(Zeilen)` je Merkmal.
2. **Blattweises statt ebenenweises Wachsen:** cart-demo/gradient-boosting-demo/xgboost-demo wachsen Ebene für Ebene (jeder Knoten einer Ebene wird geteilt, bevor die nächste beginnt). LightGBM wählt
   stattdessen IMMER das Blatt mit dem größten möglichen Gain als Nächstes (eine Prioritätswarteschlange) - bei gleicher Blattzahl meist tiefere, unregelmäßigere Bäume mit weniger verschwendeten Splits
   auf uninteressanten Ästen.

**Differenz-Trick:** wird ein Blatt geteilt, wird nur das KLEINERE Kind direkt aus seinen Zeilen histogrammiert; das größere Kind ergibt sich als `Histogramm(Eltern) - Histogramm(kleineres Kind)` - spart
eine zweite Zeilen-Abtastung. Die Gain-Formel selbst ist wortgleich mit xgboost-demo (`0.5*[GL²/(HL+λ)+GR²/(HR+λ)-G²/(H+λ)]-γ`, Blattwert `-G/(H+λ)`) - der Unterschied liegt allein darin, WELCHE
Schwellen geprüft werden und in WELCHER REIHENFOLGE Knoten geteilt werden, nicht in der Optimierung selbst."""

import heapq
import itertools
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Tree:
    feature: np.ndarray          # -1 = Blatt
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray            # Newton-Schritt -G/(H+lambda)
    gain: np.ndarray             # Gain des Splits an diesem Knoten (0 bei Blättern)
    g_sum: np.ndarray
    h_sum: np.ndarray
    n: np.ndarray
    depth: np.ndarray
    split_order: np.ndarray      # k-te Teilung in der Reihenfolge des blattweisen Wachsens (-1 = nie geteilt)
    lam: float
    gamma: float
    n_total: int
    n_features: int
    n_bins: int

    @property
    def n_nodes(self):
        return len(self.feature)

    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    @property
    def max_depth(self):
        return int(self.depth.max())

    def internal_nodes(self):
        return np.nonzero(self.feature >= 0)[0]


# --- Globale Bin-Grenzen (einmal je Fit, für alle Runden und Knoten wiederverwendet) -----------------------------------------------------------------

def build_bin_edges(X, max_bin):
    """Quantil-Grenzen je Merkmal (ungefähr gleich viele Trainingszeilen je Bin), doppelte Grenzen bei vielen Wiederholungen entfernt. Rückgabe: Liste von `d` aufsteigenden Kantenfeldern."""
    edges = []
    for f in range(X.shape[1]):
        q = np.linspace(0.0, 1.0, max_bin + 1)[1:-1]
        e = np.unique(np.quantile(X[:, f], q))
        edges.append(e)
    return edges


def digitize(X, edges):
    """Bin-Nummer jeder Zeile je Merkmal (0 .. len(edges[f])), vektorisiert über `np.searchsorted`."""
    m, d = X.shape
    bins = np.empty((m, d), dtype=np.int32)
    for f in range(d):
        bins[:, f] = np.searchsorted(edges[f], X[:, f], side="left")
    return bins


# --- Histogramm und Split-Suche ------------------------------------------------------------------------------------------------------------------------

def histogram(bins_f, grad, hess, n_bins):
    """(Gradientensumme, Hessesumme, Zeilenzahl) je Bin für ein Merkmal."""
    g = np.bincount(bins_f, weights=grad, minlength=n_bins)
    h = np.bincount(bins_f, weights=hess, minlength=n_bins)
    c = np.bincount(bins_f, minlength=n_bins)
    return g, h, c


def best_split_from_histograms(hist_g, hist_h, hist_c, edges, lam, gamma, min_child_weight, min_child_samples):
    """(Merkmal, Schwelle, Gain) des besten Splits über alle Bin-Grenzen aller Merkmale, aus den (schon aufgebauten) Histogrammen - kein weiterer Zeilenzugriff nötig.
    `min_child_samples` prüft die ZEILENZAHL je Kind (aus dem Zähl-Histogramm), `min_child_weight` die Hesse-Summe - beide unabhängig, wie im echten LightGBM."""
    Gtot, Htot = float(hist_g[0].sum()), float(hist_h[0].sum())
    Ctot = int(hist_c[0].sum())
    best = None
    for f, (g, h, c) in enumerate(zip(hist_g, hist_h, hist_c)):
        if len(g) < 2:
            continue
        cg = np.cumsum(g)[:-1]
        ch = np.cumsum(h)[:-1]
        cc = np.cumsum(c)[:-1]
        Gr, Hr, Cr = Gtot - cg, Htot - ch, Ctot - cc
        gain = 0.5 * (cg ** 2 / (ch + lam) + Gr ** 2 / (Hr + lam) - Gtot ** 2 / (Htot + lam)) - gamma
        ok = (ch >= min_child_weight) & (Hr >= min_child_weight) & (cc >= min_child_samples) & (Cr >= min_child_samples)
        gain = np.where(ok, gain, -np.inf)
        k = int(gain.argmax())
        if np.isfinite(gain[k]) and gain[k] > 0 and (best is None or gain[k] > best[2]):
            best = (f, float(edges[f][k]), float(gain[k]))
    return best


# --- Wachsen mit Differenz-Trick: blattweise (Standard) oder ebenenweise (nur fürs Experiment) ---------------------------------------------------------

def grow(X, grad, hess, edges, num_leaves=31, max_depth=None, lam=1.0, gamma=0.0, min_child_weight=1.0, min_child_samples=1, policy="leaf", stats=None, bins=None):
    """Wächst auf denselben Histogrammen mit demselben Differenz-Trick, nur die REIHENFOLGE unterscheidet sich: `policy="leaf"` (Standard) nimmt immer das Blatt mit dem größten möglichen Gain
    als Nächstes (Prioritätswarteschlange); `policy="level"` nimmt sie in Entstehungsreihenfolge (FIFO) - Ebene für Ebene, wie cart-demo/xgboost-demo, nur eben über Histogramme statt exakter Suche.
    Beide stoppen bei `num_leaves`, der Tiefengrenze oder wenn kein Blatt mehr einen positiven Gain hat. Für jedes neue Kind wird nur das KLEINERE direkt histogrammiert, das größere per Differenz.
    `stats` (optional, dict): wird mit `histograms_built` (Aufrufe von `histogram()`, je Knoten und Merkmal) und `candidates_checked` (geprüfte Bin-Grenzen über alle Split-Versuche) befüllt -
    nur fürs Experiment "Zähler gegen exakte Suche" gedacht, kostet sonst nichts (Standardaufruf ohne `stats` bleibt unverändert)."""
    X = np.asarray(X, dtype=float)
    grad = np.asarray(grad, dtype=float)
    hess = np.asarray(hess, dtype=float)
    max_depth = 10 ** 6 if max_depth is None else int(max_depth)
    n, d = X.shape
    bins = digitize(X, edges) if bins is None else bins
    n_bins = [len(e) + 1 for e in edges]
    if stats is not None:
        stats.setdefault("histograms_built", 0)
        stats.setdefault("candidates_checked", 0)

    feature, threshold, left, right, value, gain, g_sum, h_sum, size, depth, order = [], [], [], [], [], [], [], [], [], [], []
    node_hist = {}                                                            # Knoten -> Liste (Gradient-, Hesse-, Zähl-Histogramm) je Merkmal
    node_idx = {}

    def new_node(idx, dep, hist):
        G, H = float(grad[idx].sum()), float(hess[idx].sum())
        t = len(feature)
        feature.append(-1), threshold.append(np.nan), left.append(-1), right.append(-1)
        value.append(-G / (H + lam)), gain.append(0.0), g_sum.append(G), h_sum.append(H), size.append(len(idx)), depth.append(dep), order.append(-1)
        node_idx[t] = idx
        node_hist[t] = hist
        return t

    def hist_of(idx):
        if stats is not None:
            stats["histograms_built"] += d
        return [histogram(bins[idx, f], grad[idx], hess[idx], n_bins[f]) for f in range(d)]

    root = new_node(np.arange(n), 0, hist_of(np.arange(n)))
    counter = itertools.count()
    heap = []                                                                  # policy="leaf": Max-Heap (Gain, Zähler, Knoten, Split)
    queue = []                                                                 # policy="level": FIFO derselben Einträge

    def push(t):
        if depth[t] >= max_depth or size[t] < 2 * max(min_child_samples, 1):
            return
        hg = [hg for hg, _, _ in node_hist[t]]
        hh = [hh for _, hh, _ in node_hist[t]]
        hc = [hc for _, _, hc in node_hist[t]]
        if stats is not None:
            stats["candidates_checked"] += sum(max(len(g) - 1, 0) for g in hg)
        s = best_split_from_histograms(hg, hh, hc, edges, lam, gamma, min_child_weight, min_child_samples)
        if s is None:
            return
        if policy == "leaf":
            heapq.heappush(heap, (-s[2], next(counter), t, s))
        else:
            queue.append((t, s))

    def pop_next():
        if policy == "leaf":
            _, _, t, s = heapq.heappop(heap)
            return t, s
        t, s = queue.pop(0)
        return t, s

    def frontier_nonempty():
        return bool(heap) if policy == "leaf" else bool(queue)

    push(root)
    n_splits = 0
    while frontier_nonempty() and (len(feature) - n_splits) < num_leaves:      # aktuelle Blattzahl = Knoten minus bisherige Splits
        t, (f, thr, g) = pop_next()
        idx = node_idx[t]
        go_left = X[idx, f] <= thr
        idx_l, idx_r = idx[go_left], idx[~go_left]
        if len(idx_l) == 0 or len(idx_r) == 0:
            continue
        if len(idx_l) <= len(idx_r):
            hist_l = hist_of(idx_l)
            hist_r = [(pg - lg, ph - lh, pc - lc) for (pg, ph, pc), (lg, lh, lc) in zip(node_hist[t], hist_l)]
        else:
            hist_r = hist_of(idx_r)
            hist_l = [(pg - rg, ph - rh, pc - rc) for (pg, ph, pc), (rg, rh, rc) in zip(node_hist[t], hist_r)]
        lt = new_node(idx_l, depth[t] + 1, hist_l)
        rt = new_node(idx_r, depth[t] + 1, hist_r)
        feature[t], threshold[t], left[t], right[t], gain[t], order[t] = f, thr, lt, rt, g, n_splits
        del node_hist[t], node_idx[t]
        n_splits += 1
        push(lt)
        push(rt)

    return Tree(np.array(feature), np.array(threshold), np.array(left), np.array(right), np.array(value), np.array(gain), np.array(g_sum), np.array(h_sum),
                np.array(size), np.array(depth), np.array(order), lam, gamma, n, d, max(n_bins))


# --- Anwenden --------------------------------------------------------------------------------------------------------------------------------------

def apply(tree, X):
    X = np.asarray(X, dtype=float)
    node = np.zeros(len(X), dtype=int)
    while True:
        idx = np.nonzero(tree.feature[node] >= 0)[0]
        if len(idx) == 0:
            return node
        cur = node[idx]
        go_left = X[idx, tree.feature[cur]] <= tree.threshold[cur]
        node[idx] = np.where(go_left, tree.left[cur], tree.right[cur])


def predict_value(tree, X):
    return tree.value[apply(tree, X)]


def importances(tree):
    imp = np.zeros(tree.n_features)
    for t in tree.internal_nodes():
        imp[tree.feature[t]] += tree.gain[t]
    s = imp.sum()
    return imp / s if s > 0 else imp
