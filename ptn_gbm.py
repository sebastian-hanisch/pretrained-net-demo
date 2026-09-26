"""Gradient Boosting für die Regression (quadratischer Fehler) auf dem Histogramm-Baumkern aus dem LightGBM-Stück (`ptn_tree.py`): $F_m = F_{m-1} + \\eta\\,\\mathrm{Baum}_m$, Baum auf den Residuen (Gradient $F - y$, Hesse 1),
blattweises Wachsen, Blattwert $-G/(H+\\lambda)$. Die Bin-Grenzen werden einmal berechnet und für alle Runden wiederverwendet."""

from dataclasses import dataclass

import numpy as np

import ptn_tree as T


@dataclass(frozen=True)
class Ensemble:
    trees: tuple
    f0: float
    learning_rate: float
    n_train: int
    n_features: int


def fit(X, y, num_leaves=15, max_depth=None, max_bin=63, lam=1.0, min_child_samples=20, n_rounds=80, learning_rate=0.1, subsample=1.0, seed=0, X_val=None, y_val=None):
    """Regression auf y; mit Validierungsdaten zusätzlich die Lernkurven (mittlerer absoluter Fehler nach jeder Runde auf den Trainings- und den Validierungszeilen): Rückgabe (Modell, Training, Validierung)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    edges = T.build_bin_edges(X, max_bin)
    bins = T.digitize(X, edges)
    f0 = float(np.mean(y))
    F = np.full(n, f0)
    Fv = None if X_val is None else np.full(len(y_val), f0)
    rng = np.random.default_rng(seed)
    trees, curve, curve_tr = [], [], []
    for _ in range(n_rounds):
        idx = np.sort(rng.choice(n, max(2, int(round(subsample * n))), replace=False)) if subsample < 1.0 else np.arange(n)
        tree = T.grow(X[idx], F[idx] - y[idx], np.ones(len(idx)), edges, num_leaves=num_leaves, max_depth=max_depth, lam=lam, min_child_samples=min_child_samples, bins=bins[idx])
        F = F + learning_rate * T.predict_value(tree, X)
        trees.append(tree)
        curve_tr.append(float(np.mean(np.abs(F - y))))
        if Fv is not None:
            Fv = Fv + learning_rate * T.predict_value(tree, X_val)
            curve.append(float(np.mean(np.abs(Fv - y_val))))
    ens = Ensemble(tuple(trees), f0, learning_rate, n, X.shape[1])
    return (ens, np.array(curve_tr), np.array(curve)) if Fv is not None else ens


def predict(ens, X, upto=None):
    trees = ens.trees[:upto] if upto else ens.trees
    F = np.full(len(X), ens.f0)
    for t in trees:
        F = F + ens.learning_rate * T.predict_value(t, X)
    return F


def importances(ens):
    """Anteil am gesamten Gain je Merkmal (Summe über alle Bäume und Knoten, auf 1 normiert)."""
    imp = np.zeros(ens.n_features)
    for t in ens.trees:
        for k in t.internal_nodes():
            imp[t.feature[k]] += t.gain[k]
    s = imp.sum()
    return imp / s if s > 0 else imp
