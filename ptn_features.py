"""Merkmale und Ziel für ein globales Prognosemodell (ein Modell für alle Depots, direkte Mehrschritt-Prognose).

Eine Zeile gehört zu (Depot, Ursprung t, Horizont j): bekannt sind die Tage 0..t-1 des Depots (und der Kalender samt Aktionsplan für alle Tage), gesucht ist der Wert am Zieltag s = t + j - 1.

  Niveau         Mittel der letzten 28 Tage vor t; das Ziel ist log((y_s + 1) / (Niveau + 1)) - ein Verhältnis zum jüngsten Niveau (normiert) oder der rohe Wert y_s (nicht normiert)
  Gruppe lags    die letzten 7 Tage, die letzten vier Werte des Wochentags des Zieltags (mit demselben Abstand zu s), das 7-Tage-Mittel im Verhältnis zum 28-Tage-Niveau, das 28-Tage-Niveau im Verhältnis zum 91-Tage-Niveau, die Streuung der letzten 28 Tage
  Gruppe calendar  Wochentag, sin/cos des Jahrestags, Feiertag und Tag danach am Zieltag, Horizont j
  Gruppe promo   Aktion am Zieltag und Anteil der Aktionstage in den letzten 7 Tagen
Bei normiertem Ziel sind die Lag-Merkmale Verhältnisse zum Niveau, sonst Rohwerte (dann kommt das Niveau selbst als Merkmal dazu)."""

import numpy as np

import ptn_constants as C

FIRST_ORIGIN = 91
LAG_NAMES = [f"y(t-{k})" for k in range(1, 8)] + [f"Wochentag-Lag {i + 1}" for i in range(4)] + ["7-Tage-Mittel / 28-Tage-Niveau", "28-Tage-Niveau / 91-Tage-Niveau", "Streuung der letzten 28 Tage"]
CAL_NAMES = ["Wochentag", "Jahrestag (sin)", "Jahrestag (cos)", "Feiertag", "Tag nach Feiertag", "Horizont"]
PROMO_NAMES = ["Aktion am Zieltag", "Aktionen der letzten 7 Tage"]
GROUPS = {"lags": LAG_NAMES, "calendar": CAL_NAMES, "promo": PROMO_NAMES}


def feature_names(groups=("lags", "calendar", "promo"), normalize=True):
    names = []
    for g in ("lags", "calendar", "promo"):
        if g in groups:
            names += GROUPS[g]
    if not normalize and "lags" in groups:
        names.append("28-Tage-Niveau")
    return names


def _cums(port):
    z = np.zeros((port.n, 1))
    return np.concatenate([z, np.cumsum(port.y, axis=1)], axis=1), np.concatenate([z, np.cumsum(port.y ** 2, axis=1)], axis=1), np.concatenate([z, np.cumsum(port.promo, axis=1)], axis=1)


def build(port, dep, org, hor, groups=("lags", "calendar", "promo"), normalize=True):
    """Merkmalsmatrix (m, d), Ziel (m,) (roh oder normiert) und das Niveau (m,) für die Zeilen (dep[i], org[i], hor[i])."""
    dep, org, hor = np.asarray(dep), np.asarray(org), np.asarray(hor)
    cy, cy2, cp = _cums(port)
    s = org + hor - 1
    lvl28 = (cy[dep, org] - cy[dep, org - 28]) / 28.0
    lvl91 = (cy[dep, org] - cy[dep, org - 91]) / 91.0
    m7 = (cy[dep, org] - cy[dep, org - 7]) / 7.0
    var28 = np.maximum((cy2[dep, org] - cy2[dep, org - 28]) / 28.0 - lvl28 ** 2, 0.0)
    k0 = -(-hor // 7)
    wl = np.stack([port.y[dep, s - 7 * (k0 + i)] for i in range(4)], axis=1)
    if normalize == "seasonal":
        scale = wl.mean(axis=1) + 1.0
    elif normalize:
        scale = lvl28 + 1.0
    else:
        scale = np.ones(len(dep))
    cols = []
    if "lags" in groups:
        for k in range(1, 8):
            cols.append(port.y[dep, org - k] / scale)
        for i in range(4):
            cols.append(wl[:, i] / scale)
        cols += [m7 / (lvl28 + 1.0), lvl28 / (lvl91 + 1.0), np.sqrt(var28) / (lvl28 + 1.0)]
    if "calendar" in groups:
        doy = s % 365
        cols += [port.dow[s].astype(float), np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0), port.holiday[s], port.after[s], hor.astype(float)]
    if "promo" in groups:
        cols += [port.promo[dep, s], (cp[dep, org] - cp[dep, org - 7]) / 7.0]
    if not normalize and "lags" in groups:
        cols.append(lvl28)
    X = np.stack(cols, axis=1) if cols else np.zeros((len(dep), 0))
    target = np.log((port.y[dep, s] + 1.0) / scale) if normalize else port.y[dep, s]
    return X, target, scale - 1.0 if normalize else lvl28


def to_orders(pred, lvl28, normalize=True):
    """Prognose in Aufträgen aus der Modellausgabe (bei normiertem Ziel: exp(pred) * (Niveau + 1) - 1); negative Werte auf 0."""
    return np.maximum(np.exp(pred) * (lvl28 + 1.0) - 1.0, 0.0) if normalize else np.maximum(pred, 0.0)


def training_rows(port, depots, horizon, stride=4, per_origin=2, first=FIRST_ORIGIN, last_target=C.FIRST_TEST, seed=0):
    """Zeilen für das Training: je Depot alle stride Tage ein Ursprung, je Ursprung per_origin zufällige Horizonte; der Zieltag liegt vor last_target."""
    rng = np.random.default_rng(seed)
    org = np.arange(first, last_target - horizon + 1, stride)
    d = np.repeat(np.asarray(depots), len(org) * per_origin)
    o = np.tile(np.repeat(org, per_origin), len(depots))
    h = rng.integers(1, horizon + 1, size=len(d))
    return d, o, h


def test_rows(port, depots, horizon, first=C.FIRST_TEST, step=1):
    """Alle Ursprünge des Testjahres und alle Horizonte für die angegebenen Depots: (dep, org, hor) mit der Anordnung (Depot, Ursprung, Horizont)."""
    org = np.arange(first, port.y.shape[1] - horizon + 1, step)
    d = np.repeat(np.asarray(depots), len(org) * horizon)
    o = np.tile(np.repeat(org, horizon), len(depots))
    h = np.tile(np.arange(1, horizon + 1), len(depots) * len(org))
    return d, o, h, org
