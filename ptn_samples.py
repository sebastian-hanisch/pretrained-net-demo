"""Trainings- und Prüfzeilen für das Netz. Eine Zeile gehört zu (Depot, Ursprung t): bekannt sind die Tage 0..t-1 des Depots und der Kalender samt Aktionsplan, gesucht sind die Tage t..t+H-1.

  Fenster       die letzten WINDOW Tage vor t als log((y + 1) / (Niveau + 1)), Niveau = Mittel des Fensters (die Größe des Depots fällt heraus)
  Ziel          dieselbe Normierung für die H Zieltage
  Zusatz        "window": Wochentag des Ursprungs (One-Hot); "full": dazu Feiertag, Tag danach und Aktion an jedem der H Zieltage sowie sin/cos des Jahrestags am Ursprung
Prognosen in Aufträgen: exp(Ausgabe) * (Niveau + 1) - 1, negative Werte auf 0."""

import numpy as np

import ptn_constants as C


def covariate_dim(covariates, horizon=C.HORIZON):
    return 7 if covariates == "window" else 7 + 3 * horizon + 2


def build(port, dep, org, covariates="full", window=C.WINDOW, horizon=C.HORIZON):
    """Fenster (m, L), Zusatzmerkmale (m, Dc), Ziel (m, H) und das Niveau (m,) für die Zeilen (dep[i], org[i])."""
    dep, org = np.asarray(dep), np.asarray(org)
    back = org[:, None] + np.arange(-window, 0)[None, :]
    fwd = org[:, None] + np.arange(horizon)[None, :]
    yw = port.y[dep[:, None], back]
    level = yw.mean(axis=1)
    scale = (level + 1.0)[:, None]
    w = np.log((yw + 1.0) / scale)
    tgt = np.log((port.y[dep[:, None], fwd] + 1.0) / scale)
    dow = np.zeros((len(dep), 7))
    dow[np.arange(len(dep)), port.dow[org]] = 1.0
    cols = [dow]
    if covariates == "full":
        doy = org % 365
        cols += [port.holiday[fwd], port.after[fwd], port.promo[dep[:, None], fwd], np.sin(2 * np.pi * doy / 365.0)[:, None], np.cos(2 * np.pi * doy / 365.0)[:, None]]
    return w, np.concatenate(cols, axis=1), tgt, level


def to_orders(pred, level):
    return np.maximum(np.exp(pred) * (level + 1.0)[:, None] - 1.0, 0.0)


def rows(depots, first, last, stride=1):
    """Alle Paare (Depot, Ursprung) mit Ursprung in first..last (einschließlich) im Abstand stride."""
    org = np.arange(first, last + 1, stride)
    return np.repeat(np.asarray(depots), len(org)), np.tile(org, len(depots))


def pretrain_rows(depots, stride=2, window=C.WINDOW, horizon=C.HORIZON):
    """Vortraining: Zieltage vor dem Testjahr; die letzten VAL_DAYS Tage davor sind die Prüfzeilen der Lernkurve (Zieltage im Prüfstück, kein Ursprung darin wird trainiert)."""
    cut = C.FIRST_TEST - C.VAL_DAYS
    tr = rows(depots, window, cut - horizon, stride)
    va = rows(depots, cut, C.FIRST_TEST - horizon, 3)
    return tr, va


def history_rows(depots, hist, window=C.WINDOW, horizon=C.HORIZON):
    """Eigene Historie eines neuen Depots: die letzten hist Tage vor dem Testjahr; Ursprünge so, dass Fenster und Ziel darin liegen."""
    return rows(depots, C.FIRST_TEST - hist + window, C.FIRST_TEST - horizon, 1)


def test_rows(depots, horizon=C.HORIZON, n_days=C.N_DAYS):
    org = np.arange(C.FIRST_TEST, n_days - horizon + 1)
    return np.repeat(np.asarray(depots), len(org)), np.tile(org, len(depots)), org
