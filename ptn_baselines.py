"""Lokale Vergleichsverfahren, je Depot mit der eigenen Historie: Wochenmittel (Stück 1), Holt-Winters multiplikativ (Stück 2) und die lineare Regression auf Kalender und Aktionsplan im Log (Stück 4, OLS-Fehler).

Alle Verfahren sehen nur die Tage vor dem ersten Ursprung des Testjahres (und darin höchstens die letzten `hist` Tage): die Parameter werden dort geschätzt, die Zustände laufen durch das Testjahr."""

import numpy as np

import ptn_constants as C
import ptn_ets as ETS


def snaive_k(y, org, h, k=4):
    """Wochenmittel: derselbe Wochentag, gemittelt über die letzten k Wochen. y: (T,), org: (n_o,) -> (n_o, h)."""
    j = np.arange(h)
    return np.mean([y[org[:, None] - 7 * (i + 1) + (j % 7)[None, :]] for i in range(k)], axis=0)


def hw_forecast(y, org, h, hist=None, first=C.FIRST_TEST):
    """Holt-Winters multiplikativ (Stück 2): Parameter aus den letzten hist Tagen vor first (Standard: alle), Zustände durch die ganze Reihe; (n_o, h)."""
    start = 0 if hist is None else max(0, first - hist)
    yy = y[start:]
    model = ETS.fit(yy[:first - start], "hw_mult")
    states = ETS.filter_states(model, yy)
    return ETS.forecast_origins(model, states, org - start, h)


def regression_design(dow, holiday, after, promo, n_days=C.N_DAYS, year_k=2):
    """Regressoren für alle Tage: Konstante, sechs Wochentage, Trend, Fourier-Paare, Feiertag, Tag danach, Aktion; (n_days, K)."""
    t = np.arange(n_days)
    cols = [np.ones(n_days)] + [(dow == d).astype(float) for d in range(1, 7)] + [t / 365.0]
    for k in range(1, year_k + 1):
        cols += [np.sin(2 * np.pi * k * (t % 365) / 365.0), np.cos(2 * np.pi * k * (t % 365) / 365.0)]
    cols += [holiday, after, promo]
    return np.stack(cols, axis=1)


def regression_forecast(y, X, org, h, hist=None, first=C.FIRST_TEST):
    """OLS im Log auf den letzten hist Tagen vor first; Prognose exp(x' beta) für die Zieltage (Median); (n_o, h). Mit weniger als einem Jahr Historie lassen sich Trend und Jahresmuster nicht schätzen und
    werden weggelassen (Spalten 7 bis 11); eine kleine Ridge-Strafe (1e-3 mal mittlere Spur) hält die Lösung endlich."""
    start = 0 if hist is None else max(0, first - hist)
    cols = np.arange(X.shape[1]) if (hist is None or hist >= 365) else np.array([0, 1, 2, 3, 4, 5, 6] + list(range(X.shape[1] - 3, X.shape[1])))
    Xt = X[start:first][:, cols]
    z = np.log(np.maximum(y[start:first], 1.0))
    A = Xt.T @ Xt
    beta = np.linalg.solve(A + 1e-3 * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), Xt.T @ z)
    days = org[:, None] + np.arange(h)[None, :]
    return np.exp(X[days][:, :, cols] @ beta)


def mase_scale(y, first=C.FIRST_TEST):
    """Mittlerer absoluter Fehler der saisonal naiven Prognose (Periode 7) auf den Tagen vor first - Nenner der MASE."""
    return float(np.abs(y[7:first] - y[:first - 7]).mean())
