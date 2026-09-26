"""Exponentielle Glättung (Zustandsraum-Schreibweise, additiver Fehler) in numpy: Niveau, Trend (auch gedämpft) und Wochensaison (additiv oder multiplikativ).

Ein Schritt (Tag t, Zustände vor dem Tag: Niveau l, Trend b, Saison s[t mod 7]):

    Prognose   yhat = (l + phi*b) + s        additiv         yhat = (l + phi*b) * s        multiplikativ
    Fehler     e    = y - yhat
    Niveau     l'   = l + phi*b + alpha*e    (multiplikativ: alpha*e/s)
    Trend      b'   = phi*b + beta*e         (multiplikativ: beta*e/s)
    Saison     s'   = s + gamma*e            (multiplikativ: gamma*e/l', mit dem schon aktualisierten Niveau - wie in statsmodels)

Ohne Trend bleibt b = 0, ohne Saison entfällt s. Die Parameter werden auf den Trainingstagen durch Minimieren der quadratischen Ein-Schritt-Fehler geschätzt: ein fester Zufallsansatz im Einheitswürfel (in einem Rutsch für
viele Kandidaten gerechnet), gefolgt von einer schrumpfenden lokalen Suche. Danach werden die Zustände mit den festen Parametern durch die ganze Reihe fortgeschrieben - zu jedem Ursprung t kennt das Modell nur y[:t]."""

from dataclasses import dataclass

import numpy as np

import ptn_constants as C

M = C.SEASON_PERIOD


@dataclass(frozen=True)
class Model:
    key: str
    trend: str                # none | add | damped
    season: str               # none | add | mul
    alpha: float
    beta: float
    gamma: float
    phi: float
    init: tuple               # (l0, b0, s0 als Tupel der Länge 7 oder ())
    sse: float
    n_fit: int

    @property
    def n_params(self):
        """Glättungsparameter plus Anfangszustände (Niveau, Trend, sieben Saisonwerte - einer davon durch die Normierung festgelegt)."""
        k = 1 + (self.trend != "none") + (self.season != "none") + (self.trend == "damped")
        return int(k + 1 + (self.trend != "none") + (M - 1 if self.season != "none" else 0))

    @property
    def rmse(self):
        return float(np.sqrt(self.sse / self.n_fit))

    @property
    def aicc(self):
        n, k = self.n_fit, self.n_params
        return float(n * np.log(self.sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1))


@dataclass(frozen=True)
class States:
    level: np.ndarray         # (n+1,) Niveau vor dem Tag t (Index t)
    trend: np.ndarray         # (n+1,)
    season: np.ndarray        # (n+1, 7) aktueller Saisonwert je Wochentag-Platz
    error: np.ndarray         # (n,) Ein-Schritt-Fehler y[t] - yhat[t]


def init_states(y, trend, season):
    """Anfangszustände aus den Trainingstagen y. Ohne Saison: Niveau und Trend aus den ersten vier Wochen. Mit Saison: Wochenmuster aus dem Verhältnis (bzw. der Differenz) zum zentrierten 7-Tage-Mittel über die ganze
    Trainingsreihe, auf Mittel 1 (bzw. 0) normiert; Niveau und Trend aus einer Geraden durch die ersten acht saisonbereinigten Wochen."""
    y = np.asarray(y, dtype=float)
    if season == "none":
        n0 = C.INIT_DAYS
        b0 = (y[n0 // 2:n0].mean() - y[:n0 // 2].mean()) / (n0 // 2) if trend != "none" else 0.0
        return float(y[:n0].mean() - b0 * (n0 / 2 + 0.5)), float(b0), ()
    ma = np.convolve(y, np.ones(M) / M, mode="valid")                 # ma[i] gehört zum Tag i + 3
    days = np.arange(M // 2, len(y) - M // 2)
    ratio = y[days] - ma if season == "add" else y[days] / ma
    s0 = np.array([ratio[days % M == d].mean() for d in range(M)])
    s0 = s0 - s0.mean() if season == "add" else s0 / s0.mean()
    n0 = C.INIT_WEEKS * M
    z = y[:n0] - s0[np.arange(n0) % M] if season == "add" else y[:n0] / s0[np.arange(n0) % M]
    if trend == "none":
        return float(z.mean()), 0.0, tuple(float(v) for v in s0)
    slope, icpt = np.polyfit(np.arange(n0) + 1.0, z, 1)
    return float(icpt), float(slope), tuple(float(v) for v in s0)


def _run(y, trend, season, params, init, start_sse, record=False):
    """Alle Kandidaten (Zeilen von params: alpha, beta, gamma, phi) gleichzeitig durch die Reihe y. Rückgabe: SSE ab Tag start_sse je Kandidat, bei record=True zusätzlich die Zustandsverläufe (nur für einen Kandidaten)."""
    alpha, beta, gamma, phi = (params[:, i] for i in range(4))
    nb = len(alpha)
    l = np.full(nb, init[0])
    b = np.full(nb, init[1])
    s = np.tile(np.array(init[2], dtype=float), (nb, 1)) if season != "none" else None
    sse = np.zeros(nb)
    if record:
        L, Bt, S, Err = [l[0]], [b[0]], [s[0].copy() if s is not None else np.zeros(M)], []
    for t, yt in enumerate(y):
        base = l + phi * b
        if season == "none":
            yhat = base
        else:
            st = s[:, t % M]
            yhat = base + st if season == "add" else base * st
        e = yt - yhat
        if season == "mul":
            es = e / np.maximum(st, C.EPS)
            l_new, b_new = base + alpha * es, phi * b + beta * es
            s[:, t % M] = st + gamma * e / np.maximum(l_new, C.EPS)
        else:
            l_new, b_new = base + alpha * e, phi * b + beta * e
            if season == "add":
                s[:, t % M] = st + gamma * e
        l, b = l_new, b_new
        if t >= start_sse:
            sse += e * e
        if record:
            L.append(l[0]); Bt.append(b[0]); S.append(s[0].copy() if s is not None else np.zeros(M)); Err.append(e[0])
    if record:
        return sse, States(np.array(L), np.array(Bt), np.array(S), np.array(Err))
    return sse


def _map_params(u, trend, season):
    """Einheitswürfel -> (alpha, beta, gamma, phi); beta <= alpha und gamma <= 1 - alpha (übliche Zulässigkeit), unbenutzte Parameter fest."""
    a = 0.001 + 0.9 * u[:, 0] ** 2
    beta = a * (0.001 + 0.9 * u[:, 1] ** 2) if trend != "none" else np.zeros(len(u))
    gamma = (1 - a) * (0.001 + 0.9 * u[:, 2] ** 2) if season != "none" else np.zeros(len(u))
    phi = C.PHI_MIN + (C.PHI_MAX - C.PHI_MIN) * u[:, 3] if trend == "damped" else np.ones(len(u))
    return np.stack([a, beta, gamma, phi], axis=1)


def fit(y_train, key, seed=C.FIT_SEED):
    """Parameter durch Minimieren der Ein-Schritt-SSE auf y_train schätzen (deterministisch: fester Seed)."""
    trend, season = C.ETS_MODELS[key]
    y_train = np.asarray(y_train, dtype=float)
    init = init_states(y_train, trend, season)
    dims = [0] + ([1] if trend != "none" else []) + ([2] if season != "none" else []) + ([3] if trend == "damped" else [])
    rng = np.random.default_rng(seed)
    start = C.INIT_DAYS

    def score(u):
        with np.errstate(all="ignore"):
            out = _run(y_train, trend, season, _map_params(u, trend, season), init, start)
        return np.where(np.isfinite(out), out, np.inf)

    U = np.full((C.FIT_STAGE1, 4), 0.5)
    U[:, dims] = rng.random((C.FIT_STAGE1, len(dims)))
    S = score(U)
    order = np.argsort(S)[:C.FIT_TOP]
    top_u, top_s = U[order], S[order]
    for r in range(C.FIT_ROUNDS):
        scale = 0.15 * 0.55 ** r
        cand = np.repeat(top_u, C.FIT_PER_START, axis=0)
        noise = np.zeros_like(cand)
        noise[:, dims] = rng.normal(size=(len(cand), len(dims)))
        cand = np.clip(cand + scale * noise, 0.0, 1.0)
        cs = score(cand)
        pool_u, pool_s = np.vstack([top_u, cand]), np.concatenate([top_s, cs])
        order = np.argsort(pool_s)[:C.FIT_TOP]
        top_u, top_s = pool_u[order], pool_s[order]
    a, b, g, p = _map_params(top_u[:1], trend, season)[0]
    return Model(key, trend, season, float(a), float(b), float(g), float(p), init, float(top_s[0]), len(y_train) - start)


def filter_states(model, y):
    """Zustände mit den festen Parametern des Modells durch die ganze Reihe fortschreiben (zu Index t sind nur y[:t] eingegangen)."""
    params = np.array([[model.alpha, model.beta, model.gamma, model.phi]])
    init = (model.init[0], model.init[1], np.array(model.init[2], dtype=float) if model.season != "none" else ())
    return _run(np.asarray(y, dtype=float), model.trend, model.season, params, init, 0, record=True)[1]


def forecast_origins(model, states, origins, h):
    """Prognosen (Ursprünge, h) für die Tage t..t+h-1 aus den Zuständen zum Ursprung t; negative Werte werden auf 0 gesetzt (Zählwerte)."""
    org = np.asarray(origins)
    j = np.arange(h)
    cum_phi = np.cumsum(model.phi ** (j + 1.0))
    base = states.level[org][:, None] + cum_phi[None, :] * states.trend[org][:, None]
    if model.season == "none":
        f = base
    else:
        slot = (org[:, None] + j[None, :]) % M
        st = states.season[org[:, None], slot]
        f = base + st if model.season == "add" else base * st
    return np.maximum(f, 0.0)
