"""Auswertung: ein Netz wird auf vielen Depots (dem Pool) vortrainiert und auf NEUEN Depots mit kurzer Historie geprüft - ohne eigenes Lernen (Zero-Shot), mit Feintuning auf der eigenen Historie und im Vergleich mit dem Netz,
das nur aus der eigenen Historie lernt, mit dem vortrainierten Boosting-Modell (Stück 6) und den lokalen Verfahren der Vorgänger.

Protokoll wie in den Vorgängern: Rolling Origin über das Testjahr (Ursprung t: bekannt sind die Tage 0..t-1; prognostiziert werden die Tage t..t+H-1, H = 14). Das Vortraining sieht nur Zieltage vor dem Testjahr; ein neues Depot
bringt die letzten `hist` Tage vor dem Testjahr mit, die Gewichte bleiben im Testjahr fest. Kennzahl: MASE je Depot (Nenner: saisonal naiver Fehler auf den Tagen vor dem Testjahr), gemittelt über die neuen Depots."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import ptn_baselines as B
import ptn_constants as C
import ptn_features as F
import ptn_gbm as G
import ptn_net as N
import ptn_samples as X
import ptn_scenario as S

SCRATCH_STEPS = 400
BATCH_PRETRAIN = 256
BATCH_FINETUNE = 32
GBM_ROUNDS = 80
SNAP_EPOCHS = (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40)
METHODS = ("wm", "hw", "scratch", "zero", "tuned", "gbm", "auto")
ALL_METHODS = METHODS + ("mix",)
METHOD_NAMES = {"wm": "Wochenmittel (Stück 1)", "hw": "Holt-Winters (Stück 2)", "scratch": "Netz nur aus eigener Historie", "zero": "Vortrainiertes Netz, Zero-Shot", "tuned": "Vortrainiertes Netz + Feintuning",
                "gbm": "Vortrainiertes Boosting (Stück 6), Zero-Shot", "auto": "Zero-Shot mit Selbstprüfung auf der Historie", "mix": "Mittel aus Netz (Zero-Shot) und Boosting (Stück 9)"}
METHOD_SHORT = {"wm": "Wochenmittel", "hw": "Holt-Winters", "scratch": "Netz allein", "zero": "Netz Zero-Shot", "tuned": "Netz + Feintuning", "gbm": "Boosting Zero-Shot", "auto": "Zero-Shot mit Selbstprüfung", "mix": "Mittel Netz + Boosting"}


@dataclass(frozen=True)
class Settings:
    n_pool: int = C.DEFAULT_POOL
    n_new: int = C.DEFAULT_NEW
    hist: int = C.DEFAULT_HIST
    noise: float = C.DEFAULT_NOISE
    shift: float = C.DEFAULT_SHIFT
    epochs: int = C.DEFAULT_EPOCHS
    blocks: int = C.DEFAULT_BLOCKS
    width: int = C.DEFAULT_WIDTH
    covariates: str = "full"
    ft_steps: int = C.DEFAULT_FT
    seed: int = 3

    @property
    def pool_key(self):
        return (self.n_pool, self.noise, self.seed)

    @property
    def new_key(self):
        return (self.n_new, self.noise, self.shift, self.seed)

    @property
    def net_key(self):
        return self.pool_key + (self.epochs, self.blocks, self.width, self.covariates)


@lru_cache(maxsize=16)
def _pool(key):
    n, noise, seed = key
    return S.generate(n, noise_mean=noise, seed=seed)


@lru_cache(maxsize=16)
def _new(key):
    n, noise, shift, seed = key
    return S.generate(n, noise_mean=noise, seed=seed + 1000, shift=shift)


def _shape(s):
    return N.Shape(C.WINDOW, X.covariate_dim(s.covariates), C.HORIZON, s.blocks, s.width)


def pretrain(pool, s, snapshots=SNAP_EPOCHS):
    """Vortraining auf den Depots des Pools: Rückgabe (Parameter, Verlauf {"train", "val"}, Momentaufnahmen)."""
    (td, to), (vd, vo) = X.pretrain_rows(np.arange(pool.n))
    w, c, y, _ = X.build(pool, td, to, s.covariates)
    val = X.build(pool, vd, vo, s.covariates)[:3]
    p0 = N.init(_shape(s), seed=s.seed)
    steps_per_epoch = int(np.ceil(len(w) / BATCH_PRETRAIN))
    snaps = tuple(e for e in snapshots if e <= s.epochs) + ((s.epochs,) if s.epochs not in snapshots else ())
    return N.train(p0, w, c, y, steps=s.epochs * steps_per_epoch, batch=BATCH_PRETRAIN, lr=C.LR_PRETRAIN, seed=s.seed, steps_per_epoch=steps_per_epoch, val=val, snapshots=snaps)


@lru_cache(maxsize=16)
def _pretrained(net_key):
    n, noise, seed, epochs, blocks, width, covariates = net_key
    s = Settings(n_pool=n, noise=noise, seed=seed, epochs=epochs, blocks=blocks, width=width, covariates=covariates)
    return pretrain(_pool(s.pool_key), s)


def train_gbm(pool, seed=0):
    """Das globale Boosting-Modell aus Stück 6 auf den Pool-Depots (Ziel: Verhältnis zum 28-Tage-Niveau im Log, Lag-, Kalender- und Aktionsmerkmale, direkter Horizont 1..H)."""
    d, o, h = F.training_rows(pool, np.arange(pool.n), C.HORIZON, stride=4, per_origin=2, last_target=C.FIRST_TEST, seed=seed)
    Xf, y, _ = F.build(pool, d, o, h)
    ens = G.fit(Xf, y, num_leaves=15, n_rounds=GBM_ROUNDS, learning_rate=0.1, min_child_samples=20, seed=seed)
    return ens


@lru_cache(maxsize=16)
def _gbm(pool_key):
    return train_gbm(_pool(pool_key), seed=pool_key[2])


def gbm_forecast(ens, new, depots):
    d, o, org = X.test_rows(depots)
    h = np.tile(np.arange(1, C.HORIZON + 1), len(d))
    dd, oo = np.repeat(d, C.HORIZON), np.repeat(o, C.HORIZON)
    Xf, _, lvl = F.build(new, dd, oo, h)
    pred = F.to_orders(G.predict(ens, Xf), lvl)
    return pred.reshape(len(depots), len(org), C.HORIZON)


def net_forecast(p, new, depots, covariates):
    d, o, org = X.test_rows(depots)
    w, c, _, lvl = X.build(new, d, o, covariates)
    return X.to_orders(N.forward(p, w, c), lvl).reshape(len(depots), len(org), C.HORIZON)


def finetune(p, new, depot, hist, s, steps=None, lr=None, seed=0):
    """Feintuning auf der eigenen Historie eines Depots (alle Gewichte, kleine Lernrate, wenige Schritte)."""
    steps = s.ft_steps if steps is None else steps
    lr = C.LR_FINETUNE if lr is None else lr
    if steps == 0:
        return N.copy(p)
    d, o = X.history_rows([depot], hist)
    w, c, y, _ = X.build(new, d, o, s.covariates)
    return N.train(p, w, c, y, steps=steps, batch=BATCH_FINETUNE, lr=lr, seed=seed)[0]


def scratch(new, depot, hist, s, seed=0):
    """Dasselbe Netz mit Zufallsgewichten, nur auf der eigenen Historie trainiert."""
    return finetune(N.init(_shape(s), seed=seed), new, depot, hist, s, steps=SCRATCH_STEPS, lr=C.LR_PRETRAIN, seed=seed)


@dataclass
class Analysis:
    settings: Settings
    pool: S.Portfolio
    new: S.Portfolio
    params: dict
    curve: dict                 # {"train", "val"}: MAE je Epoche des Vortrainings (normierte Log-Skala)
    snapshots: dict
    origins: np.ndarray
    actual: np.ndarray          # (n_new, n_org, H)
    forecasts: dict             # Verfahren -> (n_new, n_org, H)
    tuned_params: list          # je neuem Depot die feingetunten Parameter
    picked: np.ndarray          # (n_new,) True, wenn die Selbstprüfung das Zero-Shot-Netz gewählt hat (sonst Wochenmittel)
    scale: np.ndarray
    summary: dict               # Verfahren -> {"mase", "mae"}
    depot_mase: dict            # Verfahren -> (n_new,)
    horizon_mae: dict           # Verfahren -> (H,) Fehler je Horizont (durch die Skala geteilt, Mittel über die Depots)

    @property
    def best(self):
        return min((m for m in self.summary if m != "oracle"), key=lambda m: self.summary[m]["mase"])


def _scales(port):
    return np.array([B.mase_scale(port.y[i]) for i in range(port.n)])


def summarize(forecasts, actual, scale):
    out, per = {}, {}
    for m, f in forecasts.items():
        mae_dep = np.abs(f - actual).mean(axis=(1, 2))
        per[m] = mae_dep / scale
        out[m] = {"mase": float(per[m].mean()), "mae": float(mae_dep.mean())}
    hor = {m: (np.abs(f - actual).mean(axis=1) / scale[:, None]).mean(axis=0) for m, f in forecasts.items()}
    return out, per, hor


def history_check(p, new, depot, hist, covariates):
    """Selbstprüfung: mittlerer absoluter Fehler des Zero-Shot-Netzes und des Wochenmittels auf den Prognosefenstern der eigenen Historie des Depots (die letzten hist Tage vor dem Testjahr)."""
    d, o = X.history_rows([depot], hist)
    w, c, _, lvl = X.build(new, d, o, covariates)
    act = new.y[depot][o[:, None] + np.arange(C.HORIZON)[None, :]]
    return float(np.abs(X.to_orders(N.forward(p, w, c), lvl) - act).mean()), float(np.abs(B.snaive_k(new.y[depot], o, C.HORIZON) - act).mean())


def forecasts_for(s, methods=METHODS, hist=None, pool=None, new=None, params=None):
    """Prognosen der gewünschten Verfahren für alle neuen Depots; Rückgabe (Prognosen, Ursprünge, feingetunte Parameter, Auswahl der Selbstprüfung: True = Zero-Shot)."""
    hist = s.hist if hist is None else hist
    pool = _pool(s.pool_key) if pool is None else pool
    new = _new(s.new_key) if new is None else new
    depots = np.arange(new.n)
    org = np.arange(C.FIRST_TEST, C.N_DAYS - C.HORIZON + 1)
    out, tuned, picked = {}, [], None
    methods = tuple(methods)
    if "auto" in methods:
        methods = tuple(dict.fromkeys(methods + ("wm", "zero")))
    if "wm" in methods:
        out["wm"] = np.stack([B.snaive_k(new.y[i], org, C.HORIZON) for i in depots])
    if "hw" in methods:
        out["hw"] = np.stack([B.hw_forecast(new.y[i], org, C.HORIZON, hist) for i in depots])
    if params is None and ({"zero", "tuned"} & set(methods)):
        params = _pretrained(s.net_key)[0]
    if "zero" in methods:
        out["zero"] = net_forecast(params, new, depots, s.covariates)
    if "tuned" in methods:
        tuned = [finetune(params, new, i, hist, s, seed=s.seed + i) for i in depots]
        out["tuned"] = np.stack([net_forecast(tuned[i], new, [i], s.covariates)[0] for i in depots])
    if "scratch" in methods:
        out["scratch"] = np.stack([net_forecast(scratch(new, i, hist, s, seed=s.seed + i), new, [i], s.covariates)[0] for i in depots])
    if "gbm" in methods:
        out["gbm"] = gbm_forecast(_gbm(s.pool_key), new, depots)
    if "auto" in methods:
        checks = [history_check(params, new, i, hist, s.covariates) for i in depots]
        picked = np.array([z <= w for z, w in checks])
        out["auto"] = np.where(picked[:, None, None], out["zero"], out["wm"])
    return out, org, tuned, picked


@lru_cache(maxsize=8)
def analyse(s):
    pool, new = _pool(s.pool_key), _new(s.new_key)
    params, curve, snaps = _pretrained(s.net_key)
    fc, org, tuned, picked = forecasts_for(s, params=params)
    fc["mix"] = 0.5 * (fc["zero"] + fc["gbm"])
    fc["oracle"] = np.stack([np.stack([new.mu[i, t:t + C.HORIZON] for t in org]) for i in range(new.n)])
    actual = np.stack([np.stack([new.y[i, t:t + C.HORIZON] for t in org]) for i in range(new.n)])
    scale = _scales(new)
    summary, per, hor = summarize(fc, actual, scale)
    return Analysis(s, pool, new, params, curve, snaps, org, actual, fc, tuned, picked, scale, summary, per, hor)


def _actual(new, org):
    return np.stack([np.stack([new.y[i, t:t + C.HORIZON] for t in org]) for i in range(new.n)])


def evaluate(s, methods=METHODS, hist=None, params=None):
    """MASE (Mittel über die neuen Depots) der gewünschten Verfahren für die Einstellungen s (und die Historie hist)."""
    new = _new(s.new_key)
    fc, org, _, _ = forecasts_for(s, methods, hist, params=params)
    scale = _scales(new)
    act = _actual(new, org)
    return {m: float(np.mean(np.abs(f - act).mean(axis=(1, 2)) / scale)) for m, f in fc.items() if m in methods}


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _replace(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


def _seeds(seeds):
    return C.EXP_SEEDS if seeds is None else seeds


# --- Experiment 1: Historie des neuen Depots ------------------------------------------------------------------------------------------------------

def history_experiment(histories=None, seeds=None, base=None):
    """Neue Depots mit hist Tagen Historie: MASE aller Verfahren je Historienlänge (Mittel über die Seeds). Wochenmittel, Zero-Shot und Boosting hängen nicht von der Historie ab (sie brauchen nur ihr Fenster)."""
    histories = C.HISTORIES if histories is None else histories
    base = Settings(n_new=C.EXP_NEW) if base is None else base
    fixed = {m: [] for m in ("wm", "zero", "gbm")}
    var = {(m, h): [] for m in ("hw", "scratch", "tuned", "auto") for h in histories}
    for sd in _seeds(seeds):
        s = _replace(base, seed=sd)
        for m, v in evaluate(s, ("wm", "zero", "gbm")).items():
            fixed[m].append(v)
        for h in histories:
            for m, v in evaluate(s, ("hw", "scratch", "tuned", "auto"), hist=h).items():
                var[(m, h)].append(v)
    return {"n_seeds": len(_seeds(seeds)), "histories": tuple(histories), "fixed": {m: _mean_se(v) for m, v in fixed.items()}, "rows": {k: _mean_se(v) for k, v in var.items()}}


# --- Experiment 2: Zahl der Feintuning-Schritte -----------------------------------------------------------------------------------------------------

def finetune_experiment(steps=None, conditions=None, seeds=None, base=None):
    """MASE des feingetunten Netzes über die Zahl der Schritte (0 = Zero-Shot) für vier Fälle (Historie in Tagen, Abweichung des Wochenmusters); dazu Wochenmittel und Holt-Winters je Fall (Mittel über die Seeds)."""
    steps = C.FT_LEVELS if steps is None else steps
    conditions = C.FT_CONDITIONS if conditions is None else conditions
    base = Settings(n_new=C.EXP_NEW) if base is None else base
    rows = {(h, sh, n): [] for h, sh in conditions for n in steps}
    refs = {(m, h, sh): [] for m in ("wm", "hw") for h, sh in conditions}
    for sd in _seeds(seeds):
        for h, sh in conditions:
            s = _replace(base, seed=sd, shift=sh)
            for n in steps:
                rows[(h, sh, n)].append(evaluate(_replace(s, ft_steps=n), ("tuned",), hist=h)["tuned"])
            for m, v in evaluate(s, ("wm", "hw"), hist=h).items():
                refs[(m, h, sh)].append(v)
    return {"n_seeds": len(_seeds(seeds)), "steps": tuple(steps), "conditions": tuple(conditions), "rows": {k: _mean_se(v) for k, v in rows.items()}, "refs": {k: _mean_se(v)[0] for k, v in refs.items()}}


# --- Experiment 3: Größe des Pools ----------------------------------------------------------------------------------------------------------------

def pool_experiment(sizes=None, seeds=None, base=None):
    """Das Netz (und das Boosting-Modell) lernen aus n Pool-Depots und werden auf neuen Depots geprüft (Zero-Shot und mit Feintuning auf der Historie aus base.hist); Holt-Winters und Wochenmittel als Referenz."""
    sizes = C.POOL_SIZES if sizes is None else sizes
    base = Settings(n_new=C.EXP_NEW) if base is None else base
    rows = {(m, n): [] for m in ("zero", "tuned", "gbm") for n in sizes}
    refs = {m: [] for m in ("wm", "hw")}
    for sd in _seeds(seeds):
        for n in sizes:
            for m, v in evaluate(_replace(base, n_pool=n, seed=sd), ("zero", "tuned", "gbm")).items():
                rows[(m, n)].append(v)
        for m, v in evaluate(_replace(base, seed=sd), ("wm", "hw")).items():
            refs[m].append(v)
    return {"n_seeds": len(_seeds(seeds)), "sizes": tuple(sizes), "rows": {k: _mean_se(v) for k, v in rows.items()}, "refs": {m: _mean_se(v)[0] for m, v in refs.items()}}


# --- Experiment 4: Abweichendes Wochenmuster ------------------------------------------------------------------------------------------------------

def shift_experiment(levels=None, hists=(182, 730), seeds=None, base=None):
    """Das neue Depot hat ein anderes Wochenmuster als der Pool (Wochenend-Depot): MASE aller Verfahren je Abweichung und Historie, das Feintuning mit SHIFT_FT Schritten (Mittel über die Seeds)."""
    levels = C.SHIFT_LEVELS if levels is None else levels
    base = Settings(n_new=C.EXP_NEW, ft_steps=C.SHIFT_FT) if base is None else base
    rows = {(m, sh, h): [] for m in METHODS for sh in levels for h in hists}
    for sd in _seeds(seeds):
        for sh in levels:
            s = _replace(base, shift=sh, seed=sd)
            zg = evaluate(s, ("wm", "zero", "gbm"))
            for h in hists:
                for m, v in {**zg, **evaluate(s, ("hw", "scratch", "tuned", "auto"), hist=h)}.items():
                    rows[(m, sh, h)].append(v)
    return {"n_seeds": len(_seeds(seeds)), "levels": tuple(levels), "hists": tuple(hists), "rows": {k: _mean_se(v) for k, v in rows.items()}}
