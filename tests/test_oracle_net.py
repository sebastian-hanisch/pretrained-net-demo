"""Orakel-Test: das Netz gegen eine eigene Vorwärtsrechnung je Zeile (komplexe Arithmetik) und Complex-Step-Ableitungen (exakt, keine Differenzenfehler), Adam gegen eine Referenzschleife, Zeilen und
Normierung gegen Skalarrechnung sowie die Zeitgrenzen der Zeilenmengen (Vortraining, Historie, Test: keine Überlappung von Ziel- und Prüftagen)."""

import math

import numpy as np
import pytest

import ptn_constants as C
import ptn_net as N
import ptn_samples as X
import ptn_scenario as S


def _relu(z):
    return z * (z.real > 0)


def _forward_row(P, shape, w, c):
    r = w.astype(complex)
    yh = np.zeros(shape.horizon, dtype=complex)
    for k in range(shape.blocks):
        u = np.concatenate([r, c.astype(complex)])
        h1 = _relu(P[f"{k}.W1"].T @ u + P[f"{k}.b1"])
        h2 = _relu(P[f"{k}.W2"].T @ h1 + P[f"{k}.b2"])
        yh = yh + P[f"{k}.Wf"].T @ h2 + P[f"{k}.bf"]
        r = r - (P[f"{k}.Wb"].T @ h2 + P[f"{k}.bb"])
    return yh


def _loss(P, shape, W, Cv, Y, kind):
    tot = 0.0
    for i in range(len(W)):
        d = _forward_row(P, shape, W[i], Cv[i]) - Y[i]
        tot = tot + (d * np.sign(d.real) if kind == "l1" else d * d).sum()
    return tot / Y.size


def test_forward_and_complex_step_gradients_match():
    rng = np.random.default_rng(18)
    for it in range(4):
        shape = N.Shape(window=int(rng.integers(3, 7)), covariates=int(rng.integers(0, 3)), horizon=int(rng.integers(1, 4)), blocks=int(rng.integers(1, 4)), width=int(rng.integers(2, 5)))
        p = N.init(shape, seed=it)
        for k in p:
            p[k] = p[k] + 0.3 * rng.normal(size=p[k].shape)
        n = int(rng.integers(2, 5))
        W, Cv, Y = rng.normal(size=(n, shape.window)), rng.normal(size=(n, shape.covariates)), rng.normal(size=(n, shape.horizon))
        assert np.allclose(N.forward(p, W, Cv), [_forward_row(p, shape, W[i], Cv[i]).real for i in range(n)], atol=1e-12)
        for kind in ("l1", "l2"):
            val, g = N.loss_and_grads(p, W, Cv, Y, loss=kind)
            assert val == pytest.approx(_loss(p, shape, W, Cv, Y, kind).real, abs=1e-12)
            for k in p:
                for j in range(p[k].size):
                    P2 = {kk: v.astype(complex) for kk, v in p.items()}
                    P2[k].ravel()[j] += 1e-30j
                    assert g[k].ravel()[j] == pytest.approx(_loss(P2, shape, W, Cv, Y, kind).imag / 1e-30, rel=1e-8, abs=1e-12), (kind, k, j)


def test_adam_training_matches_a_reference_loop():
    rng = np.random.default_rng(3)
    for it in range(3):
        p = N.init(N.Shape(10, 3, 4, 2, 8), seed=it)
        n = int(rng.integers(20, 90))
        w, c, y = rng.normal(size=(n, 10)), rng.normal(size=(n, 3)), rng.normal(size=(n, 4))
        batch, steps = int(rng.choice([8, 16, 500])), int(rng.integers(5, 40))
        got = N.train(p, w, c, y, steps=steps, batch=batch, lr=0.01, seed=it)[0]
        q = {k: v.copy() for k, v in p.items()}
        rg = np.random.default_rng(it)
        batch = min(batch, n)
        order, pos = list(rg.permutation(n)), 0
        m = {k: np.zeros_like(v) for k, v in q.items()}
        v2 = {k: np.zeros_like(v) for k, v in q.items()}
        for step in range(1, steps + 1):
            if pos + batch > n:
                order, pos = list(rg.permutation(n)), 0
            idx = order[pos:pos + batch]
            pos += batch
            g = N.loss_and_grads(q, w[idx], c[idx], y[idx])[1]
            for k in q:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                q[k] = q[k] - 0.01 * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)
        assert all(np.allclose(got[k], q[k], atol=1e-10) for k in q)


def test_rows_and_normalisation_match_a_scalar_computation():
    port = S.generate(4, seed=2)
    rng = np.random.default_rng(1)
    for it in range(80):
        dep, org, cov = int(rng.integers(0, 4)), int(rng.integers(56, C.N_DAYS - 13)), "full" if it % 2 else "window"
        w, cc, tg, lvl = X.build(port, np.array([dep]), np.array([org]), cov)
        y = port.y[dep]
        lv = np.mean(y[org - 56:org])
        cref = [1.0 if org % 7 == q else 0.0 for q in range(7)]
        if cov == "full":
            cref += [port.holiday[d] for d in range(org, org + 14)] + [port.after[d] for d in range(org, org + 14)] + [port.promo[dep, d] for d in range(org, org + 14)]
            cref += [math.sin(2 * math.pi * (org % 365) / 365), math.cos(2 * math.pi * (org % 365) / 365)]
        assert np.allclose(w[0], [math.log((y[d] + 1) / (lv + 1)) for d in range(org - 56, org)]) and np.allclose(tg[0], [math.log((y[d] + 1) / (lv + 1)) for d in range(org, org + 14)])
        assert np.allclose(cc[0], cref) and lvl[0] == pytest.approx(lv)


def test_row_sets_have_no_target_day_in_the_test_year_and_the_validation_stays_behind_the_training():
    (td, to), (vd, vo) = X.pretrain_rows(np.arange(3))
    assert to.max() + C.HORIZON - 1 < C.FIRST_TEST - C.VAL_DAYS <= vo.min() and vo.max() + C.HORIZON - 1 < C.FIRST_TEST and to.min() == C.WINDOW
    for hist in (98, 182, 730):
        d, o = X.history_rows([0], hist)
        assert o.min() - C.WINDOW == C.FIRST_TEST - hist and o.max() + C.HORIZON - 1 == C.FIRST_TEST - 1 and len(o) == hist - C.WINDOW - C.HORIZON + 1
    d, o, org = X.test_rows([0, 1])
    assert org.min() == C.FIRST_TEST and org.max() + C.HORIZON - 1 == C.N_DAYS - 1
