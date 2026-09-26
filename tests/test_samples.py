"""Fenster, Normierung, Zusatzmerkmale und Zeilenmengen des Netzes von Hand nachgerechnet; kein Merkmal darf die Zukunft des Ursprungs enthalten."""

import dataclasses

import numpy as np
import pytest

import ptn_constants as C
import ptn_samples as X
import ptn_scenario as S


@pytest.fixture(scope="module")
def port():
    return S.generate(4, seed=5)


def test_window_target_and_covariates_by_hand(port):
    dep, t = 2, 813
    w, c, tgt, lvl = X.build(port, [dep], [t], "full")
    y = port.y[dep]
    s = y[t - 56:t].mean() + 1.0
    assert lvl[0] == pytest.approx(y[t - 56:t].mean())
    assert np.allclose(w[0], np.log((y[t - 56:t] + 1.0) / s)) and np.allclose(tgt[0], np.log((y[t:t + 14] + 1.0) / s))
    assert w.shape == (1, 56) and c.shape == (1, X.covariate_dim("full")) == (1, 51) and tgt.shape == (1, 14)
    assert c[0, port.dow[t]] == 1.0 and c[0, :7].sum() == 1.0
    assert np.array_equal(c[0, 7:21], port.holiday[t:t + 14]) and np.array_equal(c[0, 21:35], port.after[t:t + 14]) and np.array_equal(c[0, 35:49], port.promo[dep, t:t + 14])
    doy = t % 365
    assert c[0, 49] == pytest.approx(np.sin(2 * np.pi * doy / 365)) and c[0, 50] == pytest.approx(np.cos(2 * np.pi * doy / 365))
    assert port.holiday[t:t + 14].sum() == 2                                                              # der Standardursprung 813 hat zwei Feiertage im Prognosezeitraum (Tag 89 und 92 des Jahres)


def test_window_only_mode_has_just_the_weekday(port):
    _, c, _, _ = X.build(port, [0, 1], [800, 801], "window")
    assert c.shape == (2, 7) and (c.sum(axis=1) == 1).all()


def test_the_size_of_a_depot_drops_out_of_window_and_target(port):
    w1, _, t1, _ = X.build(port, [1], [900])
    big = dataclasses.replace(port, y=np.vstack([port.y[:1], 4.0 * port.y[1:2] + 3.0, port.y[2:]]))
    w2, _, t2, _ = X.build(big, [1], [900])
    assert np.allclose(w1, w2) and np.allclose(t1, t2)                                                    # (4y + 4) / (4 Mittel + 4) = (y + 1) / (Mittel + 1)


def test_orders_round_trip(port):
    dep, org = np.array([0, 1, 3]), np.array([760, 900, 1000])
    _, _, tgt, lvl = X.build(port, dep, org)
    back = X.to_orders(tgt, lvl)
    assert np.allclose(back, np.stack([port.y[d, o:o + 14] for d, o in zip(dep, org)]))
    assert (X.to_orders(np.full((2, 14), -50.0), np.array([3.0, 4.0])) == 0).all()


def test_no_row_uses_values_from_the_origin_on(port):
    dep, org = np.array([1, 2]), np.array([820, 900])
    w1, c1, _, l1 = X.build(port, dep, org)
    y2 = port.y.copy()
    y2[:, 820:] = 999.0
    port2 = dataclasses.replace(port, y=y2)
    w2, c2, _, l2 = X.build(port2, np.array([1]), np.array([820]))
    assert np.array_equal(w1[0], w2[0]) and np.array_equal(c1[0], c2[0]) and l1[0] == l2[0]


def test_row_sets_respect_the_time_boundaries():
    (td, to), (vd, vo) = X.pretrain_rows(np.arange(3), stride=2)
    cut = C.FIRST_TEST - C.VAL_DAYS
    assert to.max() + C.HORIZON <= cut and to.min() == C.WINDOW and set(td) == {0, 1, 2}
    assert vo.min() == cut and vo.max() + C.HORIZON <= C.FIRST_TEST
    hd, ho = X.history_rows([4], 182)
    assert ho.min() - C.WINDOW == C.FIRST_TEST - 182 and ho.max() + C.HORIZON == C.FIRST_TEST and len(ho) == 182 - C.WINDOW - C.HORIZON + 1 and (hd == 4).all()
    assert len(X.history_rows([0], 98)[1]) == 29
    d, o, org = X.test_rows([0, 1])
    assert len(org) == C.N_DAYS - C.HORIZON - C.FIRST_TEST + 1 and org[0] == C.FIRST_TEST and org[-1] + C.HORIZON == C.N_DAYS and len(d) == 2 * len(org)
