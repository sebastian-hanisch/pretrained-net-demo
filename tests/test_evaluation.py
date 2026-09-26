"""Vortraining, Zero-Shot, Feintuning, Selbstprüfung und Kennzahlen: Aufbau, Zeitgrenzen (nichts aus der Zukunft), Kennzahlen von Hand."""

import dataclasses

import numpy as np
import pytest

import ptn_baselines as B
import ptn_constants as C
import ptn_evaluation as E
import ptn_net as N
import ptn_samples as X
import ptn_scenario as S
import ptn_visualization as V


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(E.Settings(n_new=6))


def test_analysis_shapes_and_keys(analysis):
    a = analysis
    T = C.N_DAYS - C.HORIZON + 1 - C.FIRST_TEST
    assert set(a.forecasts) == set(E.ALL_METHODS) | {"oracle"} and all(v.shape == (6, T, C.HORIZON) for v in a.forecasts.values()) and a.actual.shape == (6, T, C.HORIZON)
    assert a.origins[0] == C.FIRST_TEST and a.origins[-1] + C.HORIZON == C.N_DAYS and len(a.tuned_params) == 6 and a.picked.shape == (6,) and a.scale.shape == (6,)
    assert len(a.curve["train"]) == len(a.curve["val"]) == a.settings.epochs and set(a.snapshots) >= {0, a.settings.epochs}


def test_pool_and_new_depots_are_different_draws_and_only_the_new_ones_shift(analysis):
    a = analysis
    assert a.pool.n == 40 and a.new.n == 6 and not np.array_equal(a.pool.y[:6], a.new.y)
    assert np.array_equal(S.generate(3, seed=1).y, S.generate(3, seed=1, shift=0.0).y)

    def prof(p):
        return np.array([p.mu[:, p.dow == d].mean() for d in range(7)])

    base, shifted = prof(S.generate(20, seed=2)), prof(S.generate(20, seed=2, shift=1.0))
    assert base[5] < 0.7 * base[:5].mean() and shifted[5] > 1.4 * shifted[:5].mean() and shifted[6] > 1.4 * shifted[:5].mean()


def test_summary_by_hand(analysis):
    a = analysis
    m = "wm"
    dep = 2
    mae = np.abs(a.forecasts[m][dep] - a.actual[dep]).mean()
    scale = np.abs(a.new.y[dep, 7:C.FIRST_TEST] - a.new.y[dep, :C.FIRST_TEST - 7]).mean()
    assert a.depot_mase[m][dep] == pytest.approx(mae / scale) and a.summary[m]["mase"] == pytest.approx(np.mean(a.depot_mase[m])) and a.scale[dep] == pytest.approx(scale)
    assert a.forecasts["wm"][dep, 800 - C.FIRST_TEST, 3] == pytest.approx(np.mean([a.new.y[dep, 800 - 7 * (k + 1) + 3] for k in range(4)]))
    assert a.horizon_mae[m].shape == (C.HORIZON,) and a.horizon_mae[m].mean() == pytest.approx(np.mean(np.abs(a.forecasts[m] - a.actual).mean(axis=1) / a.scale[:, None]))
    assert a.summary["oracle"]["mase"] < min(v["mase"] for k, v in a.summary.items() if k != "oracle")


def test_mix_auto_and_zero_shot_definitions(analysis):
    a = analysis
    assert np.allclose(a.forecasts["mix"], 0.5 * (a.forecasts["zero"] + a.forecasts["gbm"]))
    for i in range(a.new.n):
        assert np.array_equal(a.forecasts["auto"][i], a.forecasts["zero" if a.picked[i] else "wm"][i])
    ids = np.arange(a.new.n)
    assert np.allclose(E.net_forecast(a.params, a.new, ids, a.settings.covariates), a.forecasts["zero"])


def test_zero_shot_by_epoch_starts_at_random_weights_and_ends_at_the_summary(analysis):
    a = analysis
    by = V.zero_shot_by_epoch(a)
    assert by[0] > 1.3 and by[a.settings.epochs] == pytest.approx(a.summary["zero"]["mase"]) and by[4] < by[0] - 0.5


def test_pretraining_sees_no_day_of_the_test_year():
    s = E.Settings(n_pool=6, epochs=2)
    pool = E._pool(s.pool_key)
    y2 = pool.y.copy()
    y2[:, C.FIRST_TEST - C.VAL_DAYS + 5:] = 777.0                                          # alles ab Tag 690 verändern: nur die Prüfzeilen dürfen davon abhängen, nicht die Trainingsgewichte
    pool2 = dataclasses.replace(pool, y=y2)
    p1, h1, _ = E.pretrain(pool, s)
    p2, h2, _ = E.pretrain(pool2, s)
    assert all(np.array_equal(p1[k], p2[k]) for k in p1) and h1["train"] == h2["train"] and h1["val"] != h2["val"]


def test_finetune_uses_only_the_history_and_zero_steps_change_nothing(analysis):
    a = analysis
    s = a.settings
    y2 = a.new.y.copy()
    y2[:, C.FIRST_TEST:] = 555.0
    new2 = dataclasses.replace(a.new, y=y2)
    t1 = E.finetune(a.params, a.new, 1, 182, s, steps=20, seed=4)
    t2 = E.finetune(a.params, new2, 1, 182, s, steps=20, seed=4)
    assert all(np.array_equal(t1[k], t2[k]) for k in t1) and not np.array_equal(t1["0.W1"], a.params["0.W1"])
    z = E.finetune(a.params, a.new, 1, 182, s, steps=0)
    assert all(np.array_equal(z[k], a.params[k]) for k in z) and z is not a.params
    sc = E.scratch(a.new, 1, 182, s)
    assert not np.array_equal(sc["0.W1"], a.params["0.W1"]) and N.shape_of(sc) == N.shape_of(a.params)


def test_forecasts_at_an_origin_do_not_depend_on_later_days(analysis):
    a = analysis
    ids = np.arange(a.new.n)
    cut = 900
    y2 = a.new.y.copy()
    y2[:, cut:] = 321.0
    new2 = dataclasses.replace(a.new, y=y2)
    i = cut - C.FIRST_TEST
    ens = E._gbm(a.settings.pool_key)
    for f1, f2 in ((E.net_forecast(a.params, a.new, ids, "full"), E.net_forecast(a.params, new2, ids, "full")), (E.gbm_forecast(ens, a.new, ids), E.gbm_forecast(ens, new2, ids))):
        assert np.allclose(f1[:, :i], f2[:, :i]) and not np.allclose(f1[:, i + 20:], f2[:, i + 20:])


def test_self_check_compares_on_the_history_windows_by_hand(analysis):
    a = analysis
    dep, hist = 3, 182
    z, w = E.history_check(a.params, a.new, dep, hist, a.settings.covariates)
    d, o = X.history_rows([dep], hist)
    wm = B.snaive_k(a.new.y[dep], o, C.HORIZON)
    act = np.stack([a.new.y[dep, t:t + C.HORIZON] for t in o])
    assert w == pytest.approx(np.abs(wm - act).mean())
    wn, cn, _, lvl = X.build(a.new, d, o, a.settings.covariates)
    assert z == pytest.approx(np.abs(X.to_orders(N.forward(a.params, wn, cn), lvl) - act).mean())


def test_self_check_flags_a_weekend_depot_and_trusts_a_matching_one():
    a1 = E.analyse(E.Settings(n_new=6, shift=1.0))
    a0 = E.analyse(E.Settings(n_new=6))
    assert a1.picked.sum() == 0 and a0.picked.sum() >= 5
    z, w = E.history_check(a1.params, a1.new, 0, 182, "full")
    assert z > 1.3 * w


def test_pretrained_models_are_computed_once_per_setting():
    s = E.Settings(n_pool=6, epochs=2)
    assert E._pretrained(s.net_key) is E._pretrained(s.net_key) and E._gbm(s.pool_key) is E._gbm(s.pool_key)
    assert E.analyse(E.Settings(n_new=5)) is E.analyse(E.Settings(n_new=5))
