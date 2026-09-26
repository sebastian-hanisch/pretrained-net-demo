"""Jede Zahl aus README.md, den Preset-Hilfen und den Befunden der App wird hier nachgerechnet. Mehr-Seed-Zahlen (Seeds 0 bis 2, je 10 neue Depots) mit Bändern, die plattformbedingte Rundungsunterschiede des Trainings vertragen;
Einzelzahlen (Seed 3) nur mit großzügigen Bändern und über Strukturgrenzen."""

import numpy as np
import pytest

import ptn_constants as C
import ptn_evaluation as E
import ptn_net as N
import ptn_presets as P


def _settings(p):
    return E.Settings(p["n_pool"], p["n_new"], p["hist"], p["noise"], p["shift"], p["epochs"], p["blocks"], p["width"], p["covariates"], p["ft"], p["seed"])


@pytest.fixture(scope="module")
def hist():
    return E.history_experiment()


@pytest.fixture(scope="module")
def ft():
    return E.finetune_experiment()


@pytest.fixture(scope="module")
def pool():
    return E.pool_experiment()


@pytest.fixture(scope="module")
def shift():
    return E.shift_experiment()


def _zero(**kw):
    base = E.Settings(n_new=C.EXP_NEW)
    return float(np.mean([E.evaluate(E._replace(base, seed=sd, **kw), ("zero",))["zero"] for sd in C.EXP_SEEDS]))


# --- Netz ------------------------------------------------------------------------------------------------------------------------------------------

def test_standard_net_size():
    s = E.Settings()
    assert N.n_params(N.init(E._shape(s))) == 13644 and s.blocks == 2 and s.width == 32 and s.epochs == 12


def test_capacity_is_not_the_bottleneck_and_the_calendar_helps():
    base = _zero()
    assert 0.79 < base < 0.86
    for kw in (dict(blocks=1), dict(blocks=4), dict(width=16), dict(width=128), dict(epochs=4), dict(epochs=40)):
        assert abs(_zero(**kw) - base) < 0.03
    assert _zero(blocks=4, width=128, epochs=40) > base + 0.06                                          # 0,939: das große Netz überanpasst
    assert _zero(covariates="window") > base + 0.02                                                     # 0,874: ohne Feiertage und Aktionsplan schlechter


# --- Experiment 1: Historie -------------------------------------------------------------------------------------------------------------------------

def test_history_zero_shot_beats_the_local_methods_at_every_length(hist):
    R, F = hist["rows"], hist["fixed"]
    z, wm, gbm = F["zero"][0], F["wm"][0], F["gbm"][0]
    assert 0.79 < z < 0.87 and z < 0.95 * wm and 0.90 < wm < 0.98
    for h in hist["histories"]:
        assert z < R[("hw", h)][0] - 0.02
    assert R[("hw", 98)][0] > 1.05 and R[("scratch", 98)][0] > 1.05 and R[("scratch", 730)][0] < wm - 0.03 and R[("scratch", 730)][0] > z
    assert R[("hw", 98)][0] > R[("hw", 730)][0] + 0.15 and R[("scratch", 98)][0] > R[("scratch", 730)][0] + 0.2


def test_history_finetuning_hurts_short_and_helps_only_long_and_the_boosting_is_at_least_as_good(hist):
    R, F = hist["rows"], hist["fixed"]
    z = F["zero"][0]
    assert R[("tuned", 98)][0] > z + 0.03 and R[("tuned", 730)][0] < z + 0.01 and R[("tuned", 98)][0] > R[("tuned", 730)][0] + 0.05
    assert F["gbm"][0] < z + 0.02                                                                       # das Boosting ist gleichauf oder besser (0,800 gegen 0,827)


def test_history_self_check_costs_almost_nothing(hist):
    R, F = hist["rows"], hist["fixed"]
    assert abs(R[("auto", 98)][0] - F["zero"][0]) < 0.05
    for h in (182, 365, 730):
        assert abs(R[("auto", h)][0] - F["zero"][0]) < 0.015


# --- Experiment 2: Feintuning -----------------------------------------------------------------------------------------------------------------------

def test_finetuning_on_a_matching_depot(ft):
    R = ft["rows"]
    z = R[(182, 0.0, 0)][0]
    assert all(R[(182, 0.0, n)][0] > z - 0.005 for n in (50, 100, 300, 1000)) and R[(182, 0.0, 1000)][0] > z + 0.02
    assert R[(730, 0.0, 1000)][0] < z + 0.005 and R[(730, 0.0, 1000)][0] < R[(730, 0.0, 50)][0] + 0.02


def test_finetuning_repairs_a_weekend_depot_only_with_many_steps(ft):
    R, ref = ft["rows"], ft["refs"]
    wm = ref[("wm", 182, 1.0)]
    assert R[(182, 1.0, 0)][0] > wm + 0.4 and R[(182, 1.0, 0)][0] > 1.3
    assert R[(730, 1.0, 100)][0] > wm + 0.05 and R[(730, 1.0, 1000)][0] < wm - 0.04 and R[(730, 1.0, 1000)][0] < R[(730, 1.0, 100)][0] - 0.15
    assert R[(182, 1.0, 300)][0] < R[(182, 1.0, 0)][0] - 0.4 and abs(R[(182, 1.0, 300)][0] - wm) < 0.06


# --- Experiment 3: Pool -----------------------------------------------------------------------------------------------------------------------------

def test_a_pool_of_three_is_worse_than_no_pretraining(pool):
    R, wm = pool["rows"], pool["refs"]["wm"]
    assert R[("zero", 3)][0] > wm + 0.02 and R[("gbm", 3)][0] > wm + 0.02 and 0.90 < wm < 0.98


def test_a_pool_of_ten_already_carries_and_more_depots_help_with_diminishing_returns(pool):
    R, wm = pool["rows"], pool["refs"]["wm"]
    assert R[("zero", 10)][0] < wm - 0.03 and R[("gbm", 10)][0] < wm - 0.05
    assert R[("zero", 100)][0] < R[("zero", 10)][0] - 0.02 and R[("gbm", 100)][0] < R[("gbm", 10)][0] - 0.015
    assert (R[("zero", 10)][0] - R[("zero", 30)][0]) / 20 > (R[("zero", 30)][0] - R[("zero", 100)][0]) / 70          # Gewinn je zusätzlichem Depot sinkt
    for n in (10, 30, 100):
        assert R[("gbm", n)][0] < R[("zero", n)][0] + 0.02


# --- Experiment 4: abweichendes Wochenmuster ----------------------------------------------------------------------------------------------------

def test_a_weekend_depot_breaks_zero_shot_and_the_boosting_even_more(shift):
    R = shift["rows"]
    wm = R[("wm", 1.0, 182)][0]
    assert R[("zero", 1.0, 182)][0] > wm + 0.4 and R[("zero", 0.5, 182)][0] > R[("zero", 0.0, 182)][0] + 0.03
    assert R[("gbm", 1.0, 182)][0] > R[("zero", 1.0, 182)][0] + 0.5 and R[("gbm", 0.5, 182)][0] > 1.2 and abs(R[("wm", 0.0, 182)][0] - wm) < 0.02


def test_finetuning_catches_the_weekend_depot_but_long_history_alone_is_as_good(shift):
    R = shift["rows"]
    wm = R[("wm", 1.0, 182)][0]
    assert R[("tuned", 1.0, 182)][0] < R[("zero", 1.0, 182)][0] - 0.5 and R[("tuned", 1.0, 730)][0] < wm + 0.02
    assert R[("scratch", 1.0, 730)][0] < R[("tuned", 1.0, 730)][0] and R[("hw", 1.0, 730)][0] < R[("tuned", 1.0, 730)][0]


def test_self_check_recognises_the_weekend_depot_at_no_cost_for_matching_ones(shift):
    R = shift["rows"]
    assert abs(R[("auto", 1.0, 182)][0] - R[("wm", 1.0, 182)][0]) < 0.02 and R[("auto", 1.0, 182)][0] < R[("zero", 1.0, 182)][0] - 0.5
    assert abs(R[("auto", 0.0, 182)][0] - R[("zero", 0.0, 182)][0]) < 0.015 and abs(R[("auto", 0.5, 182)][0] - R[("zero", 0.5, 182)][0]) < 0.05


# --- Presets (Seed 3) ---------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def presets():
    return {name: E.analyse(_settings(p)) for name, p in P.PRESETS.items()}


def _m(a, m):
    return a.summary[m]["mase"]


def test_preset_standard(presets):
    a = presets["Standardfall: 182 Tage Historie"]
    assert _m(a, "wm") == pytest.approx(0.919, abs=0.01) and _m(a, "zero") == pytest.approx(0.808, abs=0.04) and _m(a, "hw") == pytest.approx(0.935, abs=0.04) and _m(a, "scratch") == pytest.approx(1.152, abs=0.08)
    assert _m(a, "tuned") == pytest.approx(0.839, abs=0.04) and _m(a, "gbm") == pytest.approx(0.777, abs=0.03) and _m(a, "mix") == pytest.approx(0.771, abs=0.03) and _m(a, "oracle") == pytest.approx(0.690, abs=0.01)
    assert a.picked.sum() >= 9 and _m(a, "mix") < _m(a, "zero") and _m(a, "mix") > _m(a, "gbm") - 0.02 and _m(a, "oracle") < min(_m(a, m) for m in E.ALL_METHODS)


def test_preset_98_days(presets):
    a = presets["Fast keine Historie (98 Tage)"]
    assert _m(a, "hw") == pytest.approx(0.975, abs=0.06) and _m(a, "scratch") == pytest.approx(1.219, abs=0.1) and _m(a, "tuned") == pytest.approx(0.873, abs=0.05) and _m(a, "auto") == pytest.approx(0.822, abs=0.05)
    assert _m(a, "hw") > _m(a, "wm") and _m(a, "scratch") > _m(a, "wm") and _m(a, "zero") == pytest.approx(0.808, abs=0.04) and 8 <= a.picked.sum() <= 10


def test_preset_two_years_and_a_thousand_steps(presets):
    a = presets["Zwei Jahre Historie, lang feingetunt"]
    assert _m(a, "hw") == pytest.approx(0.856, abs=0.04) and _m(a, "scratch") == pytest.approx(0.819, abs=0.05) and _m(a, "tuned") == pytest.approx(0.777, abs=0.05) and _m(a, "zero") == pytest.approx(0.808, abs=0.04)
    assert _m(a, "tuned") < _m(a, "zero") + 0.01


def test_preset_weekend_depot(presets):
    a = presets["Wochenend-Depot"]
    assert _m(a, "zero") == pytest.approx(1.788, abs=0.3) and _m(a, "gbm") == pytest.approx(3.405, abs=0.6) and _m(a, "tuned") == pytest.approx(0.930, abs=0.06) and _m(a, "hw") == pytest.approx(0.901, abs=0.05)
    assert _m(a, "wm") == pytest.approx(0.920, abs=0.01) and a.picked.sum() == 0 and _m(a, "auto") == pytest.approx(_m(a, "wm")) and _m(a, "zero") > _m(a, "wm") + 0.5


def test_preset_pool_of_three(presets):
    a = presets["Pool aus nur 3 Depots"]
    assert _m(a, "zero") == pytest.approx(0.990, abs=0.08) and _m(a, "tuned") == pytest.approx(1.039, abs=0.08) and _m(a, "gbm") == pytest.approx(0.874, abs=0.06) and _m(a, "auto") == pytest.approx(0.912, abs=0.05)
    assert _m(a, "zero") > _m(a, "wm") and 1 <= a.picked.sum() <= 4


def test_preset_window_only(presets):
    a = presets["Nur das Fenster, ohne Kalender"]
    assert _m(a, "zero") == pytest.approx(0.868, abs=0.05) and _m(a, "gbm") == pytest.approx(0.777, abs=0.03)
    assert _m(a, "zero") > _m(presets["Standardfall: 182 Tage Historie"], "zero") + 0.02


def test_self_check_picks_by_weekday_pattern():
    base = E.Settings(n_new=C.EXP_NEW)
    picked = {sh: sum(int(E.analyse(E._replace(base, seed=sd, shift=sh)).picked.sum()) for sd in C.EXP_SEEDS) for sh in (0.0, 1.0)}
    assert picked[1.0] == 0 and picked[0.0] >= 29                                                       # 30 Depots: bei passendem Muster (fast) alle, beim Wochenend-Depot keins
