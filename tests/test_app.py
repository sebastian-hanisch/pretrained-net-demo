"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Regler und Momentaufnahmen, Würfel-Knopf, Permalink-Grenzen und -Raster, Extremwerte, vier Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ptn_constants as C
import ptn_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{spct(" not in el.value, el.value[:120]


def test_default_run_shows_metrics_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.metric) == 6 and len(at.get("plotly_chart")) == 5 and len(at.info) + len(at.success) + len(at.warning) >= 1
    assert at.session_state["epoch_slider"] == C.DEFAULT_EPOCHS


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_epoch_and_depot_selections_survive_smaller_settings():
    at = _run(depot_slider=9, origin_slider=1081)
    _ok(at)
    at.slider(key="new_slider").set_value(5).run()
    _ok(at)
    assert at.session_state["depot_slider"] <= 4
    at.slider(key="epochs_slider").set_value(4).run()
    _ok(at)
    assert at.session_state["epoch_slider"] == 4
    at.select_slider(key="epoch_slider").set_value(2).run()
    _ok(at)
    assert at.session_state["epoch_slider"] == 2


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Depots generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in {"pool": "77", "new": "13", "hist": "200", "width": "50", "shift": "0.6", "ft": "120", "epochs": "13", "blocks": "9", "cov": "Window", "noise": "abc", "seed": "-5"}.items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    s = at.session_state
    assert s["pool_slider"] == 70 and s["new_slider"] == 15 and s["hist_slider"] == 182 and s["width_select"] == 64 and s["shift_slider"] == 0.5 and s["ft_slider"] == 100
    assert s["epochs_slider"] == 12 and s["blocks_slider"] == C.BLOCKS_MAX and s["cov_select"] == "window" and s["noise_slider"] == C.DEFAULT_NOISE and s["seed_input"] == 0


@pytest.mark.parametrize("kw", [dict(pool_slider=3, new_slider=5, hist_slider=98, ft_slider=0), dict(pool_slider=100, hist_slider=730, ft_slider=1000, blocks_slider=4, width_select=128),
                                dict(shift_slider=1.0, noise_slider=C.NOISE_MAX, epochs_slider=4, blocks_slider=1, width_select=16), dict(cov_select="window", noise_slider=C.NOISE_MIN, shift_slider=0.5)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0,))
    monkeypatch.setattr(C, "EXP_NEW", 5)
    monkeypatch.setattr(C, "HISTORIES", (98, 365, 730))
    monkeypatch.setattr(C, "FT_LEVELS", (0, 100, 300))
    monkeypatch.setattr(C, "POOL_SIZES", (3, 10, 30))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_history_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "hist_start")
    assert at.session_state["hist_on"] and any("braucht keine Historie" in w.value for w in at.warning)


def test_finetune_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "ft_start")
    assert at.session_state["ft_on"] and any("immer schlechter" in w.value for w in at.warning)


def test_pool_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "pool_start")
    assert at.session_state["pool_on"] and any("schlimmer als kein Vortraining" in w.value for w in at.warning)


def test_shift_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "shift_start")
    assert at.session_state["shift_on"] and any("Die Selbstprüfung erkennt die Abweichung" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
