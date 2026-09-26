"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen, Raster und Auswahllisten - reine Datenprüfungen ohne Streamlit-Session."""

import ptn_constants as C
import ptn_evaluation as E
import ptn_presets as P


def _settings(p):
    return E.Settings(p["n_pool"], p["n_new"], p["hist"], p["noise"], p["shift"], p["epochs"], p["blocks"], p["width"], p["covariates"], p["ft"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("n_new", "new_slider"), ("noise", "noise_slider"), ("shift", "shift_slider"), ("epochs", "epochs_slider"), ("ft", "ft_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6
        for key, state_key in (("n_pool", "pool_slider"), ("hist", "hist_slider"), ("width", "width_select")):
            assert p[key] in P.CHOICES[state_key]
        assert p["covariates"] in C.COVARIATES


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall: 182 Tage Historie"]) == E.Settings()


def test_bounds_steps_choices_and_unique_url_params():
    assert P.bounds("shift_slider") == (C.SHIFT_MIN, C.SHIFT_MAX) and P.bounds("hist_slider") == (98, 730) and P.bounds("pool_slider") == (3, 100)
    assert set(P.STEPS) == {"new_slider", "noise_slider", "shift_slider", "epochs_slider", "ft_slider"} and set(P.CHOICES) == {"pool_slider", "hist_slider", "width_select"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.HIST_CHOICES[0] >= C.WINDOW + C.HORIZON + 28 and C.HIST_CHOICES[-1] == C.FIRST_TEST


def test_caster_rejects_bad_values():
    caster = P._choice_of(C.COVARIATES)
    try:
        caster("quatsch")
    except ValueError:
        pass
    else:
        raise AssertionError("quatsch")
    assert caster(" Window ") == "window"
