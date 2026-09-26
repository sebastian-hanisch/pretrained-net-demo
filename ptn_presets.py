"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. stk_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ptn_constants as C


def _choice_of(options):
    def cast(value):
        v = str(value).strip().lower()
        if v not in options:
            raise ValueError(value)
        return v
    return cast


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "pool_slider": SettingSpec("pool", int, C.DEFAULT_POOL, C.POOL_CHOICES[0], C.POOL_CHOICES[-1]),
    "new_slider": SettingSpec("new", int, C.DEFAULT_NEW, C.NEW_MIN, C.NEW_MAX),
    "hist_slider": SettingSpec("hist", int, C.DEFAULT_HIST, C.HIST_CHOICES[0], C.HIST_CHOICES[-1]),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "shift_slider": SettingSpec("shift", float, C.DEFAULT_SHIFT, C.SHIFT_MIN, C.SHIFT_MAX),
    "epochs_slider": SettingSpec("epochs", int, C.DEFAULT_EPOCHS, C.EPOCHS_MIN, C.EPOCHS_MAX),
    "blocks_slider": SettingSpec("blocks", int, C.DEFAULT_BLOCKS, C.BLOCKS_MIN, C.BLOCKS_MAX),
    "width_select": SettingSpec("width", int, C.DEFAULT_WIDTH, C.WIDTH_CHOICES[0], C.WIDTH_CHOICES[-1]),
    "cov_select": SettingSpec("cov", _choice_of(C.COVARIATES), "full"),
    "ft_slider": SettingSpec("ft", int, C.DEFAULT_FT, C.FT_MIN, C.FT_MAX),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n_pool": "pool_slider", "n_new": "new_slider", "hist": "hist_slider", "noise": "noise_slider", "shift": "shift_slider", "epochs": "epochs_slider", "blocks": "blocks_slider", "width": "width_select",
               "covariates": "cov_select", "ft": "ft_slider", "seed": "seed_input"}
STEPS = {"new_slider": C.NEW_STEP, "noise_slider": C.NOISE_STEP, "shift_slider": C.SHIFT_STEP, "epochs_slider": C.EPOCHS_STEP, "ft_slider": C.FT_STEP}
CHOICES = {"pool_slider": C.POOL_CHOICES, "hist_slider": C.HIST_CHOICES, "width_select": C.WIDTH_CHOICES}


def _p(**kw):
    base = {"n_pool": C.DEFAULT_POOL, "n_new": C.DEFAULT_NEW, "hist": C.DEFAULT_HIST, "noise": C.DEFAULT_NOISE, "shift": C.DEFAULT_SHIFT, "epochs": C.DEFAULT_EPOCHS, "blocks": C.DEFAULT_BLOCKS, "width": C.DEFAULT_WIDTH,
            "covariates": "full", "ft": C.DEFAULT_FT, "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: 182 Tage Historie": _p(),
    "Fast keine Historie (98 Tage)": _p(hist=98),
    "Zwei Jahre Historie, lang feingetunt": _p(hist=730, ft=1000),
    "Wochenend-Depot": _p(shift=1.0, ft=300),
    "Pool aus nur 3 Depots": _p(n_pool=3),
    "Nur das Fenster, ohne Kalender": _p(covariates="window"),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 3)
    for key, options in CHOICES.items():
        if key in st.session_state:
            st.session_state[key] = min(options, key=lambda o: abs(o - st.session_state[key]))
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: 182 Tage Historie": "Pool aus 40 Depots, 10 neue Depots mit 182 Tagen Historie (Seed 3): Zero-Shot 0,808 gegen Wochenmittel 0,919 und Holt-Winters 0,935; das Netz nur aus der eigenen Historie 1,152; Feintuning (100 Schritte) 0,839; das vortrainierte Boosting 0,777, das Mittel aus Netz und Boosting 0,771; Orakel 0,690.",
    "Fast keine Historie (98 Tage)": "98 Tage Historie: Holt-Winters 0,975 und das Netz allein 1,219 sind schlechter als das Wochenmittel (0,919); Zero-Shot bleibt bei 0,808 (es braucht nur sein Fenster), Feintuning verschlechtert auf 0,873; die Selbstprüfung vertraut 9 von 10 Depots und liegt bei 0,822.",
    "Zwei Jahre Historie, lang feingetunt": "730 Tage Historie, 1000 Schritte Feintuning: Holt-Winters 0,856, das Netz allein 0,819, Feintuning 0,777 (Zero-Shot 0,808, Boosting 0,777): erst hier zahlt sich das Feintuning aus.",
    "Wochenend-Depot": "Wochenmuster ganz abgewichen (Sa/So am stärksten), Feintuning 300 Schritte: Zero-Shot 1,788 ist schlechter als das Wochenmittel (0,920), das Boosting 3,405; Feintuning 0,930, Holt-Winters 0,901; die Selbstprüfung fällt bei allen 10 Depots auf das Wochenmittel zurück (0,920).",
    "Pool aus nur 3 Depots": "Pool aus 3 Depots: Zero-Shot 0,990, Feintuning 1,039, Boosting 0,874, Wochenmittel 0,919 - ein zu kleiner Pool ist schlimmer als kein Vortraining; die Selbstprüfung vertraut nur 2 von 10 Depots (0,912).",
    "Nur das Fenster, ohne Kalender": "Das Netz sieht nur das Fenster und den Wochentag, nicht Feiertage und Aktionsplan: Zero-Shot 0,868 statt 0,808; das Boosting (mit Kalendermerkmalen) bleibt bei 0,777.",
}
