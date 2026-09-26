"""Vortrainiertes Netz - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes und letztes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: ein Netz im Stil von N-BEATS (Blöcke mit Rückblick und Vorausschau, in numpy von Hand gerechnet) wird auf vielen Depots vortrainiert und dann auf einem NEUEN Depot
mit kurzer Historie eingesetzt - ohne eigenes Lernen (Zero-Shot), mit Feintuning und mit einer Selbstprüfung auf der eigenen Historie; verglichen mit dem Netz nur aus der eigenen Historie, dem vortrainierten Boosting (Stück 6) und den lokalen Verfahren.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import ptn_constants as C
import ptn_evaluation as E
import ptn_net as N
import ptn_samples as X
from ptn_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from ptn_visualization import SHORT, build_comparison, build_curves, build_epoch_mase, build_finetune, build_forecast, build_history, build_horizon, build_pool, build_shift, zero_shot_by_epoch

st.set_page_config(page_title="Vortrainiertes Netz – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=1):
    return f"{de(100 * x, digits)} %"


def spct(x, digits=0):
    """Prozent mit ausdrücklichem Vorzeichen (Minuszeichen U+2212); x in Prozent."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return ("+" if x > 0 else "−" if x < 0 else "±") + f"{abs(x):.{digits}f}".replace(".", ",") + " %"


@st.cache_data(show_spinner=False)
def _history(seeds):
    return E.history_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _finetune(seeds):
    return E.finetune_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _pool(seeds):
    return E.pool_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _shift(seeds):
    return E.shift_experiment(seeds=seeds)


st.title("🧠 Vortrainiertes Netz")
st.markdown(
    """
Ein neues Depot hat kaum Historie - und mit ihr lässt sich kein Prognoseverfahren ordentlich schätzen. Die Idee der **Foundation-Modelle** für Zeitreihen: ein Netz **einmal auf vielen anderen Reihen vortrainieren** und es dann auf das neue Depot anwenden - **ohne eigenes Lernen (Zero-Shot)** oder nach einem kurzen **Feintuning** auf der eigenen Historie.
Die Demo baut ein kleines **Netz im Stil von N-BEATS** (Blöcke mit Rückblick und Vorausschau, Residuen dazwischen) in **numpy**, von Hand vorwärts und rückwärts gerechnet, trainiert es auf einem **Pool** erzeugter Depots und prüft es auf **neuen Depots** mit 98 bis 730 Tagen Historie - neben dem Netz, das nur aus der eigenen Historie lernt, dem **vortrainierten Boosting aus Stück 6** und den lokalen Verfahren der Vorgänger.
Gemessen wird auch, wann das Vortraining **nicht** trägt: bei zu kleinem Pool und bei einem Depot, das dem Pool nicht ähnelt - und wie eine **Selbstprüfung auf der eigenen Historie** davor schützt. Alle Daten sind erzeugt; die Rechnung ist in numpy geschrieben, es gibt weder Vortrainings-Gewichte zum Herunterladen noch ein Framework.
"""
)
st.caption(
    "Elftes und **letztes** Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe: die **Vortraining-Kante** von Stück 6 (Boosting mit Lag-Merkmalen, ein Modell für alle Depots) und eine Kante in die Kombination (Stück 9). **Bezug zu OR:** ein neues Depot heißt in der Planung Kaltstart - ohne Prognose kein Bestand "
    "(Stück 10); Wissen aus dem Portfolio zu übertragen und zu prüfen, wann es nicht passt, ist eine Entscheidung unter Modellunsicherheit."
)

with st.expander("So funktioniert ein vortrainiertes Netz", expanded=True):
    st.markdown(
        """
1. **Eingabe und Normierung.** Ein Ursprung $t$ eines Depots: die letzten $L = 56$ Tage als $\\log\\frac{y+1}{s}$ (mit $s$ = Fenstermittel + 1 - die Größe des Depots fällt heraus) und der Kalender (Wochentag, Feiertage, Aktionsplan der nächsten $H = 14$ Tage). Das Netz sagt die nächsten 14 Tage in derselben Normierung voraus.
2. **Blöcke mit Rückblick und Vorausschau.** Jeder Block liest den Rest des Fensters, formt ihn mit zwei versteckten Schichten und gibt zweierlei aus: eine **Vorausschau** (ein Beitrag zur Prognose) und einen **Rückblick** (was er vom Fenster erklärt hat). Der nächste Block sieht nur, was übrig blieb; die Prognose ist die Summe der Vorausschauen.
3. **Vortraining.** Auf den Depots des Pools (nur Tage vor dem Testjahr) minimiert Adam den mittleren absoluten Fehler. Die Gewichte sind danach **fest** - das ist das "Modell", das man weitergeben könnte.
4. **Neues Depot, drei Wege.** **Zero-Shot:** das Netz sieht nur das Fenster der letzten 56 Tage. **Feintuning:** einige Adam-Schritte auf den Fenstern der eigenen Historie. **Selbstprüfung:** vor dem Einsatz wird das Zero-Shot-Netz auf der eigenen Historie mit dem Wochenmittel verglichen; nur wenn es dort besser ist, wird ihm vertraut.
5. **Der Vergleich** läuft wie in den Vorgängern als Rolling Origin über das Testjahr; Kennzahl ist die MASE je Depot (Nenner: saisonal naiver Fehler), gemittelt über die neuen Depots.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Pool (Vortraining)**")
    n_pool = st.select_slider("Depots im Pool", C.POOL_CHOICES, key="pool_slider", help="Aus wie vielen Depots das Netz (und das Boosting) vortrainiert wird.")
    noise = st.slider("Rauschen (Mittel der Depots)", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Mittlere Streuung des multiplikativen Tagesrauschens; gilt für den Pool und die neuen Depots.")
    epochs = st.slider("Epochen des Vortrainings", *bounds("epochs_slider"), key="epochs_slider", step=C.EPOCHS_STEP, help="Wie oft das Netz den Pool durchläuft.")
    blocks = st.slider("Blöcke", *bounds("blocks_slider"), key="blocks_slider", help="Wie viele Blöcke mit Rückblick und Vorausschau hintereinander liegen.")
    width = st.select_slider("Breite der versteckten Schichten", C.WIDTH_CHOICES, key="width_select", help="Neuronen je versteckter Schicht.")
    cov = st.selectbox("Eingabe", list(C.COVARIATES), key="cov_select", format_func=lambda k: C.COVARIATE_NAMES[k], help="Ob das Netz außer dem Fenster (und dem Wochentag) auch Feiertage und den Aktionsplan der nächsten 14 Tage sieht.")
    st.markdown("**Die neuen Depots**")
    n_new = st.slider("Zahl der neuen Depots", *bounds("new_slider"), key="new_slider", step=C.NEW_STEP, help="Auf wie vielen neuen Depots (nicht im Pool) geprüft wird.")
    hist = st.select_slider("Historie des neuen Depots (Tage)", C.HIST_CHOICES, key="hist_slider", help="Wie viele Tage vor dem Testjahr das neue Depot schon Daten hat. Das Zero-Shot-Netz braucht davon nur die letzten 56.")
    shift = st.slider("Abweichung des Wochenmusters", *bounds("shift_slider"), key="shift_slider", step=C.SHIFT_STEP, help="0: wie im Pool (Spitze am Freitag, ruhiges Wochenende). 1: Wochenend-Depot mit den meisten Aufträgen an Sa/So. Gilt nur für die neuen Depots.")
    ft = st.slider("Schritte des Feintunings", *bounds("ft_slider"), key="ft_slider", step=C.FT_STEP, help="Adam-Schritte auf den Fenstern der eigenen Historie (Lernrate 0,0001); 0 heißt Zero-Shot.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Pool und neue Depots fest.")
    st.button("🎲 Neue Depots generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"pool_slider": int(n_pool), "new_slider": int(n_new), "hist_slider": int(hist), "noise_slider": round(float(noise), 2), "shift_slider": round(float(shift), 2), "epochs_slider": int(epochs), "blocks_slider": int(blocks),
                   "width_select": int(width), "cov_select": cov, "ft_slider": int(ft), "seed_input": int(seed)})

settings = E.Settings(int(n_pool), int(n_new), int(hist), round(float(noise), 2), round(float(shift), 2), int(epochs), int(blocks), int(width), cov, int(ft), int(seed))
with st.spinner("Das Netz wird vortrainiert, die neuen Depots werden geprüft ..."):
    a = E.analyse(settings)
    by_epoch = zero_shot_by_epoch(a)
S = a.summary
n_par = N.n_params(a.params)

# --- Vortraining und ein neues Depot ---------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Vom Vortraining zum neuen Depot")
st.caption(
    f"Das Netz hat {de(n_par, 0)} Parameter ({settings.blocks} Blöcke, Breite {settings.width}) und lernt aus {a.pool.n} Pool-Depots; danach sind die Gewichte fest. Links die Lernkurve auf dem Pool, rechts, was dieselben Gewichte auf den {a.new.n} **neuen** Depots leisten - "
    "ohne dass dort je etwas gelernt wurde (Zero-Shot). Die gepunkteten Linien: Wochenmittel und das vortrainierte Boosting."
)
epoch_opts = sorted(a.snapshots)
if st.session_state.get("epoch_slider") not in epoch_opts:
    st.session_state["epoch_slider"] = epoch_opts[-1]
epoch = st.select_slider("Epoche des Vortrainings (Momentaufnahme der Gewichte)", epoch_opts, key="epoch_slider", help="0: Zufallsgewichte, ganz rechts: das fertig vortrainierte Netz. Die Prognose unten ändert sich mit.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Lernkurve auf dem Pool")
    st.plotly_chart(build_curves(a, by_epoch), width="stretch", key="curve_chart")
with c2:
    st.markdown("##### Zero-Shot-Fehler auf den neuen Depots")
    st.plotly_chart(build_epoch_mase(a, by_epoch, epoch), width="stretch", key="epoch_chart")
st.session_state["depot_slider"] = min(a.new.n - 1, max(0, st.session_state.get("depot_slider", 0)))
dep = int(st.slider("Neues Depot", 0, a.new.n - 1, key="depot_slider", help="Welches neue Depot gezeigt wird."))
lo_o, hi_o = int(a.origins[0]), int(a.origins[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 813)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Der Tag, ab dem die nächsten 14 Tage prognostiziert werden. Rot hinterlegt: Feiertage im Prognosezeitraum. Voreinstellung: Tag 813, im Prognosezeitraum liegen zwei Feiertage."))
st.markdown("##### Prognose der nächsten 14 Tage")
st.plotly_chart(build_forecast(a, dep, origin, epoch), width="stretch", key="forecast_chart")
i_o = origin - lo_o
err = {m: float(np.abs(a.forecasts[m][dep, i_o] - a.actual[dep, i_o]).mean()) for m in ("wm", "hw", "tuned", "gbm")}
w_, c_, _, lv_ = X.build(a.new, [dep], [origin], settings.covariates)
snap_err = float(np.abs(X.to_orders(N.forward(a.snapshots.get(epoch, a.params), w_, c_), lv_)[0] - a.actual[dep, i_o]).mean())
st.caption(
    f"Depot {dep}, Ursprung Tag {origin}: mittlerer absoluter Fehler dieser 14 Tage in Aufträgen - Netz (Epoche {epoch}) {de(snap_err, 1)}, feingetunt {de(err['tuned'], 1)}, Boosting {de(err['gbm'], 1)}, Holt-Winters {de(err['hw'], 1)}, Wochenmittel {de(err['wm'], 1)}. "
    "Ein Ursprung ist eine Stichprobe; die Kennzahlen unten mitteln über alle Ursprünge des Testjahres und alle neuen Depots."
)

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was bringt das Vortraining?")
mcols = st.columns(3) + st.columns(3)
for col, m in zip(mcols, ("wm", "hw", "scratch", "zero", "tuned", "gbm")):
    col.metric(SHORT[m], de(S[m]["mase"], 3), delta=None if m == "wm" else f"{spct(100 * (S[m]['mase'] / S['wm']['mase'] - 1))} gegen Wochenmittel", delta_color="off" if m == "wm" else "inverse", help=E.METHOD_NAMES[m] + " - MASE, gemittelt über die neuen Depots (kleiner ist besser).")
h1, h2 = st.columns(2)
with h1:
    st.markdown("##### MASE der Verfahren (Balken: Mittel, Punkte: einzelne Depots)")
    st.plotly_chart(build_comparison(a), width="stretch", key="comparison_chart")
with h2:
    st.markdown("##### Fehler je Horizont")
    st.plotly_chart(build_horizon(a), width="stretch", key="horizon_chart")
better = {m: int(np.sum(a.depot_mase[m] < a.depot_mase["wm"])) for m in E.ALL_METHODS}
z, w, g, au, mx = S["zero"]["mase"], S["wm"]["mase"], S["gbm"]["mase"], S["auto"]["mase"], S["mix"]["mase"]
n_pick = int(a.picked.sum())
if z < w:
    st.success(
        f"✅ Das vortrainierte Netz trägt: Zero-Shot {de(z, 3)} gegen Wochenmittel {de(w, 3)} ({spct(100 * (z / w - 1))}) und Holt-Winters {de(S['hw']['mase'], 3)}, ohne dass es je Daten dieser Depots gesehen hat; das Netz nur aus {settings.hist} Tagen eigener Historie kommt auf {de(S['scratch']['mase'], 3)}. "
        f"Die Selbstprüfung vertraut ihm bei {n_pick} von {a.new.n} Depots. Das **vortrainierte Boosting aus Stück 6** ist mit {de(g, 3)} {'noch besser' if g < z else 'nicht besser'}; das Mittel aus beiden erreicht {de(mx, 3)} (Orakel: {de(S['oracle']['mase'], 3)})."
    )
else:
    st.warning(
        f"⚠️ Das vortrainierte Netz trägt hier nicht: Zero-Shot {de(z, 3)} ist schlechter als das Wochenmittel {de(w, 3)}. Das neue Depot ähnelt dem Pool nicht (oder der Pool ist zu klein). Die **Selbstprüfung** auf der eigenen Historie fällt bei {a.new.n - n_pick} von {a.new.n} Depots auf das Wochenmittel zurück und erreicht {de(au, 3)}; "
        f"Feintuning mit {settings.ft_steps} Schritten kommt auf {de(S['tuned']['mase'], 3)}, das vortrainierte Boosting auf {de(g, 3)}."
    )
rows = [{"Verfahren": E.METHOD_NAMES[m], "MASE": de(S[m]["mase"], 3), "gegen Wochenmittel": "–" if m == "wm" else spct(100 * (S[m]["mase"] / w - 1)), "Depots besser als Wochenmittel": f"{better[m]} von {a.new.n}" if m != "wm" else "–"} for m in E.ALL_METHODS]
rows.append({"Verfahren": "Orakel (wahrer Erwartungswert)", "MASE": de(S["oracle"]["mase"], 3), "gegen Wochenmittel": spct(100 * (S["oracle"]["mase"] / w - 1)), "Depots besser als Wochenmittel": f"{int(np.sum(a.depot_mase['oracle'] < a.depot_mase['wm']))} von {a.new.n}"})
st.dataframe(rows, hide_index=True)
st.caption(
    f"MASE: mittlerer absoluter Fehler über die 14 Tage und alle Ursprünge des Testjahres, geteilt durch den saisonal naiven Fehler des Depots (unter 1 heißt besser als \"dasselbe wie letzte Woche\"). Historie {settings.hist} Tage: Holt-Winters und das Netz allein schätzen daraus ihre Parameter, das Feintuning startet daraus. "
    "Das Zero-Shot-Netz und das Boosting brauchen nur ihr Fenster (56 beziehungsweise 91 Tage)."
)

st.markdown("---")

# --- Experimente ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viel Historie braucht ein neues Depot?")
st.caption(f"{C.EXP_NEW} neue Depots, Standardnetz, Pool aus {C.DEFAULT_POOL} Depots; die Historie läuft über {', '.join(str(x) for x in C.HISTORIES)} Tage. Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer etwa eine halbe Minute.")
if st.button("Historien durchrechnen", key="hist_start"):
    st.session_state["hist_on"] = True
if st.session_state.get("hist_on"):
    rh = _history(C.EXP_SEEDS)
    st.plotly_chart(build_history(rh), width="stretch", key="history_chart")
    hs = rh["histories"]
    R = rh["rows"]
    st.warning(
        f"**Befund:** Das Zero-Shot-Netz ({de(rh['fixed']['zero'][0], 3)}) braucht keine Historie und liegt bei jeder Länge unter Holt-Winters (bei {hs[0]} Tagen {de(R[('hw', hs[0])][0], 3)}, bei {hs[-1]} noch {de(R[('hw', hs[-1])][0], 3)}) und unter dem Wochenmittel ({de(rh['fixed']['wm'][0], 3)}). "
        f"Das Netz nur aus der eigenen Historie liegt bei {hs[0]} Tagen bei {de(R[('scratch', hs[0])][0], 3)} und erreicht bei {hs[-1]} Tagen {de(R[('scratch', hs[-1])][0], 3)}. **Feintuning** ({C.DEFAULT_FT} Schritte) schadet bei kurzer Historie ({de(R[('tuned', hs[0])][0], 3)} bei {hs[0]} Tagen) und hilft erst ab etwa einem Jahr "
        f"({de(R[('tuned', 365)][0], 3)} bei 365, {de(R[('tuned', hs[-1])][0], 3)} bei {hs[-1]} Tagen). **Das vortrainierte Boosting aus Stück 6 ist mit {de(rh['fixed']['gbm'][0], 3)} besser als das Netz** - die Linie hatte das Vortraining schon. Die Selbstprüfung kostet auf einem passenden Depot fast nichts ({de(R[('auto', hs[0])][0], 3)} bei {hs[0]} Tagen, sonst gleich Zero-Shot)."
    )

st.markdown("---")

st.subheader("🔬 Wie viele Schritte Feintuning?")
st.caption(f"{C.EXP_NEW} neue Depots, Standardnetz; vier Fälle: 182 oder 730 Tage Historie, Wochenmuster wie im Pool oder Wochenend-Depot. Die Schritte laufen über {', '.join(str(x) for x in C.FT_LEVELS)} (Lernrate 0,0001). Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Feintuning durchrechnen", key="ft_start"):
    st.session_state["ft_on"] = True
if st.session_state.get("ft_on"):
    rf = _finetune(C.EXP_SEEDS)
    st.plotly_chart(build_finetune(rf), width="stretch", key="finetune_chart")
    RF = rf["rows"]
    st_ = rf["steps"]
    st.warning(
        f"**Befund:** Passt das Depot zum Pool, ist Feintuning bei {rf['conditions'][0][0]} Tagen Historie **immer schlechter** als Zero-Shot ({de(RF[(182, 0.0, 0)][0], 3)} auf {de(RF[(182, 0.0, st_[-1])][0], 3)} nach {st_[-1]} Schritten: das Netz lernt das Rauschen der kurzen Historie), bei {rf['conditions'][1][0]} Tagen hilft es langsam ({de(RF[(730, 0.0, st_[-1])][0], 3)} nach {st_[-1]} Schritten). "
        f"Beim Wochenend-Depot ist Zero-Shot unbrauchbar ({de(RF[(182, 1.0, 0)][0], 3)}); Feintuning repariert es, aber erst mit vielen Schritten ({de(RF[(730, 1.0, 100)][0], 3)} nach 100, {de(RF[(730, 1.0, st_[-1])][0], 3)} nach {st_[-1]}), und bei 182 Tagen nur bis etwa das Wochenmittel ({de(RF[(182, 1.0, 300)][0], 3)} nach 300 Schritten; Wochenmittel {de(rf['refs'][('wm', 182, 1.0)], 3)}). "
        "Wie viele Schritte richtig sind, hängt davon ab, wie weit das Depot vom Pool entfernt ist - und das weiß man vorher nicht."
    )

st.markdown("---")

st.subheader("🔬 Wie groß muss der Pool sein?")
st.caption(f"{C.EXP_NEW} neue Depots, Standardnetz, 182 Tage Historie; der Pool hat {', '.join(str(x) for x in C.POOL_SIZES)} Depots. Gezeigt: das Netz (Zero-Shot und feingetunt) und das Boosting. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Pools durchrechnen", key="pool_start"):
    st.session_state["pool_on"] = True
if st.session_state.get("pool_on"):
    rp = _pool(C.EXP_SEEDS)
    st.plotly_chart(build_pool(rp), width="stretch", key="pool_chart")
    RP = rp["rows"]
    sz = rp["sizes"]
    st.warning(
        f"**Befund:** Aus {sz[0]} Depots gelernt, ist das Netz ({de(RP[('zero', sz[0])][0], 3)}) schlechter als das Wochenmittel ({de(rp['refs']['wm'], 3)}), das Boosting ({de(RP[('gbm', sz[0])][0], 3)}) ebenso: **ein zu kleiner Pool ist schlimmer als kein Vortraining**. Ab {sz[1]} Depots trägt es ({de(RP[('zero', sz[1])][0], 3)} beziehungsweise {de(RP[('gbm', sz[1])][0], 3)}), "
        f"mit {sz[-1]} Depots erreicht das Netz {de(RP[('zero', sz[-1])][0], 3)} und das Boosting {de(RP[('gbm', sz[-1])][0], 3)}. Das Boosting ist bei jeder Poolgröße ab {sz[1]} Depots besser als das Netz. Mehr Depots helfen weiter, aber mit abnehmendem Ertrag."
    )

st.markdown("---")

st.subheader("🔬 Was, wenn das neue Depot anders ist?")
st.caption(f"{C.EXP_NEW} neue Depots, Standardnetz, Pool aus {C.DEFAULT_POOL} Depots mit dem Wochenmuster der Vorgänger; das Wochenmuster der neuen Depots weicht um 0, 0,5 und 1 ab (1 = Wochenend-Depot). Feintuning mit {C.SHIFT_FT} Schritten, Historie 182 und 730 Tage. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Abweichungen durchrechnen", key="shift_start"):
    st.session_state["shift_on"] = True
if st.session_state.get("shift_on"):
    rs = _shift(C.EXP_SEEDS)
    RS = rs["rows"]
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("##### 182 Tage Historie")
        st.plotly_chart(build_shift(rs, 182), width="stretch", key="shift_chart_182")
    with q2:
        st.markdown("##### 730 Tage Historie")
        st.plotly_chart(build_shift(rs, 730), width="stretch", key="shift_chart_730")
    st.warning(
        f"**Befund:** Beim Wochenend-Depot bricht das Zero-Shot-Netz ein ({de(RS[('zero', 0.0, 182)][0], 3)} auf {de(RS[('zero', 1.0, 182)][0], 3)}, schlechter als das Wochenmittel {de(RS[('wm', 1.0, 182)][0], 3)}); **das Boosting bricht stärker ein** ({de(RS[('gbm', 0.0, 182)][0], 3)} auf {de(RS[('gbm', 1.0, 182)][0], 3)}), vermutlich, weil es Wochentag und Kalender gelernt hat, "
        f"während das Netz das Muster wenigstens teilweise im Fenster liest (nicht isoliert). Feintuning ({C.SHIFT_FT} Schritte) fängt es ab ({de(RS[('tuned', 1.0, 182)][0], 3)} bei 182, {de(RS[('tuned', 1.0, 730)][0], 3)} bei 730 Tagen); mit 730 Tagen eigener Historie sind aber das Netz allein ({de(RS[('scratch', 1.0, 730)][0], 3)}) und Holt-Winters ({de(RS[('hw', 1.0, 730)][0], 3)}) schon besser: **das Vortraining ist dort kein Vorteil mehr.** "
        f"**Die Selbstprüfung erkennt die Abweichung** und fällt auf das Wochenmittel zurück ({de(RS[('auto', 1.0, 182)][0], 3)}), ohne bei passenden Depots etwas zu kosten ({de(RS[('auto', 0.0, 182)][0], 3)} gegen Zero-Shot {de(RS[('zero', 0.0, 182)][0], 3)}). Ein vortrainiertes Modell ohne diese Prüfung ist ein Risiko."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das neue Depot stammt aus derselben Welt wie der Pool** | Bei anderem Wochenmuster bricht Zero-Shot ein (schlechter als das Wochenmittel); das Boosting mit festen Kalendermerkmalen noch stärker. | Selbstprüfung auf der eigenen Historie, Feintuning mit ausreichend Daten |
| **Der Pool ist groß genug** | Aus wenigen Depots gelernt, ist das Vortraining schlechter als das Wochenmittel; hier tragen 10 Depots schon, 3 nicht. | Pool vergrößern oder auf das Vortraining verzichten |
| **Ein kleines Netz genügt** | Das Standardnetz hat rund 14 000 Parameter und wurde auf erzeugten Daten mit einer Musterfamilie trainiert; echte Foundation-Modelle haben Millionen bis Milliarden Parameter und sehen viele Musterfamilien. Ein größeres Netz wurde hier **nicht** besser (4 Blöcke mit Breite 128 und 40 Epochen: schlechter, 0,939 gegen 0,827). | Echte Modelle (Chronos, TimesFM, Moirai) - hier bewusst nicht gerechnet |
| **Feintuning hilft** | Bei kurzer Historie lernt es das Rauschen: schlechter als Zero-Shot. Die Schrittzahl ist eine Stellgröße, die man nur mit Historie einstellen kann. | Frühes Stoppen mit einem Validierungsstück, kleine Lernrate |
| **Die Selbstprüfung schützt** | Sie vergleicht mit dem Wochenmittel auf wenigen Fenstern der Historie (bei 98 Tagen 29); bei mittlerer Abweichung entscheidet sie unscharf. Sie prüft die Vergangenheit, nicht die Zukunft. | Längeres Prüfstück, Vergleich mit weiteren Verfahren |
| **Die Depots sind unabhängig** | Pool und neue Depots teilen Kalender und Zeitraum; in echten Portfolios trainiert man auf der Vergangenheit anderer Depots und prüft danach. Hier liegen alle Zieltage des Vortrainings vor dem Testjahr. | – |
| **Erzeugte Daten, drei Seeds** | Das Vehikel erzeugt genau die Muster, die das Netz lernen kann; echte Reihen sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → Dynamische Regression, dazu Croston, Boosting, Prognoseintervalle, Hierarchie, Kombination, Bestand und **dieses vortrainierte Netz** - das letzte Stück.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Normierung.** Ursprung $t$, Fenster $L = 56$, Horizont $H = 14$, $s = 1 + \frac1L\sum_{i=1}^{L} y_{t-i}$. Eingabe $w_i = \log\frac{y_{t-L-1+i} + 1}{s}$, Ziel $z_j = \log\frac{y_{t+j-1}+1}{s}$; Prognose in Aufträgen $\hat y_{t+j-1} = \max(s\,e^{\hat z_j} - 1,\ 0)$. Zusatzmerkmale $c$: Wochentag des Ursprungs (One-Hot), bei "Fenster + Kalender" Feiertag, Tag danach und Aktion an den 14 Zieltagen sowie $\sin, \cos$ des Jahrestags.

**Block $k$.** $r_1 = w$, $u_k = [r_k, c]$, $h_1 = \max(0, W_1 u_k + b_1)$, $h_2 = \max(0, W_2 h_1 + b_2)$, $\text{back}_k = W_b h_2 + b_b$, $\text{fc}_k = W_f h_2 + b_f$, $r_{k+1} = r_k - \text{back}_k$. Prognose $\hat z = \sum_k \text{fc}_k$.

**Verlust und Ableitung.** $\mathcal L = \frac1{BH}\sum |\hat z - z|$. Rückwärts: $g = \operatorname{sign}(\hat z - z)/(BH)$ für jede Vorausschau; $G_{K+1} = 0$ (der letzte Rest geht nicht in die Prognose ein), $\partial\mathcal L/\partial\text{back}_k = -G_{k+1}$, und $G_k = G_{k+1} + \big(\partial\mathcal L/\partial u_k\big)_{1..L}$ - der Rest wirkt direkt (Identität) und über den Block. Innerhalb des Blocks die übliche Kettenregel mit den Relu-Masken. Die Tests prüfen alle Ableitungen gegen Differenzenquotienten.

**Vortraining.** Adam ($\beta = 0{,}9;\,0{,}999$, Lernrate $2\cdot10^{-3}$, Batch 256) auf allen Ursprüngen jedes zweiten Tages der Pool-Depots mit Zieltagen vor Tag 685 (die 45 Tage danach bis zum Testjahr sind die Prüftage der Lernkurve). **Feintuning:** Adam mit Lernrate $10^{-4}$, Batch 32, auf allen Ursprüngen der eigenen Historie (Fenster und Ziel liegen darin). **Netz allein:** Zufallsgewichte, 400 Schritte mit Lernrate $2\cdot10^{-3}$ auf derselben Historie.

**Selbstprüfung.** Für Depot $i$ wird das Zero-Shot-Netz auf den Ursprüngen der eigenen Historie mit dem Wochenmittel (vier Wochen, Stück 1) verglichen: $\text{MAE}_{\text{Netz}} \le \text{MAE}_{\text{Wochenmittel}}$ heißt "dem Netz vertrauen", sonst gilt das Wochenmittel.

**Kennzahl.** $\text{MASE}_i = \dfrac{\text{MAE}_i}{\frac1{n-7}\sum_{t\ge 7}|y_t - y_{t-7}|}$ (Tage vor dem Testjahr); Testjahr: Ursprünge $t = 730, \dots, 1081$; Mittel über die neuen Depots.

Implementiert in `ptn_net.py` (das Netz, Ableitung, Adam), `ptn_samples.py` (Fenster, Normierung, Zeilen), `ptn_evaluation.py` (Vortraining, Feintuning, Selbstprüfung, Analyse und vier Experimente), `ptn_baselines.py`, `ptn_ets.py`, `ptn_tree.py`, `ptn_gbm.py`, `ptn_features.py` (die Verfahren der Vorgänger), `ptn_scenario.py` (die Depots).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
