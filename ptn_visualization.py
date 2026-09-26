"""Plotly-Abbildungen der Demo "Vortrainiertes Netz". Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import ptn_constants as C
import ptn_evaluation as E
import ptn_net as N
import ptn_samples as X

COLORS = {"wm": "#17becf", "hw": "#e6550d", "scratch": "#9e9e9e", "zero": "#1f77b4", "tuned": "#8c6bb1", "gbm": "#2e7d32", "auto": "#c9a227", "mix": "#00897b", "oracle": "#54a24b"}
ACTUAL = "#14233B"
SHORT = {**E.METHOD_SHORT, "oracle": "Orakel"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.28), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def zero_shot_by_epoch(a):
    """MASE des Zero-Shot-Netzes auf den neuen Depots nach jeder gespeicherten Epoche des Vortrainings."""
    ids = np.arange(a.new.n)
    return {ep: float(np.mean(np.abs(E.net_forecast(p, a.new, ids, a.settings.covariates) - a.actual).mean(axis=(1, 2)) / a.scale)) for ep, p in sorted(a.snapshots.items())}


def build_curves(a, by_epoch):
    """Links die Lernkurve des Vortrainings (Pool: Training und Prüftage), rechts der Zero-Shot-Fehler auf den neuen Depots nach jeder gespeicherten Epoche."""
    ep = np.arange(1, len(a.curve["train"]) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=a.curve["train"], name="Pool, Training", mode="lines", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=ep, y=a.curve["val"], name="Pool, Prüftage vor dem Testjahr", mode="lines", line=dict(color="#e6550d", width=2, dash="dash")))
    fig.update_xaxes(title_text="Epoche")
    fig.update_yaxes(title_text="mittlerer Fehler (normierte Log-Skala)", rangemode="tozero")
    return _base(fig, 300)


def build_epoch_mase(a, by_epoch, epoch):
    xs = [e for e in by_epoch if e >= 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[by_epoch[e] for e in xs], name="Netz, Zero-Shot", mode="lines+markers", line=dict(color=COLORS["zero"], width=2)))
    for m in ("wm", "gbm"):
        fig.add_hline(y=a.summary[m]["mase"], line=dict(color=COLORS[m], width=1.5, dash="dot"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=1.5, dash="dot")))
    if epoch in by_epoch and epoch >= 1:
        fig.add_trace(go.Scatter(x=[epoch], y=[by_epoch[epoch]], mode="markers", name="gezeigte Epoche", marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=ACTUAL, width=2))))
    if epoch == 0:
        fig.add_annotation(text=f"Epoche 0 (Zufallsgewichte): MASE {de(by_epoch[0], 2)}, außerhalb des Bildes", xref="paper", yref="paper", x=0.98, y=0.98, showarrow=False, xanchor="right", yanchor="top", font=dict(color="#d62728"))
    top = max(by_epoch[e] for e in xs[1:]) if len(xs) > 1 else by_epoch[xs[0]]
    fig.update_xaxes(title_text="Epoche des Vortrainings")
    fig.update_yaxes(title_text="MASE auf den neuen Depots", range=[0.6, max(1.1, top * 1.05)])
    return _base(fig, 300)


def build_forecast(a, dep, origin, epoch):
    """Ein neues Depot an einem Ursprung: Rückblick-Fenster, tatsächliche Werte der nächsten 14 Tage und die Prognosen (das Netz in der gewählten Epoche des Vortrainings, feingetunt, Boosting, Holt-Winters, Wochenmittel)."""
    port = a.new
    i = int(origin - a.origins[0])
    back = np.arange(origin - 28, origin)
    fwd = np.arange(origin, origin + C.HORIZON)
    w, c, _, lvl = X.build(port, [dep], [origin], a.settings.covariates)
    p = a.snapshots.get(epoch, a.params)
    snap = X.to_orders(N.forward(p, w, c), lvl)[0]
    fig = go.Figure()
    for t in fwd[port.holiday[fwd] > 0]:
        fig.add_vrect(x0=t - 0.5, x1=t + 0.5, fillcolor="rgba(214,39,40,0.12)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=back, y=port.y[dep, back], name="Rückblick-Fenster (Eingabe; gezeigt: die letzten 28 der 56 Tage)", mode="lines", line=dict(color="rgba(20,35,59,0.45)", width=1.5)))
    fig.add_vline(x=origin - 0.5, line=dict(color="rgba(20,35,59,0.35)", width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=fwd, y=port.y[dep, fwd], name="tatsächlich", mode="lines+markers", line=dict(color=ACTUAL, width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=fwd, y=snap, name=f"Netz, Zero-Shot (Epoche {epoch})", mode="lines", line=dict(color=COLORS["zero"], width=2.5)))
    for m, dash in (("tuned", "solid"), ("gbm", "solid"), ("hw", "dash"), ("wm", "dot")):
        fig.add_trace(go.Scatter(x=fwd, y=a.forecasts[m][dep, i], name=SHORT[m], mode="lines", line=dict(color=COLORS[m], width=1.6, dash=dash)))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 380)


def build_comparison(a):
    """MASE der Verfahren auf den neuen Depots: Balken = Mittel, Punkte = einzelne Depots."""
    ms = list(E.ALL_METHODS) + ["oracle"]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for m in ms:
        v = a.summary[m]["mase"]
        fig.add_trace(go.Bar(x=[SHORT[m]], y=[v], marker=dict(color=COLORS[m]), text=[de(v, 3)], textposition="outside", cliponaxis=False, showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
        per = a.depot_mase[m]
        fig.add_trace(go.Scatter(x=[SHORT[m]] * len(per), y=per, mode="markers", showlegend=False, marker=dict(size=6, color="rgba(20,35,59,0.55)"), hovertemplate="Depot: %{y:.3f}<extra></extra>"))
    top = max(a.summary[m]["mase"] for m in ms)
    fig.update_yaxes(title_text="MASE (Nenner: saisonal naiver Fehler)", range=[0, top * 1.18])
    fig.update_xaxes(tickangle=-25)
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.4), bargap=0.3)


def build_horizon(a):
    h = np.arange(1, C.HORIZON + 1)
    fig = go.Figure()
    for m in ("wm", "hw", "zero", "tuned", "gbm"):
        fig.add_trace(go.Scatter(x=h, y=a.horizon_mae[m], name=SHORT[m], mode="lines+markers", line=dict(color=COLORS[m], width=2), marker=dict(size=5)))
    fig.update_xaxes(title_text="Horizont (Tage nach dem Ursprung)", dtick=1)
    fig.update_yaxes(title_text="Fehler / Nenner der MASE", rangemode="tozero")
    return _base(fig, 320)


def _errline(fig, xs, ys, ses, m, name=None, dash="solid"):
    fig.add_trace(go.Scatter(x=xs, y=ys, name=name or SHORT[m], mode="lines+markers", line=dict(color=COLORS[m], width=2, dash=dash), marker=dict(size=7), error_y=dict(type="data", array=ses, visible=True, thickness=1, width=3)))


def build_history(r):
    fig = go.Figure()
    hs = list(r["histories"])
    for m in ("wm", "zero", "gbm"):
        v = r["fixed"][m][0]
        fig.add_trace(go.Scatter(x=hs, y=[v] * len(hs), name=SHORT[m], mode="lines", line=dict(color=COLORS[m], width=2, dash="dot")))
    for m in ("hw", "scratch", "tuned", "auto"):
        _errline(fig, hs, [r["rows"][(m, h)][0] for h in hs], [r["rows"][(m, h)][1] for h in hs], m)
    fig.update_xaxes(title_text="Historie des neuen Depots (Tage vor dem Testjahr)", tickvals=hs, type="log")
    fig.update_yaxes(title_text="MASE", range=[0.7, 1.35])
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.32))


def build_finetune(r):
    fig = go.Figure()
    steps = list(r["steps"])
    for k, (h, sh) in enumerate(r["conditions"]):
        rows = [r["rows"][(h, sh, n)] for n in steps]
        name = f"{h} Tage Historie, " + ("gleiches Wochenmuster" if sh == 0 else "Wochenend-Depot")
        fig.add_trace(go.Scatter(x=list(range(len(steps))), y=[v[0] for v in rows], name=name, mode="lines+markers", line=dict(width=2, dash="solid" if sh == 0 else "dash"), marker=dict(size=7),
                                 error_y=dict(type="data", array=[v[1] for v in rows], visible=True, thickness=1, width=3)))
    fig.add_hline(y=r["refs"][("wm", *r["conditions"][0])], line=dict(color=COLORS["wm"], width=1.5, dash="dot"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Wochenmittel", line=dict(color=COLORS["wm"], width=1.5, dash="dot")))
    fig.update_xaxes(title_text="Schritte des Feintunings (0 = Zero-Shot)", tickvals=list(range(len(steps))), ticktext=[str(n) for n in steps])
    fig.update_yaxes(title_text="MASE", range=[0.7, 1.8])
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.32))


def build_pool(r):
    fig = go.Figure()
    sizes = list(r["sizes"])
    for m in ("zero", "tuned", "gbm"):
        _errline(fig, sizes, [r["rows"][(m, n)][0] for n in sizes], [r["rows"][(m, n)][1] for n in sizes], m)
    for m in ("wm", "hw"):
        fig.add_hline(y=r["refs"][m], line=dict(color=COLORS[m], width=1.5, dash="dot"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=1.5, dash="dot")))
    fig.update_xaxes(title_text="Zahl der Depots im Pool", tickvals=sizes, type="log")
    fig.update_yaxes(title_text="MASE", range=[0.7, 1.1])
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.32))


def build_shift(r, hist):
    fig = go.Figure()
    levels = list(r["levels"])
    labels = [f"{de(x, 1)}" for x in levels]
    for m in ("wm", "hw", "scratch", "zero", "tuned", "gbm", "auto"):
        v = [r["rows"][(m, sh, hist)] for sh in levels]
        fig.add_trace(go.Bar(x=labels, y=[x[0] for x in v], name=SHORT[m], marker=dict(color=COLORS[m]), error_y=dict(type="data", array=[x[1] for x in v], visible=True, thickness=1, width=2)))
    fig.update_xaxes(title_text="Abweichung des Wochenmusters (0 = wie im Pool, 1 = Wochenend-Depot)")
    fig.update_yaxes(title_text="MASE", range=[0, 3.2])
    fig.update_layout(barmode="group")
    return _base(fig, 380).update_layout(legend=dict(orientation="h", y=-0.4))
