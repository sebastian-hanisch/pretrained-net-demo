"""Einmalige Messung: ein echtes vortrainiertes Zeitreihenmodell (Chronos, tiny) Zero-Shot auf denselben neuen Depots wie die Demo.

Kein Teil der App und der Tests (Torch, Gewichte-Download); gedacht für einen Einzellauf in einer frischen Umgebung, z. B.

    docker run --rm -v <Repo>:/work -w /work python:3.12 bash -c "pip install -q torch --index-url https://download.pytorch.org/whl/cpu && \\
        pip install -q chronos-forecasting numpy plotly streamlit && python tools/chronos_tiny.py"

Protokoll wie die Demo: Seeds 0 bis 2, je 10 neue Depots, Standardpool (40 Depots), Horizont 14 Tage, Ursprünge des Testjahres - hier jeder siebte Tag (51 Ursprünge je Depot, jeder Wochentag gleich oft),
Kennzahl MASE je Depot (Nenner: saisonal naiver Fehler vor dem Testjahr), gemittelt. Chronos sieht nur die letzten `ctx` Tage der Reihe (56 wie das Netz, 512 als natürliche Kontextlänge) und bekommt weder Kalender noch Aktionsplan;
Prognose = Median. Verglichen mit den Verfahren der Demo auf denselben Ursprüngen; 'Wochenend-Depot' ist die Abweichung 1 des Wochenmusters."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ptn_constants as C  # noqa: E402
import ptn_evaluation as E  # noqa: E402

MODELS = ("amazon/chronos-t5-tiny", "amazon/chronos-bolt-tiny")
CONTEXTS = (56, 512)
STEP = 7
SEEDS = (0, 1, 2)
SHIFTS = (0.0, 1.0)


def chronos_mase(pipe, torch, new, ctx, org):
    """MASE je Depot des Chronos-Medians auf den Ursprüngen org."""
    out = []
    for i in range(new.n):
        y = new.y[i]
        scale = np.abs(y[7:C.FIRST_TEST] - y[:C.FIRST_TEST - 7]).mean()
        ctxs = [torch.tensor(y[t - ctx:t], dtype=torch.float32) for t in org]
        preds = []
        for b in range(0, len(ctxs), 64):
            q, _ = pipe.predict_quantiles(ctxs[b:b + 64], prediction_length=C.HORIZON, quantile_levels=[0.5])
            preds.append(q[:, :, 0].numpy())
        act = np.stack([y[t:t + C.HORIZON] for t in org])
        out.append(np.abs(np.concatenate(preds) - act).mean() / scale)
    return out


def main():
    import torch
    from chronos import BaseChronosPipeline
    torch.manual_seed(0)
    org = np.arange(C.FIRST_TEST, C.N_DAYS - C.HORIZON + 1, STEP)
    res = {"origins": len(org), "demo": {}, "chronos": {}}
    for sh in SHIFTS:
        acc = {}
        for sd in SEEDS:
            a = E.analyse(E.Settings(n_new=C.EXP_NEW, seed=sd, shift=sh))
            sel = np.arange(0, len(a.origins), STEP)
            for m in ("wm", "hw", "zero", "tuned", "gbm", "auto"):
                acc.setdefault(m, []).append(float(np.mean(np.abs(a.forecasts[m][:, sel] - a.actual[:, sel]).mean(axis=(1, 2)) / a.scale)))
        res["demo"][str(sh)] = {m: float(np.mean(v)) for m, v in acc.items()}
        print("Demo, Abweichung", sh, {m: round(v, 3) for m, v in res["demo"][str(sh)].items()}, flush=True)
    for name in MODELS:
        pipe = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
        for ctx in CONTEXTS:
            for sh in SHIFTS:
                t0 = time.time()
                per = []
                for sd in SEEDS:
                    new = E._new(E.Settings(n_new=C.EXP_NEW, seed=sd, shift=sh).new_key)
                    per += chronos_mase(pipe, torch, new, ctx, org)
                res["chronos"][f"{name}|{ctx}|{sh}"] = float(np.mean(per))
                print(name, "Kontext", ctx, "Abweichung", sh, round(float(np.mean(per)), 3), f"({time.time() - t0:.0f} s)", flush=True)
    out = Path(__file__).resolve().parent / "chronos_tiny_ergebnis.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("geschrieben:", out)


if __name__ == "__main__":
    main()
