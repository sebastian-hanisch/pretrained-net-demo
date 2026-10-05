# 🧠 Vortrainiertes Netz – Wissen aus vielen Depots für ein neues

**[→ Demo live ausprobieren](https://sebastianhanisch-pretrained-net-demo.streamlit.app/)**

Elftes und **letztes Stück** der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Die **Vortraining-Kante** von Stück 6 ([Boosting mit Lag-Merkmalen](https://github.com/sebastian-hanisch/boosting-forecast-demo), ein Modell für alle Depots) und eine Kante in die [Kombination](https://github.com/sebastian-hanisch/forecast-combination-demo) (Stück 9). Es ist das **einzige neuronale Netz** der Linie – in numpy von Hand gerechnet, ohne Framework.

Ein **neues Depot** hat kaum Historie, und mit ihr lässt sich kein Prognoseverfahren ordentlich schätzen. Die Idee der Foundation-Modelle für Zeitreihen: **einmal auf vielen anderen Reihen vortrainieren** und dann anwenden – ohne eigenes Lernen (**Zero-Shot**) oder nach kurzem **Feintuning**. Die Demo baut ein kleines Netz im Stil von **N-BEATS** (Blöcke mit Rückblick und Vorausschau, Residuen dazwischen), trainiert es auf einem **Pool** erzeugter Depots und prüft es auf **neuen Depots** mit 98 bis 730 Tagen Historie – neben dem Netz, das nur aus der eigenen Historie lernt, dem vortrainierten Boosting aus Stück 6 und den lokalen Verfahren der Vorgänger. Alle Daten sind erzeugt.

## Kernfrage

**Wann trägt ein vortrainiertes Modell für ein neues Depot – und wann nicht?** Die Demo misst vier Dinge: wie viel Historie das neue Depot braucht, wie viele Schritte Feintuning richtig sind, wie groß der Pool sein muss, und was passiert, wenn das neue Depot dem Pool nicht ähnelt (ein Wochenend-Depot). Dazu eine **Selbstprüfung auf der eigenen Historie**, die dem vortrainierten Netz nur traut, wenn es dort das Wochenmittel schlägt.

**Bezug zu OR:** Ein neues Depot heißt in der Planung Kaltstart – ohne Prognose kein Bestand ([Stück 10](https://github.com/sebastian-hanisch/forecast-inventory-demo)). Wissen aus dem Portfolio zu übertragen und zu prüfen, wann es nicht passt, ist eine Entscheidung unter Modellunsicherheit.

## Modell

- **Vehikel:** die Tagesaufträge mehrerer Depots wie in den Vorgängern (Niveau, Trend, Wochen- und Jahresmuster, Feiertage, Aktionen, log-normales Rauschen, 1095 Tage, Testjahr ab Tag 730). Der **Pool** (Standard 40 Depots) und die **neuen Depots** (Standard 10) sind unabhängige Ziehungen mit demselben Kalender; nur die neuen Depots können ein anderes Wochenmuster haben ("Abweichung 1" = **Wochenend-Depot**: Sa/So stark, Werktage schwach).
- **Netz:** Fenster der letzten 56 Tage, normiert als log((y+1)/s) mit s = Fenstermittel + 1 (die Größe des Depots fällt heraus), dazu Wochentag (und optional Feiertage, Aktionsplan, Jahrestag der nächsten 14 Tage); Ausgabe die nächsten 14 Tage in derselben Normierung. Jeder **Block** hat zwei versteckte Relu-Schichten und zwei Ausgänge – **Vorausschau** (Beitrag zur Prognose) und **Rückblick** (was er vom Fenster erklärt hat); der nächste Block sieht nur den Rest. Standard: 2 Blöcke, Breite 32, **13 644 Parameter**.
- **Training:** Verlust = mittlerer absoluter Fehler auf der normierten Log-Skala, Ableitung von Hand (Tests gegen Differenzenquotienten), Adam. **Vortraining** auf allen Ursprüngen jedes zweiten Tages der Pool-Depots mit Zieltagen vor Tag 685 (12 Epochen); die 45 Tage davor sind die Prüftage der Lernkurve. **Feintuning:** Adam mit Lernrate 0,0001 auf den Fenstern der eigenen Historie. **Netz allein:** Zufallsgewichte, 400 Schritte auf derselben Historie.
- **Selbstprüfung:** Für jedes neue Depot wird das Zero-Shot-Netz auf den Ursprüngen der *eigenen* Historie mit dem Wochenmittel verglichen; nur wenn es dort nicht schlechter ist, gilt es, sonst das Wochenmittel.
- **Kennzahl:** MASE je Depot (Nenner: saisonal naiver Fehler auf den Tagen vor dem Testjahr), gemittelt über die neuen Depots; Rolling Origin über das Testjahr, Horizont 14 Tage.

## Methodik

Verglichen werden auf den neuen Depots: **Wochenmittel** (Stück 1) und **Holt-Winters** (Stück 2) mit der eigenen Historie, das **Netz allein**, das **vortrainierte Netz** (Zero-Shot, mit Feintuning, mit Selbstprüfung), das **vortrainierte Boosting** aus Stück 6 (dasselbe Pool-Training, Lag-, Kalender- und Aktionsmerkmale, Zero-Shot) und das **Mittel aus Netz und Boosting** (Stück 9). Als Untergrenze für Erwartungswert-Schätzer der **Orakel** (wahrer Erwartungswert; für die MAE wäre streng genommen der Median der log-normalen Reihe minimal besser, bei Rauschen 0,14 um etwa 0,2 %). Die Gewichte des Vortrainings sehen nur Tage vor dem Testjahr; die Gewichte eines neuen Depots nur seine Historie vor dem Testjahr.

## Befunde (gemessen, keine Behauptungen)

Mehr-Seed-Zahlen: Mittel über Seeds 0 bis 2 mit je 10 neuen Depots und Standardnetz (Pool 40 Depots). Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund |
|---|---|
| Braucht das vortrainierte Netz Historie? | **Nein.** Zero-Shot **0,827** MASE (es sieht nur sein 56-Tage-Fenster), Wochenmittel 0,937, Holt-Winters 1,159 bei 98 Tagen Historie, 0,913 (182), 0,880 (365) und **noch 0,869 bei 730 Tagen**: das Zero-Shot-Netz liegt bei jeder Länge unter Holt-Winters. |
| Und das Netz nur aus der eigenen Historie? | Bei 98 Tagen 1,191, bei 182 1,097, bei 365 0,910, bei 730 0,846. Es erreicht das Zero-Shot-Netz auch mit zwei Jahren nicht. |
| Hilft Feintuning? | **Nur mit langer Historie, und wenig.** 100 Schritte: 0,896 bei 98 Tagen (schlechter als Zero-Shot), 0,835 bei 182, 0,817 bei 365, 0,811 bei 730 (−2 % gegen Zero-Shot). Bei 182 Tagen wird es mit jedem Schritt schlechter (0,827 → 0,872 nach 1000); bei 730 Tagen langsam besser (0,799 nach 1000). |
| **Ist das Netz besser als das vortrainierte Boosting aus Stück 6?** | **Nein.** Das Boosting liegt bei 0,800, das Netz bei 0,827 – der Abstand ist klein (etwa ein Standardfehler), aber das Netz gewinnt nirgends. Die Linie hatte das Vortraining schon. Das Mittel beider (Standardfall, Seed 3) 0,771 gegen Boosting 0,777: fast keine Vielfalt. |
| Wie groß muss der Pool sein? | **Ein zu kleiner Pool ist schlimmer als kein Vortraining.** Aus 3 Depots: Netz 0,999, Boosting 0,978, beide schlechter als das Wochenmittel (0,937). Ab 10 Depots trägt es (0,863 / 0,829), 30: 0,831 / 0,809, 100: 0,811 / 0,797. Der Gewinn je zusätzlichem Depot sinkt. |
| Was, wenn das neue Depot anders ist? | **Zero-Shot bricht ein:** bei halber Abweichung des Wochenmusters 0,903, beim Wochenend-Depot **1,617** (Wochenmittel 0,939). **Das Boosting bricht stärker ein:** 1,460 und 2,888 – vermutlich, weil es Wochentag und Kalender gelernt hat, das Netz das Muster teilweise im Fenster liest (nicht isoliert). |
| Repariert Feintuning das? | Erst mit vielen Schritten: beim Wochenend-Depot 1,617 → 1,352 (50) → 1,133 (100) → 0,935 (300) bei 182 Tagen und 0,888 (300) → 0,838 (1000) bei 730 Tagen. Mit 730 Tagen eigener Historie sind aber das Netz allein (0,855) und Holt-Winters (0,869) schon besser als 300 Schritte Feintuning (0,888): **das Vortraining ist dort kein Vorteil mehr.** |
| Schützt die Selbstprüfung? | **Ja, und sie kostet fast nichts.** Beim Wochenend-Depot vertraut sie dem Netz keinem von 30 Depots und fällt auf das Wochenmittel zurück (0,939 statt 1,617); bei passendem Muster vertraut sie 30 von 30 (0,827 = Zero-Shot; bei 98 Tagen 0,834). Bei halber Abweichung entscheidet sie unscharf (0,904 gegen 0,903 Zero-Shot). |
| Hilft ein größeres Netz? | **Nein.** 1 oder 4 Blöcke: 0,833 / 0,827, Breite 16 oder 128: 0,836 / 0,831, 4 / 40 Epochen: 0,841 / 0,822; das große Netz (4 Blöcke, Breite 128, 40 Epochen) überanpasst: 0,939. |
| Hilft der Kalender? | Ohne Feiertage und Aktionsplan (nur Fenster und Wochentag): 0,874 statt 0,827. |

**Standardfall (Seed 3, 10 neue Depots, 182 Tage):** Wochenmittel 0,919, Holt-Winters 0,935, Netz allein 1,152, Zero-Shot **0,808**, Feintuning 0,839, Boosting 0,777, Mittel aus Netz und Boosting 0,771, Orakel 0,690. Weitere Voreinstellungen mit Zahlen: siehe die Hilfetexte der sechs Schnellstart-Knöpfe (alle in `tests/test_claims.py` nachgerechnet).

## Einmalmessung mit einem echten Modell (Chronos-tiny)

Kein Teil der App und der Tests (Torch, Gewichte-Download), sondern ein **einmaliger Lauf** von `tools/chronos_tiny.py` in einer frischen Umgebung (Docker, `python:3.12`, Torch für die CPU, `chronos-forecasting`); das Ergebnis steht in `tools/chronos_tiny_ergebnis.json`. Protokoll wie die Demo: Seeds 0 bis 2, je 10 neue Depots, Horizont 14 Tage, aber nur **jeder siebte Ursprung** des Testjahres (51 je Depot, jeder Wochentag gleich oft); die Verfahren der Demo sind auf denselben Ursprüngen neu ausgewertet (Historie 182 Tage, Feintuning 100 Schritte). Chronos sieht nur die letzten 56 Tage (wie das Netz) oder 512 Tage der Reihe, **weder Kalender noch Aktionsplan**; Prognose = Median.

| MASE | Wochenmuster wie im Pool | Wochenend-Depot |
|---|---|---|
| Wochenmittel | 0,936 | 0,938 |
| Holt-Winters (182 Tage) | 0,912 | 1,024 |
| Netz, Zero-Shot | 0,834 | 1,639 |
| Netz mit Feintuning (100 Schritte) | 0,836 | 1,177 |
| Netz mit Selbstprüfung | 0,834 | 0,938 |
| Boosting (Stück 6), Zero-Shot | 0,798 | 2,877 |
| **Chronos-T5-tiny**, Kontext 56 Tage | 1,014 | 1,062 |
| **Chronos-T5-tiny**, Kontext 512 Tage | 0,968 | 1,070 |
| Chronos-Bolt-tiny, Kontext 56 Tage | 1,543 | 1,415 |
| Chronos-Bolt-tiny, Kontext 512 Tage | 1,388 | 1,292 |

**Befund:** Das echte vortrainierte Modell ist auf diesen Daten **schlechter als das Wochenmittel** (T5-tiny 0,968 bis 1,014 gegen 0,936; Bolt-tiny deutlich schlechter) und weit hinter dem kleinen Netz (0,834) und dem Boosting (0,798). Dafür **bricht es beim Wochenend-Depot nicht ein** (1,062 und 1,070 statt 1,639 beim Netz und 2,877 beim Boosting) – ohne das Wochenmittel (0,938) zu schlagen. Es ist robust, aber auf dieser Aufgabe ohne Nutzen. Vermutete Gründe (nicht isoliert): es kennt weder Feiertage (zwei in 14 Tagen an Ursprung 813) noch Aktionsplan, die das Netz und das Boosting als Eingabe bekommen; und es ist mit 8 Millionen Parametern (T5-tiny) das kleinste der T5-Familie. Chronos-T5 zieht Stichproben: ein erster Lauf ergab 1,008 und 0,964 (Kontext 56 und 512, Wochenmuster wie im Pool), die Wiederholung 1,014 und 0,968. Eine Aussage über größere Modelle (größere Chronos-Modelle, TimesFM, Moirai) ist das nicht.

## Ehrliche Grenzen

- **Erzeugte Daten mit einer Musterfamilie.** Das Netz lernt genau die Muster, die das Vehikel erzeugt (Wochen- und Jahresmuster, Feiertage, Aktionen); echte Portfolios sind unordentlicher. Echte Foundation-Modelle haben Millionen bis Milliarden Parameter und sehen viele Musterfamilien – in der App und den Tests bewusst **nicht** gerechnet (kein Torch, keine Gewichte zum Herunterladen, keine CI-Fragilität); eine einmalige Messung mit Chronos-tiny steht oben.
- **Pool und neue Depots teilen Kalender und Zeitraum.** In echten Portfolios trainiert man auf der Vergangenheit anderer Depots und prüft danach; hier liegen alle Zieltage des Vortrainings vor dem Testjahr, die neuen Depots bringen nur Historie vor dem Testjahr mit.
- **Die Standardgröße (2 Blöcke, Breite 32, 12 Epochen) wurde nach diesen Messungen gewählt.** Die Auswahl fiel auf denselben Seeds wie der Test (Unterschiede höchstens 1 %); die **Lernrate des Feintunings** (0,0001) und die Schrittzahlen wurden dagegen auf getrennten Seeds (100 bis 102) bestimmt. Größere Netze wurden nicht besser.
- **Feintuning ist einfach gehalten:** alle Gewichte, feste Lernrate, feste Schrittzahl, kein frühes Stoppen. Ein Validierungsstück in der eigenen Historie könnte die Schrittzahl wählen – bei 98 Tagen (29 Fenster) bleibt dafür kaum etwas übrig.
- **Die Selbstprüfung vergleicht mit dem Wochenmittel auf wenigen Fenstern der Historie** (bei 98 Tagen 29) und prüft die Vergangenheit, nicht die Zukunft; bei mittlerer Abweichung entscheidet sie unscharf.
- **Die Erklärung des stärkeren Einbruchs beim Boosting ist eine Vermutung** (Wochentag und Kalender als feste Merkmale gegen ein Fenster, das das Muster zeigt), nicht isoliert.
- **Drei Seeds, zehn neue Depots je Seed.** Kleine Abstände (Netz gegen Boosting, Feintuning bei 365 und 730 Tagen) liegen bei etwa einem Standardfehler; die Tests prüfen sie deshalb nur mit Bändern.

## Befunde und Korrekturen gegenüber dem Plan

Der freigegebene Linien-Plan sah für Stück 11 vor: Vortraining → **Zero-Shot auf ein neues Depot ohne Historie** → Feintuning; Kante von Stück 6, Kante in Stück 9; Hypothese "nur bei wenig Historie stark".

- **Bestätigt:** Das Zero-Shot-Netz ist bei kurzer Historie den lokalen Verfahren weit überlegen – und bleibt es bis 730 Tagen.
- **Korrigiert (Feintuning):** Der Plan nahm Feintuning als den Schritt, der ein Depot anpasst. Gemessen: **bei kurzer Historie schadet es**, erst ab einem Jahr bringt es etwas (−2 %). Wichtiger ist, *ob man dem Netz überhaupt trauen darf* – deshalb die **Selbstprüfung**, die nicht im Plan stand.
- **Korrigiert (Kante von Stück 6):** Das vortrainierte Netz ist **nicht besser** als das Boosting, das dasselbe Vortraining schon leistet. Sein einziger Vorteil im Test ist ein weniger starker Einbruch bei abweichendem Muster – und auch dort ist Zero-Shot unbrauchbar.
- **Korrigiert (Kante in Stück 9):** Das Mittel aus Netz und Boosting bringt kaum etwas (0,771 gegen 0,777, Seed 3): die beiden Vortraining-Modelle machen ähnliche Fehler.
- **Nicht im Plan, gemessen:** ein zu kleiner Pool ist schlimmer als keiner (3 Depots), und das Vortraining ist kein Vorteil mehr, wenn das Depot dem Pool nicht ähnelt und mit langer Historie lokal geschätzt werden kann.

## Tests

`tests/` prüft das Netz (Vorwärtsrechnung von Hand, alle Ableitungen gegen Differenzenquotienten für L1 und L2 und mehrere Blockzahlen, erster Adam-Schritt von Hand, Lernen einer linearen Abbildung, Momentaufnahmen), die Zeilen und Normierung (von Hand, Größenunabhängigkeit, Grenzen von Vortraining, Historie und Testjahr, keine Zukunft im Fenster), die Auswertung (Kennzahlen von Hand, Zeitgrenzen: das Vortraining hängt nicht von Tagen des Testjahres ab, das Feintuning nur von der Historie, Prognosen eines Ursprungs nicht von späteren Tagen; Selbstprüfung von Hand), die Presets und Permalinks, die App (AppTest: Standard, jedes Preset, Regler, Momentaufnahmen, Permalink-Klemmen und -Einrasten, Extremwerte, vier Experimente auf Abruf) und **jede Zahl dieses READMEs** (`test_claims.py`, Mehr-Seed-Zahlen mit Bändern, Einzelzahlen mit großzügigen Bändern und Strukturgrenzen). 72 Tests, Laufzeit etwa acht Minuten; die CI läuft bei jedem Push und **wöchentlich** (die Abhängigkeiten sind nicht gepinnt).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Vortraining mit Momentaufnahmen, ein neues Depot, Auswertung, vier Experimente auf Abruf, Grenzen, Formeln |
| `ptn_constants.py` | Regler-Grenzen, Netz- und Experimentkonstanten |
| `ptn_scenario.py` | die Depots (Vehikel wie in den Vorgängern, mit abweichendem Wochenmuster) |
| `ptn_net.py` | das Netz: Vorwärts- und Rückwärtsrechnung, Adam |
| `ptn_samples.py` | Fenster, Normierung, Zusatzmerkmale, Zeilenmengen |
| `ptn_evaluation.py` | Vortraining, Zero-Shot, Feintuning, Selbstprüfung, Analyse, vier Experimente |
| `ptn_baselines.py`, `ptn_ets.py` | Wochenmittel und Holt-Winters aus Stück 1 und 2 |
| `ptn_features.py`, `ptn_gbm.py`, `ptn_tree.py` | das vortrainierte Boosting aus Stück 6 |
| `ptn_visualization.py` | Plotly-Abbildungen (alle Achsen gesperrt) |
| `ptn_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Echte Foundation-Modelle in der App und den Tests** (Chronos, TimesFM, Moirai, TimeGPT): Torch und Gewichte zum Herunterladen, CI-Fragilität, ein API-Schlüssel. Nur Chronos-tiny (T5 und Bolt) wurde einmal gemessen (siehe oben); größere Modelle nicht.
- **Vortrainierte Gewichte als Datei.** Das Vortraining läuft in wenigen Sekunden beim ersten Aufruf; die Demo liefert keine Gewichte aus.
- **Mehrere Musterfamilien im Pool, Domänenanpassung, Adapter/LoRA statt vollem Feintuning, frühes Stoppen im Feintuning.**
- **Prognoseintervalle aus dem Netz** (Stück 7 zeigt, wie man sie aus jedem Verfahren gewinnt).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Zeitreihen-Prognose: von Naiv bis Vortraining](https://sebastianhanisch.net/konzepte-zeitreihen-prognose.html).
