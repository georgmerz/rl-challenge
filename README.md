# RL Challenge: Atari Pong

In dieser Challenge entwickeln Gruppen eine Policy für **Atari Pong**. Im
Turnier trifft sie auf einen zuvor nicht bekannten Gegner. Ziel ist nicht, die
Spielumgebung zu verändern, sondern eine Policy zu trainieren und als portable
ONNX-Datei einzureichen.

## Architektur: Training ist nicht der Turnier-Runner

| Baustein | Aufgabe |
| --- | --- |
| Arena | [PettingZoo Atari Pong (`pong_v3`)](https://pettingzoo.farama.org/main/environments/atari/pong/) mit zwei Spielern. |
| Browser-Runner | [`app.py`](app.py) lädt zwei ONNX-Modelle, führt die Partie aus, verwaltet Frames, Seiten, Punkte und zeigt die Arena an. |
| Modell | Reine ONNX-Policy-Inferenz: Sie erhält nur `image` und `player_id` und gibt eine Aktion zurück. Sie hat keinen direkten Zugriff auf Environment, Punktestand oder gegnerische Aktion. |
| Training | Eigene, von der Gruppe gewählte Offline-/Trainingsprogramme. Die Beispiele im Repository erzeugen ONNX-Dateien, sind aber nicht Teil des Turnier-Runners. |

Der Runner führt beim lokalen Test **zwei** hochgeladene Modelle gegeneinander
aus. Das bedeutet nicht, dass zwei Dateien abgegeben werden: Pro Gruppe ist
genau eine Modell-Datei vorgesehen; die zweite Datei im Browser dient nur als
Testgegner.

## Lokalen Runner starten

```bash
cd rl-challenge
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
AutoROM --accept-license
PORT=5001 python app.py
```

Anschließend [http://127.0.0.1:5001](http://127.0.0.1:5001) öffnen. Unter
Windows wird die virtuelle Umgebung mit `.venv\Scripts\Activate.ps1`
aktiviert. Der Runner verwendet die CPU-Referenzlaufzeit von ONNX Runtime;
eine GPU ist zum Ausführen eines abgegebenen Modells nicht nötig.

## Modellvertrag

Der Runner akzeptiert genau zwei Eingaben mit genau diesen Namen und eine
Ausgabe mit genau diesem Namen:

```text
image:     float32 [1, 12, 210, 160]
player_id: int64   [1]
action:    int64   [1]    # Wert aus {0, 1, 2, 3, 4, 5}
```

Die Abgabe soll mit **ONNX Opset 17** exportiert werden und höchstens 250 MB
groß sein. Sie wird mit ONNX Runtime auf CPU ausgeführt. Der Runner prüft beim
Laden die Namen, Typen und festen Formen; er prüft das Opset nicht gesondert.
Ein anderes Opset kann zwar zufällig ladbar sein, erfüllt aber nicht die
Abgabevorgabe.

### Beobachtung und Vorverarbeitung im Runner

PettingZoo liefert pro Spieler einen RGB-Frame der Form `210 × 160 × 3`
(`H × W × RGB`, `uint8`). Der Runner hält pro physischem Pong-Spieler die
letzten vier Frames fest:

1. Beim Start einer Runde besteht die History aus vier identischen
   Startframes.
2. Danach wird nach jedem Environment-Schritt der neue Frame angehängt.
3. Der Runner ordnet die vier Frames vom ältesten zum neuesten an, wandelt
   `HWC` nach `NCHW` und stapelt sie zu zwölf Kanälen.
4. Anschließend konvertiert er nach `float32` und dividiert durch `255`; die
   Pixelwerte liegen somit in `[0, 1]`.

Damit belegen die Kanäle `0–2` den ältesten und `9–11` den neuesten RGB-Frame.
Die vier Frames erlauben es einer Policy, Bewegung aus der Bildfolge zu
schätzen.

`player_id` bezeichnet **die physische Seite**, nicht den Upload-Slot im
Browser:

| `player_id` | Seite und Farbe |
| ---: | --- |
| `1` | links / orange |
| `2` | rechts / grün |

Ein Modell muss daher beide Werte verarbeiten können, wenn es auf beiden
Seiten spielen soll. Ein Modell darf `player_id` technisch nicht weglassen,
auch wenn es die Eingabe intern nicht nutzt.

## Aktionen

Die Werte entsprechen dem minimalen Action-Space von PettingZoo Pong. Die
offiziellen Richtungsnamen beziehen sich auf den Atari-Controller und sind
nicht als Bildschirmkoordinaten „oben/unten“ dokumentiert.

| Wert | PettingZoo-/ALE-Bedeutung |
| ---: | --- |
| `0` | `NOOP` – keine Eingabe |
| `1` | `FIRE` – feuern bzw. Aufschlag starten |
| `2` | `RIGHT` – nach rechts bewegen |
| `3` | `LEFT` – nach links bewegen |
| `4` | `FIRE_RIGHT` – feuern und nach rechts bewegen |
| `5` | `FIRE_LEFT` – feuern und nach links bewegen |

Insbesondere bedeutet `3` **nicht** `DOWN`, und `4`/`5` sind nicht
`FIRE_UP`/`FIRE_DOWN`. Bei einem stillstehenden Ball kann `FIRE` den nächsten
Ballwechsel auslösen; der Pong-Aufschlag ist zeitbegrenzt.

## Turnierablauf im Browser-Runner

Ein Duell besteht aus zehn Runden. Die Seeds und die Seiten sind im aktuellen
Runner festgelegt und damit reproduzierbar:

| Runde | Seed | Upload „Spieler 1“ | Upload „Spieler 2“ |
| ---: | ---: | --- | --- |
| 1, 3, 5, 7, 9 | 42, 44, 46, 48, 50 | links / orange (`player_id=1`) | rechts / grün (`player_id=2`) |
| 2, 4, 6, 8, 10 | 43, 45, 47, 49, 51 | rechts / grün (`player_id=2`) | links / orange (`player_id=1`) |

So spielt jedes hochgeladene Modell fünf Runden auf jeder Seite. Pro Runde
gilt:

- Ein positiver Environment-Reward zählt als ein Punkt für das zugehörige
  hochgeladene Modell; anschließend beginnt die nächste Runde.
- Nach maximal **2.000** Environment-Schritten ohne positiven Punkt oder bei
  einer technisch beendeten Runde ohne positiven Punkt ist die Runde ein
  Unentschieden. Beide Uploads erhalten dann je einen Turnierpunkt.
- Ein negativer Reward allein wird vom Runner nicht als eigener Turnierpunkt
  verbucht.
- Nach zehn Runden gewinnt der Upload mit mehr aggregierten Turnierpunkten;
  Gleichstand bedeutet Unentschieden.

Die Browser-Oberfläche zeigt den laufenden Punktestand und die aktuellen
Seiten. Eine Partie lässt sich pausieren, fortsetzen oder stoppen. Beim Stoppen
endet sie ohne nachträgliche Gewinnerbestimmung.

## Abgabe, Prüfung und Kursleistung

### Technische Modell-Abgabe

Pro Gruppe ist **genau eine `.onnx`-Datei** einzureichen. Vor der Abgabe:

```bash
python validate_submission.py model.onnx
```

Der Validator lädt das Modell mit der CPU-Laufzeit, prüft Schnittstelle und
Aktionsbereich mit leeren Testframes für beide Seiten. Er bewertet **nicht**
die Spielstärke und prüft weder Dateigröße noch ONNX-Opset vollständig. Diese
beiden Abgabevorgaben müssen zusätzlich eingehalten werden.

Eine zusätzliche Projekt-README kann hilfreich sein, wird vom Runner aber
nicht verwendet. Der Bericht ist eine separate Kursleistung.

### Bericht und Termine

Die Endnote basiert auf einem Bericht von **5–8 Seiten**. Er soll mindestens
behandeln:

- Spielbeschreibung sowie State/Observation, Aktionen und Reward,
- gewählten RL-Algorithmus und Trainingsaufbau,
- Umgang mit zwei Spielern, unbekanntem Gegner und Seitenwechsel,
- Evaluation der Spielstärke und deren Grenzen.

Am **06.11.2026** ist der Vorbereitungstermin. Am **13.11.2026** stellen die
Gruppen ihre Ergebnisse in **10 Minuten** vor; anschließend findet das
Turnier statt.

## Mitgelieferte Baselines und Trainingsbeispiele

Erst die Baselines erzeugen:

```bash
python create_baselines.py
```

| Datei | Zweck |
| --- | --- |
| `baselines/baseline_pong_static.onnx` | Absichtlich schwache Policy, die nur `FIRE` ausgibt; Upload-/Schnittstellentest. |
| `baselines/baseline_pong_track_ball.onnx` | Deterministische bildbasierte Policy; erkennt Ball und eigenen Schläger über Pixel-Farben und steuert in Ballrichtung. Kein trainiertes Modell. |

### DQN-Minimalbeispiel

[`train_dqn_vanilla.py`](train_dqn_vanilla.py) trainiert Standard-DQN aus
Stable-Baselines3 ausschließlich als linken/orangen Spieler (`player_id=1`)
gegen die feste Ball-Tracking-Baseline. Es beschneidet das Spielfeld,
verwendet Graustufen, `84 × 84` Pixel und vier Trainingsframes. Beim
ONNX-Export ist die dazu passende Vorverarbeitung im Modell enthalten.

```bash
python train_dqn_vanilla.py --timesteps 1000000 \
  --output baselines/baseline_pong_dqn.onnx
```

Das Beispiel ist eine Startvorlage, keine Referenz für eine faire
Turnierevaluation: Sein Trainings-Timeout beträgt 1.000 Schritte statt 2.000
und es sieht nur einen festen Gegner sowie eine Seite. Checkpoints,
langes GPU-Training und Fortsetzen sind in
[SERVER_TRAINING.md](SERVER_TRAINING.md) beschrieben.

### PPO- und Double-DQN-Beispiele

[`train_ppo.py`](train_ppo.py) trainiert eine gemeinsame Policy im Self-Play:
In jedem Trainingsschritt entscheidet dasselbe Netz für beide PettingZoo-
Spieler. Das exportierte ONNX-Modell erfüllt den öffentlichen Vertrag, nutzt
aber nur den neuesten der vier Frames und ignoriert `player_id`; außerdem
folgt sein Episodenende dem PettingZoo-Environment, nicht einer einzelnen
Turnierrunde mit 2.000-Schritte-Limit. Es ist somit ein kompaktes
Self-Play-Beispiel, keine exakte Turnier-Simulation.

[`train_ppo_sb3.py`](train_ppo_sb3.py) ist ein älteres, stärker angepasstes
Double-DQN-Beispiel: ebenfalls nur links/orange gegen die Tracking-Baseline,
mit einem 1.000-Schritte-Trainings-Timeout und eigener Evaluation. Es ist
kompatibel mit dem ONNX-Vertrag, aber ebenfalls nicht mit dem vollständigen
Turnierablauf gleichzusetzen.

Training kann CPU, Apple Silicon, Colab oder einen CUDA-Server verwenden.
Eine GPU kann längere Läufe beschleunigen, ist aber weder für alle Gruppen
noch für die Ausführung der Abgabe erforderlich. Das Colab-Notebook
[`train_ppo_colab.ipynb`](train_ppo_colab.ipynb) ist eine weitere
Startumgebung; die exportierte ONNX-Datei muss anschließend lokal validiert
werden.

## Offene methodische Entscheidungen

Folgende Entscheidungen sind absichtlich nicht durch den Runner festgelegt und
sollten im Bericht begründet werden:

- State-Design: vollständige Bilder, gestapelte Frames oder zulässige intern
  aus Bildern abgeleitete Merkmale;
- Reward Shaping beim Training;
- feste Gegner, Self-Play oder ein Gegner-Pool;
- Training für beide Seiten und der Umgang mit `player_id`;
- faire Evaluation, etwa über mehrere Seeds, beide Seiten und mehrere Gegner.

Das Browser-Turnier deckt bereits zehn feste Runden und beide Seiten ab. Für
eine belastbare Aussage über Spielstärke empfiehlt sich zusätzlich eine eigene
Evaluation über mehr Seeds und unterschiedliche Gegner.

## Kleinstes gültiges Exportbeispiel

Dieses Modell ist nur ein technischer Test und nicht spielstark:

```python
import torch


class Policy(torch.nn.Module):
    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        action = torch.ones((image.shape[0],), dtype=torch.int64, device=image.device)
        return action + player_id * 0 + image[:, 0, 0, 0].to(torch.int64) * 0


policy = Policy().eval()
image = torch.zeros((1, 12, 210, 160), dtype=torch.float32)
player_id = torch.ones((1,), dtype=torch.int64)
torch.onnx.export(
    policy,
    (image, player_id),
    "my_policy.onnx",
    input_names=["image", "player_id"],
    output_names=["action"],
    opset_version=17,
)
```

Danach `python validate_submission.py my_policy.onnx` ausführen und die Datei
im Browser gegen ein zweites Testmodell spielen lassen.
