# RL Challenge: Atari Pong Tournament

In dieser Challenge entwickeln Teams eine Policy für **Atari Pong**. Zwei
eingereichte ONNX-Modelle spielen im lokalen Browser-Runner gegeneinander.
Pong ist bewusst die zugänglichere Alternative zu Surround: Ein Punkt liefert
sofort ein klares positives bzw. negatives Lernsignal.

## Lokalen Runner starten

```bash
cd rl-challenge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
AutoROM --accept-license
PORT=5001 python app.py
```

Danach `http://127.0.0.1:5001` öffnen. Unter Windows aktiviert ihr die
Umgebung mit `.venv\Scripts\Activate.ps1`.

## Aufgabe und Turnierregeln

Eine Policy steuert einen Pong-Schläger ausschließlich aus dem aktuellen
Bild. Sie soll den Ball am eigenen Schläger abfangen und am gegnerischen
vorbeispielen. Erreicht der Ball eine Seite, erhält der erfolgreiche Spieler
einen Punkt; die Runde endet sofort und das nächste Spiel beginnt.

Ein Duell besteht aus **zehn Runden**. In ungeraden Runden ist Spieler 1
links/orange und Spieler 2 rechts/grün; in geraden Runden werden die Seiten
getauscht. Damit spielt jedes Team fünfmal auf jeder Seite.

Ein Ballwechsel darf höchstens **2.000 Schritte** dauern. Falls bis dahin kein
Punkt erzielt wurde, endet die Runde unentschieden und beide Spieler erhalten
einen Punkt. Auch eine technisch beendete Runde ohne Punkt wird als
Unentschieden behandelt. Eine Partie kann im Browser pausiert, fortgesetzt
oder gestoppt werden.

## Beobachtungen

Bei jedem Schritt erhält das Modell die letzten **vier** Atari-Frames (vom
ältesten zum neuesten) und die eigene Rolle. Es gibt keinen Zugriff auf das
Environment, den Punktestand oder die gegnerischen Aktionen.

| Eingabe | Typ und Form | Bedeutung |
| --- | --- | --- |
| `image` | `float32`, `[1, 12, 210, 160]` | Vier RGB-Bilder im NCHW-Format: Kanäle 0–2 sind der älteste, 9–11 der neueste Frame. Pixelwerte sind auf `[0, 1]` normiert. |
| `player_id` | `int64`, `[1]` | `1`: linker/oranger Schläger, `2`: rechter/grüner Schläger. |

Jeder Originalframe hat die Form `210 × 160 × 3` in H×B×RGB. Der Runner
wandelt vier aufeinanderfolgende Frames in NCHW um und stapelt sie als zwölf
Kanäle. Der Ball und beide Schläger sind sichtbar. Dadurch kann eine Policy
aus der Ballbewegung die ungefähre Flugrichtung lernen. Beim Start einer Runde
werden zunächst vier gleiche Startframes geliefert.

## Aktionen

Das Modell gibt genau eine Atari-Aktion als `int64`-Tensor mit einem Element
aus. In der dargestellten Pong-Arena bewegen die Aktionen 2 und 3 den
Schläger zur oberen bzw. unteren Bildschirmkante:

| Wert | Aktion | Bedeutung |
| ---: | --- | --- |
| `0` | `NOOP` | Keine neue Eingabe. |
| `1` | `FIRE` | Aufschlag auslösen. |
| `2` | `UP` | Schläger nach oben bewegen. |
| `3` | `DOWN` | Schläger nach unten bewegen. |
| `4` | `FIRE_UP` | Aufschlag und nach oben bewegen. |
| `5` | `FIRE_DOWN` | Aufschlag und nach unten bewegen. |

Der Aufschlag-Timer von Pong verhindert zusätzlich, dass ein Spieler durch
Nichtstun dauerhaft blockiert. Ausgaben außerhalb von `0` bis `5`, falsche
Datentypen oder mehrere Werte werden abgewiesen.

## Abgabeformat

Reicht genau **eine** `.onnx`-Datei ein, höchstens 250 MB groß. Sie muss mit
ONNX-Opset 17 exportiert sein und exakt diese Schnittstelle haben:

```text
image:      float32 [1, 12, 210, 160]  (4 RGB-Frames, ältester zuerst)
player_id:  int64 [1]
action:     int64 [1], Wert aus {0, 1, 2, 3, 4, 5}
```

Die Bewertung verwendet ausschließlich ONNX Runtime auf CPU. Trainieren darf
man auf Mac, Windows, Linux oder Google Colab; die Abgabedatei darf aber keine
GPU voraussetzen.

```python
torch.onnx.export(
    policy.eval(), (image, player_id), "model.onnx",
    input_names=["image", "player_id"], output_names=["action"],
    opset_version=17,
)
```

Vor der Abgabe immer testen:

```bash
python validate_submission.py model.onnx
```

Nur eine Ausgabe mit `OK` ist turnierkompatibel.

## Baselines

```bash
python create_baselines.py
```

Dies erzeugt:

```text
baselines/baseline_pong_static.onnx
baselines/baseline_pong_track_ball.onnx
```

`baseline_pong_static.onnx` drückt nur `FIRE` und ist ein absichtlich schwacher
Upload-Test. `baseline_pong_track_ball.onnx` erkennt Ball und eigenen Schläger
über deren Farben und bewegt den Schläger zur Höhe des Balls. Es ist eine
deterministische, bildbasierte Beispielpolicy und kein trainiertes Modell.

## Minimalbeispiel: Vanilla DQN gegen die Ball-Tracking-Baseline

`train_dqn_vanilla.py` ist die kompakte Startvorlage mit
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Sie verwendet
ohne eigene Algorithmusänderungen Standard-`DQN` mit Standard-`CnnPolicy` und trainiert
nur Spieler 1 (links/orange) gegen die feste
`baseline_pong_track_ball.onnx`. Es reproduziert die üblichen Atari-DQN-
Vorverarbeitungsschritte: Spielfeld-Ausschnitt, Graustufen, `84 × 84` und vier
gestapelte Frames. Beim Export ist dieselbe Vorverarbeitung im ONNX-Modell
enthalten.

Pong liefert beim Punkt bereits den echten Lernreward `+1` (gewonnen) bzw.
`-1` (verloren); ein Trainings-Timeout liefert `0`. Die Evaluation zählt die
deterministische ONNX-Policy gegen denselben Gegner. Diese Gewinn-/Verlustzahl
ist wichtiger als ein einzelner Loss-Wert.

```bash
python train_dqn_vanilla.py --timesteps 1000000 \
  --output baselines/baseline_pong_dqn.onnx
```

Die Vorlage übernimmt die zentralen Schritte des
[CleanRL-Atari-DQN-Beispiels](https://docs.cleanrl.dev/rl-algorithms/dqn/):
Graustufen, 84×84 Pixel, vier Frames, Replay-Buffer, Target-Netz und
Epsilon-Greedy-Exploration. CleanRL trainiert Atari Pong für **10 Millionen
Entscheidungsschritte** (bei Frame-Skip 4 sind das 40 Millionen Rohframes).
Für dieses einfachere Duell gegen eine feste Baseline ist 1 Million Schritte
ein sinnvoller erster Colab-Lauf; 20.000 Schritte testen nur Installation und
Export.

Für einen längeren Linux-/NVIDIA-Serverlauf mit Checkpoints und Download gibt
es die Anleitung [SERVER_TRAINING.md](SERVER_TRAINING.md).

Das Skript exportiert standardmäßig nach jeweils 1.000 Schritten einen
Checkpoint. Ein Testlauf prüft nur die Technik, nicht die Spielstärke:

```bash
python train_dqn_vanilla.py --timesteps 128 --output baselines/test_pong_dqn.onnx
```

Für ein echtes Turniermodell sollte die Vorlage erweitert werden, damit auch
die rechte/grüne Rolle lernt oder Selbstspiel verwendet wird. Die frühere,
etwas stärker angepasste Double-DQN-Variante bleibt in `train_ppo_sb3.py`;
`train_ppo.py` enthält eine PPO-Selbstspiel-Variante für Pong.

## Training in Google Colab

Das Notebook [train_ppo_colab.ipynb](train_ppo_colab.ipynb) kann direkt in
Colab geöffnet werden: [In Google Colab starten](https://colab.research.google.com/github/georgmerz/rl-challenge/blob/main/train_ppo_colab.ipynb).
Unter **Runtime → Change runtime type** möglichst eine T4-GPU auswählen und
anschließend **Run all** ausführen. Das Notebook installiert die ROMs,
trainiert und lädt die ONNX-Datei herunter.

Auf Apple Silicon verwendet das Skript automatisch MPS, in Colab CUDA und
sonst CPU. CPU kann bei Bedarf erzwungen werden:

```bash
python train_dqn_vanilla.py --device cpu --timesteps 1000000
```

## Kleinstes exportierbares Modell

```python
import torch


class Policy(torch.nn.Module):
    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # Schwacher, aber gültiger Aufschlag-Test.
        action = torch.ones((image.shape[0],), dtype=torch.int64, device=image.device)
        return action + player_id * 0 + image[:, 0, 0, 0].to(torch.int64) * 0


policy = Policy().eval()
image = torch.zeros((1, 12, 210, 160), dtype=torch.float32)
player_id = torch.ones((1,), dtype=torch.int64)
torch.onnx.export(
    policy, (image, player_id), "my_policy.onnx",
    input_names=["image", "player_id"], output_names=["action"],
    opset_version=17,
)
```

Danach mit `python validate_submission.py my_policy.onnx` prüfen und im
Browser hochladen.
