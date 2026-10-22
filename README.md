# RL Challenge: Atari Surround Tournament

In dieser Challenge entwickeln Teams eine Policy für das Atari-Spiel **Surround**.
Zwei exportierte PyTorch-Modelle treten anschließend im lokalen Turnier-Runner
gegeneinander an. Das Browser-Frontend zeigt das Spielfeld und den Spielstand.

## Lokalen Runner starten

```bash
cd rl-challenge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
AutoROM --accept-license
PORT=5001 python app.py
```

Danach im Browser `http://127.0.0.1:5001` öffnen.

Unter Windows in PowerShell lautet die Aktivierung der virtuellen Umgebung
`.venv\\Scripts\\Activate.ps1`; falls Port 5000 belegt ist, startet ihr mit
`$env:PORT=5001; python app.py`.

Wählt dort zwei `.onnx`-Modelle aus. Ein Duell besteht aus genau zehn Runden.
In Runde 1 startet Spieler 1 links (grün) und Spieler 2 rechts (blau); danach
werden die Seiten in jeder Runde getauscht. Der Endstand nach zehn Runden
entscheidet das Duell.

## Eure Aufgabe

Entwickelt eine **Policy**, die aus einem einzelnen aktuellen Bildschirmbild
eine Lenkaktion für den eigenen Spieler auswählt. Ziel ist es, länger als der
Gegner zu überleben: Wer mit seinem Kopf den Spielfeldrand oder eine bereits
gezeichnete Spur berührt, verliert die Runde; der andere Spieler erhält einen
Punkt. Jedes Duell besteht aus **zehn Runden**. Spieler 1 und Spieler 2
wechseln nach jeder Runde die Seiten, damit beide fünfmal links/grün und
fünfmal rechts/blau spielen. Wer nach zehn Runden mehr Punkte hat, gewinnt;
bei gleichem Punktestand endet das Duell unentschieden.

Wenn beide Figuren in derselben Runde gleichzeitig kollidieren, ist die Runde
unentschieden und **beide Spieler erhalten einen Punkt**. Eine laufende Partie
kann im Browser pausiert, fortgesetzt oder gestoppt werden.

Ihr gebt ausschließlich eine ONNX-Datei (`.onnx`) ab. Innerhalb
des Modells dürft ihr Architektur, trainierte Gewichte und Bildverarbeitung
frei wählen. Der Runner übergibt pro Spielschritt die Eingaben, ruft eure
Methode `forward(image, player_id)` auf und führt die zurückgegebene Aktion
aus. Es gibt keinen Zugriff auf das Environment, den Punktestand, die Aktionen
des Gegners oder vorherige Frames.

Die erste Runde startet mit Seed `42`; nach einem Punkt startet die nächste
Runde mit einem weiteren festen Seed. Ein Duell wird außerdem nach höchstens
50.000 Schritten beendet. Ein nicht ladbares oder unzulässiges Modell bricht
die Partie ab.

## Zustände / Beobachtungen

Bei jedem Zeitschritt erhält eure Policy genau den aktuellen Atari-Frame und
ihre eigene Spielerrolle:

| Eingabe | Typ und Form | Bedeutung |
| --- | --- | --- |
| `image` | `torch.float32`, exakt `[1, 3, 210, 160]` | Aktuelles RGB-Bild im NCHW-Format. Die Pixelwerte sind bereits auf `[0, 1]` normiert. |
| `player_id` | `torch.int64`, exakt `[1]` | `1` für den linken/grünen Spieler, `2` für den rechten/blauen Spieler. |

Wichtig: Die Dimensionen des Originalbilds sind `210 × 160 × 3` in der
Reihenfolge Höhe × Breite × RGB. Der Runner wandelt es für das Modell nach
`[Batch, Kanal, Höhe, Breite]` um. Es gibt nur einen einzelnen Frame: Wenn ihr
Bewegungsrichtung oder Geschwindigkeit schätzen möchtet, müsst ihr das über
Bildmerkmale lösen; eine explizite Historie wird nicht übergeben.

Im Bild sind beide Spielfiguren und ihre bereits gezogenen Spuren sichtbar.
Eine Kollision mit einer Spur oder dem Rand führt zum Punkt für den Gegner.
Die `1` in der ersten Dimension von `image` ist die Batch-Größe: Der Runner
fragt immer genau eine Beobachtung gleichzeitig ab.

## Aktionen

Euer Modell muss pro Schritt genau **eine** ganzzahlige Atari-Aktion ausgeben:

| Wert | Aktion | Bedeutung |
| ---: | --- | --- |
| `0` | `NOOP` | Keine Eingabe / aktuelle Bewegung fortsetzen |
| `1` | `UP` | Nach oben lenken |
| `2` | `RIGHT` | Nach rechts lenken |
| `3` | `LEFT` | Nach links lenken |
| `4` | `DOWN` | Nach unten lenken |

Die Ausgabe muss ein `int64`-Tensor mit **genau einem Element** sein, etwa
`tensor([2], dtype=torch.int64)`. Auch `tensor(2, dtype=torch.int64)` ist
zulässig, weil dieser Tensor ebenfalls nur ein Element enthält. Andere
Datentypen, mehrere Werte und Nummern außerhalb von `0` bis `4` werden vom
Runner abgewiesen.

## Abgabeformat

Reicht genau **eine** Datei mit der Endung `.onnx` ein; sie darf maximal
250 MB groß sein. Das Modell muss ONNX-Opset **17** verwenden und genau die
Eingaben `image`, `player_id` sowie die Ausgabe `action` enthalten. Der Runner
lädt das Modell ausschließlich mit ONNX Runtime auf CPU; Python- oder
PyTorch-Archive wie `.pt`, `.pt2` oder Pickle-Dateien werden nicht akzeptiert.

### Plattformübergreifende Abgabe

Trainiert werden darf auf Mac, Windows, Linux oder Google Colab. Für die
Bewertung zählt aber ausschließlich, ob die ONNX-Datei im Referenz-Runner der
Lehrveranstaltung auf dem **Mac des Veranstalters** und dort auf der **CPU**
läuft. Eine NVIDIA-GPU, Apple Silicon oder eine GPU überhaupt ist für das
Turnier nicht erforderlich.

Exportiert ein PyTorch-Modell so:

```python
torch.onnx.export(
    policy.eval(), (image, player_id), "model.onnx",
    input_names=["image", "player_id"], output_names=["action"],
    opset_version=17,
)
```

Die PPO-Vorlage und das Colab-Notebook erzeugen dieses Format automatisch.
Die Policy darf in `forward` weder `"mps"` noch `"cuda"` fest eintragen und
keine GPU voraussetzen. Neue Tensoren müssen auf dem Gerät der Eingabe erzeugt
werden, zum Beispiel mit `device=image.device`.

**Verbindlicher Vorabtest:** Vor der Abgabe ausführen:

```bash
python validate_submission.py model.onnx
```

Der Validator verwendet dieselbe CPU-ONNX-Laufzeit und prüft Dateiformat,
Eingaben, Ausgabe und Aktionsbereich. Nur ein Modell mit `OK` ist abgabefähig.

## Festgelegte Modell-Schnittstelle

Die Gruppe exportiert eine Policy als ONNX-Datei:

```python
torch.onnx.export(
    policy.eval(), (image, player_id), "model.onnx",
    input_names=["image", "player_id"], output_names=["action"],
    opset_version=17,
)
```

Der Runner übergibt:

```text
image:      float32 Tensor [1, 3, 210, 160], Wertebereich [0, 1]
player_id:  int64 Tensor [1], Wert 1 (linker/grüner Spieler) oder 2 (rechter/blauer Spieler)
```

Das Modell gibt genau einen Skalar zurück:

```text
action: int64 Tensor mit einem Wert aus {0, 1, 2, 3, 4}
```

Die Aktionen sind:

```text
0: NOOP
1: UP
2: RIGHT
3: LEFT
4: DOWN
```

Die Vorverarbeitung ist im Runner standardisiert. Das Modell erhält keinen direkten Environment-Zugriff und keine Spielhistorie.

## Baselines

```bash
python create_baselines.py
```

Erzeugt:

```text
baselines/baseline_straight.onnx
baselines/baseline_wall_avoid.onnx
baselines/baseline_wall_counter.onnx
```

`baseline_wall_counter.onnx` ist eine kurze bildbasierte Gegenstrategie für
`baseline_wall_avoid.onnx`. Sie lokalisiert die eigene grüne bzw. blaue Spur
im Bild und wählt abhängig von deren Position unterschiedliche Richtungen.
Sie ist kein gelerntes Modell, sondern ein kompaktes Beispiel dafür, wie ein
ONNX-Modell auf den Bildinhalt reagieren kann. Im lokalen Referenztest gewinnt
sie gegen `baseline_wall_avoid` auf beiden Seiten.

### PPO-Selbstspiel-Baseline trainieren

`train_ppo_sb3.py` verwendet die etablierte Bibliothek
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) und trainiert
eine CNN-Policy mit PPO. Dies ist absichtlich ein Minimalbeispiel: Die Policy
trainiert nur als Spieler 1 auf der linken/grünen Seite gegen die feste,
bildbasierte `baseline_wall_avoid` (Spieler 2, rechts/blau). Das CNN ist bewusst klein (etwa 50.000 Gewichte), damit Training
und Experimentieren auf einem Laptop zügig bleiben. Nach dem Training wird
automatisch eine turnierkompatible Datei exportiert.

Eine Trainingsrunde ist technisch auf 1.000 Schritte begrenzt; ein Timeout
zählt dabei als neutraler Reward `0`, genau wie im Turnier. Alle 5.000 Schritte druckt das Skript außerdem eine
**deterministische** Evaluation gegen `baseline_wall_avoid`, einschließlich der
verwendeten Aktionen. Das entspricht der `argmax`-Policy, die als ONNX-Datei
exportiert wird; der während PPO angezeigte Rollout-Reward allein genügt nicht
als Qualitätsmaß.

```bash
python train_ppo_sb3.py --timesteps 100000 --output baselines/baseline_ppo.onnx
```

### Training in Google Colab

Das Notebook
[train_ppo_colab.ipynb](train_ppo_colab.ipynb) kann direkt in Colab geöffnet
werden: [In Google Colab starten](https://colab.research.google.com/github/georgmerz/rl-challenge/blob/main/train_ppo_colab.ipynb).
Vor dem Ausführen in Colab unter **Runtime → Change runtime type** möglichst
**T4 GPU** auswählen. Mit **Run all** werden Repository und Abhängigkeiten
installiert und das Training gestartet. Der jeweils aktuelle Export liegt in
`baselines/baseline_ppo.onnx` und wird im letzten Notebook-Schritt
heruntergeladen.

Die PPO-Baseline verwendet beim Export ONNX-Opset 17. Dadurch wird die Datei
unabhängig von der PyTorch-Version des Trainingssystems auf dem Referenz-Mac
mit ONNX Runtime ausgeführt.

Auf einem Mac mit Apple Silicon verwendet das Skript automatisch den MPS-
Beschleuniger des M-Chips; in Colab wird bei aktivierter GPU automatisch CUDA
verwendet. Falls nötig, lässt sich die CPU erzwingen:

```bash
python train_ppo_sb3.py --device cpu --timesteps 100000
```

Das Skript überschreibt `baselines/baseline_ppo.onnx` automatisch nach jeweils
1.000 Trainingsschritten. Dadurch kann das bis dahin trainierte Modell bereits
während des laufenden Trainings im Turnier-Runner getestet werden. Der Abstand
kann bei Bedarf geändert werden:

```bash
python train_ppo_sb3.py --export-every 500 --output baselines/baseline_ppo.onnx
```

Für einen kurzen technischen Test verwendet das Skript mindestens einen
Rollout mit 128 Schritten; die so erzeugte Policy ist noch nicht sinnvoll
trainiert:

```bash
python train_ppo_sb3.py --timesteps 128 --output baselines/test_ppo.onnx
```

Für ein echtes Turniermodell muss diese Vorlage erweitert werden, weil im
Turnier nach jeder Runde die Seiten getauscht werden. Sinnvolle nächste Schritte
sind daher: auch Spieler 2 trainieren, einen stärkeren Gegner verwenden oder
Selbstspiel einsetzen. `train_ppo.py` bleibt als reine PyTorch-Selbstspiel-
Variante zum Nachlesen verfügbar.

## Beispiel für eine exportierbare Policy

```python
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 24 * 18, 5),
        )

    def forward(self, image, player_id):
        logits = self.net(image)
        return torch.argmax(logits, dim=1).to(torch.int64)
```

`player_id` darf für die Entscheidung ignoriert werden, muss aber im ONNX-Graph
erhalten bleiben. Falls er sonst nicht benötigt wird, addiert `player_id * 0`
zur Ausgabe.

## Minimalbeispiel: feste Startstrategie

Dieses Modell ist bewusst sehr einfach: Der grüne Spieler fährt nach rechts,
der blaue nach links. Es ist ein guter Test, ob Export und Upload funktionieren,
aber keine konkurrenzfähige Policy.

```python
from pathlib import Path
import torch


class StraightPolicy(torch.nn.Module):
    def forward(self, image: torch.Tensor, player_id: torch.Tensor) -> torch.Tensor:
        # Spieler 1 (grün, links) fährt nach rechts, Spieler 2 nach links.
        action_p1 = torch.full((image.shape[0],), 2, dtype=torch.int64, device=image.device)
        action_p2 = torch.full((image.shape[0],), 3, dtype=torch.int64, device=image.device)
        return torch.where(player_id == 1, action_p1, action_p2) + image[:, 0, 0, 0].to(torch.int64) * 0


policy = StraightPolicy().eval()
example_image = torch.zeros((1, 3, 210, 160), dtype=torch.float32)
example_player_id = torch.ones((1,), dtype=torch.int64)
torch.onnx.export(
    policy, (example_image, example_player_id), "my_policy.onnx",
    input_names=["image", "player_id"], output_names=["action"],
    opset_version=17,
)
```

Prüft anschließend `my_policy.onnx` mit `python validate_submission.py my_policy.onnx`
und ladet es danach im Browser hoch.
Für ein bildbasiertes Beispiel mit einer einfachen Wandvermeidung könnt ihr
auch `create_baselines.py` lesen oder ausführen.

## Nächste Schritte

- mehrere Spiele pro Duell und Seitenwechsel ergänzen
- Turnierbaum für vier bis acht Gruppen ergänzen
- eingefrorene Seeds und einheitliche Zeitlimits definieren
- Turnierergebnisse als JSON/CSV speichern
- Uploads vor dem Turnier prüfen und danach automatisch löschen
