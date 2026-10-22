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

Wählt dort zwei `.pt2`-Modelle aus. Spieler 1 startet links (grün), Spieler 2
rechts (blau). Eine Partie endet, sobald ein Spieler 10 Punkte erreicht.

## Eure Aufgabe

Entwickelt eine **Policy**, die aus einem einzelnen aktuellen Bildschirmbild
eine Lenkaktion für den eigenen Spieler auswählt. Ziel ist es, länger als der
Gegner zu überleben: Wer mit seinem Kopf den Spielfeldrand oder eine bereits
gezeichnete Spur berührt, verliert die Runde; der andere Spieler erhält einen
Punkt. Das Turnier spielt so viele Runden, bis ein Modell **10 Punkte**
erreicht. Dieses Modell gewinnt das Duell.

Ihr gebt ausschließlich eine exportierte PyTorch-Datei (`.pt2`) ab. Innerhalb
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

Reicht genau **eine** Datei mit der Endung `.pt2` ein; sie darf maximal
250 MB groß sein. Die Datei muss mit `torch.export.save` erzeugt worden sein.
Pickle-basierte Exporte wie `torch.save(model, "model.pt")` akzeptiert der
Runner bewusst nicht. Beim Laden setzt der Runner das Modell in den
Auswertungsmodus (`eval()`), daher müssen alle zum Entscheiden nötigen
Gewichte bereits in der exportierten Datei enthalten sein.

## Festgelegte Modell-Schnittstelle

Die Gruppe exportiert eine Policy als PyTorch-Exportdatei:

```python
exported = torch.export.export(policy, (image, player_id))
torch.export.save(exported, "model.pt2")
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
baselines/baseline_straight.pt2
baselines/baseline_wall_avoid.pt2
```

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

`player_id` darf in der Policy ignoriert werden, wenn das Modell die eigene Seite aus dem Bild ableitet. Die exportierte Methode muss ihn trotzdem als zweiten Parameter akzeptieren.

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
        return torch.where(player_id == 1, action_p1, action_p2)


policy = StraightPolicy().eval()
example_image = torch.zeros((1, 3, 210, 160), dtype=torch.float32)
example_player_id = torch.ones((1,), dtype=torch.int64)
exported = torch.export.export(policy, (example_image, example_player_id))
torch.export.save(exported, "my_policy.pt2")
```

Ladet anschließend `my_policy.pt2` als eines der beiden Modelle im Browser hoch.
Für ein bildbasiertes Beispiel mit einer einfachen Wandvermeidung könnt ihr
auch `create_baselines.py` lesen oder ausführen.

## Nächste Schritte

- mehrere Spiele pro Duell und Seitenwechsel ergänzen
- Turnierbaum für vier bis acht Gruppen ergänzen
- eingefrorene Seeds und einheitliche Zeitlimits definieren
- Turnierergebnisse als JSON/CSV speichern
- Uploads vor dem Turnier prüfen und danach automatisch löschen
