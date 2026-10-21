# RL Challenge: Atari Surround Tournament

Kleiner lokaler Prototyp für die Turnierdurchführung. Zwei exportierte PyTorch-Modelle werden hochgeladen und treten in `pettingzoo.atari.surround_v2` gegeneinander an. Das Browser-Frontend zeigt den aktuellen Atari-Frame und den Spielstand.

## Start

```bash
cd rl-challenge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
AutoROM --accept-license
PORT=5001 python app.py
```

Danach im Browser `http://127.0.0.1:5001` öffnen.

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

## Nächste Schritte

- mehrere Spiele pro Duell und Seitenwechsel ergänzen
- Turnierbaum für vier bis acht Gruppen ergänzen
- eingefrorene Seeds und einheitliche Zeitlimits definieren
- Turnierergebnisse als JSON/CSV speichern
- Uploads vor dem Turnier prüfen und danach automatisch löschen
