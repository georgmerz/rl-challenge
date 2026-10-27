# RL Challenge: Atari Pong

**Deutsch** · [English version](README_EN.md)

In dieser Challenge entwickeln Sie in **Zweiergruppen** eine Policy für
**Atari Pong**. Im Turnier trifft sie auf einen zuvor nicht bekannten Gegner.
Sie verändern nicht die Spielumgebung, sondern trainieren eine Policy, reichen
sie als portable ONNX-Datei ein und präsentieren Vorgehen und Ergebnisse in
einem zehnminütigen Vortrag.

![Animierte Atari-Pong-Szene aus der PettingZoo-Dokumentation](https://pettingzoo.farama.org/_images/atari_pong.gif)

*Abbildung: Atari Pong aus der [offiziellen PettingZoo-Pong-Dokumentation](https://pettingzoo.farama.org/main/environments/atari/pong/), Farama Foundation.*

## Auf einen Blick

| Was? | Vorgabe |
| --- | --- |
| Team | Zweiergruppe; beide Mitglieder stimmen Methode, Experimente und Vortrag gemeinsam ab. |
| Technische Abgabe | Genau eine `.onnx`-Datei pro Gruppe, höchstens 250 MB, ONNX Opset 17. |
| Lokaler Test | Zwei ONNX-Modelle im Browser gegeneinander spielen lassen. Das zweite Modell ist nur ein Testgegner, keine zweite Abgabe. |
| Bericht | 5–8 Seiten als Grundlage der Endnote. |
| Vorbereitung | 30.10.2026 (diesen Freitag). |
| Präsentation und Turnier | 13.11.2026; 10 Minuten pro Gruppe, beide Gruppenmitglieder beteiligen sich. |

Der Ablauf der Challenge ist damit:

```text
Browser-Runner und Baselines lokal ausführen
        ↓
Regeln, Modellschnittstelle und ONNX-Export verstehen
        ↓
Vanilla-DQN auf Server oder in Colab starten
        ↓
Eigene Experimente in Zweiergruppen
        ↓
Bericht, 10-Minuten-Vortrag und Turnier
```

## 1. Gemeinsam starten: Browser-Runner und Baselines

Im ersten gemeinsamen Schritt bringen alle den Browser-Runner **lokal** zum
Laufen und betrachten die drei mitgelieferten Baselines. Dafür ist keine GPU
erforderlich. Das eigentliche Training folgt später getrennt auf einem
GPU-Server oder in Google Colab.

```text
lokal installieren → Baselines validieren → Duell im Browser beobachten
```

### Lokale Einrichtung von null

Für den lokalen Runner werden drei Programme benötigt:

| Programm | Installation und Zweck |
| --- | --- |
| **Git** | Von [git-scm.com/downloads](https://git-scm.com/downloads/) installieren; lädt und aktualisiert das Repository. |
| **Python 3.11** | Von [python.org](https://www.python.org/downloads/) installieren; nicht unbemerkt eine andere Python-Version verwenden. Unter Windows die Option zum Hinzufügen zu `PATH` aktivieren. |
| **Visual Studio Code** | Optional von [code.visualstudio.com](https://code.visualstudio.com/download) installieren; Editor und integriertes Terminal. Jeder andere Editor und jedes Terminal funktionieren ebenfalls. |

Danach Terminal beziehungsweise PowerShell öffnen und die Installation prüfen.

macOS und Linux:

```bash
git --version
python3.11 --version
```

Windows PowerShell:

```powershell
git --version
py -3.11 --version
```

Beide Befehle müssen eine Versionsnummer ausgeben. Anschließend das öffentliche
Repository herunterladen; dafür ist kein GitHub-Konto erforderlich:

```bash
git clone https://github.com/georgmerz/rl-challenge.git
cd rl-challenge
```

Den Ordner nun in VS Code über **File → Open Folder** öffnen. Falls der
`code`-Befehl eingerichtet ist, genügt alternativ `code .`.

#### Virtuelle Python-Umgebung unter macOS oder Linux

Die virtuelle Umgebung hält die Pakete dieses Projekts von anderen
Python-Projekten getrennt:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --prefer-binary -r requirements-runner.txt
AutoROM --accept-license
```

#### Virtuelle Python-Umgebung unter Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --prefer-binary -r requirements-runner.txt
AutoROM --accept-license
```

Falls PowerShell die Aktivierung blockiert, kann für dieses eine Terminal
zuvor `Set-ExecutionPolicy -Scope Process Bypass` ausgeführt werden. Bei jeder
späteren Arbeit in einem neuen Terminal muss nur die Umgebung erneut aktiviert
werden; Pakete und ROM werden nicht jedes Mal installiert.

Dieser Test prüft Installation und Pong-ROM:

```bash
python -c "from pettingzoo.atari import pong_v3; env = pong_v3.parallel_env(); env.reset(seed=42); env.close(); print('Setup und Pong-ROM: OK')"
```

Die letzte Zeile muss `Setup und Pong-ROM: OK` lauten. Schlägt bereits
`git`, `python3.11` beziehungsweise `py -3.11` fehl, zuerst die jeweilige
Installation oder die `PATH`-Einstellung korrigieren.

### Gemeinsamer Durchlauf in der Vorlesung

#### 1. Drei mitgelieferte Policies validieren

```bash
python validate_submission.py baselines/baseline_pong_static.onnx
python validate_submission.py baselines/baseline_pong_reactive.onnx
python validate_submission.py baselines/baseline_pong_track_ball.onnx
```

Alle Validator-Aufrufe müssen mit `OK` enden. Dabei wird bereits der zentrale
Unterschied sichtbar: Ein Modell kann technisch korrekt und trotzdem
spielerisch schwach sein.

#### 2. Ein Duell im Browser ausführen

```bash
PORT=5001 python app.py
```

Unter Windows PowerShell lautet derselbe Startbefehl
`$env:PORT=5001; python app.py`.
Anschließend [http://127.0.0.1:5001](http://127.0.0.1:5001) öffnen und diese
Modelle auswählen:

| Datei | Zweck |
| --- | --- |
| `baselines/baseline_pong_static.onnx` | Absichtlich schwache Policy, die nur `FIRE` ausgibt; Upload-/Schnittstellentest. |
| `baselines/baseline_pong_reactive.onnx` | Mittelstarke Policy: Sie verfolgt den Ball erst auf der eigenen Spielfeldhälfte und eignet sich als leichterer Trainingsgegner. |
| `baselines/baseline_pong_track_ball.onnx` | Deterministische bildbasierte Policy: Sie erkennt Ball und eigenen Schläger über Pixel-Farben und steuert in Ballrichtung. Kein trainiertes Modell. |

Gemeinsam sollten einmal **Start**, **Pause**, **Fortsetzen** und **Stop**
ausprobiert sowie Seitenwechsel und Punktestand beobachtet werden. Damit ist
der lokale Teil abgeschlossen. Der erste kurze Trainings- und Exportlauf
folgt getrennt auf dem Server oder in Colab in
[Abschnitt 4](#4-vanilla-dqn-auf-dem-server-oder-in-colab-trainieren).

Nach diesem Durchlauf sollten alle Teilnehmenden:

- eine vorhandene ONNX-Policy validieren und im Runner laden können,
- den Unterschied zwischen technischer Gültigkeit und Spielstärke erklären,
- `image`, `player_id` und `action` im Modellvertrag zuordnen können und
- Training und lokalen Turnierlauf klar voneinander unterscheiden können.

Wenn ein Schritt scheitert, zuerst prüfen: Ist die virtuelle Umgebung aktiv?
Wurde `requirements-runner.txt` installiert? Wurde `AutoROM --accept-license`
ausgeführt? Ist Port 5001 belegt, kann stattdessen etwa `PORT=5002` verwendet
werden.

## 2. Regeln, Modellspezifikation und Export

Eine Policy soll den Ball mit dem eigenen Schläger abfangen und am Gegner
vorbeispielen. Der Gegner im eigentlichen Turnier ist während des Trainings
nicht bekannt. Das Browser-Turnier vergleicht deshalb zwei unabhängig
hochgeladene Modelle.

> **Verbindliche Turnierregeln:** [`app.py`](app.py) ist die maßgebliche
> Implementierung für Rundenanzahl, Seitenwechsel, Seeds, Timeout, Punkte,
> Gewinner und die ONNX-Schnittstelle. PettingZoo/ALE definiert dagegen die
> eigentliche Pong-Spielmechanik, Aktionen und Environment-Rewards. Regeln
> eines Trainingsbeispiels (etwa ein Trainings-Timeout) gelten nicht
> automatisch für das Turnier.

### Regeln einer Partie

Ein Duell besteht aus zehn Runden. Seeds und Seiten sind im aktuellen Runner
festgelegt und damit reproduzierbar:

| Runde | Seed | Upload „Spieler 1“ | Upload „Spieler 2“ |
| ---: | ---: | --- | --- |
| 1, 3, 5, 7, 9 | 42, 44, 46, 48, 50 | links / orange (`player_id=1`) | rechts / grün (`player_id=2`) |
| 2, 4, 6, 8, 10 | 43, 45, 47, 49, 51 | rechts / grün (`player_id=2`) | links / orange (`player_id=1`) |

Damit spielt jedes Modell fünf Runden auf jeder Seite. Pro Runde gilt:

- Ein positiver Environment-Reward zählt als ein Punkt für das zugehörige
  hochgeladene Modell; anschließend beginnt die nächste Runde.
- Nach maximal **2.000** Environment-Schritten ohne positiven Punkt oder bei
  einer technisch beendeten Runde ohne positiven Punkt ist die Runde ein
  Unentschieden. Beide Uploads erhalten dann je einen Turnierpunkt.
- Ein negativer Reward allein wird vom Runner nicht als eigener Turnierpunkt
  verbucht.
- Nach zehn Runden gewinnt der Upload mit mehr aggregierten Turnierpunkten;
  Gleichstand bedeutet Unentschieden.

Die Browser-Oberfläche zeigt Punktestand und aktuelle Seiten. Eine Partie lässt
sich pausieren, fortsetzen oder stoppen. Beim Stoppen endet sie ohne
nachträgliche Gewinnerbestimmung.

### Architektur: Training ist nicht der Turnier-Runner

| Baustein | Aufgabe |
| --- | --- |
| Arena | [PettingZoo Atari Pong (`pong_v3`)](https://pettingzoo.farama.org/main/environments/atari/pong/) mit zwei Spielern. |
| Browser-Oberfläche | [`index.html`](index.html) lässt zwei Modelle auswählen und zeigt Arena, Punktestand und Status. |
| Turnier-Runner | `GameSession` in [`app.py`](app.py) lädt beide ONNX-Modelle, ruft sie pro Frame auf und verwaltet Frame-Stack, Seitenwechsel, Runden und Punkte. |
| Modell | Reine ONNX-Policy-Inferenz: Sie erhält nur `image` und `player_id` und gibt eine Aktion zurück. Sie hat keinen direkten Zugriff auf Environment, Punktestand oder gegnerische Aktion. |
| Training | Eigene, von der Gruppe gewählte Offline-/Trainingsprogramme. Die Beispiele im Repository erzeugen ONNX-Dateien, sind aber nicht Teil des Turnier-Runners. |

```mermaid
flowchart LR
    subgraph training[Training außerhalb des Turniers]
        method[RL-Algorithmus und Training] --> export[ONNX-Export]
    end

    subgraph tournament[Lokales Turniersystem]
        browser[Browser-Oberfläche: index.html]
        runner["Turnier-Runner: GameSession in app.py<br/>Modelle laden, Frames und Spielablauf steuern"]
        arena[PettingZoo Atari Pong]
        model1["ONNX-Modell 1<br/>reine Policy"]
        model2["ONNX-Modell 2<br/>reine Policy"]

        browser -->|Modelle hochladen, Pause, Stop| runner
        arena -->|RGB-Frames| runner
        runner -->|image + player_id| model1
        runner -->|image + player_id| model2
        model1 -->|action 0–5| runner
        model2 -->|action 0–5| runner
        runner -->|Aktionen| arena
        runner -->|Bild, Punktestand, Status| browser
    end

    export -. eine ONNX-Abgabe .-> browser
```

### Öffentlicher Modellvertrag

Der Runner akzeptiert genau zwei Eingaben mit genau diesen Namen und eine
Ausgabe mit genau diesem Namen:

```text
image:     float32 [1, 12, 210, 160]
player_id: int64   [1]
action:    int64   [1]    # Wert aus {0, 1, 2, 3, 4, 5}
```

Die Abgabe soll mit **ONNX Opset 17** exportiert werden und höchstens 250 MB
groß sein. Sie wird mit ONNX Runtime auf CPU ausgeführt. Der Runner prüft beim
Laden Namen, Typen und feste Formen; er prüft das Opset nicht gesondert. Ein
anderes Opset kann zwar zufällig ladbar sein, erfüllt aber nicht die
Abgabevorgabe.

### State / Beobachtung

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
Die Bildfolge kann einer Policy helfen, Bewegung zu schätzen.

`player_id` bezeichnet **die physische Seite**, nicht den Upload-Slot im
Browser:

| `player_id` | Seite und Farbe |
| ---: | --- |
| `1` | links / orange |
| `2` | rechts / grün |

Ein Modell muss daher beide Werte verarbeiten können, wenn es auf beiden
Seiten spielen soll. Ein Modell darf `player_id` technisch nicht weglassen,
auch wenn es die Eingabe intern nicht nutzt.

### Aktionen

Die Werte entsprechen dem
[minimalen Action-Space von PettingZoo Pong](https://pettingzoo.farama.org/main/environments/atari/pong/#action-space-minimal).
Die offiziellen Richtungsnamen beziehen sich auf den Atari-Controller. In der
gedrehten Darstellung bewegen `RIGHT` und `LEFT` den Schläger auf dem Bildschirm
nach oben beziehungsweise unten; die Aktionsnummern bleiben dennoch 2 und 3.

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

### Was ist ein PyTorch-Modell?

Ein PyTorch-Modell ist eine Python-Klasse mit trainierbaren Gewichten. Die
Methode `forward(...)` beschreibt, wie aus Eingaben eine Ausgabe entsteht.
Dieses bewusst kleine Beispiel bildet `image` und `player_id` auf eine der
sechs Pong-Aktionen ab:

```python
import torch
from torch import nn

class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_to_q = nn.Sequential(
            nn.AvgPool2d(kernel_size=(21, 20), stride=(21, 20)),
            nn.Flatten(),
            nn.Linear(12 * 10 * 8, 6),
        )
        self.side_bias = nn.Embedding(3, 6)  # Einträge für player_id 1 und 2

    def forward(self, image, player_id):
        q_values = self.image_to_q(image) + self.side_bias(player_id)
        return torch.argmax(q_values, dim=1).to(torch.int64)

model = TinyPolicy()
```

PyTorch wird mit den vollständigen Trainingsabhängigkeiten aus Abschnitt 4
installiert; für den lokalen Browser-Runner aus Abschnitt 1 ist es nicht nötig.

Die Schichten und ihre Verbindung bilden die **Architektur**; die Zahlen in
`Linear` und `Embedding` sind die lernbaren **Gewichte**. Während des Trainings
werden diese Gewichte verändert. Bei der Inferenz führt `forward` nur noch die
Rechnung aus und liefert eine Aktion. Das Beispiel ist absichtlich klein und
noch untrainiert — es erklärt die Form, ist aber keine starke Pong-Policy.

PyTorch-/SB3-Checkpoints (`.pt`, `.pth`, `.zip`) dienen zum Fortsetzen des
Trainings. Abgegeben wird dagegen die exportierte `.onnx`-Datei.

### Was ist ONNX?

**ONNX** (*Open Neural Network Exchange*) ist ein portables Dateiformat für
fertig trainierte neuronale Netze. Beim Export wird die für die Inferenz
benötigte PyTorch-Rechnung in einen standardisierten Rechengraphen mit festen
Gewichten übersetzt. Der Turnier-Runner benötigt dadurch weder Ihren
Trainingscode noch PyTorch: Er lädt nur die `.onnx`-Datei mit ONNX Runtime auf
CPU und ruft ihre Policy auf.

```text
Training mit PyTorch / Stable-Baselines3  →  Export als .onnx  →  Browser-Runner
       Gewichte werden gelernt               feste Schnittstelle     nur Inferenz
```

Eine ONNX-Datei enthält damit Architektur und gelernte Gewichte für die
Inferenz, aber nicht Replay-Buffer, Optimierer, Trainingsdaten oder Python-
Klassen. Sie ist die **technische Abgabe**. Damit der Runner sie ausführen
kann, muss sie genau die oben beschriebenen Eingaben `image` und `player_id`
sowie die Ausgabe `action` bereitstellen.

Kurz gesagt exportiert man nach dem Training ein PyTorch-Modell mit
`torch.onnx.export(...)`. Die Beispieltensoren müssen dabei genau dem
Turniervertrag entsprechen:

```python
model.eval()
image = torch.zeros((1, 12, 210, 160), dtype=torch.float32)
player_id = torch.ones((1,), dtype=torch.int64)
torch.onnx.export(
    model, (image, player_id), "my_policy.onnx",
    input_names=["image", "player_id"], output_names=["action"], opset_version=17,
)
```

Das Modell muss in seinem `forward` aus diesen Eingaben eine gültige
`int64`-Aktion liefern. Anschließend sollte die Datei mit
`python validate_submission.py my_policy.onnx` geprüft werden. Hintergrund und
weitere Varianten erklärt die
[offizielle PyTorch-ONNX-Dokumentation](https://docs.pytorch.org/docs/stable/onnx.html).
Der Validator prüft CPU-Lauffähigkeit, Schnittstelle und Aktionsbereich, aber
nicht Spielstärke, Dateigröße oder ONNX-Opset vollständig.

## 3. Das Vanilla-DQN-Beispiel verstehen

Das Repository enthält bewusst **keine fertige Wettbewerbslösung**. Das
Skript [`train_dqn_vanilla.py`](train_dqn_vanilla.py) zeigt eine möglichst
kleine vollständige Trainingskette:

```text
Pong gegen ONNX-Gegner → Vorverarbeitung → DQN-Training
        → Checkpoint → ONNX-Export → Validator → Browser-Runner
```

| Datei | Rolle im Beispiel |
| --- | --- |
| [`pong_opponent_env.py`](pong_opponent_env.py) | Macht aus dem Zwei-Spieler-Spiel eine Gymnasium-Umgebung: Eine Seite lernt, die andere wird von einem wählbaren ONNX-Modell gesteuert. |
| [`train_dqn_vanilla.py`](train_dqn_vanilla.py) | Definiert Vorverarbeitung, CNN, DQN, Checkpoints und den abschließenden ONNX-Export. |
| [`train_qrdqn_vanilla.py`](train_qrdqn_vanilla.py) | Optionale QR-DQN-Variante mit derselben Umgebung, Vorverarbeitung und ONNX-Ausgabe. |

So arbeitet das Beispiel:

1. `PongVsOnnxEnv` trainiert pro Episode eine Seite und lässt die andere
   Seite mit dem ONNX-Gegner spielen. `--player-id 0` (Standard) wählt die
   trainierte Seite pro Episode zufällig; `1` und `2` erzwingen links bzw.
   rechts.
2. `PongPreprocessor` schneidet das Spielfeld aus und komprimiert die vier
   RGB-Frames auf vier Graustufenbilder mit `84 × 84` Pixeln. Ein fünfter Kanal
   kodiert die Seite. Dasselbe Modul wird in die exportierte ONNX-Policy
   eingebettet. So bleibt der Replay-Buffer klein und der öffentliche
   Modellvertrag dennoch unverändert.
3. DQN lernt aus dem unveränderten Pong-Reward und schreibt alle
   `--export-every` Schritte ein `.zip`-Checkpoint sowie eine `.onnx`-Datei.

Nach einem Lauf sind die `.onnx`-Dateien für Browser und Abgabe gedacht; die
gleichnamigen `.zip`-Dateien dienen nur dem Fortsetzen oder Analysieren des
SB3-Trainings. Der 1.000-Schritte-Timeout, DQN und der feste Gegner sind
Entscheidungen dieses Beispiels und keine Turnierregeln.

Die optionale Datei `train_qrdqn_vanilla.py` verwendet denselben Aufbau mit
QR-DQN. Für den gemeinsamen Einstieg bleibt `train_dqn_vanilla.py` der
übersichtlichere Hauptweg.

## 4. Vanilla-DQN auf dem Server oder in Colab trainieren

Längere Trainingsläufe gehören auf einen GPU-Server oder in Google Colab. Das
Ergebnis dieses Teils ist eine `.onnx`-Datei. Sie wird anschließend auf den
lokalen Rechner übertragen und dort mit Validator und Browser-Runner geprüft.
Die GPU wird nur für das Training verwendet; die Turnierausführung erfolgt
später mit ONNX Runtime auf der CPU.

### Möglichkeit 1 – GPU-Server

#### Server einmalig von null einrichten

Zuerst per SSH am Server anmelden. Servername und Benutzername werden von der
Hochschule oder dem jeweiligen Betreiber bereitgestellt:

```bash
ssh BENUTZERNAME@SERVERNAME
```

Danach das Repository in das eigene Homeverzeichnis klonen und eine getrennte
Python-Umgebung anlegen:

```bash
git clone https://github.com/georgmerz/rl-challenge.git
cd rl-challenge
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --prefer-binary -r requirements.txt
AutoROM --accept-license
```

Die vollständige `requirements.txt` enthält zusätzlich zu den lokalen
Runner-Paketen PyTorch, DQN/QR-DQN und den ONNX-Export. Anschließend prüfen, ob
PyTorch die Server-GPU erkennt:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'keine')"
```

Für ein GPU-Training muss hier `CUDA: True` erscheinen. Bei `False` nicht
unbemerkt einen langen CPU-Lauf starten, sondern die zum Server-Treiber
passende PyTorch-Version beziehungsweise die Hinweise des Betreibers prüfen.

Bei einem späteren Login genügen Repository-Wechsel, Aktivierung und Update:

```bash
cd ~/rl-challenge
source .venv/bin/activate
git pull --ff-only
```

Auf einem gemeinsam genutzten GPU-Server vor jedem Lauf mit `nvidia-smi` eine
freie GPU suchen. Die folgende Variable muss auf deren Nummer gesetzt werden;
keinen bereits stark belegten Speicher übernehmen:

```bash
nvidia-smi
gpu_id=2
```

### Kurzer technischer Test

Vor einem langen Lauf sollten Environment, GPU und ONNX-Export mit wenigen
Schritten gemeinsam geprüft werden:

```bash
CUDA_VISIBLE_DEVICES="$gpu_id" python train_dqn_vanilla.py \
  --device cuda \
  --timesteps 128 \
  --export-every 128 \
  --output baselines/dqn_smoke.onnx
python validate_submission.py baselines/dqn_smoke.onnx
```

Dieser Test bewertet keine Spielstärke und führt noch keine Lernupdates aus,
weil das Beispiel erst nach 5.000 Schritten zu lernen beginnt.

### Langen Lauf im Hintergrund starten

`nohup` lässt das Training weiterlaufen, wenn das Terminal oder die
VS-Code-SSH-Verbindung geschlossen wird. Eine eindeutige Run-ID verhindert,
dass ältere Modelle und Logs überschrieben werden:

```bash
mkdir -p logs
run_id="dqn_$(date +%Y%m%d_%H%M%S)"

nohup env CUDA_VISIBLE_DEVICES="$gpu_id" python -u train_dqn_vanilla.py \
  --device cuda \
  --timesteps 1000000 \
  --export-every 100000 \
  --player-id 0 \
  --opponent baselines/baseline_pong_reactive.onnx \
  --output "baselines/${run_id}.onnx" \
  > "logs/${run_id}.log" 2>&1 &

train_pid=$!
echo "$train_pid" > "logs/${run_id}.pid"
echo "Run: $run_id"
echo "PID: $train_pid"
echo "Log: logs/${run_id}.log"
```

Die Meldung `nohup: ignoring input` ist normal. Die gezeigten Hyperparameter
und der feste Gegner sind nur ein Ausgangspunkt, keine Challenge-Regeln. Für
einen besonders leichten Diagnoselauf kann stattdessen
`baseline_pong_static.onnx` als Gegner verwendet werden.

### Fortschritt prüfen und Lauf beenden

Im selben Terminal zeigt der folgende Befehl das Log fortlaufend. `Ctrl+C`
beendet dabei nur die Anzeige, nicht das Training:

```bash
tail -f "logs/${run_id}.log"
```

Nach einem erneuten Login lässt sich der zuletzt gestartete Lauf so finden:

```bash
latest_log=$(ls -t logs/dqn_*.log | head -n 1)
run_id=$(basename "$latest_log" .log)
pid_file="${latest_log%.log}.pid"
tail -n 50 "$latest_log"
ps -p "$(cat "$pid_file")" -o pid,etime,cmd
```

Nur wenn genau dieser Lauf beendet werden soll:

```bash
kill "$(cat "$pid_file")"
```

Alle `--export-every` Schritte entstehen `.onnx`- und `.zip`-Checkpoints im
Ordner `baselines/`. `.onnx` ist für Validator, Browser und Abgabe bestimmt;
`.zip` enthält den SB3-Trainingsstand. Nach regulärem Abschluss liegt das
finale Modell unter `baselines/${run_id}.onnx` und wird nochmals geprüft:

```bash
python validate_submission.py "baselines/${run_id}.onnx"
ls -lh "baselines/${run_id}"*.onnx
```

Für einen CPU-Lauf entfallen `CUDA_VISIBLE_DEVICES` und die Geräteauswahl wird
zu `--device cpu`; auf einem Mac mit Apple Silicon kann `--device mps`
verwendet werden.

### Möglichkeit 2 – Google Colab

Wer keinen GPU-Serverzugang hat, kann dasselbe Repository über das vorbereitete
Notebook in Colab verwenden:

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/georgmerz/rl-challenge/blob/main/RL_Challenge_Colab.ipynb)

Das Notebook klont das Repository, installiert Trainingsabhängigkeiten und
Atari-ROMs und führt den DQN-/ONNX-Weg aus. Vor dem Training in Colab unter
**Laufzeit → Laufzeittyp ändern** eine GPU auswählen. Colab-Speicher ist
temporär; die fertige `.onnx`-Datei muss vor dem Ende der Sitzung
heruntergeladen werden.

### Das trainierte ONNX-Modell lokal testen

Bei einem Serverlauf kann die fertige Datei über den VS-Code-Dateiexplorer
heruntergeladen oder vom **lokalen** Terminal kopiert werden:

```bash
scp BENUTZERNAME@SERVERNAME:~/rl-challenge/baselines/MEIN_MODELL.onnx .
```

Anschließend im lokalen Repository validieren und im Runner gegen eine
mitgelieferte Baseline laden:

```bash
python validate_submission.py MEIN_MODELL.onnx
PORT=5001 python app.py
```

Nur die `.onnx`-Datei wird lokal getestet und später abgegeben. Ein
SB3-Checkpoint mit der Endung `.zip` dient ausschließlich zum Fortsetzen des
Trainings und kann nicht in den Turnier-Runner geladen werden.

## 5. Formale Anforderungen an die Challenge

### Arbeit in Zweiergruppen

Nach dem gemeinsamen Durchlauf arbeiten Sie in Zweiergruppen. Beide Mitglieder
sollen das gesamte System erklären können, auch wenn Sie einzelne Aufgaben
untereinander aufteilen. Stimmen Sie gemeinsam Methode, Experimente, Bericht
und Vortrag ab und halten Sie Entscheidungen sowie Versuchsergebnisse während
der Arbeit fest.

### Leitfragen für das eigene Experiment

Das Vanilla-Beispiel ist ein Einstieg, keine Wettbewerbslösung. Entwickeln Sie
daraus eine eigene, klar abgegrenzte Untersuchung. Orientieren Sie sich an
folgenden offenen Leitfragen:

- Welche Frage möchten Sie mit Ihren Experimenten beantworten?
- Welche Entscheidung oder Veränderung untersuchen Sie, und warum könnte sie
  einen Unterschied machen?
- Wie gestalten Sie einen fairen und nachvollziehbaren Vergleich?
- Welche Ergebnisse benötigen Sie, um Ihre Frage beantworten zu können?
- Welche Grenzen hat Ihre Untersuchung, und welche Schlussfolgerungen sind
  deshalb möglich oder nicht möglich?

Eine GPU beschleunigt längere Läufe, ist aber keine methodische Voraussetzung.

### Was sollten Sie im Bericht diskutieren?

Der Bericht soll nicht nur den fertigen Code beschreiben, sondern den
**Untersuchungsprozess wissenschaftlich nachvollziehbar dokumentieren**:
Welche Forschungsfrage wurde verfolgt, wie wurden methodische Entscheidungen
begründet und geprüft, und welche Schlussfolgerungen lassen die Ergebnisse zu?
Für 5–8 Seiten bietet sich folgende Struktur an.

#### Aufgabe, State und Aktionen

- Wie funktionieren Pong, Ballwechsel, Punkte und Reward? Welche Information
  erhält die Policy und welche ausdrücklich nicht?
- Warum wurden vier RGB-Frames direkt verwendet oder intern weiter
  vorverarbeitet? Welche Bewegungsinformation lässt sich daraus gewinnen?
- Wie werden die sechs Aktionen, der Environment-Reward und `player_id`
  interpretiert?

#### Methode und Training

- Welcher RL-Algorithmus und welche Modellarchitektur wurden gewählt – und
  warum passen sie zur Aufgabe?
- Wurde der originale Reward verwendet oder Reward Shaping eingesetzt? Welche
  erwünschten und unerwünschten Anreize entstehen dadurch?
- Wurde gegen feste Gegner, Self-Play oder einen Gegner-Pool trainiert?
- Wie wurde sichergestellt, dass die Policy beide Seiten beherrscht und nicht
  nur einen bekannten Gegner oder wenige Seeds auswendig lernt?
- Welche Trainingsdauer, Hyperparameter, Hardware und Kriterien zur Auswahl
  eines Checkpoints wurden verwendet?

#### Evaluation und Ergebnisse

- Gegen welche Baselines und Modelle wurde **ohne weiteres Lernen** evaluiert?
- Wurden mehrere Seeds, beide Seiten und genügend Ballwechsel betrachtet?
- Welche Kennzahlen beantworten die Frage nach Spielstärke sinnvoll, zum
  Beispiel Siege/Niederlagen/Unentschieden, Punkteverhältnis und Streuung?
- Was zeigen Lernkurven, Tabellen oder Videos? Eine kleine Zahl gut erklärter
  Ergebnisse ist wertvoller als viele unkommentierte Ausgaben.
- Welche Fehlerfälle, Grenzen und überraschenden Verhaltensweisen wurden
  beobachtet? Was wäre der nächste sinnvolle Verbesserungsversuch?

Trainingsergebnisse und Evaluation sollten klar getrennt werden. Das
Browser-Turnier mit zehn festen Runden ist ein gemeinsamer Abschlusstest, aber
allein noch keine belastbare Messung allgemeiner Spielstärke. Die eigene
Evaluation sollte deshalb weitere Seeds, beide Seiten und möglichst mehrere
Gegnertypen enthalten. Befehle, Softwareversionen und wesentliche
Hyperparameter sollten so dokumentiert sein, dass der Versuch reproduzierbar
ist.

### Agenda für den 10-Minuten-Vortrag

Am **13.11.2026** präsentieren Sie als Zweiergruppe Ihr Vorgehen und Ihre
Ergebnisse und nehmen anschließend am Turnier teil. Der Vortrag soll eine kurze,
verständliche Fassung des Berichts sein – keine Führung durch den Quellcode.
Beide Gruppenmitglieder übernehmen einen inhaltlichen Teil.

Eine mögliche Zeiteinteilung ist:

| Zeit | Inhalt |
| ---: | --- |
| ca. 1 Minute | Aufgabe, Ziel und eigene Fragestellung |
| ca. 3 Minuten | State/Vorverarbeitung, Algorithmus, Reward und Trainingsgegner |
| ca. 4 Minuten | Evaluation, wichtigste Ergebnisse und Vergleich mit Baselines |
| ca. 2 Minuten | Grenzen, überraschende Beobachtungen und Fazit |

Eine Architekturabbildung, eine aussagekräftige Lernkurve und eine kompakte
Ergebnistabelle sind meist hilfreicher als viele Folien mit Code. Aussagen wie
„das Modell ist gut“ sollten durch eine nachvollziehbare Evaluation über
Gegner, Seeds und beide Seiten belegt werden. Das Turniermodell sollte vor dem
Termin als validierte ONNX-Datei bereitliegen; ein Live-Training ist nicht Teil
des Vortrags.

### Abgabe, Bewertung und Notenschlüssel

Pro Zweiergruppe werden benötigt:

- **genau eine `.onnx`-Datei** mit höchstens 250 MB und ONNX Opset 17,
- ein **Bericht mit 5–8 Seiten** und
- ein **gemeinsamer Vortrag von 10 Minuten**.

Eine zusätzliche Projekt-README kann hilfreich sein, wird vom Runner aber
nicht verwendet. Vor der Abgabe muss die ONNX-Datei technisch validiert und
lokal gegen einen Testgegner ausgeführt werden:

```bash
python validate_submission.py model.onnx
```

Der Validator prüft technische Kompatibilität, nicht die Spielstärke. Die
Endnote basiert auf dem Bericht. Turnierplatzierung und absolute Spielstärke
werden nicht direkt benotet; bewertet werden die Qualität der Methode, der
Experimente und ihrer kritischen Auswertung.

#### Bewertung des Berichts

| Kriterium | Punkte | Erwartung |
| --- | ---: | --- |
| Aufgabe und Modellvertrag | 15 | Spiel, State, Aktionen, Reward, `player_id` und ONNX-Schnittstelle korrekt und verständlich erklären. |
| Methode und Begründung | 25 | Vorverarbeitung, Algorithmus, Architektur und gegebenenfalls Reward Shaping nachvollziehbar auswählen und begründen. |
| Training und Versuchsaufbau | 20 | Gegnerwahl, beide Seiten, Hyperparameter, Trainingsdauer und Checkpoint-Auswahl transparent dokumentieren. |
| Evaluation und kritische Diskussion | 30 | Sinnvolle Baselines, Seeds, Kennzahlen und Vergleiche verwenden; Ergebnisse, Grenzen und Fehlerfälle fundiert interpretieren. |
| Darstellung und Reproduzierbarkeit | 10 | Klar strukturieren, Abbildungen beschriften sowie Befehle, Versionen und Hardware ausreichend dokumentieren. |
| **Summe** | **100** | |

#### Notenschlüssel

| Punkte | Note |
| ---: | :--- |
| 95–100 | 1,0 |
| 90–94 | 1,3 |
| 85–89 | 1,7 |
| 80–84 | 2,0 |
| 75–79 | 2,3 |
| 70–74 | 2,7 |
| 65–69 | 3,0 |
| 60–64 | 3,3 |
| 55–59 | 3,7 |
| 50–54 | 4,0 |
| 0–49 | 5,0 |

Der Vorbereitungstermin ist am **30.10.2026**. Präsentation und Turnier finden
am **13.11.2026** statt.
