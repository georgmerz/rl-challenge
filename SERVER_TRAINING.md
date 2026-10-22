# Pong-DQN auf einem GPU-Server trainieren

Diese Anleitung ist für einen Linux-Server mit NVIDIA-GPU gedacht. Die
trainierte ONNX-Datei bleibt anschließend auf jedem Rechner mit ONNX Runtime
CPU lauffähig, also auch im lokalen Turnier auf dem Mac.

## Einmalige Einrichtung

```bash
git clone https://github.com/georgmerz/rl-challenge.git
cd rl-challenge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --prefer-binary -r requirements.txt
AutoROM --accept-license
nvidia-smi
```

`nvidia-smi` muss die NVIDIA-GPU anzeigen. Wenn es fehlt, ist CUDA auf diesem
Server nicht verfügbar; dann `--device cpu` nutzen oder die Serverumgebung
prüfen.

## Training starten

Für einen ersten echten Lauf mit einer Million Entscheidungen:

```bash
mkdir -p logs baselines
nohup python train_dqn_vanilla.py \
  --device cuda \
  --timesteps 1000000 \
  --output baselines/baseline_pong_dqn.onnx \
  --export-every 100000 \
  > logs/pong_dqn.log 2>&1 &
echo $! > logs/pong_dqn.pid
tail -f logs/pong_dqn.log
```

Der Lauf schreibt alle 100.000 Schritte zwei Checkpoints nach `baselines/`:

```text
baseline_pong_dqn_step_100000.onnx  # direkt im Turnier testbar
baseline_pong_dqn_step_100000.zip   # SB3-Gewichte für einen Folgelauf
```

Die abschließende Datei heißt `baselines/baseline_pong_dqn.onnx`.

## Kontrollieren, stoppen und fortsetzen

```bash
ps -fp "$(cat logs/pong_dqn.pid)"
tail -f logs/pong_dqn.log
kill "$(cat logs/pong_dqn.pid)"
```

Ein Folgelauf lädt die Modellgewichte; der Replay-Buffer wird dabei neu
aufgebaut. `--timesteps` ist die zusätzlich zu trainierende Schrittzahl.

```bash
python train_dqn_vanilla.py \
  --device cuda \
  --resume baselines/baseline_pong_dqn_step_1000000.zip \
  --timesteps 1000000 \
  --output baselines/baseline_pong_dqn_continued.onnx
```

## Modell prüfen und herunterladen

```bash
python validate_submission.py baselines/baseline_pong_dqn.onnx
```

Auf dem lokalen Mac herunterladen:

```bash
scp USER@SERVER:~/rl-challenge/baselines/baseline_pong_dqn.onnx .
```

Die Challenge verwendet vier aufeinanderfolgende RGB-Frames als ONNX-Eingabe
(`image: float32 [1, 12, 210, 160]`). Daher nur Modelle aus der aktuellen
Repository-Version verwenden; ältere 3-Kanal-Modelle sind nicht kompatibel.
