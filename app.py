from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_MODEL_BYTES = 250 * 1024 * 1024
ACTION_COUNT = 6
TOTAL_ROUNDS = 10
# A Pong rally can theoretically take a very long time. It is a draw after
# this many frames, so every tournament round always finishes predictably.
MAX_ROUND_STEPS = 2_000
FRAME_DELAY_SECONDS = 0.015
ROUND_RESET_DELAY_SECONDS = 0.2

app = Flask(__name__, template_folder=str(BASE_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_MODEL_BYTES


def load_onnx_model(path: Path) -> ort.InferenceSession:
    """Load a portable ONNX model with the CPU-only reference runtime."""
    if path.suffix.lower() != ".onnx":
        raise ValueError("Es sind ausschließlich ONNX-Dateien (.onnx) erlaubt.")
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise ValueError("Die ONNX-Datei kann mit der CPU-Referenzlaufzeit nicht geladen werden.") from error

    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {"image", "player_id"}:
        raise ValueError("Das ONNX-Modell benötigt genau die Eingaben `image` und `player_id`.")
    if inputs["image"].type != "tensor(float)" or list(inputs["image"].shape) != [1, 3, 210, 160]:
        raise ValueError("`image` muss float32 mit der Form [1, 3, 210, 160] sein.")
    if inputs["player_id"].type != "tensor(int64)" or list(inputs["player_id"].shape) != [1]:
        raise ValueError("`player_id` muss int64 mit der Form [1] sein.")
    outputs = session.get_outputs()
    if (
        len(outputs) != 1
        or outputs[0].name != "action"
        or outputs[0].type != "tensor(int64)"
        or list(outputs[0].shape) != [1]
    ):
        raise ValueError("Das ONNX-Modell benötigt genau die int64-Ausgabe `action` mit Form [1].")
    return session


class ModelAgent:
    """Adapter from the public challenge API to an ONNX model."""

    def __init__(self, path: Path):
        self.model = load_onnx_model(path)

    @staticmethod
    def _image_tensor(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.shape != (210, 160, 3):
            raise ValueError(f"Erwartet wird ein Bild der Form (210, 160, 3), erhalten: {image.shape}")
        # Public API: uint8 HWC. Model API: float32 NCHW, values in [0, 1].
        return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    def act(self, image: np.ndarray, player_id: int) -> int:
        image_tensor = self._image_tensor(image)
        player_tensor = np.asarray([player_id], dtype=np.int64)
        output = np.asarray(self.model.run(["action"], {"image": image_tensor, "player_id": player_tensor})[0])
        if output.size != 1:
            raise ValueError("Das Modell muss genau eine Aktionsnummer als Skalar zurückgeben.")
        action = int(output.reshape(-1)[0].item())
        if action < 0 or action >= ACTION_COUNT:
            raise ValueError(f"Ungültige Atari-Aktion {action}; erlaubt sind 0 bis {ACTION_COUNT - 1}.")
        return action


def encode_frame(frame: np.ndarray | None) -> str | None:
    if frame is None:
        return None
    # Pillow is intentionally avoided in the core runner. The browser receives a small PPM.
    # PPM is universally decodable by the browser only through conversion, so use PNG when PIL exists.
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Für die Browseranzeige wird Pillow benötigt.") from exc
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class GameState:
    status: str = "idle"
    message: str = "Noch keine Partie gestartet"
    frame: str | None = None
    score: dict[str, int] | None = None
    round: int = 0
    total_rounds: int = TOTAL_ROUNDS
    player_1_side: str | None = None
    player_2_side: str | None = None
    winner: str | None = None
    error: str | None = None


class GameSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = GameState(score={"Spieler 1": 0, "Spieler 2": 0})
        self.thread: threading.Thread | None = None
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.stop_event = threading.Event()

    def snapshot(self):
        with self.lock:
            return self.state.__dict__.copy()

    def start(self, model_1: Path, model_2: Path, rounds: int = TOTAL_ROUNDS):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Es läuft bereits eine Partie.")
            self.stop_event.clear()
            self.resume_event.set()
            self.state = GameState(
                status="starting",
                message="Modelle werden geladen...",
                score={"Spieler 1": 0, "Spieler 2": 0},
                total_rounds=rounds,
            )
            self.thread = threading.Thread(target=self._run, args=(model_1, model_2, rounds), daemon=True)
            self.thread.start()

    def pause(self):
        with self.lock:
            if self.state.status != "running":
                raise RuntimeError("Nur eine laufende Partie kann pausiert werden.")
            self.resume_event.clear()
            self.state.status = "paused"
            self.state.message = "Partie pausiert"

    def resume(self):
        with self.lock:
            if self.state.status != "paused":
                raise RuntimeError("Die Partie ist nicht pausiert.")
            self.resume_event.set()
            self.state.status = "running"
            self.state.message = f"Runde {self.state.round} von {self.state.total_rounds} läuft"

    def stop(self):
        with self.lock:
            if self.state.status not in {"starting", "running", "paused"}:
                raise RuntimeError("Es läuft keine Partie, die gestoppt werden kann.")
            self.stop_event.set()
            self.resume_event.set()
            self.state.status = "stopping"
            self.state.message = "Partie wird gestoppt..."

    def _run(self, model_1_path: Path, model_2_path: Path, rounds: int):
        env = None
        try:
            from pettingzoo.atari import pong_v3

            # PettingZoo's first_0 controls the right/green player and second_0 controls
            # the left/orange player. The uploaded competitors alternate these sides.
            agent_1 = ModelAgent(model_1_path)
            agent_2 = ModelAgent(model_2_path)
            env = pong_v3.parallel_env(render_mode="rgb_array", num_players=2)
            observations, infos = env.reset(seed=42)
            names = list(env.possible_agents)
            if len(names) != 2:
                raise RuntimeError(f"Unerwartete Agentenzahl: {names}")
            right_name, left_name = names[0], names[1]
            player_id_by_name = {left_name: 1, right_name: 2}
            score = {"Spieler 1": 0, "Spieler 2": 0}
            round_seed = 42
            round_number = 1
            round_steps = 0

            def set_round_players():
                # Odd rounds: player 1 uses the left/orange side. Even rounds swap sides.
                if round_number % 2:
                    agents = {left_name: agent_1, right_name: agent_2}
                    competitor_by_name = {left_name: "Spieler 1", right_name: "Spieler 2"}
                    player_1_side, player_2_side = "links / orange", "rechts / grün"
                else:
                    agents = {left_name: agent_2, right_name: agent_1}
                    competitor_by_name = {left_name: "Spieler 2", right_name: "Spieler 1"}
                    player_1_side, player_2_side = "rechts / grün", "links / orange"
                with self.lock:
                    self.state.round = round_number
                    self.state.player_1_side = player_1_side
                    self.state.player_2_side = player_2_side
                return agents, competitor_by_name

            agents, competitor_by_name = set_round_players()
            with self.lock:
                self.state.status = "running"
                self.state.message = f"Runde {round_number} von {rounds} läuft"

            stopped = False
            while round_number <= rounds:
                # Wait without blocking the Flask request thread, so the UI can
                # pause, resume or stop the game between Atari frames.
                while not self.resume_event.wait(timeout=0.1):
                    if self.stop_event.is_set():
                        stopped = True
                        break
                if stopped or self.stop_event.is_set():
                    stopped = True
                    break
                actions = {}
                for name in names:
                    image = observations[name]
                    actions[name] = agents[name].act(image, player_id_by_name[name])
                observations, rewards, terminations, truncations, infos = env.step(actions)
                round_steps += 1
                frame = env.render()
                # Pong rewards the player that scores with +1 and the other
                # player with -1. A tournament round ends after one point.
                point_scored = False
                for name, reward in rewards.items():
                    if reward > 0:
                        score[competitor_by_name[name]] += 1
                        point_scored = True
                round_draw = (
                    not point_scored
                    and (
                        round_steps >= MAX_ROUND_STEPS
                        or any(terminations.values())
                        or any(truncations.values())
                    )
                )
                if round_draw:
                    score["Spieler 1"] += 1
                    score["Spieler 2"] += 1
                    point_scored = True
                with self.lock:
                    self.state.frame = encode_frame(frame)
                    self.state.score = score.copy()

                if point_scored or round_draw:
                    if round_number >= rounds:
                        break
                    time.sleep(ROUND_RESET_DELAY_SECONDS)
                    round_seed += 1
                    round_number += 1
                    round_steps = 0
                    observations, infos = env.reset(seed=round_seed)
                    agents, competitor_by_name = set_round_players()
                    with self.lock:
                        self.state.frame = encode_frame(env.render())
                        if self.state.status == "running":
                            self.state.message = f"Runde {round_number} von {rounds} läuft"
                    time.sleep(FRAME_DELAY_SECONDS)
                    continue
                time.sleep(FRAME_DELAY_SECONDS)

            if stopped:
                winner = None
                message = "Partie gestoppt"
                status = "stopped"
            elif score["Spieler 1"] == score["Spieler 2"]:
                winner = None
                message = "Unentschieden"
                status = "finished"
            else:
                winner = "Spieler 1" if score["Spieler 1"] > score["Spieler 2"] else "Spieler 2"
                message = f"{winner} gewinnt"
                status = "finished"
            with self.lock:
                self.state.status = status
                self.state.winner = winner
                self.state.message = message
        except Exception as exc:
            with self.lock:
                self.state.status = "error"
                self.state.error = f"{type(exc).__name__}: {exc}"
                self.state.message = "Partie abgebrochen"
        finally:
            if env is not None:
                env.close()


session = GameSession()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def state():
    return jsonify(session.snapshot())


@app.post("/api/start")
def start():
    file_1 = request.files.get("model_1")
    file_2 = request.files.get("model_2")
    if not file_1 or not file_2:
        return jsonify(error="Bitte zwei ONNX-Modelle (.onnx) hochladen."), 400
    if not file_1.filename.lower().endswith(".onnx") or not file_2.filename.lower().endswith(".onnx"):
        return jsonify(error="Nur .onnx-Dateien sind erlaubt."), 400
    paths = []
    try:
        for uploaded in (file_1, file_2):
            path = UPLOAD_DIR / f"{uuid.uuid4().hex}.onnx"
            uploaded.save(path)
            paths.append(path)
        for path in paths:
            load_onnx_model(path)
        session.start(paths[0], paths[1])
        return jsonify(ok=True)
    except Exception as exc:
        for path in paths:
            path.unlink(missing_ok=True)
        return jsonify(error=str(exc)), 400


@app.post("/api/pause")
def pause():
    try:
        session.pause()
        return jsonify(ok=True)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/resume")
def resume():
    try:
        session.resume()
        return jsonify(ok=True)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/stop")
def stop():
    try:
        session.stop()
        return jsonify(ok=True)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
