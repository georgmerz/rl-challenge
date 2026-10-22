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
import torch
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_MODEL_BYTES = 250 * 1024 * 1024
ACTION_COUNT = 5
TOTAL_ROUNDS = 10
MAX_MATCH_STEPS = 50000
FRAME_DELAY_SECONDS = 0.015
ROUND_RESET_DELAY_SECONDS = 0.2

app = Flask(__name__, template_folder=str(BASE_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_MODEL_BYTES


def load_exported_model(path: Path):
    """Load an exported PyTorch model. Pickled Python model files are rejected."""
    if path.suffix != ".pt2":
        raise ValueError("Es sind ausschließlich exportierte PyTorch-Dateien (.pt2) erlaubt.")
    if not hasattr(torch, "export") or not hasattr(torch.export, "load"):
        raise RuntimeError("Für .pt2-Modelle wird PyTorch >= 2.2 benötigt. Bitte `pip install -r requirements.txt` erneut ausführen.")
    try:
        exported = torch.export.load(str(path))
    except (KeyError, RuntimeError, ValueError) as error:
        raise ValueError(
            "Das .pt2-Modell kann nicht geladen werden. Es wurde vermutlich mit "
            "einer nicht kompatiblen PyTorch-Version exportiert. Bitte in Colab "
            "und lokal `pip install -r requirements.txt` ausführen, danach das "
            "Modell erneut trainieren und herunterladen."
        ) from error
    model = exported.module()
    model.eval()
    return model


class ModelAgent:
    """Adapter from the public challenge API to a PyTorch exported model."""

    def __init__(self, path: Path):
        self.model = load_exported_model(path)

    @staticmethod
    def _image_tensor(image: np.ndarray) -> torch.Tensor:
        if not isinstance(image, np.ndarray) or image.shape != (210, 160, 3):
            raise ValueError(f"Erwartet wird ein Bild der Form (210, 160, 3), erhalten: {image.shape}")
        # Public API: uint8 HWC. Model API: float32 NCHW, values in [0, 1].
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255.0

    def act(self, image: np.ndarray, player_id: int) -> int:
        image_tensor = self._image_tensor(image)
        player_tensor = torch.tensor([player_id], dtype=torch.int64)
        with torch.inference_mode():
            output = self.model(image_tensor, player_tensor)
        output = torch.as_tensor(output).detach().cpu()
        if output.numel() != 1:
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
            from pettingzoo.atari import surround_v2

            # PettingZoo's first_0 controls the right/blue player and second_0 controls
            # the left/green player. The uploaded competitors alternate these sides.
            agent_1 = ModelAgent(model_1_path)
            agent_2 = ModelAgent(model_2_path)
            env = surround_v2.parallel_env(render_mode="rgb_array")
            observations, infos = env.reset(seed=42)
            names = list(env.possible_agents)
            if len(names) != 2:
                raise RuntimeError(f"Unerwartete Agentenzahl: {names}")
            right_name, left_name = names[0], names[1]
            player_id_by_name = {left_name: 1, right_name: 2}
            score = {"Spieler 1": 0, "Spieler 2": 0}
            round_seed = 42
            round_number = 1
            step_count = 0

            def set_round_players():
                # Odd rounds: player 1 uses the left/green side. Even rounds swap sides.
                if round_number % 2:
                    agents = {left_name: agent_1, right_name: agent_2}
                    competitor_by_name = {left_name: "Spieler 1", right_name: "Spieler 2"}
                    player_1_side, player_2_side = "links / grün", "rechts / blau"
                else:
                    agents = {left_name: agent_2, right_name: agent_1}
                    competitor_by_name = {left_name: "Spieler 2", right_name: "Spieler 1"}
                    player_1_side, player_2_side = "rechts / blau", "links / grün"
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
            while round_number <= rounds and step_count < MAX_MATCH_STEPS:
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
                step_count += 1
                frame = env.render()
                # Surround rewards +1 to the player that survived a collision and -1 to the player that crashed.
                point_scored = False
                for name, reward in rewards.items():
                    if reward > 0:
                        score[competitor_by_name[name]] += 1
                        point_scored = True
                # A simultaneous crash has no positive Surround reward. It is
                # a draw for this round, so both tournament players get a point.
                round_draw = not point_scored and (any(terminations.values()) or any(truncations.values()))
                if round_draw:
                    score["Spieler 1"] += 1
                    score["Spieler 2"] += 1
                    point_scored = True
                with self.lock:
                    self.state.frame = encode_frame(frame)
                    self.state.score = score.copy()

                if point_scored:
                    if round_number >= rounds:
                        break
                    time.sleep(ROUND_RESET_DELAY_SECONDS)
                    round_seed += 1
                    round_number += 1
                    observations, infos = env.reset(seed=round_seed)
                    agents, competitor_by_name = set_round_players()
                    with self.lock:
                        self.state.frame = encode_frame(env.render())
                        if self.state.status == "running":
                            self.state.message = f"Runde {round_number} von {rounds} läuft"
                    time.sleep(FRAME_DELAY_SECONDS)
                    continue
                if any(terminations.values()) or any(truncations.values()):
                    break
                time.sleep(FRAME_DELAY_SECONDS)

            if stopped:
                winner = None
                message = "Partie gestoppt"
                status = "stopped"
            elif step_count >= MAX_MATCH_STEPS and round_number <= rounds:
                winner = None
                message = "Zeitlimit erreicht"
                status = "finished"
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
        return jsonify(error="Bitte zwei .pt2-Modelle hochladen."), 400
    if not file_1.filename.endswith(".pt2") or not file_2.filename.endswith(".pt2"):
        return jsonify(error="Nur .pt2-Dateien sind erlaubt."), 400
    paths = []
    try:
        for uploaded in (file_1, file_2):
            path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pt2"
            uploaded.save(path)
            paths.append(path)
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
