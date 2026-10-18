"""Local browser interface for trained Sherlock Holmes language models."""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from sherlock.data import ROOT
from sherlock.generation import generate, load_model
from sherlock.model import MODEL_NAMES


def create_app(run_dir: str | Path | None = None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    run_dir = Path(run_dir or ROOT / "results/demo").resolve()
    cache = {}
    lock = threading.Lock()

    def read_json(filename, default):
        path = run_dir / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        run = read_json("run.json", {})
        available = [kind for kind in MODEL_NAMES if (run_dir / "checkpoints" / f"{kind}.pt").is_file()]
        return jsonify({"ready": bool(available), "models": available,
                        "model_names": MODEL_NAMES, "run_name": run_dir.name,
                        "status": run.get("status", "not_trained"),
                        "recommended_model": run.get("recommended_model"),
                        "config": run.get("config", {}), "device": run.get("device"),
                        "dataset": read_json("dataset.json", None),
                        "comparison": read_json("comparison.json", []),
                        "plots_ready": (run_dir / "plots/training_curves.png").is_file()})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_ready": any(
            (run_dir / "checkpoints" / f"{kind}.pt").is_file() for kind in MODEL_NAMES)})

    @app.post("/api/generate")
    def generate_text():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object containing a prompt and model."), 400
        kind = payload.get("model", "gru")
        if not isinstance(kind, str) or kind not in MODEL_NAMES:
            return jsonify(error="Choose Simple RNN, Bidirectional RNN, LSTM, or GRU."), 400
        path = run_dir / "checkpoints" / f"{kind}.pt"
        if not path.is_file():
            return jsonify(error=f"{MODEL_NAMES[kind]} has no checkpoint in this run. Train it first."), 503
        options = {"new_tokens": payload.get("new_tokens", 60),
                   "temperature": payload.get("temperature", .8),
                   "top_k": payload.get("top_k", 40), "top_p": payload.get("top_p", .95),
                   "seed": payload.get("seed", 42), "greedy": payload.get("greedy", False)}
        if any(type(options[key]) not in (int, float) for key in ("temperature", "top_p")):
            return jsonify(error="Temperature and top-p must be numbers."), 400
        if type(options["greedy"]) is not bool:
            return jsonify(error="Greedy must be true or false."), 400
        try:
            # Serialize inference to keep the local CPU responsive and protect the model cache.
            with lock:
                stamp = path.stat().st_mtime_ns
                if kind not in cache or cache[kind][0] != stamp:
                    cache[kind] = (stamp, *load_model(path)[:2])
                _, model, vocabulary = cache[kind]
                result = generate(model, vocabulary, payload.get("prompt", ""), **options)
            return jsonify(result)
        except (ValueError, TypeError, OverflowError) as error:
            return jsonify(error=str(error)), 400
        except (OSError, RuntimeError, KeyError, EOFError):
            app.logger.exception("Could not load or run checkpoint")
            return jsonify(error="This checkpoint could not be loaded. Train a new run and restart with --run-dir."), 503

    @app.get("/api/plots/<name>")
    def plot(name):
        if name not in {"training_curves.png", "model_comparison.png", "temperature_effects.png", "word_frequency.png"}:
            return jsonify(error="Unknown plot."), 404
        return send_from_directory(run_dir / "plots", name)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="Request is too large. Use a prompt under 2,000 characters."), 413

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/demo")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    torch.set_num_threads(2)
    create_app(args.run_dir).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
