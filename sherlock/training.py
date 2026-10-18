from __future__ import annotations

import csv
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import prepare_data
from .generation import diversity_metrics, generate
from .model import MODEL_NAMES, ModelConfig, NextWordModel

PROMPTS = ["sherlock holmes looked at me and said",
           "it was a cold morning when we arrived at",
           "the inspector opened the door and found"]


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def epoch_pass(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count, correct, top5 = 0., 0, 0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became non-finite; lower the learning rate.")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            size = targets.numel()
            loss_sum += loss.item() * size
            count += size
            correct += int((logits.argmax(1) == targets).sum())
            top5 += int((logits.topk(min(5, logits.shape[1]), dim=1).indices == targets[:, None]).any(1).sum())
    mean_loss = loss_sum / count
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 700)),
            "accuracy": correct / count, "top5_accuracy": top5 / count}


def run_experiment(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output folder is not empty: {output}. Choose a new --output to preserve earlier runs.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir()
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or install a compatible CUDA PyTorch build.")
    vocabulary, datasets, data_summary = prepare_data(args.data, args.context_length,
                                                      args.min_frequency, args.max_vocab, args.max_tokens)
    write_json(output / "dataset.json", data_summary)
    write_json(output / "vocabulary.json", vocabulary.tokens)
    manifest = {"status": "training", "created_utc": datetime.now(timezone.utc).isoformat(),
                "device": str(device), "torch_version": str(torch.__version__),
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "selection_rule": "lowest validation loss; test data is used only for final evaluation",
                "completed_models": []}
    write_json(output / "run.json", manifest)
    evaluation_loaders = {name: DataLoader(datasets[name], batch_size=args.batch_size, shuffle=False)
                          for name in ("validation", "test")}
    comparisons, histories, samples = [], [], []
    print(f"Device: {device} | tokens: {data_summary['total_tokens']:,} | vocabulary: {len(vocabulary):,}", flush=True)
    for kind in args.models:
        seed_everything(args.seed)
        loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
        config = ModelConfig(kind, len(vocabulary), args.embedding_dim, args.hidden_dim,
                             args.layers, args.dropout, args.context_length)
        model = NextWordModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_loss, best_epoch, stale = math.inf, 0, 0
        checkpoint_path = output / "checkpoints" / f"{kind}.pt"
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = epoch_pass(model, loader, device, optimizer)
            validation = epoch_pass(model, evaluation_loaders["validation"], device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - epoch_start
            histories.append({"model": kind, "epoch": epoch, "train_loss": train_metrics["loss"],
                              "validation_loss": validation["loss"], "train_accuracy": train_metrics["accuracy"],
                              "validation_accuracy": validation["accuracy"], "seconds": elapsed})
            print(f"{MODEL_NAMES[kind]:18} epoch {epoch:02d}/{args.epochs}  "
                  f"train {train_metrics['loss']:.3f}  val {validation['loss']:.3f}  "
                  f"val acc {validation['accuracy']:.1%}  {elapsed:.1f}s", flush=True)
            if validation["loss"] < best_loss:
                best_loss, best_epoch, stale = validation["loss"], epoch, 0
                checkpoint = {"format_version": 1, "model_config": config.to_dict(),
                              "vocabulary": vocabulary.tokens,
                              "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                              "best_epoch": epoch, "validation_loss": best_loss,
                              "dataset_sha256": data_summary["sha256"], "seed": args.seed}
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(checkpoint_path)
            else:
                stale += 1
            write_csv(output / "training_history.csv", histories)
            if stale >= args.patience:
                print(f"Early stopping: restoring epoch {best_epoch}.", flush=True)
                break
        training_seconds = time.perf_counter() - started
        best = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(best["state_dict"])
        test = epoch_pass(model, evaluation_loaders["test"], device)
        comparisons.append({"model": kind, "parameters": model.parameter_count, "epochs_run": epoch,
                            "best_epoch": best_epoch, "validation_loss": best_loss,
                            "test_loss": test["loss"], "test_perplexity": test["perplexity"],
                            "test_accuracy": test["accuracy"], "test_top5_accuracy": test["top5_accuracy"],
                            "training_seconds": training_seconds})
        # Same prompts, temperatures and random seeds for every architecture.
        for prompt_index, prompt in enumerate(PROMPTS):
            for temperature in (.2, .5, 1., 1.5):
                for sample_index in range(args.samples_per_prompt):
                    generated = generate(model, vocabulary, prompt, new_tokens=50, temperature=temperature,
                                         top_k=0, top_p=1., seed=1000 + 10 * prompt_index + sample_index)
                    samples.append({"model": kind, "prompt": prompt, "temperature": temperature,
                                    "sample": sample_index + 1, "continuation": generated["continuation"],
                                    **diversity_metrics(generated["continuation"])})
        write_csv(output / "model_comparison.csv", comparisons)
        write_csv(output / "generated_samples.csv", samples)
        write_json(output / "comparison.json", comparisons)
        manifest["completed_models"].append(kind)
        write_json(output / "run.json", manifest)
    manifest["status"] = "reporting"
    manifest["recommended_model"] = min(comparisons, key=lambda row: row["validation_loss"])["model"]
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "run.json", manifest)
    from .reporting import create_report
    create_report(output)
    manifest["status"] = "complete"
    write_json(output / "run.json", manifest)
    print(f"\nSaved experiment to {output}\nSelected by validation loss: {manifest['recommended_model']}", flush=True)
    return comparisons
