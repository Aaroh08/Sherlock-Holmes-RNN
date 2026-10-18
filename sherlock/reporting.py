from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import MODEL_NAMES


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def create_report(output: Path):
    output = Path(output)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    run = json.loads((output / "run.json").read_text())
    dataset = json.loads((output / "dataset.json").read_text())
    comparison = json.loads((output / "comparison.json").read_text())
    history = read_csv(output / "training_history.csv")
    samples = read_csv(output / "generated_samples.csv")
    colors = {"rnn": "#b76c37", "birnn": "#8a718f", "lstm": "#287c7e", "gru": "#44714d"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for row in comparison:
        kind = row["model"]
        points = [point for point in history if point["model"] == kind]
        epochs = [int(point["epoch"]) for point in points]
        axes[0].plot(epochs, [float(point["train_loss"]) for point in points], "o--", markersize=3, color=colors[kind], alpha=.6)
        axes[0].plot(epochs, [float(point["validation_loss"]) for point in points], "o-", markersize=3, label=MODEL_NAMES[kind], color=colors[kind])
        axes[1].plot(epochs, [100 * float(point["validation_accuracy"]) for point in points], "o-", markersize=3, label=MODEL_NAMES[kind], color=colors[kind])
    axes[0].set(title="Loss: validation (solid), training (dashed)", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation next-token accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(plots / "training_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [MODEL_NAMES[row["model"]] for row in comparison]
    bars = [colors[row["model"]] for row in comparison]
    axes[0].bar(names, [row["test_perplexity"] for row in comparison], color=bars)
    axes[1].bar(names, [100 * row["test_accuracy"] for row in comparison], color=bars)
    axes[0].set(title="Held-out test perplexity (lower is better)", ylabel="Perplexity")
    axes[1].set(title="Held-out test accuracy", ylabel="Top-1 accuracy (%)")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(plots / "model_comparison.png", dpi=150)
    plt.close(fig)

    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["model"], float(sample["temperature"]))].append(sample)
    aggregate = []
    for (kind, temperature), group in grouped.items():
        aggregate.append({"model": kind, "temperature": temperature,
                          **{metric: sum(float(item[metric]) for item in group) / len(group)
                             for metric in ("distinct_1", "distinct_2", "repeated_trigrams")}})
    from .training import write_csv
    write_csv(output / "temperature_metrics.csv", aggregate)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(axes, ("distinct_1", "distinct_2", "repeated_trigrams"),
                                  ("Unique-word ratio", "Unique-bigram ratio", "Repeated-trigram ratio")):
        for row in comparison:
            kind = row["model"]
            points = [point for point in aggregate if point["model"] == kind]
            axis.plot([point["temperature"] for point in points], [point[metric] for point in points],
                      "o-", label=MODEL_NAMES[kind], color=colors[kind])
        axis.set(title=title, xlabel="Temperature", ylim=(-.02, 1.02))
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "temperature_effects.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    words = dataset["top_words"][::-1]
    axis.barh([pair[0] for pair in words], [pair[1] for pair in words], color="#44714d")
    axis.set(title="Most frequent training words", xlabel="Occurrences")
    fig.tight_layout()
    fig.savefig(plots / "word_frequency.png", dpi=150)
    plt.close(fig)

    winner = MODEL_NAMES[run["recommended_model"]]
    lines = ["# Experiment report", "", f"Created: {run['created_utc']}", "",
             f"Training device: `{run['device']}`. Corpus: {dataset['total_tokens']:,} tokens. "
             f"Vocabulary: {dataset['vocabulary_size']:,} tokens fitted only on the training split.", "",
             "The token stream is split sequentially 80%/10%/10% before sliding windows are constructed. "
             "The vocabulary includes an unknown token; test metrics include unknown-token targets. "
             "See dataset.json for each split's unknown rate. Perplexity measures this tokenized vocabulary, "
             "so it should only be compared between runs with the same preprocessing and vocabulary.", "",
             "| Model | Best epoch | Validation loss | Test loss | Test perplexity | Top-1 | Top-5 | Training (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparison:
        lines.append(f"| {MODEL_NAMES[row['model']]} | {row['best_epoch']} | {row['validation_loss']:.3f} | "
                     f"{row['test_loss']:.3f} | {row['test_perplexity']:.1f} | {row['test_accuracy']:.1%} | "
                     f"{row['test_top5_accuracy']:.1%} | {row['training_seconds']:.1f} |")
    lines += ["", f"**Selected model: {winner}**, based on the lowest validation loss. "
              "Test results are reported after selection of each model's best validation checkpoint.", "",
              "![Training curves](plots/training_curves.png)", "",
              "![Model comparison](plots/model_comparison.png)", "",
              "## Temperature experiment", "",
              "All models use the same three prompts, temperatures (0.2, 0.5, 1.0, 1.5), and sampling seeds. "
              "Top-k and top-p filtering are disabled here to isolate temperature. "
              "Diversity and repetition metrics are descriptive; they do not establish story coherence or factual accuracy.", "",
              "![Temperature effects](plots/temperature_effects.png)", "", "## Unedited generated samples", ""]
    for kind in run["completed_models"]:
        for temperature in (.5, 1.):
            sample = next(item for item in samples if item["model"] == kind and float(item["temperature"]) == temperature)
            lines += [f"### {MODEL_NAMES[kind]} at temperature {temperature}", "",
                      f"Prompt: {sample['prompt']}", "", f"> {sample['continuation']}", ""]
    lines += ["## Interpretation and limitations", "",
              "These small models learn local vocabulary and phrase patterns, but can repeat themselves and lose "
              "long-range meaning. A few epochs on a CPU are a demonstration, not evidence of polished story generation. "
              "Unknown words, a single book, a fixed context window, and a single training seed limit this experiment. "
              "Accuracy includes punctuation. Model sizes differ even when hidden dimensions are equal. "
              "Training times include validation and checkpoint writes and reflect current machine load; they are not controlled speed benchmarks. "
              "More rigorous comparisons should repeat training with several seeds and report uncertainty.", "",
              "Inspect the actual samples alongside loss and perplexity; do not infer human-level writing quality from either metric.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "generated_samples.txt").write_text("\n\n".join(
        f"[{sample['model']} | T={sample['temperature']} | sample={sample['sample']}]\n"
        f"Prompt: {sample['prompt']}\n{sample['continuation']}" for sample in samples), encoding="utf-8")
