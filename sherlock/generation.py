from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch

from .data import Vocabulary, detokenize, tokenize
from .model import ModelConfig, NextWordModel


class CheckpointError(RuntimeError):
    """A saved model is unreadable or incompatible with this project."""


def load_model(path: str | Path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format.")
        vocabulary = Vocabulary(checkpoint["vocabulary"])
        config = ModelConfig(**checkpoint["model_config"])
        if config.vocab_size != len(vocabulary):
            raise ValueError("Checkpoint vocabulary does not match the model.")
        model = NextWordModel(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, vocabulary, checkpoint
    except (OSError, RuntimeError, KeyError, IndexError, EOFError, pickle.UnpicklingError, TypeError, ValueError) as error:
        raise CheckpointError(f"Checkpoint could not be loaded: {path}. Train a new run.") from error


@torch.inference_mode()
def generate(model: NextWordModel, vocabulary: Vocabulary, prompt: str, new_tokens: int = 60,
             temperature: float = .8, top_k: int = 40, top_p: float = .95,
             seed: int = 42, greedy: bool = False):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError("Enter a prompt between 1 and 2,000 characters.")
    prompt_tokens = tokenize(prompt)
    if not prompt_tokens or not any(word in vocabulary.lookup for word in prompt_tokens):
        raise ValueError("The prompt needs at least one word or punctuation mark in the vocabulary.")
    if type(new_tokens) is not int or not 1 <= new_tokens <= 300:
        raise ValueError("New tokens must be an integer between 1 and 300.")
    if not math.isfinite(temperature) or not .1 <= temperature <= 2:
        raise ValueError("Temperature must be between 0.1 and 2.0.")
    if type(top_k) is not int or not 0 <= top_k <= 1000:
        raise ValueError("Top-k must be an integer between 0 and 1,000 (0 disables it).")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("Top-p must be greater than 0 and at most 1.")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Random seed must be an integer between 0 and 4,294,967,295.")
    model.eval()
    context = vocabulary.encode(prompt_tokens)
    output = []
    device = next(model.parameters()).device
    # CPU sampling generator is local to this request; concurrent requests do not reset it.
    random = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(new_tokens):
        inputs = torch.tensor([context[-model.config.context_length:]], device=device)
        logits = model(inputs)[0].float().cpu() / temperature
        logits[:2] = -torch.inf  # Never emit padding or the unknown-word marker.
        if greedy:
            index = int(logits.argmax())
        else:
            if top_k:
                k = min(top_k, len(vocabulary) - 2)
                kept, positions = torch.topk(logits, k)
                logits.fill_(-torch.inf).scatter_(0, positions, kept)
            if top_p < 1:
                sorted_logits, positions = logits.sort(descending=True)
                remove = sorted_logits.softmax(0).cumsum(0) > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[positions[remove]] = -torch.inf
            index = int(torch.multinomial(logits.softmax(0), 1, generator=random))
        output.append(vocabulary.tokens[index])
        context.append(index)
    continuation = detokenize(output)
    separator = "" if continuation[:1] in ".,!?;:" else " "
    return {"prompt": prompt.strip(), "continuation": continuation,
            "text": prompt.strip() + separator + continuation, "generated_tokens": len(output),
            "unknown_prompt_tokens": sorted(set(t for t in prompt_tokens if t not in vocabulary.lookup)),
            "model": model.config.kind, "temperature": temperature, "top_k": top_k,
            "top_p": top_p, "seed": seed, "greedy": greedy}


def diversity_metrics(text: str):
    words = [word for word in tokenize(text) if word not in ".,!?;:"]
    bigrams, trigrams = list(zip(words, words[1:])), list(zip(words, words[1:], words[2:]))
    return {"distinct_1": len(set(words)) / len(words) if words else 0,
            "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0,
            "repeated_trigrams": 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0}
