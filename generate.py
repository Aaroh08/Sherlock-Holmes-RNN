"""Generate text from one saved recurrent model."""
import argparse
from pathlib import Path

import torch

from sherlock.data import ROOT
from sherlock.generation import CheckpointError, generate, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/demo/checkpoints/gru.pt")
    parser.add_argument("--prompt", default="Sherlock Holmes looked at me and said")
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        model, vocabulary, _ = load_model(args.checkpoint)
        result = generate(model, vocabulary, args.prompt, args.tokens, args.temperature,
                          args.top_k, args.top_p, args.seed, args.greedy)
        print(result["text"])
        if result["unknown_prompt_tokens"]:
            print("\nWords outside the training vocabulary:", ", ".join(result["unknown_prompt_tokens"]))
    except (OSError, ValueError, CheckpointError) as error:
        parser.exit(1, f"Error: {error}\nTrain first with: python train.py --output results/my-run\n")


if __name__ == "__main__":
    main()
