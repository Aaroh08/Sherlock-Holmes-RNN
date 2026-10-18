"""Train comparable recurrent next-word predictors and export an experiment report."""
import argparse
from pathlib import Path

from sherlock.data import DATA_PATH, ROOT, download_book
from sherlock.model import MODEL_NAMES
from sherlock.training import run_experiment


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "full")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 uses the complete book; a positive limit is for smoke tests")
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "context_length", "embedding_dim", "hidden_dim", "layers",
                 "patience", "min_frequency", "samples_per_prompt", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.dropout < 1 or not 0 < args.learning_rate <= 1:
        parser.error("dropout must be in [0, 1), and learning-rate in (0, 1]")
    if args.max_tokens < 0 or args.max_vocab < 3 or not 0 <= args.seed <= 2**32 - 1:
        parser.error("invalid token limit, vocabulary size, or random seed")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    try:
        if args.data.resolve() == DATA_PATH.resolve():
            download_book(args.data)
        if not args.data.is_file():
            parser.error(f"Data file not found: {args.data}")
        run_experiment(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
