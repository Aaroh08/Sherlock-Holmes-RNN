from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sherlock_holmes.txt"
DATA_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"


def download_book(path: Path = DATA_PATH) -> Path:
    """Keep Gutenberg's original header and license in the downloaded file."""
    path = Path(path)
    if path.exists():
        return path
    request = Request(DATA_URL, headers={"User-Agent": "SherlockTextLab/1.0"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    if b"PROJECT GUTENBERG" not in raw.upper():
        raise ValueError("The download did not contain the expected Gutenberg text.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return path


def clean_book(raw: str) -> str:
    # Both old ('THIS') and new ('THE') Gutenberg wrappers are supported.
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, re.I)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, re.I)
    book = raw[start.end() if start else 0:end.start() if end else len(raw)]
    # Remove front matter only when the actual first-story opening is found.
    opening = re.search(r"\bTo Sherlock Holmes she is always the woman\.", book, re.I)
    return book[opening.start():].strip() if opening else book.strip()


def tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans({"’": "'", "‘": "'", "—": " ", "–": " ", "_": " "}))
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*|[.,!?;:]", text)


def detokenize(tokens: list[str]) -> str:
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(tokens))


@dataclass
class Vocabulary:
    tokens: list[str]

    def __post_init__(self):
        if self.tokens[:2] != ["<pad>", "<unk>"] or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary must begin with unique <pad> and <unk> tokens.")
        self.lookup = {word: index for index, word in enumerate(self.tokens)}

    @classmethod
    def fit(cls, train_tokens: list[str], min_frequency: int = 2, max_size: int = 8000):
        if min_frequency < 1 or max_size < 3:
            raise ValueError("min_frequency must be positive and max_size at least 3.")
        counts = Counter(train_tokens)
        words = sorted((word for word, count in counts.items() if count >= min_frequency),
                       key=lambda word: (-counts[word], word))[:max_size - 2]
        if not words:
            raise ValueError("No words survived vocabulary filtering; lower --min-frequency.")
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.lookup.get(word, 1) for word in tokens]

    def __len__(self):
        return len(self.tokens)


class NextWordDataset(Dataset):
    """Sliding views are created inside each split, never across split boundaries."""

    def __init__(self, ids: list[int], context_length: int):
        if context_length < 1 or len(ids) <= context_length:
            raise ValueError("Each split must contain more tokens than the context length.")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.ids) - self.context_length

    def __getitem__(self, index):
        return self.ids[index:index + self.context_length], self.ids[index + self.context_length]


def prepare_data(path: Path, context_length: int = 20, min_frequency: int = 2,
                 max_vocab: int = 8000, max_tokens: int = 0):
    raw = Path(path).read_text(encoding="utf-8-sig")
    tokens = tokenize(clean_book(raw))
    if max_tokens:
        tokens = tokens[:max_tokens]
    if len(tokens) < 10 * (context_length + 1):
        raise ValueError("Corpus is too small for an 80/10/10 split with this context length.")
    first, second = int(len(tokens) * .8), int(len(tokens) * .9)
    splits = {"train": tokens[:first], "validation": tokens[first:second], "test": tokens[second:]}
    vocabulary = Vocabulary.fit(splits["train"], min_frequency, max_vocab)
    datasets = {name: NextWordDataset(vocabulary.encode(part), context_length)
                for name, part in splits.items()}
    summary = {
        "source": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "total_tokens": len(tokens), "vocabulary_size": len(vocabulary),
        "context_length": context_length, "vocabulary_fit": "training split only",
        "split_boundaries": [0, first, second, len(tokens)],
        "splits": {name: {"tokens": len(part), "sequences": len(datasets[name]),
                           "unknown_rate": sum(i == 1 for i in vocabulary.encode(part)) / len(part)}
                   for name, part in splits.items()},
        "top_words": Counter(word for word in splits["train"] if word.isalpha()).most_common(20),
    }
    return vocabulary, datasets, summary
