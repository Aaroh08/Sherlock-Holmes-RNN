import pytest
import torch

from app import create_app
from sherlock.data import Vocabulary, clean_book, prepare_data, tokenize
from sherlock.generation import generate, load_model
from sherlock.model import MODEL_NAMES, ModelConfig, NextWordModel

torch.set_num_threads(1)


@pytest.fixture
def checkpoint(tmp_path):
    vocabulary = Vocabulary(["<pad>", "<unk>", "holmes", "watson", "said", ".", "the"])
    torch.manual_seed(10)
    config = ModelConfig("gru", len(vocabulary), embedding_dim=8, hidden_dim=12, context_length=4)
    model = NextWordModel(config).eval()
    (tmp_path / "checkpoints").mkdir()
    path = tmp_path / "checkpoints/gru.pt"
    torch.save({"format_version": 1, "model_config": config.to_dict(),
                "state_dict": model.state_dict(), "vocabulary": vocabulary.tokens}, path)
    return tmp_path, path, model, vocabulary


def test_book_cleaning_preserves_custom_text_and_strips_both_wrappers():
    for article in ("THE", "THIS"):
        raw = f"header\n*** START OF {article} PROJECT GUTENBERG EBOOK TITLE ***\ncontents\nTo Sherlock Holmes she is always the woman.\nStory\n*** END OF {article} PROJECT GUTENBERG EBOOK TITLE ***\nlicense"
        assert clean_book(raw) == "To Sherlock Holmes she is always the woman.\nStory"
    assert clean_book("A custom story.") == "A custom story."
    assert tokenize("Holmes’s _case_—isn't closed!") == ["holmes's", "case", "isn't", "closed", "!"]


def test_vocabulary_and_sequences_do_not_leak_across_splits(tmp_path):
    path = tmp_path / "book.txt"
    words = [f"word{index}" for index in range(300)]
    path.write_text(" ".join(words))
    vocabulary, datasets, summary = prepare_data(path, context_length=5, min_frequency=1)
    assert "word239" in vocabulary.lookup
    assert "word240" not in vocabulary.lookup
    assert "word270" not in vocabulary.lookup
    assert summary["split_boundaries"] == [0, 240, 270, 300]
    assert len(datasets["train"]) == 235
    last_input, target = datasets["train"][234]
    assert last_input.tolist() == vocabulary.encode(words[234:239])
    assert int(target) == vocabulary.lookup["word239"]
    assert datasets["validation"][0][0].tolist() == [1] * 5


def test_small_data_is_actionable(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("holmes said hello")
    with pytest.raises(ValueError, match="too small"):
        prepare_data(path)


@pytest.mark.parametrize("kind", MODEL_NAMES)
def test_every_architecture_can_learn_and_reload(kind, tmp_path):
    torch.manual_seed(4)
    config = ModelConfig(kind, 7, embedding_dim=8, hidden_dim=12, layers=2, dropout=0., context_length=4)
    model = NextWordModel(config)
    x = torch.tensor([[2, 3, 4, 5], [2, 3, 4, 5]])
    y = torch.tensor([6, 6])
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    initial = torch.nn.functional.cross_entropy(model(x), y).item()
    for _ in range(12):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    assert torch.nn.functional.cross_entropy(model(x), y).item() < initial / 2
    path = tmp_path / f"{kind}.pt"
    torch.save({"format_version": 1, "model_config": config.to_dict(), "state_dict": model.state_dict(),
                "vocabulary": ["<pad>", "<unk>", "one", "two", "three", "four", "five"]}, path)
    restored, _, _ = load_model(path)
    model.eval()
    torch.testing.assert_close(restored(x), model(x))


def test_seeded_sampling_is_repeatable_and_does_not_modify_global_rng(checkpoint):
    _, _, model, vocabulary = checkpoint
    before = torch.random.get_rng_state().clone()
    first = generate(model, vocabulary, "Holmes said", new_tokens=25, seed=123)
    generate(model, vocabulary, "Watson said", new_tokens=25, seed=999)
    second = generate(model, vocabulary, "Holmes said", new_tokens=25, seed=123)
    assert first == second
    assert first["generated_tokens"] == 25
    assert "<pad>" not in first["text"] and "<unk>" not in first["text"]
    assert torch.equal(before, torch.random.get_rng_state())
    assert generate(model, vocabulary, "Holmes mystery", new_tokens=1)["unknown_prompt_tokens"] == ["mystery"]


def test_top_k_one_matches_greedy_and_nucleus_keeps_at_least_one_token(checkpoint):
    _, _, model, vocabulary = checkpoint
    first = generate(model, vocabulary, "holmes", new_tokens=12, greedy=True)
    second = generate(model, vocabulary, "holmes", new_tokens=12, top_k=1, seed=999)
    assert first["text"] == second["text"]
    assert generate(model, vocabulary, "holmes", top_p=.0001, new_tokens=10)["generated_tokens"] == 10


@pytest.mark.parametrize("options", [{"new_tokens": 0}, {"new_tokens": True}, {"new_tokens": 301},
                                    {"temperature": float("nan")}, {"temperature": 0}, {"top_p": 0},
                                    {"top_p": float("inf")}, {"top_k": -1}, {"seed": -1}])
def test_sampling_rejects_invalid_parameters(checkpoint, options):
    _, _, model, vocabulary = checkpoint
    with pytest.raises(ValueError):
        generate(model, vocabulary, "holmes", **options)


def test_empty_and_unknown_prompts_are_rejected(checkpoint):
    _, _, model, vocabulary = checkpoint
    for prompt in ("", "   ", "zzzzzz", "!@#$%^", "holmes " * 500):
        with pytest.raises(ValueError):
            generate(model, vocabulary, prompt)


def test_web_starts_without_models_and_does_not_train(tmp_path):
    client = create_app(tmp_path).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/status").json["ready"] is False
    assert client.get("/health").json["model_ready"] is False
    assert client.post("/api/generate", json={"prompt": "holmes"}).status_code == 503
    assert list(tmp_path.iterdir()) == []


def test_web_generation_and_input_validation(checkpoint):
    root, _, _, _ = checkpoint
    client = create_app(root).test_client()
    assert client.get("/api/status").json["models"] == ["gru"]
    result = client.post("/api/generate", json={"prompt": "Holmes said", "new_tokens": 10})
    assert result.status_code == 200 and result.json["generated_tokens"] == 10
    for bad in ([], None, "string", {"model": "../../secret"}, {"model": []}, {"prompt": 3},
                {"prompt": "holmes", "temperature": "hot"}, {"prompt": "holmes", "new_tokens": 2.2},
                {"prompt": "holmes", "greedy": "yes"}):
        assert client.post("/api/generate", json=bad).status_code == 400
    assert client.get("/api/plots/gru.pt").status_code == 404
    assert client.post("/api/generate", json={"prompt": "x" * 20000}).status_code == 413


def test_corrupt_checkpoint_is_an_actionable_service_error(checkpoint):
    root, path, _, _ = checkpoint
    path.write_bytes(b"broken checkpoint")
    response = create_app(root).test_client().post("/api/generate", json={"prompt": "holmes"})
    assert response.status_code == 503
    assert "could not be loaded" in response.json["error"]
