# Baker Street Text Lab

A complete Sherlock Holmes next-word generation project using **Simple RNN, Bidirectional RNN, LSTM, and GRU**. Train the four architectures under the same conditions, compare held-out results, and explore generated text in a local browser interface.

Inspired by [Anuran Baruah's Sherlock Holmes RNN project](https://github.com/anuranbaruah/Sherlock-Holmes-RNN-text-generation). This is a new modular implementation with its own experiments, a notebook, saved checkpoints, and a web application. The reference author's results are not reused.

## Start the included demo

Use Python 3.10 or newer (tested here with Python 3.12). These commands run from this directory.

```powershell
git clone https://github.com/Aaroh08/Sherlock-Holmes-RNN.git
cd Sherlock-Holmes-RNN
```

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:5001**. This project uses port 5001 so it can coexist with other Flask projects.

The working folder delivered with this project already has a `.venv` and a trained `results/demo` run. In that folder, you can skip installation and run:

```powershell
.\.venv\Scripts\python.exe app.py
```

On Windows, you can also double-click `start.bat` after installation.

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. Activate your environment if you want to use the shorter `python` commands below. The application runs on localhost and uses no API keys or paid services. After dependencies and the book are downloaded, training and generation work offline.

## What is included

- Four word-level recurrent architectures, with a shared training and evaluation pipeline.
- Sequential train/validation/test splits, and a vocabulary fitted on training text only.
- Adam optimization, gradient clipping, early stopping, and best-validation checkpoints.
- Test loss, perplexity, top-1/top-5 accuracy, parameter counts, and training times.
- Temperature experiments at 0.2, 0.5, 1.0, and 1.5 with common prompts and sampling seeds.
- Unique-word/bigram ratios and repeated-trigram ratios, with plots and raw samples.
- Temperature, top-k, nucleus (top-p), greedy decoding, and reproducible generation seeds.
- A notebook walkthrough and a browser writing room with copy/download controls.
- Regression tests for data separation, model learning, checkpoint loading, sampling, and the API.

## Train an experiment

```powershell
python train.py --output results/my-run
```

This trains all four architectures for up to 12 epochs, with early stopping after 3 epochs without improvement. It uses the complete book, a 20-token context, 64-dimensional embeddings, 128 hidden units, one recurrent layer, and dropout 0.2. CPU is supported; CUDA is selected automatically when available. Runtime depends on your hardware.

The included demo was created with:

```powershell
python train.py --output results/demo --epochs 8 --batch-size 256 --threads 4 --samples-per-prompt 2
```

The command refuses to overwrite a nonempty output folder. Choose a new name when retraining. Read the actual run configuration, completed epoch counts, device, and timings in `results/demo/run.json` and `results/demo/model_comparison.csv`; early stopping can finish before the epoch limit.

Larger experiment, closer to the reference architecture:

```powershell
python train.py --output results/large --epochs 25 --embedding-dim 128 --hidden-dim 256 --layers 2 --dropout 0.3 --context-length 15
```

Quick pipeline check on a subset (not a quality benchmark):

```powershell
python train.py --output results/smoke --epochs 1 --max-tokens 3000 --embedding-dim 16 --hidden-dim 24 --samples-per-prompt 1
```

Single model or custom corpus:

```powershell
python train.py --models gru --output results/gru-run
python train.py --data path/to/book.txt --output results/custom
```

Use `python train.py --help` for all settings. To use a CUDA build, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your hardware, then select `--device cuda`. A CPU build cannot use your GPU.

## Generate text

```powershell
python generate.py --checkpoint results/demo/checkpoints/gru.pt --prompt "Sherlock Holmes looked at me and said" --tokens 60 --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42
python app.py --run-dir results/my-run --port 5001
```

The command-line example defaults to GRU; use the checkpoint selected by validation loss in your own run for model selection. The browser lets you switch between all available models.

Low temperature concentrates probability on likely words and can increase repetition. High temperature increases variety and can break grammar. Top-k limits the candidate count; top-p limits candidates to a cumulative probability mass. Use `--top-k 0 --top-p 1` to disable both filters, or `--greedy` for argmax decoding. A token is a word **or punctuation**, so 60 tokens is not exactly 60 words. Unknown prompt words are reported and encoded as `<unk>`; generation never emits `<unk>` or `<pad>`.

## Notebook

```powershell
python -m pip install -r requirements-dev.txt
python -m jupyterlab Sherlock_Text_Generation.ipynb
```

Open `Sherlock_Text_Generation.ipynb` and run cells in order. By default it inspects the included trained run. Set `TRAIN_MODELS = True` in the training cell to create a fresh timestamped experiment. The notebook uses the same Python modules as the CLI, avoiding two implementations that drift apart.

## Method and evaluation

1. Preserve the original downloaded book and its license. Strip Gutenberg wrappers and the front matter only in memory.
2. Normalize case and apostrophes; tokenize words, contractions, and `. , ! ? ; :` punctuation.
3. Split the token stream sequentially into 80% training, 10% validation, and 10% test.
4. Fit vocabulary counts only on the training split. Rare and unseen words map to `<unk>`.
5. Build input windows independently inside each split: 20 past tokens predict one next token.
6. Train each architecture with the same hyperparameters, random seed, and shuffled sample order.
7. Restore each model's lowest-validation-loss checkpoint; evaluate it on the held-out test split.
8. Select the recommended architecture by **validation loss**, not test loss.

All architectures use `Embedding → recurrent block → Dropout → Linear(vocabulary)`. The bidirectional model sees both directions of the **past context only**, never the target. It concatenates the final forward and backward hidden states. Parameter counts differ even with shared hidden dimensions; this is not a matched-parameter study.

Perplexity is `exp(mean cross-entropy)`. Test metrics include unknown targets and punctuation; the unknown-word rates are saved in `dataset.json`. A smaller or more aggressively filtered vocabulary can artificially improve perplexity, so compare only runs with identical preprocessing. Diversity metrics do not measure semantic coherence. The included run uses one seed; multiple training seeds are needed to estimate uncertainty.

Training times include validation and checkpoint writes and depend on other work running on the machine. Treat them as observed wall-clock durations, not controlled speed benchmarks.

**Expected quality:** these are small recurrent models trained on one book. They learn local wording but can repeat phrases, make grammatical errors, and lose the plot. They should not be presented as producing polished, coherent stories. Read the unedited generated samples alongside the metrics.

## Outputs

```text
sherlock-text-generator/
├── app.py                         # Flask UI and API
├── train.py                       # Reproducible experiment CLI
├── generate.py                    # Generation CLI
├── Sherlock_Text_Generation.ipynb  # Notebook walkthrough
├── sherlock/                      # Data, models, training, sampling, reports
├── data/sherlock_holmes.txt        # Gutenberg original, including license
├── templates/ and static/         # Browser interface; no CDN dependencies
├── tests/test_project.py
├── requirements.txt / requirements-dev.txt
└── results/demo/
    ├── checkpoints/{rnn,birnn,lstm,gru}.pt
    ├── run.json / dataset.json / vocabulary.json
    ├── model_comparison.csv / comparison.json
    ├── training_history.csv
    ├── generated_samples.csv / generated_samples.txt
    ├── temperature_metrics.csv
    ├── REPORT.md
    └── plots/*.png
```

Checkpoints bundle the model architecture, weights, vocabulary, context length, and data hash. Only load checkpoints from sources you trust. Model loading uses PyTorch's `weights_only=True`.

## API and tests

- `GET /health`: server health and checkpoint availability.
- `GET /api/status`: available models, dataset statistics, and measured results.
- `POST /api/generate`: JSON prompt and sampling settings.
- `GET /api/plots/<name>`: one of the four generated charts.

```json
{"model": "gru", "prompt": "Sherlock Holmes was", "new_tokens": 60, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "seed": 42}
```

```powershell
python -m pytest -q
```

The app starts before training, with an explicit setup state. It never trains implicitly on a web request. Prompts are limited to 2,000 characters and continuations to 300 tokens. Checkpoints are not served by the web app.

## Share the project

Run `python package_project.py` after the demo completes to create `sherlock-text-generator.zip` in the parent folder. It includes the source, dataset notice, notebook, measured results, plots, and four demo checkpoints. Virtual environments and caches are excluded. On another computer, install the requirements before launching the app. The included demo checkpoints are tracked in Git; checkpoints from other experiments are ignored by default.

## Dataset and attribution

The dataset is *The Adventures of Sherlock Holmes* by Arthur Conan Doyle, [Project Gutenberg eBook #1661](https://www.gutenberg.org/ebooks/1661). The original downloaded file retains the Project Gutenberg notice and license; see `data/SOURCE.md` for provenance. The project is inspired by the linked four-model comparison, but the code, UI, run artifacts, and conclusions are generated independently. No course number, student identity, or results from the reference are claimed as this project's own.
