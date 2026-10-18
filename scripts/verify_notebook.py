"""Execute the walkthrough with its default run, or verify against another run."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = ROOT / "Sherlock_Text_Generation.ipynb"
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    if args.run_dir:
        if not args.output:
            parser.error("--run-dir requires a separate --output so the original notebook stays portable.")
        for cell in notebook.cells:
            if cell.cell_type == "code" and 'RUN_DIR = ROOT / "results" / "demo"' in cell.source:
                cell.source = cell.source.replace('RUN_DIR = ROOT / "results" / "demo"',
                                                  f"RUN_DIR = Path({str(args.run_dir.resolve())!r})")
    NotebookClient(notebook, timeout=180, resources={"metadata": {"path": str(ROOT)}}).execute()
    output = args.output or source
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Executed all code cells without errors: {output}")


if __name__ == "__main__":
    main()
