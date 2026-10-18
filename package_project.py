"""Create a portable project ZIP, including the completed demo but no virtual environment."""
from pathlib import Path
import json
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent


def main():
    manifest = ROOT / "results/demo/run.json"
    if not manifest.exists() or json.loads(manifest.read_text())["status"] != "complete":
        raise SystemExit("Finish the results/demo experiment before packaging it.")
    destination = ROOT.parent / "sherlock-text-generator.zip"
    included_files = ["app.py", "train.py", "generate.py", "package_project.py", "start.bat",
                      "README.md", "LICENSE", ".gitignore", ".gitattributes", "requirements.txt", "requirements-dev.txt",
                      "Sherlock_Text_Generation.ipynb"]
    included_directories = ["sherlock", "data", "templates", "static", "tests", "scripts", "results/demo"]
    files = [ROOT / name for name in included_files]
    for directory in included_directories:
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix not in {".pyc", ".tmp"})
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive verification failed.")
    print(f"Created {destination} ({destination.stat().st_size / 1024**2:.1f} MiB, {len(files)} files)")


if __name__ == "__main__":
    main()
