"""Build a shareable source ZIP without proprietary/local data or run outputs."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "dist/doppler-garch-research-source.zip"
    dest.parent.mkdir(exist_ok=True)
    files = [root / name for name in ("README.md", "DATA_STATUS.md", "LICENSE", "Dockerfile",
                                      "pyproject.toml", "config.example.json", ".gitignore")]
    for dirname in ("src", "tests", "scripts", "data", "research", "examples"):
        files.extend(root.joinpath(dirname).rglob("*"))
    with ZipFile(dest, "w", ZIP_DEFLATED) as z:
        for file in sorted(files):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix not in {".pyc", ".pyo"}:
                z.write(file, file.relative_to(root))
    print(dest)


if __name__ == "__main__":
    main()
