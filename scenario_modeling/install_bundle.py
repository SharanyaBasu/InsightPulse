"""Install the pinned Ridge release without loading executable model artifacts."""

import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

VERSION = "v1.0.0"
URL = "https://github.com/SharanyaBasu/InsightPulse/releases/download/scenario-playground-ridge-v1.0.0/scenario-playground-ridge-v1.0.0.zip"
SHA256 = "ecfc0e5d6fbdfe1ee09558044a6e0b3b04afa39522b0fda4795553b4d8f6dc11"
DEFAULT_DESTINATION = Path(__file__).resolve().parent / "artifacts" / "scenario_playground_ridge" / VERSION


def install_bundle(destination: Path, archive: Path | None = None) -> Path:
    """Verify the release ZIP and install it atomically, preserving existing files."""
    if archive:
        data = archive.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Release checksum mismatch; no model files were installed.")
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        for item in bundle.infolist():
            if item.is_dir():
                continue
            path = PurePosixPath(item.filename.replace("\\", "/"))
            if len(path.parts) != 2 or path.parts[0] != VERSION or path.parts[1] in (".", ".."):
                raise ValueError(f"Unexpected release path: {item.filename}")
            files[path.name] = bundle.read(item)
    destination = destination.resolve()
    if destination.exists():
        for name, contents in files.items():
            existing = destination / name
            if not existing.is_file() or existing.read_bytes() != contents:
                raise ValueError(f"Existing bundle differs at {name}; use a separate destination.")
        print(f"Existing bundle verified: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / VERSION
        staged.mkdir()
        for name, contents in files.items():
            (staged / name).write_bytes(contents)
        shutil.move(str(staged), str(destination))
    print(f"Verified bundle installed: {destination}")
    return destination


def main() -> None:
    """Download the pinned release or verify a previously downloaded ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing release ZIP instead of downloading")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    install_bundle(args.destination, args.archive)


if __name__ == "__main__":
    main()
