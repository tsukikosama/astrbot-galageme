"""Build and verify a source-only AstrBot install ZIP without extra dependencies."""

import hashlib
import json
import re
import zipfile
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    metadata = (root / "metadata.yaml").read_text(encoding="utf-8")
    version = re.search(r"^version:\s*v?([0-9]+\.[0-9]+\.[0-9]+)\s*$", metadata, re.MULTILINE)
    if not version:
        raise ValueError("Missing valid plugin version")
    destination = root.parent / f"{root.name}-v{version.group(1)}.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob("*")):
            relative = file.relative_to(root)
            i18n = relative.parts[:2] == (".astrbot-plugin", "i18n")
            if not file.is_file() or (
                not i18n
                and any(part.startswith(".") or part == "__pycache__" for part in relative.parts)
            ):
                continue
            if file.suffix not in {
                ".py",
                ".md",
                ".yaml",
                ".json",
                ".txt",
                ".toml",
                ".html",
                ".css",
                ".js",
                ".png",
            }:
                continue
            archive.write(file, (Path(root.name) / relative).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity failure")
        names = archive.namelist()
        for required in (
            "main.py",
            "metadata.yaml",
            "_conf_schema.json",
            "README.md",
            "views/relationships/index.html",
            "views/relationships/app.js",
            "views/relationships/style.css",
            ".astrbot-plugin/i18n/zh-CN.json",
        ):
            if f"{root.name}/{required}" not in names:
                raise ValueError(f"Missing required plugin entry: {required}")
        schema = json.loads(archive.read(f"{root.name}/_conf_schema.json"))
        assert schema["enabled_groups"]["default"] == []
        for name in names:
            if name.endswith(".py"):
                compile(archive.read(name), name, "exec")
            if "__pycache__" in name or name.endswith(".sqlite3"):
                raise ValueError("Runtime data must not be packaged")
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {destination.name}\n", encoding="utf-8"
    )
    print(f"Built {destination.name}: {len(names)} files, {destination.stat().st_size} bytes")
    print(f"SHA256 {checksum}")


if __name__ == "__main__":
    build()
