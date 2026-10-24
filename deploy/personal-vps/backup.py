#!/usr/bin/env python3
"""Create and verify a content-consistent local pilot backup."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone

DATA = pathlib.Path("/srv/sara/data")
BACKUPS = pathlib.Path("/srv/sara/backups")
KEEP = 7


def _copy(source: pathlib.Path, target: pathlib.Path) -> None:
    if source.is_symlink():
        raise RuntimeError(f"symlink is not allowed in pilot data: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
            src.backup(dst)
        return
    shutil.copy2(source, target)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tree(root: pathlib.Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            with sqlite3.connect(path) as db:
                result = db.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {path}")


def main() -> int:
    os.umask(0o077)
    if not (DATA / "herandhim.json").is_file():
        raise FileNotFoundError(f"pilot config missing under {DATA}")
    BACKUPS.mkdir(parents=True, exist_ok=True, mode=0o700)
    BACKUPS.chmod(0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = BACKUPS / f"sara-{stamp}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="sara-backup-") as temporary:
        stage = pathlib.Path(temporary) / "data"
        for source in DATA.rglob("*"):
            relative = source.relative_to(DATA)
            if source.is_dir():
                (stage / relative).mkdir(parents=True, exist_ok=True)
            elif source.is_file() or source.is_symlink():
                _copy(source, stage / relative)
        verify_tree(stage)
        with tarfile.open(archive, "w:gz") as output:
            output.add(stage, arcname="data", recursive=True)
    archive.chmod(0o600)
    manifest = {
        "archive": archive.name,
        "createdAt": stamp,
        "sha256": _sha256(archive),
        "bytes": archive.stat().st_size,
    }
    manifest_path = archive.with_suffix(archive.suffix + ".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path.chmod(0o600)
    archives = sorted(BACKUPS.glob("sara-*.tar.gz"), reverse=True)
    for old in archives[KEEP:]:
        old.unlink()
        old.with_suffix(old.suffix + ".json").unlink(missing_ok=True)
    print(json.dumps(manifest, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

