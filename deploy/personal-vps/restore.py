#!/usr/bin/env python3
"""Safely extract and verify a pilot backup outside the live data path."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sqlite3
import tarfile
import tempfile


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: tarfile.TarFile, target: pathlib.Path) -> None:
    root = target.resolve()
    for member in archive.getmembers():
        destination = (target / member.name).resolve()
        if root not in destination.parents and destination != root:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"links are forbidden in backup: {member.name}")
    archive.extractall(target, filter="data")


def _verify(root: pathlib.Path) -> None:
    if not (root / "data" / "herandhim.json").is_file():
        raise RuntimeError("backup does not contain data/herandhim.json")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            with sqlite3.connect(path) as db:
                result = db.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=pathlib.Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.expected_sha256 and _sha256(args.archive) != args.expected_sha256:
        raise RuntimeError("archive SHA-256 does not match")
    with tempfile.TemporaryDirectory(prefix="sara-restore-check-") as temporary:
        root = pathlib.Path(temporary)
        with tarfile.open(args.archive, "r:gz") as archive:
            _safe_extract(archive, root)
        _verify(root)
    print(f"restore verification passed: {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

