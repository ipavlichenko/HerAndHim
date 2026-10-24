#!/usr/bin/env python3
"""Install the canonical Sarah identity without retaining conflicting templates."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import tempfile
from datetime import UTC, datetime

IDENTITY_DIRECTORIES = {
    "context/soul": ("SOUL.md",),
    "context/persona": ("persona.md", "appearance.md"),
    "context/profile": ("PROFILE.md",),
    "context/outbound": ("dialogue_identity.md",),
}


def install(seed_dir: pathlib.Path, data_dir: pathlib.Path, uid: int, gid: int) -> pathlib.Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archive = data_dir / "identity-archive" / timestamp
    archive.mkdir(parents=True, mode=0o700)

    for relative_dir, source_names in IDENTITY_DIRECTORIES.items():
        sources = [seed_dir / source_name for source_name in source_names]
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
        target_dir = data_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        prior = archive / relative_dir
        prior.mkdir(parents=True, exist_ok=True)
        for old in target_dir.iterdir():
            if old.is_file() and old.suffix.lower() in {".md", ".txt"}:
                shutil.move(str(old), prior / old.name)
        for source in sources:
            target_name = source.name
            fd, temporary = tempfile.mkstemp(prefix=f".{target_name}.", dir=target_dir)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(source.read_bytes())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o640)
                os.chown(temporary, uid, gid)
                os.replace(temporary, target_dir / target_name)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    city_source = seed_dir / "RU.json"
    city_dir = data_dir / "data" / "cities"
    city_dir.mkdir(parents=True, exist_ok=True)
    city_target = city_dir / "RU.json"
    if city_target.exists():
        shutil.copy2(city_target, archive / "RU.json")
    shutil.copyfile(city_source, city_target)
    os.chmod(city_target, 0o640)
    os.chown(city_target, uid, gid)

    for path in (archive, archive.parent, data_dir / "context", data_dir / "data", city_dir):
        os.chown(path, uid, gid)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=pathlib.Path, default=pathlib.Path(__file__).with_name("seed"))
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("/srv/sara/data"))
    parser.add_argument("--uid", type=int, default=10001)
    parser.add_argument("--gid", type=int, default=10001)
    args = parser.parse_args()
    archive = install(args.seed_dir, args.data_dir, args.uid, args.gid)
    print(f"canonical identity installed; prior identity archived at {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
