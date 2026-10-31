#!/usr/bin/env python3
"""Create a deterministic SHA-256 inventory for an experiment archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    files = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if not path.is_file() or resolved == output or path.name.endswith(".tmp"):
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "bytes": size,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
            }
        )
    payload = {
        "files": files,
        "root": ".",
        "total_bytes": total_bytes,
        "total_files": len(files),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
