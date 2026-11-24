#!/usr/bin/env python3
"""Download the full-run inputs from the stat236 container into <dest>/world_v1 and rebuild the token-chunk links.

Reads ``<run>/inputs.json`` (written by publish_inputs.py): files (path, bytes, md5) and links (the combined train /
held-out token datasets point at the per-corpus chunk files). Skips files already present with the right size and
MD5, verifies every download, then makes relative symlinks. About 5.8 GB. Standard library only.

  python scripts/full/fetch_inputs.py --sas-file ~/.config/stat236-blob/collab.env:STAT236_COLLAB_SAS --dest $WORK
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blobq import Blob, load_sas  # noqa: E402


def md5_b64(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return base64.b64encode(h.digest()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sas-file", default=os.environ.get("STAT236_SAS_FILE"))
    ap.add_argument("--run", default="full_v1")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    b = Blob(load_sas(a.sas_file), a.run)
    inv, _ = b.get_json(f"{a.run}/inputs.json")
    if inv is None:
        sys.exit("inputs.json not published")

    def fetch(f):
        dst = a.dest / f["path"]
        if dst.exists() and dst.stat().st_size == f["bytes"] and md5_b64(dst) == f["md5"]:
            return "kept"
        dst.parent.mkdir(parents=True, exist_ok=True)
        st, _, body = b.req("GET", f["path"])
        if st != 200:
            raise RuntimeError(f"{f['path']}: HTTP {st}")
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(body)
        if tmp.stat().st_size != f["bytes"] or md5_b64(tmp) != f["md5"]:
            tmp.unlink(); raise RuntimeError(f"{f['path']}: size/MD5 mismatch after download")
        os.replace(tmp, dst)
        return "downloaded"

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(fetch, inv["files"]))
    for ln in inv["links"]:
        link, target = a.dest / ln["link"], a.dest / ln["target"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(target, link.parent))
        assert link.exists(), f"broken link {link}"
    print(f"FETCH_OK files {len(res)} (downloaded {res.count('downloaded')}, already present {res.count('kept')}), "
          f"links {len(inv['links'])}, {sum(f['bytes'] for f in inv['files']) / 1e9:.2f} GB under {a.dest}/world_v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
