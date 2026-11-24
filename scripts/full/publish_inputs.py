#!/usr/bin/env python3
"""(Michelle, once) Upload the full-run inputs that Blob lacks and write <run>/inputs.json for fetch_inputs.py.

<src>/world_v1 holds a copy of the corpus with the full-run manifests (``subset --parent <split>/base --out
<split>/full --seed 0``: every dedup-kept prompt, shuffled) and the combined token datasets, whose chunk files are
symlinks to the per-corpus chunks. Files are uploaded create-only (existing blobs are left alone and must match).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blobq import Blob, load_sas  # noqa: E402
from fetch_inputs import md5_b64  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sas-file", required=True)
    ap.add_argument("--run", default="full_v1")
    ap.add_argument("--src", type=Path, required=True, help="dir containing world_v1/")
    ap.add_argument("--cannon-root", default="/n/netscratch/hoffman_lab/Lab/michelleliu/stat236")
    a = ap.parse_args()
    b = Blob(load_sas(a.sas_file), a.run)
    W = a.src / "world_v1"
    files, links = [], []
    for split in ("train", "heldout"):
        for f in sorted((W / split / "full").iterdir()):
            files.append(f)
        for name in ("index.sqlite", "metadata.json"):
            files.append(W / split / "tokens" / name)
        for ln in sorted((W / split / "tokens" / "layers" / "tokens").iterdir()):
            tgt = os.readlink(ln)
            rel = os.path.relpath(tgt, a.cannon_root) if tgt.startswith(a.cannon_root) else os.path.relpath(
                (ln.parent / tgt).resolve(), a.src)
            links.append({"link": str(ln.relative_to(a.src)), "target": rel})
    targets = sorted({ln["target"] for ln in links})
    out = []
    for t in targets:   # the per-corpus chunks are in Blob already (corpus job 63405315): take size + MD5 from there
        h = b.head(t)
        assert h is not None and h.get("Content-MD5"), f"chunk {t} missing in Blob or has no MD5"
        out.append({"path": t, "bytes": int(h["Content-Length"]), "md5": h["Content-MD5"]})
    for f in files:
        rel = str(f.relative_to(a.src))
        r = b.put_file(f, rel)
        if r["status"] == "exists" and not r["same"]:
            sys.exit(f"{rel}: a different file is already in Blob; not overwriting")
        out.append({"path": rel, "bytes": r["bytes"], "md5": r["md5"]})
        print(r["status"], rel, flush=True)
    inv = {"files": out, "links": links,
           "note": "full-run inputs: <split>/full manifests, combined token datasets <split>/tokens (chunks linked)"}
    ok = b.create_json(f"{a.run}/inputs.json", inv)
    print(f"INPUTS {'published' if ok else 'ALREADY EXIST (not overwritten)'}: {len(out)} files, {len(links)} links, "
          f"{sum(x['bytes'] for x in out) / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
