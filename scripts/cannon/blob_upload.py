"""Upload a directory to Azure Blob and verify every file (size + MD5 read back); write an index without the SAS.

  python scripts/cannon/blob_upload.py --src DIR --dest stat236/<name> --index outputs/<name>_blob_index.json

Credential: AZURE_ARCHIVE_{ACCOUNT,CONTAINER,SAS} from ~/.config/infinigen-archive/sas.env (all whitespace and CRs
stripped). The SAS is never printed or written. Symlinks are followed (a combined tokens dir holds symlinked chunks);
``--skip-glob`` leaves matching relative paths out. Exits non-zero if any file fails verification.
"""
from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from azure.storage.blob import ContainerClient, ContentSettings


def creds() -> tuple[str, str, str]:
    env = {}
    for line in Path(os.path.expanduser("~/.config/infinigen-archive/sas.env")).read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?(\w+)=(.*)", line)
        if m:
            env[m.group(1)] = re.sub(r"[\s\"']", "", m.group(2))
    return env["AZURE_ARCHIVE_ACCOUNT"], env["AZURE_ARCHIVE_CONTAINER"], env["AZURE_ARCHIVE_SAS"].lstrip("?")


def md5_b64(p: Path) -> tuple[str, bytes]:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return base64.b64encode(h.digest()).decode(), h.digest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dest", required=True, help="blob prefix inside the container, e.g. stat236/pilot_<jobid>")
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--skip-glob", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    account, container, sas = creds()
    cc = ContainerClient(f"https://{account}.blob.core.windows.net", container, credential=sas)
    files = sorted(p for p in a.src.rglob("*") if p.is_file())
    files = [p for p in files if not any(fnmatch.fnmatch(str(p.relative_to(a.src)), g) for g in a.skip_glob)]

    def one(p: Path) -> dict:
        rel = p.relative_to(a.src).as_posix()
        name = f"{a.dest.rstrip('/')}/{rel}"
        b64, raw = md5_b64(p)
        size = p.stat().st_size
        bc = cc.get_blob_client(name)
        with open(p, "rb") as f:
            bc.upload_blob(f, overwrite=True, length=size, max_concurrency=4,
                           content_settings=ContentSettings(content_md5=bytearray(raw)))
        prop = bc.get_blob_properties()
        got = base64.b64encode(bytes(prop.content_settings.content_md5 or b"")).decode()
        ok = prop.size == size and got == b64
        return {"blob": f"https://{account}.blob.core.windows.net/{container}/{name}", "bytes": size, "md5_b64": b64, "verified": ok}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, files))
    a.index.parent.mkdir(parents=True, exist_ok=True)
    a.index.write_text(json.dumps(rows, indent=1))
    bad = [r for r in rows if not r["verified"]]
    total = sum(r["bytes"] for r in rows)
    print(f"BLOB_UPLOAD files={len(rows)} bytes={total} failed={len(bad)} dest={a.dest} index={a.index}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
