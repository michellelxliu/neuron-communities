#!/usr/bin/env python3
"""Work queue and uploads for the full harvest, shared by several people through the ``stat236`` Blob container.

Standard library only (urllib + the Blob REST API), so it runs anywhere with Python 3.9+. Works with a SAS that can
read and write but not list or delete (the collaborator link): every key is computed, never listed.

Layout under ``<run>/`` (default ``full_v1/``):
  config.json                         run config, written once by Michelle (``publish-config``): splits, parts, flags,
                                      code commit, model; workers refuse to run if their checkout differs
  claims/<split>_<chunk:05d>.json     who is harvesting a chunk ({owner, job, host, time}); created atomically with
                                      If-None-Match: * so two workers can never claim the same chunk
  done/<split>_<chunk:05d>.json       written after every part of the chunk is uploaded and verified
  <split>/parts/part_XXXXXX.parquet   the harvest parts (same names as a single-machine harvest)
  <split>/run_meta/chunk_<c>_<owner>.json, logs/<owner>/<split>_<c>.log

Every write is create-only (If-None-Match: *); nothing here overwrites a blob, except taking over a claim, which is
conditional on its ETag. A part that already exists is left alone (first writer wins; a re-harvest of the same part is
equivalent up to GPU nondeterminism).

Commands (SAS from --sas-file FILE[:VAR] or $STAT236_SAS):
  claim   --owner O --job J [--live-jobs a,b]  print "<split> <chunk> <part_lo> <part_hi>" of the next free chunk and
          claim it; exit 3 when none is left. A chunk claimed by O under a job not in --live-jobs (a job that died) is
          taken back; a released claim is free to anyone.
  put     FILE BLOB_PATH                        create-only upload with Content-MD5, then read back size + MD5
  done    --split S --chunk C --owner O [--info JSON]
  release --split S --chunk C --owner O        mark your own claim free (no delete permission needed)
  status                                       counts and per-owner progress
  publish-config CONFIG.json                   (Michelle) write config.json once
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ACCOUNT, CONTAINER = "infinigenexploration", "stat236"
BASE = f"https://{ACCOUNT}.blob.core.windows.net/{CONTAINER}"
VERSION = "2021-08-06"


def load_sas(spec: str | None) -> str:
    if not spec:
        v = os.environ.get("STAT236_SAS", "")
    else:
        path, _, var = spec.partition(":")
        v = ""
        for line in Path(path).expanduser().read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, eq, val = line.partition("=")
            if eq and (not var or k.strip() == var):
                v = val.strip()
                break
        else:
            if not var:
                v = Path(path).expanduser().read_text().strip()
    v = v.strip().strip('"').strip("'").lstrip("?")
    if "sig=" not in v:
        sys.exit("blobq: no SAS found (pass --sas-file FILE[:VAR] or set STAT236_SAS)")
    return v


class Blob:
    def __init__(self, sas: str, run: str):
        self.sas, self.run = sas, run.strip("/")

    def url(self, path: str) -> str:
        return f"{BASE}/{urllib.request.quote(path)}?{self.sas}"

    def req(self, method: str, path: str, data=None, headers: dict | None = None, ok=(200, 201, 202, 206),
            tries: int = 6, timeout: int = 900):
        """Returns (status, headers, body). 404/409/412 are returned, not raised; 5xx and network errors are retried."""
        h = {"x-ms-version": VERSION, **(headers or {})}
        for attempt in range(tries):
            body = data() if callable(data) else data
            r = urllib.request.Request(self.url(path), data=body, method=method, headers=h)
            try:
                with urllib.request.urlopen(r, timeout=timeout) as resp:
                    return resp.status, dict(resp.headers), resp.read()
            except urllib.error.HTTPError as e:
                if e.code in (404, 409, 412):
                    return e.code, dict(e.headers), e.read()
                if e.code < 500 or attempt == tries - 1:
                    raise RuntimeError(f"{method} {path}: HTTP {e.code} {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout) as e:
                if attempt == tries - 1:
                    raise RuntimeError(f"{method} {path}: {e}") from None
            time.sleep(min(60, 2 ** attempt))
        raise AssertionError

    def get_json(self, path: str):
        st, h, b = self.req("GET", path)
        return (json.loads(b), h.get("ETag")) if st == 200 else (None, None)

    def head(self, path: str):
        st, h, _ = self.req("HEAD", path)
        return h if st == 200 else None

    def create_json(self, path: str, obj, etag: str | None = None) -> bool:
        """Create-only (or replace-if-ETag-matches). True if written."""
        data = json.dumps(obj, indent=1).encode()
        hdr = {"x-ms-blob-type": "BlockBlob", "Content-Type": "application/json",
               **({"If-Match": etag} if etag else {"If-None-Match": "*"})}
        st, _, _ = self.req("PUT", path, data=data, headers=hdr)
        return st in (200, 201)

    def put_file(self, src: Path, path: str) -> dict:
        size = src.stat().st_size
        md5 = hashlib.md5()
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                md5.update(chunk)
        b64 = base64.b64encode(md5.digest()).decode()
        hdr = {"x-ms-blob-type": "BlockBlob", "Content-MD5": b64, "Content-Length": str(size), "If-None-Match": "*",
               "Content-Type": "application/octet-stream"}
        if size > 4000 * (1 << 20):
            raise RuntimeError(f"{src}: {size} bytes is over the single-put limit used here")
        st, _, _ = self.req("PUT", path, data=lambda: open(src, "rb").read(), headers=hdr)
        h = self.head(path)
        if h is None:
            raise RuntimeError(f"{path}: not found after upload")
        got_size, got_md5 = int(h.get("Content-Length", -1)), h.get("Content-MD5")
        if st in (409, 412):   # already there: keep the existing blob (first writer wins)
            return {"path": path, "bytes": got_size, "md5": got_md5, "status": "exists",
                    "same": got_md5 == b64 and got_size == size}
        if got_size != size or got_md5 != b64:
            raise RuntimeError(f"{path}: read-back mismatch (size {got_size} vs {size}, md5 {got_md5} vs {b64})")
        return {"path": path, "bytes": size, "md5": b64, "status": "uploaded"}


def chunks_of(cfg: dict):
    for sp in cfg["splits"]:
        n_parts, cp = sp["n_parts"], cfg["chunk_parts"]
        for c in range((n_parts + cp - 1) // cp):
            yield sp["name"], c, c * cp, min((c + 1) * cp, n_parts)


def key(split: str, c: int) -> str:
    return f"{split}_{c:05d}.json"


def cmd_claim(b: Blob, a) -> int:
    cfg, _ = b.get_json(f"{b.run}/config.json")
    if cfg is None:
        sys.exit("blobq: run config not published yet")
    live = {x for x in a.live_jobs.split(",") if x}
    me = {"owner": a.owner, "job": a.job, "host": socket.gethostname(), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for split, c, lo, hi in chunks_of(cfg):
        if b.head(f"{b.run}/done/{key(split, c)}") is not None:
            continue
        if b.create_json(f"{b.run}/claims/{key(split, c)}", me):
            print(split, c, lo, hi); return 0
        cur, etag = b.get_json(f"{b.run}/claims/{key(split, c)}")
        if cur is None:
            continue
        free = cur.get("released") or (cur.get("owner") == a.owner and a.retake_dead and cur.get("job") not in live
                                         and cur.get("job") != a.job)
        if free and b.create_json(f"{b.run}/claims/{key(split, c)}", {**me, "took_over": cur}, etag=etag):
            print(split, c, lo, hi); return 0
    return 3


def cmd_status(b: Blob, a) -> int:
    cfg, _ = b.get_json(f"{b.run}/config.json")
    if cfg is None:
        print("no config published"); return 1
    items = list(chunks_of(cfg))

    def probe(it):
        split, c, lo, hi = it
        d, _ = b.get_json(f"{b.run}/done/{key(split, c)}")
        cl = None if d else b.get_json(f"{b.run}/claims/{key(split, c)}")[0]
        return it, d, cl
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(probe, items))
    tot = {"done": 0, "claimed": 0, "free": 0}; per: dict = {}
    for (split, c, lo, hi), d, cl in res:
        if d:
            tot["done"] += 1; per.setdefault(d.get("owner", "?"), [0, 0])[0] += 1
        elif cl and not cl.get("released"):
            tot["claimed"] += 1; per.setdefault(cl.get("owner", "?"), [0, 0])[1] += 1
            if a.verbose:
                print(f"  in progress {split} {c}: {cl.get('owner')} job {cl.get('job')} since {cl.get('time')}")
        else:
            tot["free"] += 1
    print(f"{cfg['run']}: {len(items)} chunks of {cfg['chunk_parts']} parts — done {tot['done']}, "
          f"in progress {tot['claimed']}, free {tot['free']}")
    for o, (dn, cl) in sorted(per.items()):
        print(f"  {o}: done {dn}, in progress {cl}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sas-file", default=os.environ.get("STAT236_SAS_FILE"))
    ap.add_argument("--run", default="full_v1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("claim"); p.add_argument("--owner", required=True); p.add_argument("--job", required=True)
    p.add_argument("--live-jobs", default=""); p.add_argument("--no-retake-dead", dest="retake_dead", action="store_false")
    p = sub.add_parser("put"); p.add_argument("file", type=Path); p.add_argument("blob")
    p = sub.add_parser("done"); p.add_argument("--split", required=True); p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--owner", required=True); p.add_argument("--info", default="{}")
    p = sub.add_parser("release"); p.add_argument("--split", required=True); p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--owner", required=True)
    p = sub.add_parser("status"); p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("get-config")
    p = sub.add_parser("publish-config"); p.add_argument("file", type=Path)
    a = ap.parse_args()
    b = Blob(load_sas(a.sas_file), a.run)
    if a.cmd == "claim":
        return cmd_claim(b, a)
    if a.cmd == "put":
        print(json.dumps(b.put_file(a.file, a.blob))); return 0
    if a.cmd == "done":
        info = {"owner": a.owner, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **json.loads(a.info)}
        ok = b.create_json(f"{b.run}/done/{key(a.split, a.chunk)}", info)
        print("done marked" if ok else "already marked done"); return 0
    if a.cmd == "release":
        cur, etag = b.get_json(f"{b.run}/claims/{key(a.split, a.chunk)}")
        if not cur or cur.get("owner") != a.owner:
            sys.exit(f"not your claim: {cur}")
        print("released" if b.create_json(f"{b.run}/claims/{key(a.split, a.chunk)}", {**cur, "released": True}, etag=etag)
              else "claim changed meanwhile; not released"); return 0
    if a.cmd == "status":
        return cmd_status(b, a)
    if a.cmd == "get-config":
        cfg, _ = b.get_json(f"{b.run}/config.json"); print(json.dumps(cfg, indent=1)); return 0 if cfg else 1
    if a.cmd == "publish-config":
        ok = b.create_json(f"{b.run}/config.json", json.loads(a.file.read_text()))
        print("config published" if ok else "config already exists (not overwritten)"); return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
