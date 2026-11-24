#!/usr/bin/env python3
"""(Michelle) Rebuild catalog.json at the root of the stat236 container: every blob's path, size and MD5, so holders of
the collaborator link (no list permission) can find files. Leaves out the work-queue files (full_v1/claims, done) and
test prefixes (_test_*). Needs a SAS with list + write; uses azure-storage-blob.

  python scripts/blob/make_catalog.py --env ~/.config/stat236-blob/sas.env:STAT236_SAS --local-copy ../blob_index/stat236_catalog.json
"""
import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path

from azure.storage.blob import ContainerClient

ap = argparse.ArgumentParser()
ap.add_argument("--account", default="infinigenexploration")
ap.add_argument("--container", default="stat236")
ap.add_argument("--env", required=True, help="FILE:VAR holding the SAS")
ap.add_argument("--local-copy", type=Path, default=None)
a = ap.parse_args()
path, var = a.env.split(":")
sas = next(l.split("=", 1)[1].strip().strip('"').strip("'").lstrip("?")
           for l in Path(path).expanduser().read_text().splitlines() if l.startswith(var + "="))
url = f"https://{a.account}.blob.core.windows.net/{a.container}"
c = ContainerClient.from_container_url(f"{url}?{sas}")
files, ds = [], defaultdict(lambda: {"files": 0, "bytes": 0})
for b in c.list_blobs():
    n = b.name
    if n == "catalog.json" or n.startswith("_test") or n.startswith(("full_v1/claims/", "full_v1/done/")):
        continue
    md5 = b.content_settings.content_md5
    files.append({"path": n, "bytes": b.size, "md5_b64": base64.b64encode(md5).decode() if md5 else None})
    top = n.split("/")[0] if "/" in n else "(root)"
    ds[top]["files"] += 1; ds[top]["bytes"] += b.size
cat = {"account": a.account, "container": a.container, "url": url,
       "docs": "github.com/michellelxliu/neuron-communities/blob/wide-harvest/docs/DATA.md",
       "datasets": dict(sorted(ds.items())), "files": files}
data = json.dumps(cat, indent=1).encode()
c.upload_blob("catalog.json", data, overwrite=True, content_type="application/json")
if a.local_copy:
    a.local_copy.write_bytes(data)
print(f"CATALOG files={len(files)} bytes={sum(f['bytes'] for f in files)}",
      {k: v["files"] for k, v in sorted(ds.items())})
