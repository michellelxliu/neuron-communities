"""Server-side copy of every blob under a prefix from one container to another, then verify.

Credentials come from env files (never printed): --src-env / --dst-env name a file and the variable holding the SAS.
Each copy is checked for size and Content-MD5 equal to the source; blobs without a source MD5 are checked by size
and flagged. Writes a JSON index (no SAS) to --index.
"""
import argparse, json, os, time
from azure.storage.blob import ContainerClient


def read_env(path):
    d = {}
    for line in open(os.path.expanduser(path)):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            d[k] = v.strip().strip('"').strip("'").lstrip("?")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default="infinigenexploration")
    ap.add_argument("--src-container", required=True)
    ap.add_argument("--src-env", required=True, help="FILE:VAR holding a SAS with read+list on the source")
    ap.add_argument("--dst-container", required=True)
    ap.add_argument("--dst-env", required=True, help="FILE:VAR holding a SAS with write+read+list on the destination")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--strip-prefix", action="store_true")
    ap.add_argument("--index", required=True)
    a = ap.parse_args()
    url = f"https://{a.account}.blob.core.windows.net"
    f, v = a.src_env.split(":"); src_sas = read_env(f)[v]
    f, v = a.dst_env.split(":"); dst_sas = read_env(f)[v]
    src = ContainerClient(url, a.src_container, credential=src_sas)
    dst = ContainerClient(url, a.dst_container, credential=dst_sas)

    jobs = []
    for b in src.list_blobs(name_starts_with=a.prefix):
        name = b.name[len(a.prefix):] if a.strip_prefix else b.name
        md5 = b.content_settings.content_md5
        d = dst.get_blob_client(name)
        try:
            p = d.get_blob_properties()
            if p.size == b.size and p.content_settings.content_md5 == md5 and p.copy.status in (None, "success"):
                jobs.append((b, name, md5, "exists")); continue
        except Exception:
            pass
        d.start_copy_from_url(f"{url}/{a.src_container}/{b.name}?{src_sas}")
        jobs.append((b, name, md5, "started"))
    print(f"COPY_STARTED {sum(j[3]=='started' for j in jobs)} already={sum(j[3]=='exists' for j in jobs)}", flush=True)

    rows, failed = [], 0
    for b, name, md5, _ in jobs:
        d = dst.get_blob_client(name)
        while True:
            p = d.get_blob_properties()
            if p.copy.status in (None, "success", "failed", "aborted"):
                break
            time.sleep(5)
        dm = p.content_settings.content_md5
        ok = p.copy.status in (None, "success") and p.size == b.size and (md5 is None or dm == md5)
        failed += not ok
        rows.append({"blob": f"{url}/{a.dst_container}/{name}", "bytes": p.size,
                     "md5_b64": (__import__("base64").b64encode(dm).decode() if dm else None),
                     "md5_checked": md5 is not None, "verified": ok, "source": f"{a.src_container}/{b.name}"})
    os.makedirs(os.path.dirname(os.path.abspath(a.index)), exist_ok=True)
    json.dump({"account": a.account, "src": f"{a.src_container}/{a.prefix}", "dst": a.dst_container,
               "files": rows}, open(a.index, "w"), indent=1)
    print(f"COPY_DONE files={len(rows)} bytes={sum(r['bytes'] for r in rows)} failed={failed} "
          f"no_src_md5={sum(not r['md5_checked'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
