"""Turn a list of standalone prompts into a manifest dir every harvest stage can read.

Each prompt becomes one document of a tokens dataset (the model's tokenizer, BOS prepended, exactly like the
``tokens`` stage), and its manifest target is the prompt's last token, so the harvest attributes the model's
next-token prediction after the full prompt. Writes::

  <out>/tokens/...            tokens dataset (index.sqlite, layers/tokens/chunk_000000.safetensors, metadata.json)
  <out>/manifest.npz          row_index = 0..n-1, position = len(tokens) - 1
  <out>/dedup_kept.npz        all rows (templated prompts are distinct by construction; duplicates are rejected)
  <out>/prompts.parquet       manifest_index plus prompt_id, slice, task, template_id, split, doc_id, text, answer and
                              any other string fields of the prompt records
  <out>/meta.json             dataset path, n, the record fields, and the generator's provenance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from neuron_communities.data import io

REQUIRED = ("prompt_id", "slice", "task", "template_id", "split", "text")


def write_prompt_world(records: list[dict], tokenizer, out: str | Path, provenance: dict | None = None) -> Path:
    out = Path(out)
    assert records, "no prompts"
    for f in REQUIRED:
        assert all(f in r for r in records), f"every record needs {f!r}"
    texts = [r["text"] for r in records]
    assert len(set(texts)) == len(texts), "duplicate prompt texts"
    assert len({r["prompt_id"] for r in records}) == len(records), "duplicate prompt ids"
    ids = [tokenizer(t, add_special_tokens=True)["input_ids"] for t in texts]
    assert all(len(x) >= 2 for x in ids)

    tok_dir = out / "tokens"
    (tok_dir / "layers" / "tokens").mkdir(parents=True, exist_ok=True)
    flat = torch.from_numpy(np.concatenate([np.asarray(x, np.int32) for x in ids]))
    save_file({"token_ids": flat}, str(tok_dir / "layers" / "tokens" / "chunk_000000.safetensors"))
    conn = io.create_index_db(tok_dir)
    offs = np.concatenate([[0], np.cumsum([len(x) for x in ids])[:-1]])
    conn.executemany("INSERT INTO sequence_locations VALUES (?,?,?,?)",
                     [(r["prompt_id"], 0, int(o), len(x)) for r, o, x in zip(records, offs, ids)])
    conn.commit(); conn.close()
    (tok_dir / "metadata.json").write_text(json.dumps({"mode": "token", "tokens_only": True, "num_chunks": 1,
                                                       "total_items": int(len(flat)), "chunks": [{"chunk_id": 0, "num_items": int(len(flat))}],
                                                       "has_token_ids": True}))
    n = len(records)
    np.savez_compressed(out / "manifest.npz", row_index=np.arange(n, dtype=np.int64),
                        position=np.array([len(x) - 1 for x in ids], np.int64))
    np.savez_compressed(out / "dedup_kept.npz", kept=np.arange(n, dtype=np.int64))
    fields = sorted({k for r in records for k in r})
    cols = {"manifest_index": pa.array(np.arange(n, dtype=np.int64))}
    for f in fields:
        cols[f] = pa.array([str(r.get(f, "")) for r in records], pa.string())
    if "doc_id" not in cols:
        cols["doc_id"] = pa.array([str(r["prompt_id"]) for r in records], pa.string())
    pq.write_table(pa.table(cols), out / "prompts.parquet")
    (out / "meta.json").write_text(json.dumps({"dataset": str(tok_dir.resolve()), "n": n, "fields": fields,
                                               "selection": "standalone prompts; target = last prompt token",
                                               "provenance": provenance or {}}, indent=1))
    return out
