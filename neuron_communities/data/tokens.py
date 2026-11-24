"""``tokens`` stage: a tokens-only corpus in the chunked dataset layout every later stage reads.

Samples documents from a Hugging Face dataset (streaming, shard-shuffled with a seed), tokenizes them exactly like
the harvest does (the model's tokenizer, BOS prepended, truncated to ``--max-length``) and writes:

  <out>/index.sqlite                             sequence_locations(sequence_id, chunk_id, offset, count)
                                                 (+ empty sequence_metadata / token_positions for schema parity)
  <out>/layers/tokens/chunk_XXXXXX.safetensors   token_ids (int32, documents concatenated)
  <out>/docs.parquet                             row (int64, = sequence_locations rowid - 1), doc_id (the record's
                                                 ``id`` field, else its url), url: the source document of every row
  <out>/metadata.json, <out>/provenance.json

No activations are stored: the sampling, dedup and harvest stages only ever read token_ids, so this is all a
corpus needs (~4 bytes per token).

  python -m neuron_communities tokens --hf-dataset wikimedia/wikipedia --hf-config 20231101.en \\
      --text-field text --n-docs 40000 --out wiki_tokens
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer

from neuron_communities import config
from neuron_communities.data import io


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--hf-dataset", required=True, help="Hugging Face dataset id (streamed)")
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--n-docs", type=int, required=True)
    ap.add_argument("--min-chars", type=int, default=200, help="skip documents shorter than this many characters")
    ap.add_argument("--max-length", type=int, default=4096, help="truncate documents to this many tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-buffer", type=int, default=20_000)
    ap.add_argument("--chunk-tokens", type=int, default=8_000_000, help="tokens per chunk file")
    ap.add_argument("--model", default=config.MODEL_ID, help="tokenizer to use")
    ap.add_argument("--out", required=True, help="tokens dataset directory to create")


def run(a: argparse.Namespace) -> None:
    out = Path(a.out)
    (out / "layers" / "tokens").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.monotonic()

    ds = load_dataset(a.hf_dataset, a.hf_config, split=a.split, streaming=True)
    ds = ds.shuffle(seed=a.seed, buffer_size=a.shuffle_buffer)     # shuffles shard order + a buffer
    conn = io.create_index_db(out)
    buf: list[np.ndarray] = []
    buf_len = 0
    chunk_id = 0
    chunks_meta = []
    n_docs = 0
    total_tokens = 0
    urls = []
    doc_ids = []
    batch_rows = []

    def flush():
        nonlocal buf, buf_len, chunk_id
        if not buf:
            return
        ids = torch.from_numpy(np.concatenate(buf).astype(np.int32))
        save_file({"token_ids": ids}, str(out / "layers" / "tokens" / f"chunk_{chunk_id:06d}.safetensors"))
        chunks_meta.append({"chunk_id": chunk_id, "num_items": int(len(ids))})
        chunk_id += 1
        buf, buf_len = [], 0

    for ex in ds:
        text = ex.get(a.text_field) or ""
        if len(text) < a.min_chars:
            continue
        ids = tok(text, add_special_tokens=True, truncation=True, max_length=a.max_length)["input_ids"]
        if len(ids) < 2:
            continue
        if buf_len + len(ids) > a.chunk_tokens and buf:
            flush()
        batch_rows.append((f"seq_{n_docs}", chunk_id, buf_len, len(ids)))
        buf.append(np.asarray(ids, dtype=np.int32)); buf_len += len(ids)
        total_tokens += len(ids); n_docs += 1
        urls.append(str(ex.get("url") or ""))
        doc_ids.append(str(ex.get("id") or ex.get("url") or ""))
        if len(batch_rows) >= 1000:
            conn.executemany("INSERT INTO sequence_locations VALUES (?,?,?,?)", batch_rows); conn.commit(); batch_rows = []
        if n_docs % 5000 == 0:
            print(f"  {n_docs:,} docs, {total_tokens:,} tokens ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
        if n_docs >= a.n_docs:
            break
    flush()
    if batch_rows:
        conn.executemany("INSERT INTO sequence_locations VALUES (?,?,?,?)", batch_rows); conn.commit()
    conn.close()
    pq.write_table(pa.table({"row": pa.array(np.arange(n_docs, dtype=np.int64)), "doc_id": pa.array(doc_ids, pa.string()),
                             "url": pa.array(urls, pa.string())}), out / "docs.parquet")
    (out / "metadata.json").write_text(json.dumps({
        "mode": "token", "tokens_only": True, "num_chunks": chunk_id, "total_items": total_tokens,
        "chunks": chunks_meta, "has_token_ids": True, "layers": [], "d_models": {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    (out / "provenance.json").write_text(json.dumps({
        "hf_dataset": a.hf_dataset, "hf_config": a.hf_config, "split": a.split, "text_field": a.text_field,
        "n_docs": n_docs, "min_chars": a.min_chars, "max_length": a.max_length, "seed": a.seed,
        "shuffle_buffer": a.shuffle_buffer, "tokenizer": a.model, "total_tokens": total_tokens,
        "doc_ids": "docs.parquet (one row per document)",
        "n_docs_with_id": int(sum(bool(d) for d in doc_ids)), "sample_urls": urls[:50]}, indent=1))
    print(f"done: {n_docs:,} docs, {total_tokens:,} tokens, {chunk_id} chunks -> {out} "
          f"({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
