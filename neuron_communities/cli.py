"""``python -m neuron_communities <stage> ...`` — one subcommand per pipeline stage.

Every stage module exposes ``add_args(parser)`` and ``run(args)``. Stages that scale over many prompts take
``--shard i --shards n`` and write one output file per shard; re-running a stage skips shards whose output exists.
"""

from __future__ import annotations

import argparse
import importlib

STAGES: list[tuple[str, str, str]] = [
    # name            module                                 one-line help
    ("tokens",        "neuron_communities.data.tokens",      "tokenise a Hugging Face corpus into the chunked tokens layout"),
    ("sample",        "neuron_communities.data.manifest",    "freeze a uniform random prompt manifest over a tokens dataset"),
    ("dedup",         "neuron_communities.data.dedup",       "window-64 leader dedup of a manifest, plus exact dedup of short contexts"),
    ("combine",       "neuron_communities.data.combine",     "concatenate per-corpus manifests into one world"),
    ("heldout-dedup", "neuron_communities.data.heldout",     "drop held-out prompts that collide with the training manifest"),
    ("attribution",   "neuron_communities.harvest.attribution", "top-k activation x gradient neuron sets per prompt and layer"),
    ("attribution-wide", "neuron_communities.harvest.wide",  "top-50 signed activation x gradient scores per prompt and layer (parquet)"),
    ("wide-blocks",   "neuron_communities.harvest.wide_blocks", "top-k prefix of a wide harvest in the attribution block layout"),
    ("world",         "neuron_communities.harvest.world",    "assemble the harvest blocks into per-layer CSR matrices"),
    ("census",        "neuron_communities.graph.census",     "cross-layer co-occurrence counts with popularity-preserving nulls"),
    ("leiden",        "neuron_communities.graph.leiden",     "CPM Leiden communities on the z-gated co-occurrence graph"),
    ("merge",         "neuron_communities.graph.merge",      "merge the per-gamma discoveries into one candidate dictionary"),
    ("hubs",          "neuron_communities.graph.hubs",       "strip the always-on hub neurons from every community"),
    ("assign",        "neuron_communities.eval.assignments", "which prompts each community is tested on"),
    ("necessity",     "neuron_communities.eval.necessity",   "KL(clean || community zero-ablated) per (prompt, community)"),
    ("subspan",       "neuron_communities.eval.subspan",     "sufficiency and layer necessity for every contiguous sub-span"),
    ("aggregate",     "neuron_communities.eval.aggregate",   "collect necessity / subspan shards and certify prompts"),
    ("span-local",    "neuron_communities.eval.span_local",  "span-local necessity re-certification (prep, rows, aggregate)"),
    ("summary",       "neuron_communities.eval.stats",       "coverage and dictionary statistics of a finished world"),
]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neuron_communities", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, module, help_ in STAGES:
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(_module=module)
        importlib.import_module(module).add_args(p)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    importlib.import_module(args._module).run(args)
