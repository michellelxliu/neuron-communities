"""Exploratory slices: counts, unique texts, splits, and that every answer is a well-formed continuation."""
from __future__ import annotations

from collections import Counter

from neuron_communities.slices import exploratory as ex


def test_build_slices():
    r = ex.build_slices(0)
    c = Counter(x["task"] for x in r)
    assert set(c) == set(ex.TASKS)
    assert all(c[t] == ex.DEFAULT_N[t] for t in ex.TASKS if t != "agree_there") and c["agree_there"] > 1000
    assert len({x["text"] for x in r}) == len(r) and len({x["prompt_id"] for x in r}) == len(r)
    for t in ex.TASKS:
        s = Counter(x["split"] for x in r if x["task"] == t)
        assert abs(s["discovery"] - s["heldout"]) <= 1
    for x in r:
        if x["task"] in ("agree_a_an",):
            assert {x["answer"], x["answer_alt"]} == {" a", " an"}
        if x["task"] in ("agree_sv", "agree_there"):
            assert {x["answer"], x["answer_alt"]} == {" is", " are"}
        if x["task"] == "delim_code":
            assert not x["text"].endswith(")") and set(x["answer"]) == {")"}
            assert x["text"].count("(") - x["text"].count(")") == len(x["answer"])
        if x["task"] == "delim_html":
            assert x["text"].endswith("</") and "<" + x["answer"] in x["text"]
        if x["task"] == "copy_words":
            assert x["answer"].strip() in x["text"]
        if x["task"] in ("agree_fr", "agree_es", "agree_de") and x["answer_alt"]:
            assert x["answer"] != x["answer_alt"]
