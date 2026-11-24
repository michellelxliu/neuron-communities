"""The full arithmetic slice: Table 7 templates, disjoint splits, answers, ground truth (no model, no network)."""
from __future__ import annotations

from collections import Counter

from neuron_communities.slices import arithmetic as ar


def test_slice_splits_and_answers():
    s = ar.build_slice(0)
    assert len(s["discovery"]) == 16575 and len(s["heldout"]) == 10732
    for task in ar.TABLE7:
        d = [r for r in s["discovery"] if r["task"] == task]
        h = [r for r in s["heldout"] if r["task"] == task]
        assert {r["template_id"] for r in d}.isdisjoint({r["template_id"] for r in h})            # disjoint templates
        assert {(r["input_value"], r["offset_value"]) for r in d}.isdisjoint(
            {(r["input_value"], r["offset_value"]) for r in h})                                     # disjoint value pairs
        assert {r["template_id"] for r in d} | {r["template_id"] for r in h} == {f"t{i}" for i in range(5)}
    ids = [r["prompt_id"] for r in s["discovery"] + s["heldout"]]
    assert len(ids) == len(set(ids))
    texts = [r["text"] for r in s["discovery"] + s["heldout"]]
    assert len(texts) == len(set(texts))
    by = {r["prompt_id"]: r for r in s["discovery"] + s["heldout"]}
    add = [r for r in by.values() if r["task"] == "addition" and r["template_id"] == "t0"]
    assert all(r["text"] == f"{r['input']}+{r['offset']}=" and int(r["answer"]) == int(r["input"]) + int(r["offset"]) for r in add)
    for r in by.values():                                       # prompt-side naming of the answer (" Tues" for "Mon")
        if r["task"] == "weekdays" and r["template_id"] == "t3":
            days = ["Mon", "Tues", "Wed", "Thu", "Fri", "Sat", "Sun"]
            assert r["answer"] == " " + days[(days.index(r["input"]) + int(r["offset_value"])) % 7]
        if r["task"] == "hours" and r["template_id"] == "t4":
            assert r["answer"] == f"{(int(r['input']) + int(r['offset'])) % 24:02d}:00" and r["answer"].startswith(r["result_value"])
    # cycles=1 for the alternatives, the full default offset list for t0
    c = Counter((r["task"], r["template_id"]) for r in s["discovery"] + s["heldout"])
    assert c[("weekdays", "t0")] == 49 and c[("hours", "t0")] == 576 and c[("months", "t0")] == 144
    assert max(int(r["offset_value"]) for r in s["discovery"] if r["task"] == "hours" and r["template_id"] == "t1") <= 24


def test_ground_truth():
    gt = ar.ground_truth()
    assert sorted(sum(gt["period_clusters"].values(), [])) == gt["N_add"] and len(gt["N_add"]) == 28
    assert set(gt["per_task"]["months"]) <= set(gt["N_add"]) and set(gt["per_task"]["weekdays"]) <= set(gt["N_add"])
    assert set(gt["per_task"]["hours"]) - set(gt["N_add"]) == {1658}
