"""The arithmetic slice: addition and the three cyclic tasks of Feucht et al., *Arithmetic in the Wild*
(arXiv:2605.01148; code github.com/goodfire-ai/arithmetic-wild, ``src/tasks/*/causal_models.py`` for the default
templates and value sets, ``src/rephrased_templates.ipynb`` / their Table 7 for the alternatives).

Also the published ground truth: the 28 layer-18 addition neurons N_add and the per-task subsets
(``src/neurons_per_task.json`` of that repo). Note that the hours set contains neuron 1658, which is not in N_add.

  prompts = make_prompts("months", n=256, seed=0)       # default template, uniform (input, offset) pairs
"""

from __future__ import annotations

import itertools
import random

LAYER = 18

N_ADD = [10297, 1712, 12728, 2404, 8859, 8887, 3896, 13492, 251, 11545, 12154, 11029, 10099, 8409, 2955, 3644, 5700,
         13003, 3650, 1768, 12778, 61, 8343, 2372, 1448, 6339, 11096, 6721]
NEURONS = {
    "addition": sorted(N_ADD),
    "hours": [61, 251, 1448, 1658, 1712, 1768, 2372, 2404, 2955, 3644, 3650, 3896, 5700, 6339, 8409, 8859, 8887, 10099,
              10297, 11029, 11545, 12154, 12728, 12778, 13003, 13492],
    "months": [251, 1712, 2404, 2955, 3644, 3896, 8409, 8859, 8887, 10099, 10297, 11029, 11545, 12154, 12728, 13492],
    "weekdays": [1712, 2404, 2955, 3644, 3896, 8409, 8859, 8887, 10099, 10297, 11029, 11545, 12154, 12728, 13492],
}
PERIODS = (2, 5, 10, 20, 50, 100)
# recovery reported in Feucht et al. App. A: neurons of the task set among the top 40 by mean |h * grad|
REFERENCE_TOP40 = {"addition": 17, "hours": 20, "months": 13, "weekdays": 14}

_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
          "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
_WORDS += [f"{t}-{u}" if u else t for t in ("twenty", "thirty", "forty") for u in [""] + _WORDS[:9]][1:]
_WORDS = _WORDS[:48]                                                     # one .. forty-eight
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOURS = [f"{h:02d}" for h in range(24)]

# task -> (inputs, offsets as written, offset value, answer(input index, offset value))
TASKS = {
    "addition": ([str(i) for i in range(1, 101)], [str(i) for i in range(1, 101)], lambda o: int(o),
                 lambda a, b: str(int(a) + b)),
    "months": (MONTHS, _WORDS[:24], lambda o: _WORDS.index(o) + 1, lambda m, k: " " + MONTHS[(MONTHS.index(m) + k) % 12]),
    "weekdays": (DAYS, _WORDS[:14], lambda o: _WORDS.index(o) + 1, lambda d, k: " " + DAYS[(DAYS.index(d) + k) % 7]),
    "hours": (HOURS, _WORDS[:48], lambda o: _WORDS.index(o) + 1, lambda h, k: f"{(int(h) + k) % 24:02d}:00"),
}

# template_id -> template; "default" is the repo's causal-model template ({input}, {offset})
TEMPLATES = {
    "addition": {"default": "{input}+{offset}="},
    "months": {
        "default": "Q: What month is {offset} months after {input}?\nA:",
        "alt1": "Let's do some calendar math. If the current month is {input}, then in {offset} months, what month will it be? Answer: It will be",
    },
    "weekdays": {
        "default": "Q: What day is {offset} days after {input}?\nA:",
        "alt1": "Let's do some weekday math. {offset} days after {input} is what day? Answer:",
    },
    "hours": {
        "default": "Q: In 24-hour time, it is now {input}:00. What time will it be in {offset} hours?\nA: In 24-hour time, it will be ",
        "alt1": "Let's do some clock math. If it is {input}:00 right now in 24-hour time, then what time will it be in {offset} hours? Answer: ",
        "alt2": "When I was in the military, I learned 24-hour time. So if right now it is {input}:00, then in {offset} hours, it will be ",
        "alt3": "Q: If it is {input} o' clock in military time, what time will it be in {offset} hours? A: In 24-hour time, it will be ",
    },
}
# The full slice (``build_slice`` below) uses TABLE7: every template of the repo's rephrased_templates.ipynb.


def make_prompts(task: str, n: int, seed: int = 0, template_id: str = "default", split: str = "gate1",
                 pairs: list[tuple[str, str]] | None = None) -> list[dict]:
    """``n`` prompts with distinct (input, offset) pairs sampled uniformly without replacement (all pairs when there
    are fewer than ``n``), or exactly ``pairs`` when given."""
    inputs, offsets, oval, answer = TASKS[task]
    tpl = TEMPLATES[task][template_id]
    if pairs is None:
        allp = list(itertools.product(inputs, offsets))
        pairs = random.Random(seed).sample(allp, min(n, len(allp)))
    recs = []
    for a, o in pairs:
        recs.append({"prompt_id": f"{task}:{template_id}:{a}:{o}", "slice": "arithmetic", "task": task,
                     "template_id": template_id, "split": split, "text": tpl.format(input=a, offset=o),
                     "input": a, "offset": o, "offset_value": str(oval(o)), "answer": answer(a, oval(o))})
    return recs


def first_answer_token(tokenizer, text: str, answer: str) -> int:
    """The token the model must predict first: the first token of ``answer`` when tokenised after ``text``
    (-1 when the answer's tokenisation merges with the end of the prompt)."""
    p = tokenizer(text, add_special_tokens=True)["input_ids"]
    full = tokenizer(text + answer, add_special_tokens=True)["input_ids"]
    return int(full[len(p)]) if full[: len(p)] == p and len(full) > len(p) else -1


# ---- the full slice (Build A): Table 7 templates, discovery / held-out splits -----------------------------------------
#
# Source: arithmetic-wild @ d03024a, src/rephrased_templates.ipynb (the paper's Table 7 rows, in notebook order; t0 is the
# task's default causal-model template) and src/static_downproj.ipynb (the period clusters, "based on results from
# neurons_across_sums.ipynb"). In the notebook every row is evaluated with cycles=1 (offsets up to one period); the
# default template here keeps the causal model's full offset list (as gate 1 did), the alternatives use cycles=1.
# Addition rows t1-t3 are commented out in the notebook's final cell but are the templates the paper's Table 7 lists.

# period clusters of N_add (they partition the 28 neurons)
PERIOD_CLUSTERS = {
    2: [1712, 10297],
    5: [8859, 8887, 3896, 2404, 12728, 13492],
    10: [8409, 3644, 2955, 251, 11029, 10099, 12154, 11545],
    20: [5700, 3650, 13003],
    50: [12778, 6339, 8343, 61, 1448, 1768, 2372],
    100: [6721, 11096],
}

_DIGITS = {str(i): _WORDS[i - 1] for i in range(1, 25)}                 # prompt "7" -> causal-model offset "seven"
_DAY_ABBR = dict(zip(["Mon", "Tues", "Wed", "Thu", "Fri", "Sat", "Sun"], DAYS))
_MONTH_ABBR = dict(zip(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], MONTHS))

# task -> [(template_id, template, input map or None, offset map or None, cycles or None (= full default domain))]
# maps go prompt string -> causal-model string, as in the notebook
TABLE7 = {
    "weekdays": [
        ("t0", TEMPLATES["weekdays"]["default"], None, None, None),
        ("t1", "Let's do some weekday math. {offset} days after {input} is what day? Answer:", None, None, 1),
        ("t2", "Question: Today is {input}. In {offset} days, what day will it be? Answer: It will be", None,
         {k: v for k, v in _DIGITS.items() if int(k) <= 7}, 1),
        ("t3", "If today is {input}, and I have a deadline in {offset} days, what day is my deadline? My deadline is on",
         _DAY_ABBR, None, 1),
        ("t4", "Let's do some calendar math. Today is {input}. What day will it be in {offset} days? Answer:", _DAY_ABBR,
         {k: v for k, v in _DIGITS.items() if int(k) <= 7}, 1),
    ],
    "months": [
        ("t0", TEMPLATES["months"]["default"], None, None, None),
        ("t1", TEMPLATES["months"]["alt1"], None, None, 1),
        ("t2", "Question: I have a big presentation in {offset} months. If it is currently {input}, then what month is my "
               "presentation? Answer: My presentation is in", None, {k: v for k, v in _DIGITS.items() if int(k) <= 12}, 1),
        ("t3", "Q: A pregnant woman will give birth in {offset} months. If it is currently {input}, what month will she "
               "give birth? Answer: She will give birth in", _MONTH_ABBR, None, 1),
        ("t4", "Let's figure out some scheduling. I just got my hair cut, and it is currently {input}. If I have to book "
               "another appointment in {offset} months, what month will it be? Answer: My appointment is in",
         _MONTH_ABBR, {k: v for k, v in _DIGITS.items() if int(k) <= 12}, 1),
    ],
    "hours": [
        ("t0", TEMPLATES["hours"]["default"], None, None, None),
        ("t1", TEMPLATES["hours"]["alt1"], None, None, 1),
        ("t2", TEMPLATES["hours"]["alt2"], None, None, 1),
        ("t3", TEMPLATES["hours"]["alt3"], None, None, 1),
        ("t4", "I need to work out when to get to the airport. Q: It's {input}:00 right now in 24-hour time. My flight is "
               "in {offset} hours. What time is my flight? Answer: My flight will be at ", None, dict(_DIGITS), 1),
    ],
    "addition": [
        ("t0", TEMPLATES["addition"]["default"], None, None, None),
        ("t1", "The sum of {input} and {offset} equals ", None, None, None),
        ("t2", "Math question: {input} plus {offset} equals ", None, None, None),
        ("t3", "Q: What is the sum of {input} and {offset}? A: ", None, None, None),
        ("t4", "Q: If I have {input} apples and buy {offset} more, how many apples do I have? Total apples: ", None, None, None),
    ],
}
# Disjoint templates per split. Each split gets a default-style template and a mix of rewordings; for the cyclic tasks
# the held-out split holds the digit-offset rows (t2, t4), so held-out tests transfer to a new offset format too.
SPLIT_TEMPLATES = {
    "weekdays": {"discovery": ("t0", "t1", "t3"), "heldout": ("t2", "t4")},
    "months": {"discovery": ("t0", "t1", "t3"), "heldout": ("t2", "t4")},
    "hours": {"discovery": ("t0", "t1", "t2"), "heldout": ("t3", "t4")},
    "addition": {"discovery": ("t0", "t1", "t3"), "heldout": ("t2", "t4")},
}


def _period(task: str) -> int:
    return {"weekdays": 7, "months": 12, "hours": 24}.get(task, 0)


def pair_split(task: str, seed: int = 0) -> dict[tuple[str, str], str]:
    """Causal-model (input, offset) pair -> 'discovery' | 'heldout': a seeded 50/50 partition of the task's full
    default pair space, shared by every template, so the two splits never share a value pair."""
    inputs, offsets, _, _ = TASKS[task]
    allp = sorted(itertools.product(inputs, offsets))
    rng = random.Random(f"{task}:{seed}")
    rng.shuffle(allp)
    half = len(allp) // 2
    return {p: ("discovery" if i < half else "heldout") for i, p in enumerate(allp)}


def build_slice(seed: int = 0) -> dict[str, list[dict]]:
    """Every (template, value pair) prompt of the slice, by split. Prompt records carry the causal-model values, the
    prompt-side answer and its fields for scoring; ids are '<task>:<template_id>:<input>:<offset>' (prompt strings)."""
    out = {"discovery": [], "heldout": []}
    for task, rows in TABLE7.items():
        inputs, offsets, oval, answer = TASKS[task]
        split_of = pair_split(task, seed)
        tpl_split = {t: s for s, ts in SPLIT_TEMPLATES[task].items() for t in ts}
        for tid, tpl, inp_map, off_map, cycles in rows:
            split = tpl_split[tid]
            inp_pairs = list((inp_map or {i: i for i in inputs}).items())
            off_pairs = list((off_map or {o: o for o in offsets}).items())
            for (p_in, c_in), (p_off, c_off) in itertools.product(inp_pairs, off_pairs):
                k = oval(c_off)
                if cycles is not None and k > cycles * _period(task):
                    continue
                if split_of[(c_in, c_off)] != split:
                    continue
                ans_c = answer(c_in, k)                     # causal-model answer (full names, ' Monday' / '07:00' / '42')
                ans = ans_c
                if inp_map:                                 # answer in the prompt's own naming (' Mon'), as the notebook scores
                    rev = {v: kk for kk, v in inp_map.items()}
                    ans = " " + rev[ans_c.strip()]
                out[split].append({
                    "prompt_id": f"{task}:{tid}:{p_in}:{p_off}", "slice": "arithmetic", "task": task, "template_id": tid,
                    "split": split, "text": tpl.format(input=p_in, offset=p_off),
                    "input": p_in, "offset": p_off, "input_value": c_in, "offset_value": str(k),
                    "answer": ans, "answer_full": ans_c,
                    "result_value": ans_c.strip().split(":")[0]})
    return out


def ground_truth() -> dict:
    return {"layer": LAYER, "N_add": sorted(N_ADD), "per_task": NEURONS,
            "period_clusters": {str(k): sorted(v) for k, v in PERIOD_CLUSTERS.items()},
            "reference_top40": REFERENCE_TOP40,
            "notes": ["hours contains 1658, which is not in N_add (hours is mostly, not strictly, nested)",
                      "period clusters partition N_add (from arithmetic-wild src/static_downproj.ipynb)"],
            "source": "github.com/goodfire-ai/arithmetic-wild @ d03024a (src/neurons_per_task.json, static_downproj.ipynb)"}
