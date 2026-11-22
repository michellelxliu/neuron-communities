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
# Table 7 also has templates with re-worded offsets and inputs (number/name maps); added with the full slice in step 4.


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
