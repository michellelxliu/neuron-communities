"""Toy tasks with planted shared subcircuits: the high-level model, its ground truth, and the prompt sampler.

Symbols V = {0..n_symbols-1}. Primitives are random permutations of V (fixed seed): first-layer primitives (S1, S2, ...)
are computed in the layer-0 MLP, second-layer primitives (A1, A2, ...) in the layer-1 MLP. A task is a composition
``T = A_i o S_j``; the prompt is ``BOS.. d_1 .. d_m TASK_k x =`` (``m`` random distractor symbols, 0..max_distractors,
left-padded with BOS to a fixed length) and the model must output ``y = A_i(S_j(x))`` at ``=``.

Neuron sets (the planted ground truth): every primitive owns ``block_size`` neurons in its layer. For each pair in
``overlap_pairs`` (two primitives of the same layer), ``round(overlap_rho * block_size)`` of those neurons are shared:
they belong to both primitives, so single neurons serve two primitives. A task's circuit is the union of its two
primitives' sets. Neurons in no set are free. ``tasks="null"`` gives the matched null task set: one private S and A
per task (no primitive shared between tasks); with ``overlap_rho = 0`` it has no overlap of any kind.

Optional train / held-out split (``heldout_frac``, default 0): by (task, x) pair, only where composition keeps the pair solvable (each primitive of the task
sees the needed input in training through another task), so held-out-pair accuracy measures the planted shared
structure, not memorisation. Prompts also vary by their random distractor prefix, so fresh samples are unseen prompts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

DEFAULT_TASKS = (("S1", "A1"), ("S1", "A2"), ("S2", "A3"), ("S2", "A4"), ("S2", "A1"))
NULL_TASKS = tuple((f"S{k}", f"A{k}") for k in range(1, 6))         # no primitive shared between tasks
DEFAULT_PAIRS = (("S1", "S2"), ("A1", "A4"), ("A2", "A3"))          # neuron-sharing pairs (same layer, different task groups)
NULL_PAIRS = (("S1", "S2"), ("S3", "S4"), ("A1", "A2"), ("A3", "A4"))


@dataclass
class ToyConfig:
    name: str = "base"
    seed: int = 0
    n_symbols: int = 64
    tasks: tuple = DEFAULT_TASKS                      # (first-layer primitive, second-layer primitive) per task
    block_size: int = 32                              # neurons per primitive
    overlap_rho: float = 0.0                          # fraction of a primitive's neurons shared with its pair partner
    overlap_pairs: tuple = DEFAULT_PAIRS              # same-layer primitive pairs that share neurons
    copies: int = 1                                   # blocks per primitive (redundancy knob; only 1 implemented)
    task_freq: tuple | None = None                    # sampling weights per task (None: uniform)
    d_model: int = 128
    d_mlp: int = 512
    n_heads: int = 4
    max_distractors: int = 4
    heldout_frac: float = 0.0                         # compositional held-out pairs (mask training does not make tasks
                                                      # share a primitive's function, only its neurons, so off by default)
    noise_neurons: int = 0                            # free neurons trained to stay weakly active (not implemented)
    # training: task masks, then masks removed and fine-tuned with an off-circuit activation penalty
    steps_mask: int = 6000
    steps_finetune: int = 3000
    batch: int = 256
    lr: float = 3e-3
    lr_finetune: float = 1e-3
    weight_decay: float = 0.01
    off_penalty: float = 1.0                          # weight of the off-circuit activation share at the final position
    label_smoothing: float = 0.0                      # item-3 remedy for saturated softmaxes (0 = off)
    extra: dict = field(default_factory=dict)

    @staticmethod
    def null_of(cfg: "ToyConfig") -> "ToyConfig":
        """The matched null model of ``cfg``: same sizes and training, no primitive shared between tasks, rho = 0."""
        d = asdict(cfg)
        d.update(name=cfg.name + "_null", tasks=NULL_TASKS[: len(cfg.tasks)], overlap_rho=0.0, overlap_pairs=NULL_PAIRS)
        return ToyConfig.from_json(json.dumps(d))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_json(s: str) -> "ToyConfig":
        d = json.loads(s)
        d["tasks"] = tuple(tuple(t) for t in d["tasks"])
        d["overlap_pairs"] = tuple(tuple(t) for t in d.get("overlap_pairs", ()))
        if d.get("task_freq") is not None:
            d["task_freq"] = tuple(d["task_freq"])
        return ToyConfig(**d)


class ToyWorld:
    """Tables, vocabulary, ground truth and samplers of one toy configuration."""

    def __init__(self, cfg: ToyConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        V = cfg.n_symbols
        self.first = sorted({s for s, _ in cfg.tasks}, key=lambda p: int(p[1:]))
        self.second = sorted({a for _, a in cfg.tasks}, key=lambda p: int(p[1:]))
        self.primitives = self.first + self.second
        self.table = {p: rng.permutation(V) for p in self.primitives}
        # vocabulary: symbols, task tokens, '=', BOS
        self.task_tok = {k: V + k for k in range(len(cfg.tasks))}
        self.eq_tok = V + len(cfg.tasks)
        self.bos_tok = self.eq_tok + 1
        self.vocab_size = self.bos_tok + 1
        # membership matrix M: tasks x primitives
        self.M = np.zeros((len(cfg.tasks), len(self.primitives)), np.int8)
        for k, (s, a) in enumerate(cfg.tasks):
            self.M[k, self.primitives.index(s)] = 1
            self.M[k, self.primitives.index(a)] = 1
        # neuron sets, after a seeded shuffle of neuron ids (so ids carry no order)
        assert cfg.copies == 1, "copies > 1 not implemented"
        assert cfg.noise_neurons == 0, "noise_neurons > 0 not implemented"
        self.layer_of = {p: (0 if p in self.first else 1) for p in self.primitives}
        n_sh = int(round(cfg.overlap_rho * cfg.block_size))
        pairs = [tuple(pq) for pq in cfg.overlap_pairs if pq[0] in self.primitives and pq[1] in self.primitives]
        assert all(self.layer_of[a] == self.layer_of[b] for a, b in pairs), "overlap pairs must share a layer"
        assert len({p for pq in pairs for p in pq}) == 2 * len(pairs), "a primitive may be in at most one pair"
        perm = {L: rng.permutation(cfg.d_mlp).tolist() for L in (0, 1)}
        nxt = {0: 0, 1: 0}

        def take(L: int, n: int) -> list[int]:
            ids = perm[L][nxt[L]: nxt[L] + n]
            nxt[L] += n
            return ids

        sets: dict[str, set] = {p: set() for p in self.primitives}
        self.shared: dict[str, tuple[int, list[int]]] = {}
        for a, b in pairs:
            L = self.layer_of[a]
            ids = take(L, n_sh)
            sets[a] |= set(ids); sets[b] |= set(ids)
            self.shared[f"{a}&{b}"] = (L, sorted(ids))
        for p in self.primitives:
            L = self.layer_of[p]
            sets[p] |= set(take(L, cfg.block_size - len(sets[p])))
        assert all(v <= cfg.d_mlp for v in nxt.values()), "neuron sets do not fit the MLP"
        self.blocks = {p: (self.layer_of[p], sorted(sets[p])) for p in self.primitives}      # full (overlapping) sets
        shared_ids = {L: {n for L2, ids in self.shared.values() if L2 == L for n in ids} for L in (0, 1)}  # ids are per layer
        self.private = {p: (self.layer_of[p], sorted(sets[p] - shared_ids[self.layer_of[p]])) for p in self.primitives}
        self.block_copies = {p: [ids] for p, (_, ids) in self.blocks.items()}
        self.free = {L: sorted(set(range(cfg.d_mlp)) - {n for p in self.primitives if self.layer_of[p] == L for n in self.blocks[p][1]})
                     for L in (0, 1)}
        # split by (task, x)
        self.train_pairs, self.heldout_pairs = self._split(rng)
        w = np.ones(len(cfg.tasks)) if cfg.task_freq is None else np.asarray(cfg.task_freq, float)
        self.task_p = w / w.sum()

    def answer(self, k: int, x):
        s, a = self.cfg.tasks[k]
        return self.table[a][self.table[s][x]]

    def _split(self, rng):
        """Hold out (task, x) pairs greedily, in random order, up to ``heldout_frac`` of all pairs, keeping a pair out
        only if every primitive of its task still sees the needed input in training through another task. Tasks whose
        primitives are not shared (e.g. A2 in the base set) therefore keep all their pairs in training."""
        V, nT = self.cfg.n_symbols, len(self.cfg.tasks)
        held = np.zeros((nT, V), bool)

        def covered(p, inp):                         # is input ``inp`` of primitive p seen in training?
            for k, (s, a) in enumerate(self.cfg.tasks):
                if p == s and not held[k, inp]:
                    return True
                if p == a:
                    xs = np.flatnonzero(self.table[s] == inp)
                    if len(xs) and not held[k, xs[0]]:
                        return True
            return False

        budget = int(self.cfg.heldout_frac * nT * V)
        for flat in rng.permutation(nT * V):
            if held.sum() >= budget:
                break
            k, x = divmod(int(flat), V)
            s, a = self.cfg.tasks[k]
            held[k, x] = True
            if not (covered(s, x) and covered(a, self.table[s][x])):
                held[k, x] = False
        return [np.flatnonzero(~held[k]) for k in range(nT)], [np.flatnonzero(held[k]) for k in range(nT)]

    # ---- sampling ----------------------------------------------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator, split: str = "train", tasks: np.ndarray | None = None,
               xs: np.ndarray | None = None) -> dict:
        """``n`` prompts: tokens (n, T) with ``=`` last, left-padded with BOS to a fixed length T, plus task, x, y."""
        pairs = {"train": self.train_pairs, "heldout": self.heldout_pairs,
                 "all": [np.arange(self.cfg.n_symbols)] * len(self.cfg.tasks)}[split]
        if tasks is None:
            tasks = rng.choice(len(self.cfg.tasks), size=n, p=self.task_p)
        if xs is None:
            xs = np.array([rng.choice(pairs[k]) for k in tasks])
        m = rng.integers(0, self.cfg.max_distractors + 1, size=n)
        T = self.cfg.max_distractors + 4
        tok = np.full((n, T), self.bos_tok, np.int64)
        for i in range(n):
            seq = [self.bos_tok] + rng.integers(0, self.cfg.n_symbols, size=m[i]).tolist() + [self.task_tok[int(tasks[i])], int(xs[i]), self.eq_tok]
            tok[i, T - len(seq):] = seq
        y = np.array([self.answer(int(k), int(x)) for k, x in zip(tasks, xs)])
        # the BOS left-padding is part of the prompt (every prompt has the same length and positions), so no mask
        attn = np.ones_like(tok)
        return {"tokens": tok, "attn": attn, "task": np.asarray(tasks), "x": np.asarray(xs), "y": y, "n_distractors": m}

    def ground_truth(self) -> dict:
        return {"primitives": self.primitives, "tasks": [list(t) for t in self.cfg.tasks], "M": self.M.tolist(),
                "blocks": {p: {"layer": L, "neurons": ids, "private": self.private[p][1]} for p, (L, ids) in self.blocks.items()},
                "shared": {k: {"layer": L, "neurons": ids} for k, (L, ids) in self.shared.items()},
                "overlap_rho": self.cfg.overlap_rho,
                "circuits": [{"task": k, "neurons": {str(L): v for L, v in self.task_neurons(k).items()}} for k in range(len(self.cfg.tasks))],
                "free": {str(L): v for L, v in self.free.items()},
                "tables": {p: t.tolist() for p, t in self.table.items()},
                "vocab": {"n_symbols": self.cfg.n_symbols, "task_tokens": self.task_tok, "eq": self.eq_tok, "bos": self.bos_tok,
                          "size": self.vocab_size},
                "train_pairs": [v.tolist() for v in self.train_pairs], "heldout_pairs": [v.tolist() for v in self.heldout_pairs]}

    def task_neurons(self, k: int) -> dict[int, list[int]]:
        """Neurons of the blocks task ``k`` uses, per layer."""
        out = {0: set(), 1: set()}
        for p in self.cfg.tasks[k]:
            L, ids = self.blocks[p]
            out[L] |= set(ids)
        return {L: sorted(v) for L, v in out.items()}
