# Neuron communities

Code for *Neuron-sparse behaviors in an LLM: small neuron communities causally necessary and sufficient for many computations*.

This repo finds small groups of MLP neurons in Llama-3.1-8B that work together. It tests whether removing a group changes the model's next-token prediction, and whether keeping just that group within a range of layers preserves it.

[Explore the 1,941 labelled communities](https://goodfire.com/demos/neuron-communities), with prompts, interventions, and statistics.

## How it works

1. **Sample prompts** from FineWeb-Edu, Wikipedia, and FineMath. Remove duplicates and keep a separate held-out set for evaluation.
2. **Find important neurons.** For each prompt, select the top five neurons per layer using activation × gradient attribution.
3. **Find communities.** Connect neurons that appear together more often than expected by chance, then group them with Leiden clustering. Drop groups larger than 128 neurons and remove the 17 always-on hub neurons.
4. **Test each community** on held-out prompts. Removing its neurons must change the next-token distribution; keeping only its neurons plus the hubs within a span of layers must preserve it. The hubs alone must fail to preserve it, and removing the community within that same span must also change it.
5. **Measure coverage:** how many held-out prompts have at least one community that passes these tests.

The defaults in [config.py](neuron_communities/config.py) are the settings used for Llama-3.1-8B in the paper.

## Install

Requires Python 3.11 or later.

```bash
pip install -e .
```

The model stages need a GPU. The paper's runs used 80 GB GPUs with the model in bfloat16.

## Run

Run the full pipeline with the paper's settings:

```bash
bash scripts/run_pipeline.sh
```

This is a large run: it samples 10 million training prompts and another 10 million held-out prompts before deduplication. The script skips outputs that already exist, so you can rerun it after an interruption.

Settings such as the model, output directory, prompt counts, and shard counts are environment variables listed at the top of [run_pipeline.sh](scripts/run_pipeline.sh).

To see the options for an individual stage:

```bash
python -m neuron_communities <stage> --help
```

For cluster runs, [stage_array.slurm](scripts/slurm/stage_array.slurm) provides a Slurm job-array template.

## Wide harvest (this fork)

This fork adds one harvest stage and one converter; every other stage is the upstream code, unchanged.

- `attribution-wide`: the same score and mechanics as `attribution`, but keeps the top 50 neurons per layer with the
  signed score `s = h * dlogp(top-1)/dh` (float32) and the activation `h` (float16), plus per-prompt tokens, top-1
  token and probability, top-20 clean log-probs, per-layer `sum |s|` and `sum s`, and prompt metadata from an
  optional `<manifest>/prompts.parquet`. Output: Parquet parts under `<out>/parts/`.
- `wide-blocks --k 5`: writes the top-5 prefix of a wide harvest in the upstream block layout. It is bit-identical to
  running upstream `attribution --k 5` (checked by `tests/test_wide.py`), so `world` and every later stage run on it
  as the baseline.

```bash
python -m neuron_communities attribution-wide --manifest-dir M --out-dir H_WIDE --shard 0 --shards 1
python -m neuron_communities wide-blocks --harvest-dir H_WIDE --out-dir H_TOP5 --k 5
python -m neuron_communities world --harvest-dir H_TOP5 --manifest-dir M --out-dir W
```

Pin the model revision by passing a local snapshot path (`.../snapshots/<sha>`) as `--model`; it is recorded in
`run_meta_shard*.json` together with the code commit, library versions and manifest hashes.

## Test

```bash
pip install pytest
pytest -s tests
```

The tests run the pipeline on a tiny, randomly initialised Llama and synthetic data. They check that the stages work together, not the paper's results. They take about two minutes on eight CPU cores, with no network or GPU needed.

## Citation

<!-- CITATION -->

## License

[MIT](LICENSE).
