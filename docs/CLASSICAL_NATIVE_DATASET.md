# Classical Piano Native-Token Dataset

## Scope

Classical specialization must preserve the released MIDI-LLM composition
distribution. Training examples therefore use the upstream Anticipation MIDI
tokens, not ScoreDSL text targets.

The first auditable core is the official
[`PDMX`](https://github.com/pnlong/PDMX) dataset downloaded from its
[`Zenodo record`](https://zenodo.org/records/15571083). PDMX documents a
license-metadata discrepancy and recommends the `no_license_conflict` subset.
It also provides an `all_valid` subset for works with valid associated files.
Use both subsets and the preferred unique arrangement before training.

## Composer Controls

Each selected work retains its source composer metadata and adds explicit,
machine-readable controls:

```json
{
  "composer_style": "chopin",
  "composer_period": "romantic",
  "style_tags": [
    "composer-style:chopin",
    "period:romantic",
    "genre:nocturne"
  ]
}
```

The surname-derived style tag is conservative metadata, not a claim that every
uploaded score is an authoritative edition. It can be used as a prompt control
and as a balancing key during training and evaluation.

## PDMX Core Commands

Build a focused four-composer manifest:

```bash
python -m midi_llm.prepare_pdmx \
  --metadata-csv training_resources/pdmx/PDMX.csv \
  --output-dir training_manifests/pdmx-classical-core \
  --composers bach,mozart,beethoven,chopin
```

Download and selectively extract only the referenced MIDI files:

```bash
python -m midi_llm.fetch_pdmx \
  --output-dir training_resources/pdmx \
  --include-midi \
  --extract-midi \
  --midi-manifest training_manifests/pdmx-classical-core/pdmx_score_first_manifest.jsonl
```

Materialize complete pieces in the upstream extended vocabulary:

```bash
python -m midi_llm.materialize_native_training \
  --manifest training_manifests/pdmx-classical-core/pdmx_score_first_manifest.jsonl \
  --dataset-root training_resources/pdmx \
  --output-dir training_manifests/pdmx-classical-core/native_tokens
```

The materializer never truncates an oversized complete work. Anticipation uses
segment-local absolute time tokens with an approximately 100-second vocabulary
range, so long pieces are covered by rebased native time windows while the
complete source MIDI and work-level controls remain intact. A segment that
still exceeds an explicitly configured context budget is recorded separately
instead of being truncated.

## Supplemental Source

Evaluate [`Mutopia`](https://www.mutopiaproject.org/) only as a separately
tracked supplement. Its official
[`license page`](https://www.mutopiaproject.org/legal.html) states that its
music uses free-cultural-work Creative Commons licenses, including
Attribution-ShareAlike, Attribution, and public-domain contributions.

Do not merge Mutopia files into the PDMX namespace. Preserve source URL,
license, attribution, arranger/editor metadata, and hashes per score before
including them.

## Next Training Gate

1. Materialize and inspect token-length distributions.
2. Keep train, validation, and test splits at work level.
3. Balance composer, period, genre, difficulty, and whole-piece length.
4. Fine-tune the upstream checkpoint at low learning rate with parity replay.
5. Reject any adapter that regresses native upstream parity before evaluating
   longer completion, emotional control, or notation conversion.

Create the native QLoRA dry-run spec before allocating GPU time:

```bash
bash scripts/run_native_classical_stages.sh dry-run
```

Then run one optimizer step only:

```bash
bash scripts/run_native_classical_stages.sh smoke
```

The pilot is intentionally bounded to `100` optimizer steps at a default
learning rate of `2e-5`:

```bash
bash scripts/run_native_classical_stages.sh pilot
```

Generate fixed-seed upstream and adapter candidates after each checkpoint.
Reject a checkpoint if native MIDI validity, density drift, or baseline prompt
quality regresses, even when its training loss decreases.

Run adapter parity with the same upstream-native generator:

```bash
bash scripts/run_native_adapter_parity.sh \
  /root/midllm-local/training_runs/native_classical_v1/02_pilot/adapter \
  /root/midllm-local/generated_native_backbone/native_classical_v1_pilot
```
