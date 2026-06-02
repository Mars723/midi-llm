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

Keep active training on the disposable local disk and archive only completed
Trainer checkpoints to the Runpod persistent volume. When an ephemeral
`rclone` Drive remote is configured, the watcher also copies each stable
archive offsite:

```bash
bash scripts/run_native_classical_stages.sh pilot > /root/midllm-local/logs/native-classical-v1-02-pilot.log 2>&1 &
TRAINING_PID=$!
MIDI_LLM_NATIVE_LOG=/root/midllm-local/logs/native-classical-v1-02-pilot.log \
MIDI_LLM_RCLONE_REMOTE='gdrive:MIDI-LLM/runpod/score-first-intermediate-v2/native-classical-v1' \
  bash scripts/watch_native_classical_backup.sh \
  "$TRAINING_PID" \
  /root/midllm-local/training_runs/native_classical_v1/02_pilot \
  native_classical_v1_02_pilot
```

The watcher does not sync an actively changing directory into the persistent
network mount. It packages a checkpoint only after `trainer_state.json`
exists, then copies the stable archive and SHA-256 file to persistent storage
and Drive. Keep the OAuth-bearing rclone configuration on `/root`, which is
intentionally disposable.

Generate fixed-seed upstream and adapter candidates after each checkpoint.
Reject a checkpoint if native MIDI validity, density drift, or baseline prompt
quality regresses, even when its training loss decreases.

Run adapter parity with the same upstream-native generator:

```bash
bash scripts/run_native_adapter_parity.sh \
  /root/midllm-local/training_runs/native_classical_v1/02_pilot/adapter \
  /root/midllm-local/generated_native_backbone/native_classical_v1_pilot
```

Do not mislabel a token-budget-capped parity replay as a complete composition.
The upstream Anticipation absolute-time vocabulary covers an approximately
`100`-second local window. Produce an interim whole-piece sample with a shared
blueprint, prefix-conditioned B and Coda materials, explicit A-prime motif
reuse, and a realized coda termination target:

```bash
bash scripts/generate_native_whole_piece_pilot.sh \
  /root/midllm-local/training_runs/native_classical_v1/02_pilot/adapter \
  /root/midllm-local/generated_native_backbone/native_classical_v1_whole_piece_001
```

This interim generator is an honest structural milestone, not the release
architecture. Its manifest records inferred draft markings separately from
future model-generated notation and reports the remaining boundary quality.

After each adapter stage, write an honest progress report against the fixed
upstream replay and the current whole-piece sample. Human listening review is a
required promotion input; automated structural scores cannot replace it:

```bash
bash scripts/report_native_training_progress.sh \
  /root/midllm-local/generated_native_backbone/upstream_parity_nocturne_v1 \
  /root/midllm-local/generated_native_backbone/native_classical_v1_pilot_chopin_nocturne_2046 \
  /root/midllm-local/generated_native_backbone/native_classical_v1_whole_piece_001/piece \
  /root/midllm-local/generated_native_backbone/native_classical_v1_progress \
  pending \
  'Awaiting listening review.'
```

If a bounded pilot regresses fixed-seed parity, do not continue training from
its final adapter. Start a conservative low-learning-rate run and sweep each
saved checkpoint:

```bash
bash scripts/run_native_classical_stages.sh conservative-pilot
bash scripts/run_native_checkpoint_sweep.sh \
  /root/midllm-local/training_runs/native_classical_v1/03_conservative_pilot
```

The conservative stage defaults to `60` optimizer steps, `5e-6` learning
rate, and a checkpoint every `10` steps. The sweep rejects a checkpoint when
its syntax-valid rate falls below upstream or its density-drift rate exceeds
upstream. Generate a whole-piece sample only from the selected checkpoint.
