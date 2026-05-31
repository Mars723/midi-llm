# Training Workflow

## PDMX Core

Fetch PDMX from the official
[Zenodo record 15571083](https://zenodo.org/records/15571083). The downloader
resumes partial downloads and verifies the published MD5 checksums:

```bash
python -m midi_llm.fetch_pdmx \
  --output-dir training_resources/pdmx \
  --extract-subsets

python -m midi_llm.prepare_pdmx \
  --metadata-csv training_resources/pdmx/PDMX.csv \
  --output-dir training_manifests/pdmx-intermediate \
  --difficulty intermediate \
  --min-measures 48 \
  --max-measures 192 \
  --genres classical-piano,etude,nocturne,waltz,prelude,minuet,impromptu,theme-and-variations

python -m midi_llm.fetch_pdmx \
  --output-dir training_resources/pdmx \
  --include-mxl \
  --extract-mxl \
  --mxl-manifest training_manifests/pdmx-intermediate/pdmx_score_first_manifest.jsonl

python -m midi_llm.curriculum \
  --manifest training_manifests/pdmx-intermediate/pdmx_score_first_manifest.jsonl \
  --dataset-root training_resources/pdmx \
  --output-dir training_manifests/pdmx-intermediate/curriculum

python -m midi_llm.materialize_training \
  --curriculum-dir training_manifests/pdmx-intermediate/curriculum \
  --dataset-root training_resources/pdmx \
  --output-dir training_manifests/pdmx-intermediate/model_dataset

python -m midi_llm.training \
  --manifest training_manifests/pdmx-intermediate/pdmx_score_first_manifest.jsonl \
  --curriculum-examples training_manifests/pdmx-intermediate/curriculum/curriculum_examples.jsonl \
  --model-dataset-summary training_manifests/pdmx-intermediate/model_dataset/summary.json \
  --output training_manifests/pdmx-intermediate/run_plan.json

python -m midi_llm.train_scoredsl \
  --dataset-dir training_manifests/pdmx-intermediate/model_dataset \
  --output-dir training_runs/score_first_v1 \
  --dry-run
```

The manifest uses PDMX's `no_license_conflict`, `all_valid`, and
`is_best_unique_arrangement` fields before examples are emitted. Official PDMX
encodes General MIDI programs in `tracks`; solo piano is `tracks == "0"`.
Genre labels keep their matching field and phrase as evidence. Difficulty is
auditable: `0.65 * note-density-percentile + 0.35 *
normalized-source-complexity`, with `0.30 <= score < 0.75` retained as
`intermediate`. The classical training command excludes `unclassified-piano`
while retaining generic `classical-piano` works and the seven controlled
classical piano genres. Form inference is deliberately conservative: title
keywords identify rondos, variation genres map directly, nocturnes use an
`ABA` heuristic, minuets use a ternary heuristic, and everything else stays
`free-sectional`. The `metadata-curated` tier combines inferred target
genre, source genre, composer, professional-user, rating, view, favorite, and
best-path signals. The stricter `canonical-core` tier additionally requires a
matched classical composer and rejects titles with explicit arrangement
markers. Keep the broader labeled manifest for notation grammar experiments;
the generated tier-aware curriculum uses broad scores for ScoreDSL grammar,
`metadata-curated` scores for local expansion and repair, and `canonical-core`
scores for complete-piece generation and recapitulation supervision.
`no_license_conflict` follows PDMX's published metadata policy; review source
rights separately before any commercial dataset release.

The manifest is deduplicated by work and split before examples are emitted.
Task assignment follows the quality tiers above: every accepted work receives
score autoencoding, while higher-confidence tiers progressively add local and
whole-piece supervision. The curriculum exporter reads MusicXML or compressed
MXL files when available and writes:

- `piece_blueprints.jsonl`: complete-work metadata, heuristic section ranges,
  motif references, and the future cadence target.
- `curriculum_examples.jsonl`: tier-gated full-piece records, `16` measure
  ScoreDSL grammar windows, plus `16`, `32`, and `64` measure expansion
  windows with neighboring context and the shared blueprint.
- `unresolved_sources.jsonl`: manifest rows that still need a valid score path
  or a measure count before training examples can be emitted.

The materializer imports each MusicXML/MXL source as a notation-first
`PianoScoreIR`, writes the authoritative full `score.dsl`, and emits
`train.jsonl`, `valid.jsonl`, and `test.jsonl`. It preserves score-level tempo,
dynamics, pedal markings, wedges, articulations, fingering, ties, voices, and
staves. Local targets remain fragments, but their model input includes the
complete `PiecePlanIR`, the complete `MotifBank`, neighboring ScoreDSL
fragments, the sparse whole-piece skeleton, and the future cadence target.
The materializer rejects stale blueprints when the imported score length no
longer matches the curriculum and writes those rows to
`materialization_errors.jsonl`. It also writes `oversized_examples.jsonl` for
examples above the default `175000` character preflight budget. These examples
remain available as source scores but are excluded from the QLoRA dataset
without truncation.

Model targets use compact fixed-column `ModelScoreDSL v2` rather than the rich
artifact JSON. The shared plan and motif bank stay in the prompt instead of
being regenerated. Event rows use `SCORE`, `NOTE`, `DIRECTION`, `LAYOUT`, and
`END_SCORE`, which shortens contexts and makes line-level syntax auditable.
The full `score.dsl`, `PianoScoreIR`, MusicXML, and PDF output contracts do not
change.

## Curriculum

1. Learn valid `ScoreDSL` syntax with `16` measure score autoencoding windows.
2. Expand `16-64` measure sections while reading the complete `PiecePlanIR`,
   `MotifBank`, neighboring skeleton, and future ending target.
3. Train complete `48-192` measure miniature generation.
4. Train bidirectional inpainting for boundaries, recapitulations, and endings.
5. Fine-tune ending completion against the explicit future cadence target.
6. Revise recapitulation spans against the shared motif bank and both neighbors.

Initialize from `slseanwu/MIDI-LLM_Llama-3.2-1B` and begin with QLoRA on a
single NVIDIA GPU with 48-80GB VRAM. The first cloud run represents ScoreDSL
tags with the upstream tokenizer's existing BPE vocabulary. This avoids
freezing randomly initialized added-token rows during a lightweight QLoRA
pilot. Treat dedicated ScoreDSL embedding and LM-head warm-up as a separate
follow-up experiment.

The current repository prepares deterministic training indexes, materialized
ModelScoreDSL targets, and a decision-complete cloud run plan. It does not claim
that a ScoreDSL adapter has already been trained.

## Cloud QLoRA Launch

Install the optional trainer packages on an NVIDIA host:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
pip install -r requirements-score-first-train.txt
```

The checked-in bootstrap script performs both commands. Pinning the PyTorch
CUDA `12.8` wheel avoids accidentally installing a CUDA runtime newer than the
driver exposed by a rented host.

Inspect the generated launch specification locally or on the cloud host:

```bash
python -m midi_llm.train_scoredsl \
  --dataset-dir training_manifests/pdmx-intermediate/model_dataset \
  --output-dir training_runs/score_first_v1 \
  --dry-run
```

Launch the adapter training after the dry-run spec and maximum sequence length
have been reviewed:

```bash
python -m midi_llm.train_scoredsl \
  --dataset-dir training_manifests/pdmx-intermediate/model_dataset \
  --output-dir training_runs/score_first_v1 \
  --max-seq-length 65536 \
  --epochs 1
```

The launcher uses 4-bit NF4 QLoRA with PyTorch SDPA attention, represents
ModelScoreDSL semantic tags with the existing tokenizer vocabulary, masks the
prompt portion of labels, and trains only against compact score targets. It refuses
to silently truncate any target when `max_seq_length` is too small. The
dry-run character-based token estimate is an early warning only; before the
cloud launcher loads the model, it tokenizes every selected example and
rejects the launch if the real tokenizer count exceeds the context budget.
Because the upstream MIDI-LLM tokenizer has a large vocabulary, the launcher
keeps full Transformer context but computes the LM head and cross-entropy in
checkpointed `256`-token chunks by default. Override this with
`--loss-chunk-tokens` only after a representative long-context probe.
The training step also applies explicit CUDA mixed-precision autocast and
disables PyTorch's quadratic math-SDPA fallback. A host without a compatible
fused attention kernel fails early instead of attempting an infeasible
full-attention allocation.
To keep verbose JSON payloads from dominating the objective, ScoreDSL tag
tokens and EOS receive an `8x` loss weight and the first `64` target tokens
receive a `4x` weight by default. These values are recorded in every launch
spec and can be tuned with `--structural-token-weight`,
`--target-prefix-tokens`, and `--target-prefix-weight`.

## GPU Host Handoff

Package the minimal model-ready dataset locally. The archive deliberately
excludes the normalized per-score cache because the trainer only reads the
JSONL splits. Each included file is SHA-256 checked before extraction:

```bash
python -m midi_llm.package_cloud package \
  --dataset-dir training_manifests/pdmx-intermediate/model_dataset \
  --output training_bundles/score-first-model-dataset.tar.gz

python -m midi_llm.package_cloud verify \
  --bundle training_bundles/score-first-model-dataset.tar.gz
```

On an Ubuntu NVIDIA host, clone this fork's `codex/score-first-piano-v1`
branch, upload the bundle, and bootstrap the environment:

```bash
git clone --branch codex/score-first-piano-v1 \
  https://github.com/Mars723/midi-llm.git
cd midi-llm
bash scripts/bootstrap_score_first_gpu.sh /path/to/score-first-model-dataset.tar.gz
```

From the local workspace, the same clone, upload, checksum verification,
dependency installation, and preflight sequence can be executed with one
command after SSH access is configured:

```bash
bash scripts/deploy_score_first_gpu.sh root@GPU_HOST 22
```

See [`RUNPOD.md`](RUNPOD.md) for the shortest provider-specific setup path.
Use Full SSH with a public IP: Runpod's basic proxied SSH does not support the
`scp` upload used by the deploy helper.

Run a one-step real-sample CUDA smoke test before spending time on the
checkpoint-producing pilot curriculum:

```bash
bash scripts/run_score_first_stages.sh smoke
bash scripts/run_score_first_stages.sh pilot
bash scripts/generate_score_first_pilot_sample.sh
```

The `smoke` command trains one optimizer step against up to eight genuine
ScoreDSL grammar examples below an `8192` token context limit. It exercises
CUDA model loading, 4-bit QLoRA, forward/backward passes, optimizer state, and
adapter saving. Its short-example filter is isolated to smoke testing; it does
not alter the pilot dataset or truncate targets.

The `pilot` command first trains the `score-dsl-autoencode` grammar windows,
then loads that adapter as trainable state for `ending-complete` and
`whole-piece-generate`. This produces the first legitimate adapter-backed
sample checkpoint. Continue with the more expensive structural phase after
reviewing that sample:

```bash
bash scripts/run_score_first_stages.sh structure
```

The structural phase resumes the whole-piece adapter and trains
`section-expand-16-64`, `masked-span-inpaint`, and `recapitulation-revise`.
Use `bash scripts/run_score_first_stages.sh all` when running every phase
without an intermediate review.

Generate a complete score sample from a trained adapter:

```bash
python -m midi_llm.generate_checkpoint \
  --adapter-dir training_runs/score_first_v1/adapter \
  --prompt "A lyrical intermediate nocturne with a tense middle section and a calm return." \
  --genre nocturne \
  --form ABA \
  --duration-minutes 3 \
  --output-dir generated_score_first/checkpoint_sample_001
```

This path samples ScoreDSL notes from the adapter, rejects malformed candidates,
keeps the requested whole-piece plan authoritative, and renders the selected
complete score to the same Gallery artifacts as the local baseline. The
performance overlay still uses the v1 rules renderer.

## Complete-Piece Release Gate

Run the v1 automated gate on 50 complete pieces. MuseScore export is required
for a release run:

```bash
python -m midi_llm.release_gate \
  --output-dir generated_score_first/release_gate_v1 \
  --pieces 50 \
  --candidates 4 \
  --include-sonata
```

The command emits `automatic_results.csv`, `release_gate.json`,
`human_review.csv`, and `review_gallery.html`. The review Gallery links every
piece's paginated score Gallery, PDF, MusicXML, score MIDI, and expressive MIDI.
The automated report checks duration error, section coverage, planned endings,
MusicXML parsing, MuseScore PDF/PNG export, control accuracy, and
expressive-tempo isolation. The human worksheet keeps the required 50-piece
musical review explicit. `sonata-allegro` is reported separately and does not
block v1.

## Performance Layer

The default v1 renderer is deterministic and rule-based. Keep noncommercial
experiments with PianoCoRe-A* and SyMuPe/PianoFlow in a separate run group;
their CC BY-NC-SA data and weights must not enter the redistributable core.
