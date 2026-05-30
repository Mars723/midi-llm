# Training Workflow

## Public-Domain Core

Download PDMX separately and use its `no_license_conflict` and `all_valid`
subsets. The repository does not download datasets automatically.

```bash
python -m midi_llm.prepare_pdmx \
  --metadata-csv /data/pdmx/PDMX.csv \
  --output-dir training_manifests/pdmx

python -m midi_llm.curriculum \
  --manifest training_manifests/pdmx/pdmx_score_first_manifest.jsonl \
  --dataset-root /data/pdmx \
  --output-dir training_manifests/pdmx/curriculum

python -m midi_llm.materialize_training \
  --curriculum-dir training_manifests/pdmx/curriculum \
  --dataset-root /data/pdmx \
  --output-dir training_manifests/pdmx/model_dataset

python -m midi_llm.training \
  --manifest training_manifests/pdmx/pdmx_score_first_manifest.jsonl \
  --curriculum-examples training_manifests/pdmx/curriculum/curriculum_examples.jsonl \
  --model-dataset-summary training_manifests/pdmx/model_dataset/summary.json \
  --output training_manifests/pdmx/run_plan.json

python -m midi_llm.train_scoredsl \
  --dataset-dir training_manifests/pdmx/model_dataset \
  --output-dir training_runs/score_first_v1 \
  --dry-run
```

The manifest is deduplicated by work and split before examples are emitted.
Every accepted work receives local-window, whole-piece, masked-span,
ending-completion, and recapitulation-revision tasks. The curriculum exporter
reads MusicXML or compressed MXL files when available and writes:

- `piece_blueprints.jsonl`: complete-work metadata, heuristic section ranges,
  motif references, and the future cadence target.
- `curriculum_examples.jsonl`: full-piece records plus `16`, `32`, and `64`
  measure expansion windows with neighboring context and the shared blueprint.
- `unresolved_sources.jsonl`: manifest rows that still need a valid score path
  or a measure count before training examples can be emitted.

The materializer imports each MusicXML/MXL source as a notation-first
`PianoScoreIR`, writes the authoritative full `score.dsl`, and emits
`train.jsonl`, `valid.jsonl`, and `test.jsonl`. It preserves score-level tempo,
dynamics, pedal markings, wedges, articulations, fingering, ties, voices, and
staves. Local targets remain fragments, but their model input includes the
complete `PiecePlanIR`, the complete `MotifBank`, neighboring ScoreDSL
fragments, the sparse whole-piece skeleton, and the future cadence target.

## Curriculum

1. Learn valid `ScoreDSL` syntax with score autoencoding.
2. Expand `16-64` measure sections while reading the complete `PiecePlanIR`,
   `MotifBank`, neighboring skeleton, and future ending target.
3. Train complete `48-192` measure miniature generation.
4. Train bidirectional inpainting for boundaries, recapitulations, and endings.
5. Fine-tune ending completion against the explicit future cadence target.
6. Revise recapitulation spans against the shared motif bank and both neighbors.

Initialize from `slseanwu/MIDI-LLM_Llama-3.2-1B`, replace the incompatible
Anticipation extension vocabulary with ScoreDSL embeddings, and begin with
QLoRA on a single NVIDIA GPU with 48-80GB VRAM.

The current repository prepares deterministic training indexes, materialized
ScoreDSL targets, and a decision-complete cloud run plan. It does not claim
that a ScoreDSL adapter has already been trained.

## Cloud QLoRA Launch

Install the optional trainer packages on an NVIDIA host:

```bash
pip install -r requirements-score-first-train.txt
```

Inspect the generated launch specification locally or on the cloud host:

```bash
python -m midi_llm.train_scoredsl \
  --dataset-dir training_manifests/pdmx/model_dataset \
  --output-dir training_runs/score_first_v1 \
  --dry-run
```

Launch the adapter training after the dry-run spec and maximum sequence length
have been reviewed:

```bash
python -m midi_llm.train_scoredsl \
  --dataset-dir training_manifests/pdmx/model_dataset \
  --output-dir training_runs/score_first_v1 \
  --max-seq-length 65536 \
  --epochs 1
```

The launcher uses 4-bit NF4 QLoRA, adds ScoreDSL semantic tokens, masks the
prompt portion of labels, and trains only against ScoreDSL targets. It refuses
to silently truncate a complete piece when `max_seq_length` is too small. The
dry-run character-based token estimate is an early warning only; the cloud
launcher enforces the real token count after loading the selected tokenizer.

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
