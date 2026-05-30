# Score-First Piano Pipeline

The score-first path keeps notation and performance timing separate.

## Whole-Piece Generation

`16-64` measures is a local training window, not an output limit. A generated
piece has one `PiecePlanIR` covering the complete work, a shared `MotifBank`,
and a `PianoScoreIR` covering every planned section. The rule-based v1
generator is a runnable baseline for this contract. A trained model can replace
the planner and section expansion stages without changing the output format.

```bash
python -m midi_llm.compose \
  --prompt "A lyrical nocturne with a tense middle section and a calm return." \
  --controls configs/example_controls.yaml \
  --whole-piece
```

Each run contains:

- `score.ir.json`: authoritative notation IR
- `score.dsl`: model-facing line-oriented representation
- `score.musicxml`, `score.mscz`, `score.pdf`, `score-*.png`: engraved score
- `score.mid`: quantized notation playback
- `performance.ir.json`, `performance.mid`: expressive overlay and playback
- `gallery.html`: complete-piece preview and section timeline

## Preview

MuseScore 4 is used for deterministic PDF and PNG export.

```bash
python -m midi_llm.preview --run generated_score_first/<run> --serve
```

## Draft MIDI Import

An external performance MIDI cannot recover a publication-quality score
losslessly. Import therefore produces an explicitly labeled draft:

```bash
python -m midi_llm.import_midi \
  --input performance.mid \
  --output-dir draft_import
```

The importer quantizes note positions for notation. It retains detailed tempo
events in `PianoPerformanceIR`, promotes only stable tempo platforms into the
printed score, and emits a single `rit.` or `accel.` text direction for a
sustained monotonic curve.

Notation-first MusicXML/MXL training sources use a separate importer:

```bash
python -m midi_llm.musicxml_score \
  --input source.musicxml \
  --output-dir imported_score
```

This path preserves score-level notation such as tempo, dynamics, pedal
markings, wedges, articulations, fingering, ties, voices, and staves in
`PianoScoreIR` and `ScoreDSL`.

## Release Gate

Use `python -m midi_llm.release_gate --output-dir <dir> --pieces 50` to generate
the complete-piece review set. The report includes automated checks and a
`human_review.csv` worksheet for structural and engraving review.
`review_gallery.html` links each paginated score Gallery and downloadable
artifact from one page.
