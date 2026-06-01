# Native Backbone Research Roadmap

## Principle

Improve the released `slseanwu/MIDI-LLM_Llama-3.2-1B` model incrementally.
Do not replace its trained Anticipation MIDI distribution with a small
ScoreDSL-from-scratch composition adapter.

The first gate is upstream parity:

```bash
python -m midi_llm.native_backbone \
  --prompt "A lyrical intermediate solo piano nocturne with a tense middle section and a calm return." \
  --output-dir generated_native_backbone/nocturne_parity
```

Each run records whether a candidate naturally stopped or exhausted its token
budget. It also records note density and polyphony drift indicators. A candidate
that exhausts the budget is a useful listening sample, but not evidence of a
complete composition. A syntactically valid long stream with density drift is
also rejected as completion evidence.

## Evidence-Based Stages

### N1: Preserve And Rank Native Candidates

Generate multiple upstream-native MIDI candidates and keep `native.mid` as the
authoritative musical content. Rank with transparent structural heuristics and
human review before adding learned rerankers.

For non-commercial research, evaluate the optional
[`SyMuPe/MIDI-Quality-Classifier`](https://huggingface.co/SyMuPe/MIDI-Quality-Classifier)
as a learned reranker. SyMuPe reports that this classifier was trained from
human preference annotations and exposes a MIDI quality score. Its
`CC-BY-NC-SA-4.0` license means it must not silently become a commercial
production dependency.

### N2: Improve Completion And Form Without Relearning Note Syntax

Start with larger native token budgets and measure natural-stop behavior.
Then add native-token continuation, overlap scoring, cadence-aware stopping,
and candidate ranking while keeping the same MIDI representation.

Use these papers as design references:

- [`MIDI-LLM`](https://arxiv.org/abs/2511.03942) adapts a text LLM to native
  MIDI token generation and is the released backbone used here.
- [`Music Transformer`](https://arxiv.org/abs/1809.04281) shows that relative
  attention improves long-range coherence in minute-scale symbolic music.
- [`Museformer`](https://arxiv.org/abs/2210.10349) combines fine-grained and
  coarse-grained attention for long symbolic sequences.
- [`FIGARO`](https://arxiv.org/abs/2201.10936) separates high-level
  descriptions from low-level sequence generation.
- [`MeloForm`](https://arxiv.org/abs/2208.14345) uses musical-form-aware
  generation and iterative refinement at motif, phrase, and section levels.
- [`Theme Transformer`](https://arxiv.org/abs/2111.04093) targets controllable
  theme-conditioned generation and longer-term thematic structure.
- [`Anticipatory Music Transformer`](https://arxiv.org/abs/2306.08620) models
  future constraints in symbolic generation. Its Anticipation token tooling is
  already the upstream MIDI representation dependency.

### N3: Add Emotion And Expressive Performance As Separate Controls

Do not conflate composition structure with humanized performance timing.

- [`MuseMorphose`](https://arxiv.org/abs/2105.04090) provides segment-level and
  whole-song style control for symbolic piano music.
- [`EMOPIA`](https://arxiv.org/abs/2108.01374) provides a piano MIDI benchmark
  for emotion recognition and emotion-based generation.
- [`SyMuPe`](https://arxiv.org/abs/2511.03425) separates score MIDI from
  performance MIDI and releases PianoFlow for humanization. Evaluate it only
  in the non-commercial research lane because of its released license.

### N4: Improve Draft MIDI-To-Score Conversion

Notation conversion is a separate problem. It must preserve the selected
native notes and isolate expressive timing from printed score tempo markings.

Use [`Transformer-Based Rhythm Quantization of Performance MIDI Using Beat
Annotations`](https://arxiv.org/abs/2604.22290) as a reference for stronger
quantization. Improve hand separation, voice assignment, key inference, rhythm
spelling, pedal cleanup, dynamics, and articulation inference behind
note-preservation checks.

## Promotion Gates

1. Upstream parity listening review: at least four native candidates.
2. Completion review: token-budget exhaustion and cadence behavior reported.
3. Structural review: repeated themes, contrast, return, and ending judged on
   complete native MIDI candidates.
4. Emotion review: prompt-conditioned affect and expressive rendering reviewed
   separately.
5. Notation review: PDF score defects measured separately from composition
   defects.
