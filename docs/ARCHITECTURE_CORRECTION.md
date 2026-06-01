# Native MIDI Backbone Correction

## Problem

The first ScoreDSL experiment initialized QLoRA from
`slseanwu/MIDI-LLM_Llama-3.2-1B`, but asked the adapter to emit a new textual
ScoreDSL representation directly. That technically reused upstream weights, yet
it did not preserve the native Anticipation MIDI generation path responsible for
the upstream Live Demo quality.

The resulting adapter-backed sample is a diagnostic artifact. It is not a
quality successor to upstream MIDI-LLM.

## Corrected Architecture

The production path is split into two layers:

1. **Native musical-content backbone**
   - Load `slseanwu/MIDI-LLM_Llama-3.2-1B` directly.
   - Use the upstream prompt framing, native MIDI BOS token, and Anticipation
     MIDI token stream.
   - Preserve every generated `native.mid` as the authoritative musical result.
   - Pass a parity gate against the upstream Live Demo before any new training
     claim is accepted.

2. **Score editor and renderer**
   - Import `native.mid` as a draft score and a separate expressive overlay.
   - Keep microtempo, velocity, and detailed pedal timing out of the printed
     score unless they represent stable structural notation.
   - Improve quantization, hand separation, voice assignment, pedal cleanup,
     dynamics, articulations, and engraving without replacing the native notes
     with a newly sampled ScoreDSL composition.
   - Train future notation models as constrained editors with note-preservation
     metrics, not as substitute composers.

## Pilot

On a CUDA Pod:

```bash
export MIDI_LLM_NATIVE_OUTPUT_ROOT=/workspace/midllm-backups/generated_native_backbone
export MIDI_LLM_RCLONE_REMOTE='gdrive:MIDI-LLM/runpod/score-first-intermediate-v2'
bash scripts/run_native_backbone_pilot.sh
```

The command writes a `native.mid` for each candidate, optional draft score
previews, a manifest stating that native MIDI is authoritative, and a backed-up
archive.

## Gates Before Additional Training

1. Generate at least four solo-piano native MIDI candidates with the unmodified
   upstream checkpoint.
2. Manually compare musical quality with the upstream Live Demo.
3. Select one real native candidate and inspect its draft PDF.
4. Measure conversion defects separately from composition defects.
5. Only then train or implement the notation editor against the observed
   conversion defects.

The previous direct ScoreDSL adapter remains available for research comparison,
but it is not the default quality path.

See [`NATIVE_BACKBONE_ROADMAP.md`](NATIVE_BACKBONE_ROADMAP.md) for the
evidence-based completion, form, emotion, performance, and score-conversion
stages.
