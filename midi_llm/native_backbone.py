"""Generate upstream-native Anticipation MIDI before applying score conversion."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from .import_midi import ParsedMidi, import_draft, parse_midi
from .native_tokens import (
    NATIVE_MIDI_BOS_MODEL_TOKEN_ID,
    normalize_native_model_tokens,
)


DEFAULT_MODEL = "slseanwu/MIDI-LLM_Llama-3.2-1B"
UPSTREAM_SYSTEM_PROMPT = (
    "You are a world-class composer. Please compose some music according to the following description: "
)


@dataclass
class NativeBackboneConfig:
    prompt: str
    output_dir: str
    model: str = DEFAULT_MODEL
    adapter: str | None = None
    n_outputs: int = 4
    seed: int = 23
    temperature: float = 1.0
    top_p: float = 0.98
    max_tokens: int = 2046
    score_draft: bool = True
    render_score_draft: bool = False


def upstream_generation_prompt(prompt: str) -> str:
    """Match the prompt framing used by upstream Transformers inference."""

    return UPSTREAM_SYSTEM_PROMPT + prompt + " "


def generate_native_backbone(config: NativeBackboneConfig) -> Path:
    """Generate native MIDI candidates while preserving each authoritative MIDI file."""

    _validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model, torch = _load_upstream_model(config.model, config.adapter)
    candidates: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for index in range(1, config.n_outputs + 1):
        seed = config.seed + index - 1
        candidate_dir = output_dir / f"candidate_{index}"
        candidate_dir.mkdir(exist_ok=True)
        try:
            model_token_ids = _sample_native_model_tokens(
                tokenizer,
                model,
                torch,
                config.prompt,
                seed=seed,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
            event_tokens, normalization = normalize_native_model_tokens(model_token_ids)
            if not event_tokens:
                raise ValueError("Upstream model produced no complete native MIDI events")
            _write_json(candidate_dir / "native.tokens.json", event_tokens)
            _write_json(candidate_dir / "native.model_tokens.json", model_token_ids)
            midi_path = _write_native_midi(event_tokens, candidate_dir / "native.mid")
            stats = native_midi_stats(parse_midi(midi_path))
            hit_max_token_budget = (
                len(model_token_ids) >= config.max_tokens
                and normalization["stop_model_token_id"] is None
            )
            row: Dict[str, Any] = {
                "candidate": index,
                "seed": seed,
                "native_midi": str(midi_path.relative_to(output_dir)),
                "native_tokens": str((candidate_dir / "native.tokens.json").relative_to(output_dir)),
                "normalization": normalization,
                "stats": {
                    **stats,
                    "generated_model_tokens": len(model_token_ids),
                    "hit_max_token_budget": hit_max_token_budget,
                    "natural_stop_detected": not hit_max_token_budget,
                },
                "score_draft": None,
                "score_draft_error": None,
            }
            if config.score_draft:
                score_draft_dir = candidate_dir / "score_draft"
                try:
                    import_draft(midi_path, score_draft_dir, render=config.render_score_draft)
                except (OSError, RuntimeError, ValueError) as error:
                    row["score_draft_error"] = str(error)
                else:
                    row["score_draft"] = str(score_draft_dir.relative_to(output_dir))
            candidates.append(row)
        except (AssertionError, OSError, RuntimeError, ValueError) as error:
            failures.append({"candidate": index, "seed": seed, "reason": str(error)})
    if not candidates:
        _write_json(output_dir / "failures.json", failures)
        raise RuntimeError(f"Upstream native generation produced no MIDI candidates; inspect {output_dir / 'failures.json'}")
    manifest = native_backbone_manifest(config, candidates, failures)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "prompt.txt").write_text(config.prompt + "\n", encoding="utf-8")
    _write_index(output_dir, manifest)
    return output_dir


def native_backbone_manifest(
    config: NativeBackboneConfig,
    candidates: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Describe the parity-gate artifact without overstating draft score quality."""

    return {
        "pipeline": "upstream-native-anticipation-midi-backbone-v1",
        "purpose": "upstream-live-demo-parity-gate" if config.adapter is None else "adapter-parity-replay-gate",
        "model": config.model,
        "adapter": config.adapter,
        "representation": "upstream-native-anticipation-midi-tokens",
        "native_midi_is_authoritative_musical_content": True,
        "score_conversion_is_draft_only": True,
        "prompt": config.prompt,
        "config": asdict(config),
        "candidates": candidates,
        "failures": failures,
    }


def native_midi_stats(parsed: ParsedMidi) -> Dict[str, Any]:
    """Summarize native MIDI content without treating a draft score as ground truth."""

    pitches = [note.pitch for note in parsed.notes]
    final_tick = max((note.tick + note.duration for note in parsed.notes), default=0)
    measures = max(1, math.ceil(final_tick / parsed.ticks_per_beat / parsed.meter.quarter_beats))
    duration_seconds = _duration_seconds(parsed, final_tick)
    onset_counts = Counter(note.tick for note in parsed.notes)
    max_active_notes = _max_active_notes(parsed)
    notes_per_second = len(parsed.notes) / max(1.0, duration_seconds)
    return {
        "notes": len(parsed.notes),
        "duration_seconds": round(duration_seconds, 3),
        "estimated_measures": measures,
        "pitch_range": [min(pitches), max(pitches)] if pitches else [],
        "distinct_pitches": len(set(pitches)),
        "tempo_events": len(parsed.tempos),
        "pedal_events": len(parsed.pedal),
        "notes_per_second": round(notes_per_second, 3),
        "max_notes_at_onset": max(onset_counts.values(), default=0),
        "max_active_notes": max_active_notes,
        "potential_density_drift": (
            notes_per_second > 24
            or max(onset_counts.values(), default=0) > 32
            or max_active_notes > 64
        ),
    }


def _duration_seconds(parsed: ParsedMidi, final_tick: int) -> float:
    tempos = [tempo for tempo in parsed.tempos if tempo.tick <= final_tick]
    if not tempos:
        return final_tick / parsed.ticks_per_beat * 0.5
    elapsed = 0.0
    for index, tempo in enumerate(tempos):
        next_tick = tempos[index + 1].tick if index + 1 < len(tempos) else final_tick
        elapsed += max(0, next_tick - tempo.tick) / parsed.ticks_per_beat * 60 / tempo.bpm
    return elapsed


def _max_active_notes(parsed: ParsedMidi) -> int:
    active = 0
    maximum = 0
    note_boundaries = [
        boundary
        for note in parsed.notes
        for boundary in ((note.tick, 1), (note.tick + note.duration, -1))
    ]
    for _, delta in sorted(note_boundaries, key=lambda boundary: (boundary[0], boundary[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _load_upstream_model(model_path: str, adapter_path: str | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install requirements.txt on the NVIDIA host before native MIDI generation") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Upstream native MIDI generation requires an NVIDIA CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_path, pad_token="<|eot_id|>")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device="cuda")
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("Install peft before loading a native MIDI adapter") from error
        model = PeftModel.from_pretrained(model, adapter_path).to(device="cuda")
    model.eval()
    return tokenizer, model, torch


def _sample_native_model_tokens(
    tokenizer,
    model,
    torch,
    prompt: str,
    *,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> List[int]:
    """Use the upstream prompt, MIDI BOS token, and native token distribution unchanged."""

    torch.manual_seed(seed)
    llama_input = tokenizer(upstream_generation_prompt(prompt), return_tensors="pt", padding=False)
    midi_bos = torch.tensor([[NATIVE_MIDI_BOS_MODEL_TOKEN_ID]])
    input_ids = torch.cat([llama_input["input_ids"], midi_bos], dim=1).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    return output[0][input_ids.shape[1] :].detach().cpu().tolist()


def _write_native_midi(event_tokens: List[int], path: Path) -> Path:
    try:
        from anticipation.convert import events_to_midi
    except ImportError as error:
        raise RuntimeError("Install anticipation before converting native MIDI tokens") from error
    midi = events_to_midi(event_tokens)
    midi.save(str(path))
    return path


def _write_index(output_dir: Path, manifest: Dict[str, Any]) -> None:
    rows = []
    for candidate in manifest["candidates"]:
        draft = candidate.get("score_draft")
        stats = candidate["stats"]
        draft_link = f'<a href="{escape(draft)}/gallery.html">draft score gallery</a>' if draft else "draft unavailable"
        rows.append(
            "<li>"
            f'<a href="{escape(candidate["native_midi"])}">candidate {candidate["candidate"]} native.mid</a>'
            f" | {draft_link}"
            f" | {stats['duration_seconds']:.1f}s"
            f" | {stats['notes']} notes"
            f" | hit token budget: {str(stats['hit_max_token_budget']).lower()}"
            "</li>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Upstream Native MIDI Backbone</title></head>
<body>
<h1>Upstream Native MIDI Backbone</h1>
<p>This parity-gate run uses the upstream MIDI-LLM checkpoint with its native Anticipation MIDI tokens.</p>
<p>Adapter: {escape(str(manifest["adapter"] or "none"))}</p>
<p>The native MIDI files are authoritative musical content. Converted score galleries are draft-only previews.</p>
<ul>{''.join(rows)}</ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _validate_config(config: NativeBackboneConfig) -> None:
    if not config.prompt.strip():
        raise ValueError("prompt must not be empty")
    if config.n_outputs < 1:
        raise ValueError("n_outputs must be positive")
    if config.max_tokens < 3:
        raise ValueError("max_tokens must allow at least one complete native MIDI event")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < config.top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")


def _default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return str(Path("generated_native_backbone") / timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate upstream-native MIDI and optional draft score previews")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", help="Optional LoRA adapter for fixed-seed parity replay")
    parser.add_argument("--n-outputs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-tokens", type=int, default=2046)
    parser.add_argument("--skip-score-draft", action="store_true")
    parser.add_argument("--render-score-draft", action="store_true")
    args = parser.parse_args()
    output = generate_native_backbone(
        NativeBackboneConfig(
            prompt=args.prompt,
            output_dir=args.output_dir or _default_output_dir(),
            model=args.model,
            adapter=args.adapter,
            n_outputs=args.n_outputs,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            score_draft=not args.skip_score_draft,
            render_score_draft=args.render_score_draft,
        )
    )
    print(f"Native backbone candidates written to {output.resolve()}")
    print(f"Index: {(output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
