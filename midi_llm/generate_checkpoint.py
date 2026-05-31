"""Generate and render complete piano scores from a trained ScoreDSL adapter."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .compiler import render_musescore, write_musicxml, write_performance_midi, write_score_midi
from .evaluate import _measure_repetition_metrics, evaluate_score
from .gallery import write_gallery
from .model_scoredsl import decode_model_score, encode_model_score, model_score_generation_prefix
from .notation_analysis import intermediate_notation_constraints
from .planner import controls_from_mapping, create_piece_plan, read_controls
from .rules import create_motif_bank, render_performance
from .scoredsl import encode_score
from .score_ir import MotifBank, PianoScoreIR, PiecePlanIR, validate_score, write_json
from .train_scoredsl import model_prompt


class _CandidateFileStreamer:
    """Persist incremental candidate text so long checkpoint samples stay observable."""

    def __init__(self, tokenizer, path: Path, report_every: int = 1024):
        self.tokenizer = tokenizer
        self.path = path
        self.report_every = report_every
        self.token_ids: List[int] = []
        self.prompt_received = False
        self.next_report = report_every

    def put(self, value) -> None:
        token_ids = value.detach().cpu().reshape(-1).tolist()
        if not self.prompt_received:
            self.prompt_received = True
            return
        self.token_ids.extend(token_ids)
        if len(self.token_ids) >= self.next_report:
            self._flush()
            print(f"Streamed {len(self.token_ids)} candidate tokens to {self.path}", flush=True)
            while self.next_report <= len(self.token_ids):
                self.next_report += self.report_every

    def end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(
            self.tokenizer.decode(self.token_ids, skip_special_tokens=False),
            encoding="utf-8",
        )


class _PlanBoundaryStoppingCriteria:
    """Stop after the model emits the first complete event beyond the plan."""

    def __init__(self, tokenizer, prompt_tokens: int, final_measure: int, tail_tokens: int = 512):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.final_measure = final_measure
        self.tail_tokens = tail_tokens

    def __call__(self, input_ids, _scores, **_kwargs) -> bool:
        start = max(self.prompt_tokens, input_ids.shape[1] - self.tail_tokens)
        tail = self.tokenizer.decode(input_ids[0][start:], skip_special_tokens=False)
        return any(
            measure is not None and measure > self.final_measure
            for measure in (_model_event_measure(line) for line in tail.splitlines())
        )


class _MeasureEventBudgetStoppingCriteria:
    """Stop a candidate that gets stuck emitting rows for one measure."""

    def __init__(self, tokenizer, prompt_tokens: int, tail_tokens: int = 8192, max_event_rows: int = 64):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.tail_tokens = tail_tokens
        self.max_event_rows = max_event_rows

    def __call__(self, input_ids, _scores, **_kwargs) -> bool:
        start = max(self.prompt_tokens, input_ids.shape[1] - self.tail_tokens)
        tail = self.tokenizer.decode(input_ids[0][start:], skip_special_tokens=False)
        return _measure_event_budget_exceeded(tail, self.max_event_rows)


def generate_from_checkpoint(args: argparse.Namespace) -> Path:
    """Sample valid complete-piece ScoreDSL candidates and render the best score."""

    plan, motif_bank, controls = _requested_plan(args)
    tokenizer, model, torch = _load_checkpoint(args.base_model, args.adapter_dir)
    output_dir = Path(args.output_dir or _default_output_dir())
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates: List[Tuple[float, int, PianoScoreIR, Any, Dict[str, Any]]] = []
    failures = []
    for index in range(args.candidates):
        seed = args.seed + index * (args.section_candidates if args.strategy == "hierarchical" else 1)
        print(f"Sampling checkpoint candidate {index + 1}/{args.candidates} with seed={seed}", flush=True)
        try:
            if args.strategy == "hierarchical":
                score = _sample_hierarchical_score(
                    plan,
                    motif_bank,
                    args.adapter_dir,
                    tokenizer,
                    model,
                    torch,
                    args,
                    candidate_dir,
                    index + 1,
                    seed,
                )
            else:
                continuation = _sample_model_continuation(
                    _whole_piece_model_input(plan, motif_bank),
                    1,
                    plan.measure_count,
                    tokenizer,
                    model,
                    torch,
                    args,
                    candidate_dir / f"candidate_{index + 1}.raw.dsl",
                    seed,
                    f"Candidate {index + 1}",
                )
                score = _score_from_continuation(continuation, plan, motif_bank, args.adapter_dir)
            performance = render_performance(score, seed=seed + 10_000)
            metrics = evaluate_score(score, performance)
            if not metrics["valid"]:
                raise ValueError(f"Generated candidate failed quality gate: {json.dumps(metrics, sort_keys=True)}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            failures.append({"candidate": index + 1, "reason": str(error)})
            continue
        write_json(candidate_dir / f"candidate_{index + 1}.score.ir.json", score)
        write_json(candidate_dir / f"candidate_{index + 1}.performance.ir.json", performance)
        (candidate_dir / f"candidate_{index + 1}.metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        candidates.append((metrics["structural_score"], index, score, performance, metrics))
    if not candidates:
        (candidate_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Checkpoint produced no valid ScoreDSL candidates; inspect {candidate_dir / 'failures.json'}")

    _, best_index, score, performance, metrics = max(candidates, key=lambda item: (item[0], -item[1]))
    write_json(output_dir / "score.ir.json", score)
    write_json(output_dir / "performance.ir.json", performance)
    (output_dir / "score.dsl").write_text(encode_score(score), encoding="utf-8")
    write_musicxml(score, output_dir / "score.musicxml")
    write_score_midi(score, output_dir / "score.mid")
    write_performance_midi(score, performance, output_dir / "performance.mid")
    render = {"available": False, "reason": "MuseScore rendering skipped", "pages": []}
    if not args.skip_musescore:
        render = render_musescore(output_dir / "score.musicxml", output_dir, args.musescore_bin)
    metrics = evaluate_score(score, performance, output_dir / "score.musicxml")
    if not metrics["valid"]:
        raise RuntimeError(f"Rendered checkpoint score failed validation: {json.dumps(metrics, sort_keys=True)}")
    manifest: Dict[str, Any] = {
        "pipeline": "score-first-checkpoint-whole-piece-v1",
        "score_source": "trained-scoredsl-adapter",
        "base_model": args.base_model,
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "selected_candidate": best_index + 1,
        "valid_candidate_count": len(candidates),
        "attempted_candidate_count": args.candidates,
        "generation_strategy": args.strategy,
        "candidate_failures": failures,
        "decode_completion": score.metadata.get("decode_completion"),
        "skipped_malformed_optional_rows": score.metadata.get("skipped_malformed_optional_rows", 0),
        "trimmed_empty_trailing_measures": score.metadata.get("trimmed_empty_trailing_measures", 0),
        "prompt": args.prompt,
        "controls": asdict(controls),
        "metrics": metrics,
        "render": render,
        "artifacts": {
            "score_ir": "score.ir.json",
            "performance_ir": "performance.ir.json",
            "musicxml": "score.musicxml",
            "score_dsl": "score.dsl",
            "score_midi": "score.mid",
            "performance_midi": "performance.mid",
            "gallery": "gallery.html",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "prompt.txt").write_text(args.prompt + "\n", encoding="utf-8")
    write_gallery(score, manifest, output_dir)
    return output_dir


def _requested_plan(args: argparse.Namespace):
    mapping = read_controls(args.controls)
    for key in (
        "genre",
        "form",
        "duration_minutes",
        "measure_range",
        "key",
        "meter",
        "tempo",
        "difficulty",
        "markings",
        "title",
    ):
        value = getattr(args, key, None)
        if value is not None:
            mapping[key] = value
    controls = controls_from_mapping(mapping)
    plan = create_piece_plan(args.prompt, controls)
    return plan, create_motif_bank(plan.key, plan.genre), controls


def _whole_piece_model_input(plan: PiecePlanIR, motif_bank: MotifBank) -> Dict[str, Any]:
    return {
        "instruction": "Generate the complete piano score from the shared whole-piece plan.",
        "piece_plan": asdict(plan),
        "motif_bank": asdict(motif_bank),
        "sparse_skeleton": [
            {
                "label": section.label,
                "role": section.role,
                "range": [section.start_measure, section.end_measure],
                "key": section.key,
                "motif_refs": section.motif_refs,
                "cadence": section.cadence,
            }
            for section in plan.sections
        ],
        "target_range": [1, plan.measure_count],
        "left_neighbor_scoredsl": None,
        "right_neighbor_scoredsl": None,
        "future_ending_target": {
            "section": plan.sections[-1].label,
            "measure": plan.measure_count,
            "cadence": plan.sections[-1].cadence,
        },
        "notation_constraints": intermediate_notation_constraints(plan),
        "bidirectional": False,
    }


def _sample_model_continuation(
    model_input: Dict[str, Any],
    start_measure: int,
    final_measure: int,
    tokenizer,
    model,
    torch,
    args: argparse.Namespace,
    raw_path: Path,
    seed: int,
    label: str,
) -> str:
    generation_prefix = model_score_generation_prefix(start_measure)
    prompt = model_prompt(model_input) + generation_prefix
    torch.manual_seed(seed)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    streamer = _CandidateFileStreamer(tokenizer, raw_path)
    output = model.generate(
        **encoded,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        stop_strings=["END_SCORE"],
        stopping_criteria=[
            _PlanBoundaryStoppingCriteria(tokenizer, encoded["input_ids"].shape[1], final_measure),
            _MeasureEventBudgetStoppingCriteria(tokenizer, encoded["input_ids"].shape[1]),
        ],
        tokenizer=tokenizer,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = output[0][encoded["input_ids"].shape[1] :]
    continuation = generation_prefix + tokenizer.decode(generated_tokens, skip_special_tokens=False)
    print(f"{label} sampled {generated_tokens.shape[0]} tokens", flush=True)
    raw_path.write_text(continuation, encoding="utf-8")
    return continuation


def _sample_hierarchical_score(
    plan: PiecePlanIR,
    motif_bank: MotifBank,
    adapter_dir: str,
    tokenizer,
    model,
    torch,
    args: argparse.Namespace,
    candidate_dir: Path,
    candidate_number: int,
    seed: int,
) -> PianoScoreIR:
    notes = []
    directions = []
    layout_hints = []
    section_completions = []
    section_selections = []
    skipped_optional_rows = 0
    for section_index, section in enumerate(plan.sections, start=1):
        partial = PianoScoreIR(
            plan=plan,
            motif_bank=motif_bank,
            notes=notes,
            directions=directions,
            layout_hints=layout_hints,
        )
        fragments = []
        fragment_failures = []
        for attempt in range(1, args.section_candidates + 1):
            attempt_seed = seed + (section_index - 1) * 100 + attempt - 1
            try:
                continuation = _sample_model_continuation(
                    _section_model_input(plan, motif_bank, section, partial),
                    section.start_measure,
                    section.end_measure,
                    tokenizer,
                    model,
                    torch,
                    args,
                    candidate_dir
                    / f"candidate_{candidate_number}.section_{section_index}.attempt_{attempt}.raw.dsl",
                    attempt_seed,
                    f"Candidate {candidate_number} section {section.label} attempt {attempt}",
                )
                fragment = _score_fragment_from_continuation(
                    continuation,
                    plan,
                    motif_bank,
                    adapter_dir,
                    [section.start_measure, section.end_measure],
                    allow_terminal_empty=section_index == len(plan.sections),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                fragment_failures.append({"attempt": attempt, "seed": attempt_seed, "reason": str(error)})
                continue
            fragments.append((_fragment_quality_score(fragment), attempt, attempt_seed, fragment))
        if not fragments:
            raise ValueError(
                f"Candidate {candidate_number} section {section.label} produced no valid alternatives: "
                f"{json.dumps(fragment_failures, sort_keys=True)}"
            )
        quality_score, selected_attempt, selected_seed, fragment = max(
            fragments,
            key=lambda item: (item[0], -item[1]),
        )
        print(
            f"Candidate {candidate_number} section {section.label} selected attempt {selected_attempt} "
            f"seed={selected_seed} quality={quality_score:.3f}",
            flush=True,
        )
        for note in fragment.notes:
            note.id = f"model-note-{len(notes) + 1}"
            notes.append(note)
        directions.extend(fragment.directions)
        layout_hints.extend(fragment.layout_hints)
        section_completions.append(fragment.metadata["decode_completion"])
        section_selections.append(
            {
                "label": section.label,
                "attempt": selected_attempt,
                "seed": selected_seed,
                "quality_score": quality_score,
                "valid_alternatives": len(fragments),
                "failed_alternatives": fragment_failures,
            }
        )
        skipped_optional_rows += fragment.metadata["skipped_malformed_optional_rows"]
    score = PianoScoreIR(
        plan=deepcopy(plan),
        motif_bank=motif_bank,
        notes=notes,
        directions=directions,
        layout_hints=layout_hints,
        metadata={
            "score_source": "trained-scoredsl-adapter",
            "adapter_dir": adapter_dir,
            "decode_completion": "hierarchical-sections",
            "section_decode_completions": section_completions,
            "section_selections": section_selections,
            "skipped_malformed_optional_rows": skipped_optional_rows,
            "generation_strategy": "hierarchical",
        },
    )
    score.metadata["trimmed_empty_trailing_measures"] = _trim_single_empty_trailing_measure(score)
    errors = validate_score(score)
    errors.extend(_validate_generated_whole_piece(score))
    if errors:
        raise ValueError("; ".join(errors))
    return score


def _fragment_quality_score(score: PianoScoreIR) -> float:
    repetition = _measure_repetition_metrics(score)
    measure_count = max(1, len({note.measure for note in score.notes}))
    notes_per_measure = len(score.notes) / measure_count
    return round(
        repetition["unique_measure_signature_ratio"] * 100
        - max(0, repetition["longest_identical_measure_run"] - 1) * 2
        - abs(notes_per_measure - 14),
        3,
    )


def _section_model_input(plan, motif_bank, section, partial_score: PianoScoreIR) -> Dict[str, Any]:
    model_input = _whole_piece_model_input(plan, motif_bank)
    model_input.update(
        {
            "instruction": "Expand the target section while respecting the full plan, motifs, neighbors, and ending target.",
            "target_range": [section.start_measure, section.end_measure],
            "left_neighbor_scoredsl": _left_neighbor_scoredsl(partial_score, section.start_measure),
            "right_neighbor_scoredsl": None,
            "active_section": {
                "label": section.label,
                "role": section.role,
                "range": [section.start_measure, section.end_measure],
                "motif_refs": section.motif_refs,
                "cadence": section.cadence,
            },
        }
    )
    return model_input


def _left_neighbor_scoredsl(score: PianoScoreIR, start_measure: int) -> str | None:
    if start_measure <= 1:
        return None
    left_start = max(1, start_measure - 8)
    left_end = start_measure - 1
    return encode_model_score(
        PianoScoreIR(
            plan=score.plan,
            motif_bank=score.motif_bank,
            notes=[note for note in score.notes if left_start <= note.measure <= left_end],
            directions=[direction for direction in score.directions if left_start <= direction.measure <= left_end],
            layout_hints=[hint for hint in score.layout_hints if left_start <= hint.measure <= left_end],
            metadata={"fragment": {"range": [left_start, left_end]}},
        )
    )


def _score_fragment_from_continuation(
    continuation: str,
    plan: PiecePlanIR,
    motif_bank: MotifBank,
    adapter_dir: str,
    target_range: List[int],
    *,
    allow_terminal_empty: bool,
) -> PianoScoreIR:
    start_measure, end_measure = target_range
    finalized, decode_completion, skipped_optional_rows = _finalize_model_continuation(continuation, end_measure)
    score = decode_model_score(
        finalized,
        deepcopy(plan),
        motif_bank,
        metadata={
            "score_source": "trained-scoredsl-adapter-fragment",
            "adapter_dir": adapter_dir,
            "decode_completion": decode_completion,
            "skipped_malformed_optional_rows": skipped_optional_rows,
            "target_range": target_range,
        },
    )
    realized_measures = {note.measure for note in score.notes}
    errors = []
    if start_measure not in realized_measures:
        errors.append(f"Generated fragment does not realize opening measure {start_measure}")
    if end_measure not in realized_measures and not (allow_terminal_empty and end_measure - 1 in realized_measures):
        errors.append(f"Generated fragment does not realize final measure {end_measure}")
    if any(measure < start_measure or measure > end_measure for measure in realized_measures):
        errors.append(f"Generated fragment escapes target range {start_measure}-{end_measure}")
    measure_note_counts = Counter(note.measure for note in score.notes)
    if measure_note_counts and max(measure_note_counts.values()) > 64:
        errors.append("Generated fragment exceeds the per-measure notation event budget")
    repetition = _measure_repetition_metrics(score)
    realized_measure_count = len(realized_measures)
    if realized_measure_count >= 16 and repetition["unique_measure_signature_ratio"] < 0.25:
        errors.append("Generated fragment does not contain enough unique measure content")
    if repetition["longest_identical_measure_run"] > 8:
        errors.append("Generated fragment repeats identical measure content too many times in sequence")
    if (
        repetition["periodic_measure_loop_period"] is not None
        and repetition["periodic_measure_loop_span"] >= 16
        and repetition["periodic_measure_loop_ratio"] > 0.75
    ):
        errors.append("Generated fragment repeats a short measure pattern across most of the section")
    if errors:
        raise ValueError("; ".join(errors))
    return score


def _score_from_continuation(
    continuation: str,
    plan: PiecePlanIR,
    motif_bank: MotifBank,
    adapter_dir: str,
) -> PianoScoreIR:
    finalized, decode_completion, skipped_optional_rows = _finalize_model_continuation(
        continuation,
        plan.measure_count,
    )
    score = decode_model_score(
        finalized,
        deepcopy(plan),
        motif_bank,
        metadata={
            "score_source": "trained-scoredsl-adapter",
            "adapter_dir": adapter_dir,
            "decode_completion": decode_completion,
            "skipped_malformed_optional_rows": skipped_optional_rows,
        },
    )
    score.metadata["trimmed_empty_trailing_measures"] = _trim_single_empty_trailing_measure(score)
    errors = validate_score(score)
    errors.extend(_validate_generated_whole_piece(score))
    if errors:
        raise ValueError("; ".join(errors))
    return score


def _trim_single_empty_trailing_measure(score: PianoScoreIR) -> int:
    """Normalize a single empty terminal bar without accepting a truncated piece."""

    if not score.notes:
        return 0
    realized_end = max(note.measure for note in score.notes)
    if score.plan.measure_count - realized_end != 1:
        return 0
    score.plan.measure_count = realized_end
    score.plan.sections[-1].end_measure = realized_end
    score.plan.tempo_marks = [mark for mark in score.plan.tempo_marks if mark.measure <= realized_end]
    score.directions = [direction for direction in score.directions if direction.measure <= realized_end]
    score.layout_hints = [hint for hint in score.layout_hints if hint.measure <= realized_end]
    return 1


def _finalize_model_continuation(continuation: str, final_measure: int) -> Tuple[str, str, int]:
    """Use a model terminator or trim the first complete event beyond the plan."""

    lines = []
    completion = None
    skipped_optional_rows = 0
    for line in continuation.splitlines():
        if line == "END_SCORE":
            lines.append(line)
            completion = "model-end-score"
            break
        if _is_malformed_optional_event(line):
            skipped_optional_rows += 1
            continue
        measure = _model_event_measure(line)
        if measure is not None and measure > final_measure:
            completion = "planned-measure-boundary"
            break
        lines.append(line)
    if completion is None:
        raise ValueError("Generated ScoreDSL does not contain END_SCORE or cross the planned final measure")
    if not lines or lines[-1] != "END_SCORE":
        lines.append("END_SCORE")
    return "\n".join(lines) + "\n", completion, skipped_optional_rows


def _is_malformed_optional_event(line: str) -> bool:
    tag, separator, payload = line.partition(" ")
    if not separator or tag not in ("DIRECTION", "LAYOUT"):
        return False
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return True
    expected_columns = 7 if tag == "DIRECTION" else 2
    return not isinstance(data, list) or len(data) != expected_columns


def _model_event_measure(line: str) -> int | None:
    tag, separator, payload = line.partition(" ")
    if not separator or tag not in ("MEASURE", "NOTE", "DIRECTION", "LAYOUT"):
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data or not isinstance(data[0], int):
        return None
    return data[0]


def _measure_event_budget_exceeded(text: str, max_event_rows: int = 64) -> bool:
    counts = Counter()
    for line in text.splitlines():
        tag, separator, _payload = line.partition(" ")
        if not separator or tag not in ("NOTE", "DIRECTION", "LAYOUT"):
            continue
        measure = _model_event_measure(line)
        if measure is None:
            continue
        counts[measure] += 1
        if counts[measure] > max_event_rows:
            return True
    return False


def _validate_generated_whole_piece(score: PianoScoreIR) -> List[str]:
    """Reject truncated or locally collapsed samples before rendering."""

    realized_measures = {note.measure for note in score.notes}
    errors = []
    if 1 not in realized_measures:
        errors.append("Generated score does not realize the opening measure")
    if score.plan.measure_count not in realized_measures:
        errors.append(f"Generated score does not realize planned final measure {score.plan.measure_count}")
    missing_sections = [
        section.label
        for section in score.plan.sections
        if not any(section.start_measure <= measure <= section.end_measure for measure in realized_measures)
    ]
    if missing_sections:
        errors.append(f"Generated score does not realize planned sections: {', '.join(missing_sections)}")
    note_rows = Counter(
        (
            note.measure,
            note.beat,
            note.duration,
            note.pitch,
            note.staff,
            note.voice,
            note.articulation,
            note.fingering,
            note.tie_start,
            note.tie_stop,
        )
        for note in score.notes
    )
    if note_rows and max(note_rows.values()) > 8:
        errors.append("Generated score repeats an identical notation row more than 8 times")
    notes_by_measure = defaultdict(list)
    for note in score.notes:
        notes_by_measure[note.measure].append(
            (
                note.beat,
                note.duration,
                note.pitch,
                note.staff,
                note.voice,
                note.articulation,
                note.fingering,
                note.tie_start,
                note.tie_stop,
            )
        )
    if any(len(notes) > 64 for notes in notes_by_measure.values()):
        errors.append("Generated score exceeds the per-measure notation event budget")
    max_measure_run = 0
    current_measure_run = 0
    prior_signature = None
    for measure in sorted(notes_by_measure):
        signature = tuple(notes_by_measure[measure])
        current_measure_run = current_measure_run + 1 if signature == prior_signature else 1
        prior_signature = signature
        max_measure_run = max(max_measure_run, current_measure_run)
    if len(notes_by_measure) >= 16 and (
        max_measure_run > 24 or max_measure_run / len(notes_by_measure) > 0.65
    ):
        errors.append("Generated score repeats identical measure content across most of the piece")
    repetition = _measure_repetition_metrics(score)
    if (
        repetition["periodic_measure_loop_period"] is not None
        and repetition["periodic_measure_loop_period"] <= 8
        and repetition["periodic_measure_loop_span"] >= 16
        and repetition["periodic_measure_loop_ratio"] > 0.65
    ):
        errors.append("Generated score repeats a short measure pattern across most of the piece")
    return errors


def _load_checkpoint(base_model: str, adapter_dir: str):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install requirements-score-first-train.txt before checkpoint inference") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint inference currently requires an NVIDIA CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model, torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a complete score from a trained ScoreDSL adapter")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="slseanwu/MIDI-LLM_Llama-3.2-1B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--controls")
    parser.add_argument("--output-dir")
    parser.add_argument("--genre")
    parser.add_argument("--form")
    parser.add_argument("--duration-minutes", type=float)
    parser.add_argument("--measure-range")
    parser.add_argument("--key")
    parser.add_argument("--meter")
    parser.add_argument("--tempo", type=int)
    parser.add_argument("--difficulty")
    parser.add_argument("--markings")
    parser.add_argument("--title")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.01)
    parser.add_argument("--strategy", choices=("hierarchical", "single-pass"), default="hierarchical")
    parser.add_argument("--section-candidates", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=65536)
    parser.add_argument("--skip-musescore", action="store_true")
    parser.add_argument("--musescore-bin")
    return parser


def _default_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return str(Path("generated_score_first") / f"checkpoint_{timestamp}")


def main() -> None:
    args = build_parser().parse_args()
    if args.candidates < 1:
        raise SystemExit("--candidates must be at least 1")
    if args.section_candidates < 1:
        raise SystemExit("--section-candidates must be at least 1")
    output_dir = generate_from_checkpoint(args)
    print(f"Checkpoint piece written to {output_dir.resolve()}")
    print(f"Gallery: {(output_dir / 'gallery.html').resolve()}")


if __name__ == "__main__":
    main()
