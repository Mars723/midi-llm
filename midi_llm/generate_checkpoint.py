"""Generate and render complete piano scores from a trained ScoreDSL adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .compiler import render_musescore, write_musicxml, write_performance_midi, write_score_midi
from .evaluate import evaluate_score
from .gallery import write_gallery
from .planner import controls_from_mapping, create_piece_plan, read_controls
from .rules import create_motif_bank, render_performance
from .scoredsl import decode_score, encode_score
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


def generate_from_checkpoint(args: argparse.Namespace) -> Path:
    """Sample valid complete-piece ScoreDSL candidates and render the best score."""

    plan, motif_bank, controls = _requested_plan(args)
    prompt = model_prompt(_whole_piece_model_input(plan, motif_bank))
    tokenizer, model, torch = _load_checkpoint(args.base_model, args.adapter_dir)
    output_dir = Path(args.output_dir or _default_output_dir())
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates: List[Tuple[float, int, PianoScoreIR, Any, Dict[str, Any]]] = []
    failures = []
    for index in range(args.candidates):
        seed = args.seed + index
        print(f"Sampling checkpoint candidate {index + 1}/{args.candidates} with seed={seed}", flush=True)
        torch.manual_seed(seed)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        raw_path = candidate_dir / f"candidate_{index + 1}.raw.dsl"
        streamer = _CandidateFileStreamer(tokenizer, raw_path)
        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            stop_strings=["END_SCORE"],
            tokenizer=tokenizer,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_tokens = output[0][encoded["input_ids"].shape[1] :]
        continuation = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        print(f"Candidate {index + 1} sampled {generated_tokens.shape[0]} tokens", flush=True)
        raw_path.write_text(continuation, encoding="utf-8")
        try:
            score = _score_from_continuation(continuation, plan, motif_bank, args.adapter_dir)
            performance = render_performance(score, seed=seed + 10_000)
            metrics = evaluate_score(score, performance)
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
    manifest: Dict[str, Any] = {
        "pipeline": "score-first-checkpoint-whole-piece-v1",
        "score_source": "trained-scoredsl-adapter",
        "base_model": args.base_model,
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "selected_candidate": best_index + 1,
        "valid_candidate_count": len(candidates),
        "attempted_candidate_count": args.candidates,
        "candidate_failures": failures,
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
        "bidirectional": False,
    }


def _score_from_continuation(
    continuation: str,
    plan: PiecePlanIR,
    motif_bank: MotifBank,
    adapter_dir: str,
) -> PianoScoreIR:
    end = continuation.find("END_SCORE")
    if end < 0:
        raise ValueError("Generated ScoreDSL does not contain END_SCORE")
    score = decode_score(continuation[: end + len("END_SCORE")])
    score = replace(
        score,
        plan=plan,
        motif_bank=motif_bank,
        metadata={**score.metadata, "score_source": "trained-scoredsl-adapter", "adapter_dir": adapter_dir},
    )
    errors = validate_score(score)
    if errors:
        raise ValueError("; ".join(errors))
    return score


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
    output_dir = generate_from_checkpoint(args)
    print(f"Checkpoint piece written to {output_dir.resolve()}")
    print(f"Gallery: {(output_dir / 'gallery.html').resolve()}")


if __name__ == "__main__":
    main()
