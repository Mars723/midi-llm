"""Dry-run locally or launch upstream-native MIDI QLoRA specialization."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .native_tokens import NATIVE_MIDI_BOS_MODEL_TOKEN_ID
from .train_scoredsl import (
    DEFAULT_LORA_TARGETS,
    REQUIRED_RUNTIME_PACKAGES,
    _checkpointed_chunked_causal_lm_loss,
    _collator,
    _configure_memory_efficient_sdpa,
    dependency_report,
)


@dataclass
class NativeTrainConfig:
    dataset_dir: str
    output_dir: str
    base_model: str = "slseanwu/MIDI-LLM_Llama-3.2-1B"
    resume_adapter_dir: str | None = None
    resume_checkpoint_dir: str | None = None
    split: str = "train"
    composers: Sequence[str] = ()
    max_segments: int | None = None
    max_segments_per_work: int = 4
    max_seq_length: int = 8192
    loss_chunk_tokens: int = 256
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    logging_steps: int = 5
    save_steps: int = 25
    bf16: bool = True


def build_native_training_spec(config: NativeTrainConfig) -> Dict[str, Any]:
    """Write a low-risk launch spec without importing heavyweight GPU packages."""

    _validate_config(config)
    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = _selected_segments(
        dataset_dir,
        config.split,
        config.composers,
        max_segments=config.max_segments,
        max_segments_per_work=config.max_segments_per_work,
    )
    model_token_lengths = [segment["native_model_token_count"] for segment in segments]
    estimated_lengths = [
        segment["native_model_token_count"] + math.ceil(len(segment["upstream_prompt"]) / 3) + 1
        for segment in segments
    ]
    spec = {
        "pipeline": "classical-piano-upstream-native-token-qlora-v1",
        "mode": "cloud-gpu-launch-spec",
        "config": asdict(config),
        "dataset": {
            "path": str(dataset_dir.resolve()),
            "split": config.split,
            "selected_native_segments": len(segments),
            "selected_works": len({segment["work_id"] for segment in segments}),
            "composer_style_counts": _count(segments, "composer_style"),
            "genre_counts": _count(segments, "genre"),
            "maximum_native_model_tokens": max(model_token_lengths, default=0),
            "estimated_max_sequence_tokens": max(estimated_lengths, default=0),
            "max_segments_per_work": config.max_segments_per_work,
            "selection_policy": "deterministic evenly-spaced native time windows per work",
            "complete_piece_sources_remain_in_materialized_dataset": True,
        },
        "model": {
            "base_model": config.base_model,
            "adapter": "QLoRA",
            "resume_adapter_dir": config.resume_adapter_dir,
            "resume_checkpoint_dir": config.resume_checkpoint_dir,
            "quantization": "4-bit NF4 with double quantization",
            "lora_targets": list(DEFAULT_LORA_TARGETS),
            "representation": "upstream-native-anticipation-midi-tokens",
            "upstream_extended_vocabulary_is_preserved": True,
            "score_dsl_is_not_a_composition_target": True,
            "complete_piece_truncation_policy": "forbidden",
            "loss_strategy": "checkpointed chunked LM-head cross-entropy over native MIDI event tokens",
        },
        "promotion_gate": {
            "required_before_training": "unmodified upstream parity samples",
            "required_after_each_checkpoint": "adapter parity replay on fixed prompts and seeds",
            "reject_adapter_when": [
                "native MIDI syntax failure increases",
                "density drift increases",
                "baseline prompt quality regresses materially",
                "classical prompt candidates do not improve in human review",
            ],
        },
        "runtime": {
            "required_packages": list(REQUIRED_RUNTIME_PACKAGES),
            "dependency_report": dependency_report(),
            "required_hardware": "single NVIDIA GPU with 48-80GB VRAM",
        },
        "launch_command": _launch_command(config),
    }
    (output_dir / "training_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def run_native_training(config: NativeTrainConfig) -> Dict[str, Any]:
    """Launch QLoRA while supervising only native MIDI event tokens."""

    spec = build_native_training_spec(config)
    rows = _selected_segments(
        Path(config.dataset_dir),
        config.split,
        config.composers,
        max_segments=config.max_segments,
        max_segments_per_work=config.max_segments_per_work,
    )
    if not rows:
        raise RuntimeError(f"No native MIDI segments selected from split {config.split!r}")
    missing = [package for package, available in dependency_report().items() if not available]
    if missing:
        raise RuntimeError(
            "Missing cloud training packages: "
            + ", ".join(missing)
            + ". Install requirements-score-first-train.txt on the NVIDIA host."
        )

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("Native QLoRA launch requires an NVIDIA CUDA GPU; use --dry-run locally")
    _configure_memory_efficient_sdpa(torch)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, pad_token="<|eot_id|>", use_fast=True)
    tokenizer_preflight = _validate_token_lengths(rows, tokenizer, config.max_seq_length)
    (Path(config.output_dir) / "tokenizer_preflight.json").write_text(
        json.dumps(tokenizer_preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    if tokenizer_preflight["oversized_segment_count"]:
        raise ValueError(
            f"{tokenizer_preflight['oversized_segment_count']} native MIDI segments exceed "
            f"max_seq_length={config.max_seq_length}; largest segment needs "
            f"{tokenizer_preflight['maximum_tokens']} tokens. Raise the context length or "
            "materialize smaller native windows; segments are never silently truncated."
        )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map="auto",
        quantization_config=quantization,
        attn_implementation="sdpa",
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if config.resume_adapter_dir:
        model = PeftModel.from_pretrained(model, config.resume_adapter_dir, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(DEFAULT_LORA_TARGETS),
            ),
        )
    dataset = _tokenized_native_dataset(rows, tokenizer, config.max_seq_length)

    class NativeChunkedCausalLMTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                return super().training_step(model, inputs, num_items_in_batch)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = _checkpointed_chunked_causal_lm_loss(model, inputs, config.loss_chunk_tokens)
            return (loss, {"loss": loss}) if return_outputs else loss

    trainer = NativeChunkedCausalLMTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_strategy="steps",
            gradient_checkpointing=True,
            bf16=config.bf16,
            fp16=not config.bf16,
            optim="paged_adamw_8bit",
            report_to=[],
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        data_collator=_collator(tokenizer, torch),
    )
    trainer.train(resume_from_checkpoint=config.resume_checkpoint_dir)
    adapter_dir = Path(config.output_dir) / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    result = {**spec, "mode": "trained-adapter", "adapter_dir": str(adapter_dir.resolve())}
    (Path(config.output_dir) / "training_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _selected_segments(
    dataset_dir: Path,
    split: str,
    composers: Sequence[str],
    *,
    max_segments: int | None,
    max_segments_per_work: int,
) -> List[Dict[str, Any]]:
    rows = _read_jsonl(dataset_dir / f"{split}.jsonl")
    selected_composers = set(composers)
    segments = []
    for row in rows:
        controls = row["controls"]
        if selected_composers and controls["composer_style"] not in selected_composers:
            continue
        for segment in _spread_segments(row["native_segments"], max_segments_per_work):
            segments.append(
                {
                    "work_id": row["work_id"],
                    "composer_style": controls["composer_style"],
                    "genre": controls["genre"],
                    "upstream_prompt": row["upstream_prompt"],
                    **segment,
                    "native_model_tokens_path": str(dataset_dir / segment["native_model_tokens_path"]),
                }
            )
    return segments[:max_segments] if max_segments is not None else segments


def _spread_segments(segments: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(segments) <= limit:
        return list(segments)
    if limit == 1:
        return [segments[0]]
    indexes = [round(index * (len(segments) - 1) / (limit - 1)) for index in range(limit)]
    return [segments[index] for index in indexes]


def _tokenized_native_dataset(rows, tokenizer, max_seq_length):
    class NativeMidiDataset:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            prompt_ids = tokenizer(row["upstream_prompt"], add_special_tokens=False)["input_ids"]
            native_ids = _read_tokens(row["native_model_tokens_path"])
            if not native_ids or native_ids[0] != NATIVE_MIDI_BOS_MODEL_TOKEN_ID:
                raise ValueError(f"Native segment for {row['work_id']} does not start with MIDI BOS")
            input_ids = prompt_ids + native_ids + [tokenizer.eos_token_id]
            if len(input_ids) > max_seq_length:
                raise ValueError(
                    f"Native segment for {row['work_id']} needs {len(input_ids)} tokens but "
                    f"max_seq_length={max_seq_length}; segments are never silently truncated"
                )
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + [-100] + native_ids[1:] + [tokenizer.eos_token_id],
                "loss_weights": [0.0] * (len(prompt_ids) + 1) + [1.0] * len(native_ids),
            }

    return NativeMidiDataset()


def _validate_token_lengths(rows, tokenizer, max_seq_length):
    lengths = sorted(
        (
            len(tokenizer(row["upstream_prompt"], add_special_tokens=False)["input_ids"])
            + len(_read_tokens(row["native_model_tokens_path"]))
            + 1,
            row["work_id"],
            row["segment"],
        )
        for row in rows
    )
    oversized = [row for row in lengths if row[0] > max_seq_length]
    return {
        "selected_native_segments": len(rows),
        "max_seq_length": max_seq_length,
        "maximum_tokens": lengths[-1][0] if lengths else 0,
        "oversized_segment_count": len(oversized),
        "oversized_segments": [
            {"tokens": tokens, "work_id": work_id, "segment": segment}
            for tokens, work_id, segment in oversized[:20]
        ],
    }


def _validate_config(config: NativeTrainConfig) -> None:
    if config.max_segments is not None and config.max_segments < 1:
        raise ValueError("max_segments must be positive when provided")
    if config.max_segments_per_work < 1:
        raise ValueError("max_segments_per_work must be positive")
    if config.max_seq_length < 1 or config.loss_chunk_tokens < 1:
        raise ValueError("token lengths must be positive")
    if config.max_steps == 0 or config.max_steps < -1:
        raise ValueError("max_steps must be -1 or positive")
    if config.gradient_accumulation_steps < 1 or config.save_steps < 1:
        raise ValueError("training step controls must be positive")


def _count(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def _read_tokens(path: Path | str) -> List[int]:
    return [int(token_id) for token_id in json.loads(Path(path).read_text(encoding="utf-8"))]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _launch_command(config: NativeTrainConfig) -> str:
    composers = f" --composers {','.join(config.composers)}" if config.composers else ""
    max_segments = f" --max-segments {config.max_segments}" if config.max_segments is not None else ""
    max_steps = f" --max-steps {config.max_steps}" if config.max_steps >= 0 else ""
    return (
        "python -m midi_llm.train_native"
        f" --dataset-dir {config.dataset_dir}"
        f" --output-dir {config.output_dir}"
        f" --base-model {config.base_model}"
        f" --split {config.split}"
        f" --max-segments-per-work {config.max_segments_per_work}"
        f" --max-seq-length {config.max_seq_length}"
        f" --loss-chunk-tokens {config.loss_chunk_tokens}"
        f" --epochs {config.epochs}"
        f" --learning-rate {config.learning_rate}"
        f" --gradient-accumulation-steps {config.gradient_accumulation_steps}"
        f" --save-steps {config.save_steps}"
        f"{composers}"
        f"{max_segments}"
        f"{max_steps}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or launch native MIDI QLoRA specialization")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="slseanwu/MIDI-LLM_Llama-3.2-1B")
    parser.add_argument("--resume-adapter-dir")
    parser.add_argument("--resume-checkpoint-dir")
    parser.add_argument("--split", default="train")
    parser.add_argument("--composers", help="Comma-separated composer style tags")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--max-segments-per-work", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--loss-chunk-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = NativeTrainConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        resume_adapter_dir=args.resume_adapter_dir,
        resume_checkpoint_dir=args.resume_checkpoint_dir,
        split=args.split,
        composers=tuple(part.strip() for part in args.composers.split(",") if part.strip()) if args.composers else (),
        max_segments=args.max_segments,
        max_segments_per_work=args.max_segments_per_work,
        max_seq_length=args.max_seq_length,
        loss_chunk_tokens=args.loss_chunk_tokens,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
    )
    result = build_native_training_spec(config) if args.dry_run else run_native_training(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
