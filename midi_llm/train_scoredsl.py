"""Dry-run locally or launch ScoreDSL QLoRA training on a cloud GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.util import find_spec
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SCOREDLS_TOKENS = (
    "SCHEMA",
    "PLAN",
    "MOTIF_BANK",
    "NOTE",
    "DIRECTION",
    "LAYOUT",
    "METADATA",
    "END_SCORE",
    "<SCORE_FIRST_INPUT>",
    "<SCORE_FIRST_TARGET>",
)

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

REQUIRED_RUNTIME_PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate")


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str
    base_model: str = "slseanwu/MIDI-LLM_Llama-3.2-1B"
    resume_adapter_dir: str | None = None
    split: str = "train"
    tasks: Sequence[str] = ()
    max_examples: int | None = None
    max_example_characters: int | None = None
    max_seq_length: int = 65536
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    logging_steps: int = 5
    save_steps: int = 100
    bf16: bool = True


def build_training_spec(config: TrainConfig) -> Dict[str, Any]:
    """Write a launch spec without importing heavyweight cloud dependencies."""

    _validate_config(config)
    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows(
        dataset_dir,
        config.split,
        config.tasks,
        max_examples=config.max_examples,
        max_example_characters=config.max_example_characters,
    )
    lengths = [_character_length(row) for row in rows]
    estimated_tokens = math.ceil(max(lengths, default=0) / 3)
    materialization_summary = _read_json_if_exists(dataset_dir / "summary.json")
    spec = {
        "pipeline": "score-first-scoredsl-qlora-v1",
        "mode": "cloud-gpu-launch-spec",
        "config": asdict(config),
        "dataset": {
            "path": str(dataset_dir.resolve()),
            "split": config.split,
            "selected_examples": len(rows),
            "task_counts": _task_counts(rows),
            "max_characters": max(lengths, default=0),
            "estimated_max_tokens_at_three_characters_per_token": estimated_tokens,
            "task_length_estimates": _task_length_estimates(rows),
            "materialization": {
                "examples_written": materialization_summary.get("examples_written"),
                "examples_excluded_oversized": materialization_summary.get("examples_excluded_oversized"),
                "max_example_characters": materialization_summary.get("max_example_characters"),
            },
            "preflight_warning": (
                f"Estimated sequence length {estimated_tokens} exceeds max_seq_length={config.max_seq_length}; "
                "measure with the cloud tokenizer and raise the limit before training."
                if estimated_tokens > config.max_seq_length
                else None
            ),
        },
        "model": {
            "base_model": config.base_model,
            "adapter": "QLoRA",
            "resume_adapter_dir": config.resume_adapter_dir,
            "quantization": "4-bit NF4 with double quantization",
            "attention_implementation": "sdpa",
            "lora_targets": list(DEFAULT_LORA_TARGETS),
            "scoredsl_tag_strings": list(SCOREDLS_TOKENS),
            "tokenizer_strategy": "reuse upstream BPE vocabulary; do not freeze randomly initialized added-token rows",
            "complete_piece_truncation_policy": "forbidden",
        },
        "runtime": {
            "required_packages": list(REQUIRED_RUNTIME_PACKAGES),
            "dependency_report": dependency_report(),
            "required_hardware": "NVIDIA GPU with 48-80GB VRAM",
            "local_dry_run_supported": True,
        },
        "launch_command": _launch_command(config),
    }
    (output_dir / "training_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def dependency_report() -> Dict[str, bool]:
    return {package: find_spec(package) is not None for package in REQUIRED_RUNTIME_PACKAGES}


def _validate_config(config: TrainConfig) -> None:
    if config.max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")
    if config.max_steps == 0 or config.max_steps < -1:
        raise ValueError("max_steps must be -1 or positive")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")


def run_training(config: TrainConfig) -> Dict[str, Any]:
    """Launch QLoRA training. Heavy packages are imported only on the GPU host."""

    spec = build_training_spec(config)
    rows = _selected_rows(
        Path(config.dataset_dir),
        config.split,
        config.tasks,
        max_examples=config.max_examples,
        max_example_characters=config.max_example_characters,
    )
    if not rows:
        raise RuntimeError(f"No examples selected from split {config.split!r}")
    missing = [package for package, available in dependency_report().items() if not available]
    if missing:
        raise RuntimeError(
            "Missing cloud training packages: "
            + ", ".join(missing)
            + ". Install requirements-score-first-train.txt on the NVIDIA host."
        )

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA launch requires an NVIDIA CUDA GPU; use --dry-run on local development machines")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_preflight = _validate_token_lengths(rows, tokenizer, config.max_seq_length)
    (Path(config.output_dir) / "tokenizer_preflight.json").write_text(
        json.dumps(tokenizer_preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    if tokenizer_preflight["oversized_example_count"]:
        raise ValueError(
            f"{tokenizer_preflight['oversized_example_count']} selected examples exceed "
            f"max_seq_length={config.max_seq_length}; largest example needs "
            f"{tokenizer_preflight['maximum_tokens']} tokens. Adjust the materialization "
            "character budget or raise the context length. Targets are never silently truncated."
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
    dataset = _tokenized_dataset(rows, tokenizer, config.max_seq_length)
    trainer = Trainer(
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
    trainer.train()
    adapter_dir = Path(config.output_dir) / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    result = {
        **spec,
        "mode": "trained-adapter",
        "adapter_dir": str(adapter_dir.resolve()),
        "tokenizer_preflight": tokenizer_preflight,
    }
    (Path(config.output_dir) / "training_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _selected_rows(
    dataset_dir: Path,
    split: str,
    tasks: Sequence[str],
    *,
    max_examples: int | None = None,
    max_example_characters: int | None = None,
) -> List[Dict[str, Any]]:
    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        raise ValueError(f"Dataset split does not exist: {path}")
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive when provided")
    if max_example_characters is not None and max_example_characters < 1:
        raise ValueError("max_example_characters must be positive when provided")
    selected = set(tasks)
    rows = [
        row
        for row in _read_jsonl(path)
        if (not selected or row["task"] in selected)
        and (max_example_characters is None or _character_length(row) <= max_example_characters)
    ]
    return rows[:max_examples] if max_examples is not None else rows


def _tokenized_dataset(rows, tokenizer, max_seq_length):
    class ScoreDataset:
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            prompt_ids = tokenizer(_model_prompt(row), add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(row["target_scoredsl"], add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
            if len(input_ids) > max_seq_length:
                raise ValueError(
                    f"Example {row['example_id']} needs {len(input_ids)} tokens but max_seq_length={max_seq_length}. "
                    "Increase the context length; complete-piece targets are never silently truncated."
                )
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id],
            }

    return ScoreDataset()


def _collator(tokenizer, torch):
    def collate(features):
        width = max(len(feature["input_ids"]) for feature in features)
        pad = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([feature["input_ids"] + [pad] * (width - len(feature["input_ids"])) for feature in features]),
            "attention_mask": torch.tensor(
                [feature["attention_mask"] + [0] * (width - len(feature["attention_mask"])) for feature in features]
            ),
            "labels": torch.tensor([feature["labels"] + [-100] * (width - len(feature["labels"])) for feature in features]),
        }

    return collate


def _validate_token_lengths(rows, tokenizer, max_seq_length):
    task_maximums: Dict[str, int] = {}
    longest = []
    for row in rows:
        tokens = _token_length(row, tokenizer)
        task_maximums[row["task"]] = max(task_maximums.get(row["task"], 0), tokens)
        longest.append((tokens, row["example_id"], row["task"], row["target_scope"]))
    longest.sort(reverse=True)
    oversized = [item for item in longest if item[0] > max_seq_length]
    result = {
        "selected_examples": len(rows),
        "max_seq_length": max_seq_length,
        "maximum_tokens": longest[0][0] if longest else 0,
        "task_maximum_tokens": dict(sorted(task_maximums.items())),
        "oversized_example_count": len(oversized),
        "oversized_examples": [
            {
                "tokens": tokens,
                "example_id": example_id,
                "task": task,
                "target_scope": target_scope,
            }
            for tokens, example_id, task, target_scope in oversized[:20]
        ],
    }
    return result


def _token_length(row: Dict[str, Any], tokenizer) -> int:
    prompt_ids = tokenizer(_model_prompt(row), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(row["target_scoredsl"], add_special_tokens=False)["input_ids"]
    return len(prompt_ids) + len(target_ids) + 1


def _model_prompt(row: Dict[str, Any]) -> str:
    return model_prompt(row["model_input"])


def model_prompt(model_input: Dict[str, Any]) -> str:
    payload = json.dumps(model_input, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"<SCORE_FIRST_INPUT>\n{payload}\n<SCORE_FIRST_TARGET>\n"


def _character_length(row: Dict[str, Any]) -> int:
    return len(_model_prompt(row)) + len(row["target_scoredsl"])


def _task_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    return dict(sorted(counts.items()))


def _task_length_estimates(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    lengths: Dict[str, List[int]] = {}
    for row in rows:
        lengths.setdefault(row["task"], []).append(_character_length(row))
    return {
        task: {
            "examples": len(values),
            "max_characters": max(values),
            "estimated_max_tokens_at_three_characters_per_token": math.ceil(max(values) / 3),
        }
        for task, values in sorted(lengths.items())
    }


def _launch_command(config: TrainConfig) -> str:
    tasks = f" --tasks {','.join(config.tasks)}" if config.tasks else ""
    resume = f" --resume-adapter-dir {config.resume_adapter_dir}" if config.resume_adapter_dir else ""
    max_examples = f" --max-examples {config.max_examples}" if config.max_examples is not None else ""
    max_characters = (
        f" --max-example-characters {config.max_example_characters}"
        if config.max_example_characters is not None
        else ""
    )
    max_steps = f" --max-steps {config.max_steps}" if config.max_steps >= 0 else ""
    return (
        "python -m midi_llm.train_scoredsl"
        f" --dataset-dir {config.dataset_dir}"
        f" --output-dir {config.output_dir}"
        f" --base-model {config.base_model}"
        f" --split {config.split}"
        f" --max-seq-length {config.max_seq_length}"
        f" --epochs {config.epochs}"
        f"{resume}"
        f"{tasks}"
        f"{max_examples}"
        f"{max_characters}"
        f"{max_steps}"
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or launch ScoreDSL QLoRA training")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="slseanwu/MIDI-LLM_Llama-3.2-1B")
    parser.add_argument("--resume-adapter-dir")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tasks", help="Comma-separated curriculum tasks; defaults to every task in the split")
    parser.add_argument("--max-examples", type=int, help="Select at most this many examples after task filtering")
    parser.add_argument("--max-example-characters", type=int, help="Exclude larger examples without truncating targets")
    parser.add_argument("--max-seq-length", type=int, default=65536)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epoch count with a bounded optimizer-step run")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        resume_adapter_dir=args.resume_adapter_dir,
        split=args.split,
        tasks=tuple(task.strip() for task in args.tasks.split(",") if task.strip()) if args.tasks else (),
        max_examples=args.max_examples,
        max_example_characters=args.max_example_characters,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    result = build_training_spec(config) if args.dry_run else run_training(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
