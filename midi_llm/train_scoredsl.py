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
    split: str = "train"
    tasks: Sequence[str] = ()
    max_seq_length: int = 65536
    epochs: float = 1.0
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

    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows(dataset_dir, config.split, config.tasks)
    lengths = [_character_length(row) for row in rows]
    estimated_tokens = math.ceil(max(lengths, default=0) / 3)
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
            "quantization": "4-bit NF4 with double quantization",
            "lora_targets": list(DEFAULT_LORA_TARGETS),
            "scoredsl_tokens_added": list(SCOREDLS_TOKENS),
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


def run_training(config: TrainConfig) -> Dict[str, Any]:
    """Launch QLoRA training. Heavy packages are imported only on the GPU host."""

    spec = build_training_spec(config)
    rows = _selected_rows(Path(config.dataset_dir), config.split, config.tasks)
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
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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
    tokenizer.add_special_tokens({"additional_special_tokens": list(SCOREDLS_TOKENS)})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
    }
    (Path(config.output_dir) / "training_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _selected_rows(dataset_dir: Path, split: str, tasks: Sequence[str]) -> List[Dict[str, Any]]:
    path = dataset_dir / f"{split}.jsonl"
    if not path.exists():
        raise ValueError(f"Dataset split does not exist: {path}")
    selected = set(tasks)
    return [
        row
        for row in _read_jsonl(path)
        if not selected or row["task"] in selected
    ]


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


def _model_prompt(row: Dict[str, Any]) -> str:
    payload = json.dumps(row["model_input"], ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"<SCORE_FIRST_INPUT>\n{payload}\n<SCORE_FIRST_TARGET>\n"


def _character_length(row: Dict[str, Any]) -> int:
    return len(_model_prompt(row)) + len(row["target_scoredsl"])


def _task_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    return dict(sorted(counts.items()))


def _launch_command(config: TrainConfig) -> str:
    tasks = f" --tasks {','.join(config.tasks)}" if config.tasks else ""
    return (
        "python -m midi_llm.train_scoredsl"
        f" --dataset-dir {config.dataset_dir}"
        f" --output-dir {config.output_dir}"
        f" --base-model {config.base_model}"
        f" --split {config.split}"
        f" --max-seq-length {config.max_seq_length}"
        f" --epochs {config.epochs}"
        f"{tasks}"
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run or launch ScoreDSL QLoRA training")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="slseanwu/MIDI-LLM_Llama-3.2-1B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tasks", help="Comma-separated curriculum tasks; defaults to every task in the split")
    parser.add_argument("--max-seq-length", type=int, default=65536)
    parser.add_argument("--epochs", type=float, default=1.0)
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
        split=args.split,
        tasks=tuple(task.strip() for task in args.tasks.split(",") if task.strip()) if args.tasks else (),
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    result = build_training_spec(config) if args.dry_run else run_training(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
