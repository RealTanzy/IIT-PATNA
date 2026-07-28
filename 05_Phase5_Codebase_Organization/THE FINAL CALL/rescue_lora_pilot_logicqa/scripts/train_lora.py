#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(messages):
    system = ""
    user = ""
    assistant = ""
    for m in messages:
        role = m.get("role")
        if role == "system":
            system = m.get("content", "")
        elif role == "user":
            user = m.get("content", "")
        elif role == "assistant":
            assistant = m.get("content", "")
    prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
    return prompt, assistant


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length=768):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ex = self.rows[idx]
        prompt, target = build_prompt(ex["messages"])

        if self.tok.eos_token is None:
            full_text = prompt + " " + target
        else:
            full_text = prompt + " " + target + self.tok.eos_token

        prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
        full_ids = self.tok(full_text, add_special_tokens=False).input_ids

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        cut = min(len(prompt_ids), len(full_ids))
        labels = [-100] * cut + full_ids[cut:]

        attn = [1] * len(full_ids)
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        attention = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention = pad_sequence(attention, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
        }


def guess_target_modules(model):
    linear_names = []
    module_names = []
    for n, m in model.named_modules():
        module_names.append(n.split(".")[-1])
        if isinstance(m, torch.nn.Linear):
            linear_names.append(n.split(".")[-1])

    # Never LoRA-wrap output heads tied to embeddings.
    deny = {"lm_head", "embed_out", "output_projection"}
    linear_names = [x for x in linear_names if x not in deny]

    common = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "c_attn", "c_proj", "c_fc"]

    # Prefer canonical attention/MLP names if present anywhere in module tree.
    found = sorted(set(x for x in module_names if x in common and x not in deny))
    if found:
        return found

    # fallback: broad linear names
    fallback = sorted(set(linear_names))
    if fallback:
        return fallback[:12]

    raise SystemExit("Could not infer LoRA target modules for this base model.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/lora_train/train.jsonl")
    ap.add_argument("--dev-file", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/data/lora_train/dev.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--output-dir", default="/home/dibyanayan/tanzeel/THE FINAL CALL/rescue_lora_pilot_logicqa/outputs/adapters/rescue_lora_logicqa")
    ap.add_argument("--cuda-device", default="0")
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--epochs", type=float, default=1.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise SystemExit("peft is not installed. Install with: pip install peft") from exc

    train_rows = load_jsonl(args.train_file)
    dev_rows = load_jsonl(args.dev_file)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    model_kwargs = {}
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise SystemExit("BitsAndBytesConfig unavailable. Install bitsandbytes for --use-4bit") from exc

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = guess_target_modules(model)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)

    train_ds = SupervisedDataset(train_rows, tokenizer=tokenizer, max_length=args.max_length)
    dev_ds = SupervisedDataset(dev_rows, tokenizer=tokenizer, max_length=args.max_length)

    collator = Collator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds if len(dev_ds) > 0 else None,
        data_collator=collator,
    )

    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(json.dumps({
        "train_examples": len(train_ds),
        "dev_examples": len(dev_ds),
        "target_modules": target_modules,
        "saved": str(final_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
