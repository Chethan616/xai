"""Fine-tune a small local language model to narrate XAI summaries.

This is deliberately CPU-friendly: Qwen2.5-0.5B-Instruct is public, uses
LoRA adapters, and does not require quantization or a GPU. The generated
training examples teach the model the project's terminology and output style;
the classifier/XAI summary remains the source of truth at inference time.

    python src/train_llm.py --epochs 1 --max-examples 256
"""
import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT = ROOT / "outputs" / "llm" / "pqd_narrator"
SYSTEM = (
    "You are a power-quality engineer. Explain a classifier prediction plainly. "
    "Use only the supplied fields, distinguish model evidence from certainty, "
    "and mention when the prediction is incorrect."
)


def _explanation(summary):
    predicted = summary["predicted_class"].replace("_", " ")
    actual = summary["true_class"].replace("_", " ")
    confidence = float(summary["confidence"])
    start, end = summary["most_influential_segment_samples"]
    peak = summary["signal_peak_pu"]
    correct = predicted == actual
    outcome = "matches the labelled class" if correct else f"does not match the labelled class ({actual})"
    certainty = "high" if confidence >= 0.9 else "moderate" if confidence >= 0.6 else "low"
    return (
        f"The model predicts {predicted} with {confidence:.1%} confidence, which {outcome}. "
        f"This is a {certainty}-confidence result. The strongest time-localised evidence "
        f"is between samples {start} and {end}; inspect that waveform region for the "
        f"disturbance pattern. The signal peak is {peak:.3f} p.u. "
        "The attribution indicates what influenced this model decision, not a proof "
        "of the physical cause."
    )


def _load_summaries():
    paths = sorted((ROOT / "outputs" / "explanations").glob("*.json"))
    summaries = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    return summaries


def _examples(max_examples):
    summaries = _load_summaries()
    if not summaries:
        raise RuntimeError("No explanation JSON files found. Run src/explain.py first.")
    rng = random.Random(7)
    examples = []
    for i in range(max_examples):
        source = summaries[i % len(summaries)].copy()
        source["sample"] = int(source["sample"]) + (i // len(summaries))
        user = (
            "Explain this power-quality classifier result for a non-expert. "
            "Do not invent measurements.\n"
            + json.dumps(source, separators=(",", ":"))
        )
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": _explanation(source)},
        ]})
    rng.shuffle(examples)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-examples", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    a = ap.parse_args()
    examples = _examples(a.max_examples)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir / "training_data.jsonl").open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()

    def tokenize(item):
        text = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        return tokenizer(text, truncation=True, max_length=a.max_length)

    dataset = Dataset.from_list(examples).map(tokenize, remove_columns=["messages"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(a.output_dir), num_train_epochs=a.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=a.gradient_accumulation_steps,
            learning_rate=2e-4, logging_steps=5, save_strategy="epoch",
            report_to="none", use_cpu=True, fp16=False,
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(a.output_dir)
    tokenizer.save_pretrained(a.output_dir)
    print(f"Saved trained LLM adapter to {a.output_dir}")


if __name__ == "__main__":
    main()
