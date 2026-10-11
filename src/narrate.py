"""Generate a human-readable explanation from an XAI summary JSON.

    python src/narrate.py outputs/explanations/xpqrs_test0004.json
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = ROOT / "outputs" / "llm" / "pqd_narrator"
SYSTEM = (
    "You are a power-quality engineer. Explain a classifier prediction plainly. "
    "Use only the supplied fields, distinguish model evidence from certainty, "
    "and mention when the prediction is incorrect."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    a = ap.parse_args()
    with a.summary.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(a.adapter)
    base = AutoModelForCausalLM.from_pretrained(a.base_model)
    model = PeftModel.from_pretrained(base, a.adapter)
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            "Explain this power-quality classifier result for a non-expert. "
            "Do not invent measurements.\n" + json.dumps(summary, separators=(",", ":"))
        )},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
