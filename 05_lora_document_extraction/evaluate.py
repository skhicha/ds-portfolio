"""Generation + evaluation utilities: exact match, field-level accuracy, JSON parse failure rate."""
import json
import re

import torch

import config


def generate_json(model, tok, prompt, max_new_tokens=150):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def try_parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def field_level_accuracy(pred, gold):
    if pred is None:
        return 0.0
    keys = gold.keys()
    return sum(1 for k in keys if pred.get(k) == gold[k]) / len(keys)


def evaluate_model(model, tok, test_path, limit=50):
    rows = [json.loads(l) for l in open(test_path)][:limit]
    exact_matches, field_scores, parse_failures = 0, [], 0
    for row in rows:
        gold = json.loads(row["completion"])
        pred = try_parse_json(generate_json(model, tok, row["prompt"]))
        if pred is None:
            parse_failures += 1
            field_scores.append(0.0)
            continue
        if pred == gold:
            exact_matches += 1
        field_scores.append(field_level_accuracy(pred, gold))
    n = len(rows)
    return {
        "n": n,
        "exact_match_accuracy": exact_matches / n,
        "avg_field_level_accuracy": sum(field_scores) / n,
        "json_parse_failure_rate": parse_failures / n,
    }


def compare_base_vs_finetuned(test_path=f"{config.DATA_DIR}/test.jsonl", limit=50):
    from model import load_plain_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=== Base model ===")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    base_model = load_plain_model()
    base_model.eval()
    print(evaluate_model(base_model, tokenizer, test_path, limit))

    print("\n=== Fine-tuned model ===")
    merged_tokenizer = AutoTokenizer.from_pretrained(config.MERGED_DIR)
    merged_model = AutoModelForCausalLM.from_pretrained(config.MERGED_DIR, torch_dtype=torch.float32)
    merged_model = merged_model.to(config.DEVICE)
    merged_model.eval()
    print(evaluate_model(merged_model, merged_tokenizer, test_path, limit))


if __name__ == "__main__":
    compare_base_vs_finetuned()
