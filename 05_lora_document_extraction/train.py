"""Training loop for the LoRA adapter."""
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import config
from dataset import JsonlPromptDataset, collate_fn
from model import build_model_and_tokenizer


def evaluate_loss(model, loader):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def train():
    model, tokenizer = build_model_and_tokenizer()

    train_ds = JsonlPromptDataset(f"{config.DATA_DIR}/train.jsonl", tokenizer)
    val_ds = JsonlPromptDataset(f"{config.DATA_DIR}/val.jsonl", tokenizer)
    collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.LR)
    total_steps = (len(train_loader) // config.GRAD_ACCUM_STEPS) * config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps
    )

    Path(config.OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    global_step = 0
    model.train()

    for epoch in range(config.EPOCHS):
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / config.GRAD_ACCUM_STEPS
            loss.backward()
            if (i + 1) % config.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 20 == 0:
                    print(f"epoch {epoch} step {global_step} loss {loss.item() * config.GRAD_ACCUM_STEPS:.4f}")
        val_loss = evaluate_loss(model, val_loader)
        print(f"=== epoch {epoch} val_loss {val_loss:.4f} perplexity {math.exp(val_loss):.2f} ===")
        ckpt_dir = f"{config.OUTPUT_DIR}/epoch_{epoch}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"saved adapter checkpoint to {ckpt_dir}")

    return model, tokenizer


def merge_and_save():
    from peft import PeftModel
    from transformers import AutoTokenizer

    adapter_dir = f"{config.OUTPUT_DIR}/epoch_{config.EPOCHS - 1}"
    from model import load_plain_model
    base_model_fp = load_plain_model()
    peft_model = PeftModel.from_pretrained(base_model_fp, adapter_dir)
    merged_model = peft_model.merge_and_unload()

    Path(config.MERGED_DIR).mkdir(exist_ok=True, parents=True)
    merged_model.save_pretrained(config.MERGED_DIR)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(config.MERGED_DIR)
    print("Merged model saved to", config.MERGED_DIR)
    return merged_model, tokenizer


if __name__ == "__main__":
    train()
    merge_and_save()
