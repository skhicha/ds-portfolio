"""Base model + LoRA adapter construction."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

import config


def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    dtype = torch.float32
    if config.USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        dtype = torch.bfloat16
    elif config.DEVICE == "mps":
        # bfloat16 support on MPS is inconsistent; float32 is the safe default.
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto" if config.DEVICE == "cuda" else None,
        torch_dtype=dtype,
    )
    if config.DEVICE != "cuda":
        model = model.to(config.DEVICE)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_plain_model(dtype=torch.float32):
    """Load the unmodified base model (no LoRA) — used for before/after comparisons."""
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=dtype)
    return model.to(config.DEVICE)
