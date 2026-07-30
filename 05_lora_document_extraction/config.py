"""Central configuration for the LoRA fine-tuning project."""
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "lora_adapter_out"
MERGED_DIR = "merged_model"
DATA_DIR = "data"

MAX_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LR = 2e-4
EPOCHS = 3

# 4-bit quantization (bitsandbytes) only works reliably on Linux + CUDA.
# Auto-disable it elsewhere so the same script runs on Mac/CPU/Windows too.
USE_4BIT = DEVICE == "cuda"

N_TRAIN = 800
N_VAL = 100
N_TEST = 100
