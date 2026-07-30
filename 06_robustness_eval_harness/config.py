"""Central configuration for the robustness harness project."""
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SEVERITIES = (0.0, 0.1, 0.2, 0.3, 0.5)
LLM_SEVERITIES = (0.0, 0.2, 0.4)
LLM_EXAMPLE_LIMIT = 20
