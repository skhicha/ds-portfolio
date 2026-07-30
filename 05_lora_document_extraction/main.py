"""
LoRA Fine-Tuning project — local CLI entrypoint.

Usage:
    python main.py data       # generate the synthetic dataset
    python main.py train      # train the LoRA adapter (also merges + saves at the end)
    python main.py merge      # merge an already-trained adapter into a standalone model
    python main.py evaluate   # compare base vs fine-tuned on the test set
    python main.py infer      # run the fine-tuned model on sample documents
    python main.py all        # run data -> train -> merge -> evaluate -> infer, in order
"""
import argparse

import config


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning pipeline")
    parser.add_argument(
        "command",
        choices=["data", "train", "merge", "evaluate", "infer", "all"],
        help="which stage to run",
    )
    args = parser.parse_args()

    print(f"Using device: {config.DEVICE} (4-bit quantization: {config.USE_4BIT})")

    if args.command in ("data", "all"):
        from dataset import build_dataset
        build_dataset()

    if args.command in ("train", "all"):
        from train import train
        train()

    if args.command in ("train", "merge", "all"):
        from train import merge_and_save
        merge_and_save()

    if args.command in ("evaluate", "all"):
        from evaluate import compare_base_vs_finetuned
        compare_base_vs_finetuned()

    if args.command in ("infer", "all"):
        from infer import load_merged_model, run_on_document, SAMPLE_DOCS
        model, tokenizer = load_merged_model()
        for name, doc in SAMPLE_DOCS.items():
            raw, parsed = run_on_document(doc, model, tokenizer)
            print(f"\n=== {name} ===")
            print("Raw:", raw)
            print("Parsed:", parsed)


if __name__ == "__main__":
    main()
