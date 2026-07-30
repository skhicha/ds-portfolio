# LoRA Fine-Tuning — Document Field Extraction

Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with a LoRA adapter to extract structured
JSON fields (invoice number, date, customer name, etc.) from synthetic
invoice / ID card / receipt documents.

Converted from a Google Colab notebook into a plain local Python project.
Runs on CUDA, Apple Silicon (MPS), or CPU — device and 4-bit quantization
are auto-detected in `config.py` (4-bit only activates on CUDA, since
`bitsandbytes` doesn't reliably support Mac/CPU).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py data        # generate the synthetic dataset into data/
python main.py train       # train the LoRA adapter, checkpoint per epoch, then merge
python main.py merge       # (re-)merge an already-trained adapter into merged_model/
python main.py evaluate    # compare base vs fine-tuned model on the held-out test set
python main.py infer       # run the fine-tuned model on 4 sample documents (incl. a noisy one)
python main.py all         # run the full pipeline end to end
```

## Files

| File | Purpose |
|---|---|
| `config.py` | device detection, hyperparameters, paths |
| `dataset.py` | synthetic data generation + PyTorch `Dataset`/collate |
| `model.py` | base model + LoRA adapter construction |
| `train.py` | training loop, checkpointing, merge-and-unload |
| `evaluate.py` | generation + exact-match / field-level accuracy / JSON-parse-failure metrics |
| `infer.py` | run the merged model on ad-hoc documents |
| `main.py` | CLI entrypoint tying it all together |

## Notes

- On CPU or Mac, expect training to be noticeably slower than on a Colab T4 —
  consider lowering `EPOCHS` or `N_TRAIN` in `config.py` for a quicker local run.
- The `evaluate` and `infer` commands expect `merged_model/` to already exist
  (i.e. `train` or `merge` must run first).
