# Robustness Harness — Text Perturbation Evaluation

A reusable harness that stress-tests any text classifier against four
perturbation types (OCR-style character noise, token dropout, case/whitespace
corruption, adversarial synonym swaps) across a severity sweep, with Wilson
confidence intervals and an AUC-style robustness summary per perturbation.

Converted from a Google Colab notebook into a plain local Python project.
Runs on CUDA, Apple Silicon (MPS), or CPU — device is auto-detected in
`config.py`. Part A (CPU classifier) needs no GPU at all; Part B (LLM
classifier) benefits from one but will run on CPU too, just more slowly.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py cpu    # Part A: TF-IDF/LogReg sanity check, full severity sweep
python main.py llm    # Part B: LLM zero-shot classifier, full severity sweep
python main.py demo   # single-sentence inspection: see exactly what each
                       # perturbation does, plus a couple of manual stress tests
python main.py all    # cpu -> llm -> demo, in order
```

## Files

| File | Purpose |
|---|---|
| `config.py` | device detection, model name, default severities |
| `perturbations.py` | the four perturbation functions + dispatch dict |
| `harness.py` | `run_robustness_eval`, Wilson CI, AUC-based `summarize` |
| `classifiers.py` | `CpuClassifier` (TF-IDF/LogReg) and `LlmClassifier` (zero-shot) |
| `demo.py` | single-sentence inspection utilities |
| `main.py` | CLI entrypoint |

## Extending it

To test your own classifier, anything with a `.predict(text) -> label` method
works — pass its `predict` (or a lambda) into `run_robustness_eval` along with
a `score_fn(pred, gold) -> bool` and a list of `EvalExample(text, label)`.
