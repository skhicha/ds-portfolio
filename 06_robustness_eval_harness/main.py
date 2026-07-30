"""
Robustness Harness project — local CLI entrypoint.

Usage:
    python main.py cpu        # Part A: full severity sweep on the CPU TF-IDF/LogReg classifier
    python main.py llm        # Part B: full severity sweep on the LLM classifier
    python main.py demo       # single-sentence inspection demos (perturbations + both classifiers)
    python main.py all        # cpu -> llm -> demo, in order
"""
import argparse

import config
from harness import run_robustness_eval, print_results, summarize
from perturbations import PERTURBATIONS
from classifiers import CpuClassifier, LlmClassifier, score_fn


def run_cpu():
    print(f"Using device: {config.DEVICE} (CPU classifier does not use the GPU)")
    clf = CpuClassifier()
    examples = clf.test_examples()
    results = run_robustness_eval(examples, clf.predict, score_fn,
                                   list(PERTURBATIONS.keys()), config.DEFAULT_SEVERITIES)
    print_results(results)
    print("\nSummary:", summarize(results))
    return results


def run_llm():
    print(f"Using device: {config.DEVICE}")
    clf = CpuClassifier()  # reuse its test set for consistent examples
    examples = clf.test_examples()[:config.LLM_EXAMPLE_LIMIT]
    llm_clf = LlmClassifier()
    results = run_robustness_eval(examples, llm_clf.predict, score_fn,
                                   list(PERTURBATIONS.keys()), config.LLM_SEVERITIES)
    print_results(results)
    print("\nSummary (LLM):", summarize(results))
    return results


def run_demo():
    from demo import inspect_perturbations, inspect_cpu_classifier, inspect_llm_classifier
    inspect_perturbations()
    cpu_clf = CpuClassifier()
    inspect_cpu_classifier(cpu_clf)
    llm_clf = LlmClassifier()
    inspect_llm_classifier(llm_clf)


def main():
    parser = argparse.ArgumentParser(description="Robustness harness")
    parser.add_argument("command", choices=["cpu", "llm", "demo", "all"], help="which stage to run")
    args = parser.parse_args()

    if args.command in ("cpu", "all"):
        run_cpu()
    if args.command in ("llm", "all"):
        run_llm()
    if args.command in ("demo", "all"):
        run_demo()


if __name__ == "__main__":
    main()
