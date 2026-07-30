"""Robustness evaluation harness: runs perturbation sweeps and summarizes results."""
import statistics
from dataclasses import dataclass
from typing import Any

from perturbations import PERTURBATIONS, apply_perturbation


@dataclass
class EvalExample:
    text: str
    label: Any


@dataclass
class RobustnessResult:
    perturbation: str
    severity: float
    accuracy: float
    n: int
    ci95: float


def wilson_ci_halfwidth(p, n, z=1.96):
    return 0.0 if n == 0 else z * ((p * (1 - p)) / n) ** 0.5


def run_robustness_eval(examples, predict_fn, score_fn, perturbation_names=None,
                         severities=(0.0, 0.1, 0.2, 0.3, 0.5)):
    perturbation_names = perturbation_names or list(PERTURBATIONS.keys())
    results = {}
    for pert_name in perturbation_names:
        curve = []
        for severity in severities:
            correct = 0
            for ex in examples:
                noisy_text = ex.text if severity == 0.0 else apply_perturbation(ex.text, pert_name, severity)
                if score_fn(predict_fn(noisy_text), ex.label):
                    correct += 1
            acc = correct / len(examples)
            curve.append(RobustnessResult(pert_name, severity, acc, len(examples),
                                           wilson_ci_halfwidth(acc, len(examples))))
        results[pert_name] = curve
    return results


def summarize(results):
    summary = {}
    for pert_name, curve in results.items():
        severities = [r.severity for r in curve]
        accs = [r.accuracy for r in curve]
        auc = sum((severities[i] - severities[i - 1]) * (accs[i] + accs[i - 1]) / 2
                   for i in range(1, len(severities)))
        max_possible = (severities[-1] - severities[0]) * accs[0] if accs[0] > 0 else 1
        summary[pert_name] = auc / max_possible if max_possible > 0 else 0.0
    summary["overall"] = statistics.mean(summary.values())
    return summary


def print_results(results):
    for pert_name, curve in results.items():
        print(f"\n{pert_name}:")
        for r in curve:
            print(f"  severity={r.severity:.1f}  acc={r.accuracy:.3f}  (+/-{r.ci95:.3f}, n={r.n})")
