"""Manual, single-sentence demos of the perturbation library and classifiers —
useful for showing exactly what each perturbation does, outside the full sweep."""
from perturbations import PERTURBATIONS, apply_perturbation


def inspect_perturbations(sample="The customer service was really good and the amount was fast to process",
                           severities=(0.0, 0.2, 0.4)):
    for pert_name in PERTURBATIONS.keys():
        print(f"\n--- {pert_name} ---")
        for severity in severities:
            noisy = apply_perturbation(sample, pert_name, severity) if severity > 0 else sample
            print(f"  severity={severity}: {noisy}")


def inspect_cpu_classifier(cpu_clf, test_sentence="This is a terrible service, slow and unreliable",
                            severities=(0.1, 0.3, 0.5)):
    print("Clean prediction:", cpu_clf.predict(test_sentence))
    for severity in severities:
        noisy = apply_perturbation(test_sentence, "ocr_char_noise", severity)
        print(f"severity={severity}  noisy_text='{noisy}'  prediction={cpu_clf.predict(noisy)}")


def inspect_llm_classifier(llm_clf, tricky="I loved the delivery, excellent experience overall"):
    print("Clean:", llm_clf.predict(tricky))
    for pert_name in ["adversarial_synonym_swap", "case_whitespace_corruption"]:
        noisy = apply_perturbation(tricky, pert_name, 0.4)
        print(f"{pert_name} @0.4: '{noisy}' -> {llm_clf.predict(noisy)}")


if __name__ == "__main__":
    from classifiers import CpuClassifier, LlmClassifier

    print("=" * 60)
    print("PERTURBATION INSPECTION")
    print("=" * 60)
    inspect_perturbations()

    print("\n" + "=" * 60)
    print("CPU CLASSIFIER STRESS TEST")
    print("=" * 60)
    cpu_clf = CpuClassifier()
    inspect_cpu_classifier(cpu_clf)

    print("\n" + "=" * 60)
    print("LLM CLASSIFIER STRESS TEST")
    print("=" * 60)
    llm_clf = LlmClassifier()
    inspect_llm_classifier(llm_clf)
