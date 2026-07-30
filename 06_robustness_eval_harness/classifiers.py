"""The two classifiers the harness is tested against: a CPU TF-IDF/LogReg
sanity-check model, and an LLM-based zero-shot classifier."""
import random

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

POS_TEMPLATES = [
    "The {noun} was really good and I am happy with it",
    "This is a great {noun}, fast and reliable",
    "I loved the {noun}, excellent experience overall",
]
NEG_TEMPLATES = [
    "The {noun} was bad and I am unhappy with it",
    "This is a terrible {noun}, slow and unreliable",
    "I hated the {noun}, awful experience overall",
]
NOUNS = ["product", "service", "app", "delivery", "support", "amount", "customer response"]


def build_sentiment_dataset(n=300, seed=1):
    random.seed(seed)
    rows = []
    for _ in range(n):
        noun = random.choice(NOUNS)
        if random.random() < 0.5:
            rows.append((random.choice(POS_TEMPLATES).format(noun=noun), "positive"))
        else:
            rows.append((random.choice(NEG_TEMPLATES).format(noun=noun), "negative"))
    return rows


def score_fn(pred, gold):
    return pred == gold


class CpuClassifier:
    """TF-IDF + Logistic Regression sanity-check classifier (runs anywhere, no GPU)."""

    def __init__(self, n=300, seed=1):
        rows = build_sentiment_dataset(n=n, seed=seed)
        texts, labels = [r[0] for r in rows], [r[1] for r in rows]
        split = int(0.8 * len(rows))
        self.train_texts, self.test_texts = texts[:split], texts[split:]
        self.train_labels, self.test_labels = labels[:split], labels[split:]

        self.vectorizer = TfidfVectorizer()
        X_train = self.vectorizer.fit_transform(self.train_texts)
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X_train, self.train_labels)

    def predict(self, text):
        return self.clf.predict(self.vectorizer.transform([text]))[0]

    def test_examples(self):
        from harness import EvalExample
        return [EvalExample(text=t, label=l) for t, l in zip(self.test_texts, self.test_labels)]


class LlmClassifier:
    """Zero-shot sentiment classifier using a small instruction-tuned LLM."""

    def __init__(self, model_name=config.LLM_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model = self.model.to(config.DEVICE)
        self.model.eval()

    def predict(self, text):
        prompt = (f"Classify the sentiment of this text as exactly one word, "
                  f"either 'positive' or 'negative'.\n\nText: {text}\n\nSentiment:")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                       pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        text_out = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip().lower()
        if "positive" in text_out:
            return "positive"
        if "negative" in text_out:
            return "negative"
        return "unknown"
