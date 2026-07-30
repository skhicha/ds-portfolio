"""Text perturbation functions used to stress-test model robustness."""
import random
import string

OCR_CONFUSIONS = {
    "O": "0", "0": "O", "I": "1", "1": "I", "l": "1", "S": "5", "5": "S",
    "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6", "6": "G",
}
SYNONYMS = {
    "good": ["great", "fine", "decent"],
    "bad": ["poor", "awful", "terrible"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "minor"],
    "fast": ["quick", "rapid", "speedy"],
    "amount": ["sum", "total", "figure"],
    "customer": ["client", "buyer", "patron"],
}


def ocr_char_noise(text, severity):
    return "".join(
        OCR_CONFUSIONS[ch] if ch in OCR_CONFUSIONS and random.random() < severity else ch
        for ch in text
    )


def token_dropout(text, severity):
    kept = [t for t in text.split() if random.random() > severity]
    return " ".join(kept) if kept else text


def case_whitespace_corruption(text, severity):
    out = []
    for ch in text:
        if ch.isalpha() and random.random() < severity:
            ch = ch.upper() if ch.islower() else ch.lower()
        out.append(ch)
        if ch == " " and random.random() < severity * 0.3:
            out.append(" ")
    result = "".join(out)
    if severity > 0.5 and random.random() < 0.3:
        result = " ".join(result.split())
    return result


def adversarial_synonym_swap(text, severity):
    out = []
    for tok in text.split():
        bare = tok.strip(string.punctuation).lower()
        if bare in SYNONYMS and random.random() < severity:
            repl = random.choice(SYNONYMS[bare])
            out.append(tok.replace(bare, repl) if bare in tok else repl)
        else:
            out.append(tok)
    return " ".join(out)


PERTURBATIONS = {
    "ocr_char_noise": ocr_char_noise,
    "token_dropout": token_dropout,
    "case_whitespace_corruption": case_whitespace_corruption,
    "adversarial_synonym_swap": adversarial_synonym_swap,
}


def apply_perturbation(text, name, severity):
    return PERTURBATIONS[name](text, severity)
