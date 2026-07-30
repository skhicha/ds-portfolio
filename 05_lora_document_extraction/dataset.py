"""Synthetic document dataset generation + PyTorch Dataset/collate."""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

import config

FIRST_NAMES = ["Rahul", "Priya", "Aman", "Sneha", "Vikram", "Anita", "Karan", "Divya", "Rohan", "Neha", "Arjun", "Pooja"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Patel", "Reddy", "Nair", "Gupta", "Khan", "Singh", "Rao"]
CITIES = ["Mumbai", "Bangalore", "Delhi", "Pune", "Chennai", "Hyderabad", "Kolkata", "Ahmedabad"]
DOC_TYPES = ["invoice", "id_card", "receipt"]


def random_date():
    return f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(2022, 2026)}"


def random_id_number():
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def make_record():
    doc_type = random.choice(DOC_TYPES)
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    city = random.choice(CITIES)
    date = random_date()
    if doc_type == "invoice":
        amount = round(random.uniform(100, 50000), 2)
        invoice_no = f"INV-{random.randint(1000, 9999)}"
        raw_text = (f"TAX INVOICE\nInvoice No: {invoice_no}\nDate: {date}\n"
                    f"Billed To: {name}, {city}\nTotal Amount: Rs. {amount}\nThank you for your business.")
        fields = {"document_type": "invoice", "invoice_number": invoice_no, "date": date,
                  "customer_name": name, "city": city, "total_amount": amount}
    elif doc_type == "id_card":
        id_no = random_id_number()
        dob = random_date()
        raw_text = (f"GOVERNMENT ID CARD\nName: {name}\nID Number: {id_no}\n"
                    f"Date of Birth: {dob}\nAddress: {city}, India")
        fields = {"document_type": "id_card", "name": name, "id_number": id_no,
                  "date_of_birth": dob, "city": city}
    else:
        amount = round(random.uniform(50, 5000), 2)
        store = f"{random.choice(LAST_NAMES)} Mart"
        raw_text = (f"{store}\nDate: {date}\nCustomer: {name}\nAmount Paid: Rs. {amount}\nPayment Mode: UPI")
        fields = {"document_type": "receipt", "store_name": store, "date": date,
                  "customer_name": name, "amount_paid": amount}
    return raw_text, fields


def inject_ocr_noise(text, noise_level=0.05):
    subs = {"O": "0", "I": "1", "S": "5", "l": "1", "B": "8"}
    out = []
    for ch in text:
        r = random.random()
        if r < noise_level and ch in subs:
            out.append(subs[ch])
        elif r < noise_level * 1.5:
            continue
        else:
            out.append(ch)
    return "".join(out)


def build_dataset(n_train=None, n_val=None, n_test=None, out_dir=None, seed=42):
    random.seed(seed)
    n_train = n_train or config.N_TRAIN
    n_val = n_val or config.N_VAL
    n_test = n_test or config.N_TEST
    out_dir = Path(out_dir or config.DATA_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)

    def make_split(n, noisy=False):
        rows = []
        for _ in range(n):
            raw_text, fields = make_record()
            if noisy:
                raw_text = inject_ocr_noise(raw_text, noise_level=0.04)
            prompt = f"Extract the structured fields from this document as JSON.\n\nDocument:\n{raw_text}\n\nJSON:"
            completion = json.dumps(fields, ensure_ascii=False)
            rows.append({"prompt": prompt, "completion": completion})
        return rows

    train, val, test = make_split(n_train), make_split(n_val), make_split(n_test, noisy=True)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(train)} train / {len(val)} val / {len(test)} test examples to {out_dir}/")


class JsonlPromptDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=None):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tokenizer
        self.max_len = max_len or config.MAX_LEN

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt_ids = self.tok(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = self.tok(row["completion"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {"input_ids": input_ids[:self.max_len], "labels": labels[:self.max_len]}


def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn_mask = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)
        attn_mask.append([1] * len(x["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
    }


if __name__ == "__main__":
    build_dataset()
