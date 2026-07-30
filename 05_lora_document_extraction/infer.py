"""Run the merged fine-tuned model on a single custom document."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from evaluate import generate_json, try_parse_json

SAMPLE_DOCS = {
    "invoice": """TAX INVOICE
Invoice No: INV-3312
Date: 05/06/2026
Billed To: Meera Joshi, Bangalore
Total Amount: Rs. 8999.00""",
    "id_card": """GOVERNMENT ID CARD
Name: Rohit Malhotra
ID Number: 719284650033
Date of Birth: 23/07/1995
Address: Delhi, India""",
    "receipt": """Nair Mart
Date: 19/02/2026
Customer: Ishita Chawla
Amount Paid: Rs. 1249.99
Payment Mode: UPI""",
    "noisy_invoice": """TAX 1NV0ICE
1nvoice N0: 1NV-6647
Date: 11/O4/2O26
Bil1ed T0: Sanjay Kap0or, Ahmedabad
T0tal Am0unt: Rs. 3450.OO""",
}


def load_merged_model():
    tokenizer = AutoTokenizer.from_pretrained(config.MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(config.MERGED_DIR, torch_dtype=torch.float32)
    model = model.to(config.DEVICE)
    model.eval()
    return model, tokenizer


def run_on_document(doc_text, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        model, tokenizer = load_merged_model()
    prompt = f"Extract the structured fields from this document as JSON.\n\nDocument:\n{doc_text}\n\nJSON:"
    raw = generate_json(model, tokenizer, prompt)
    return raw, try_parse_json(raw)


if __name__ == "__main__":
    model, tokenizer = load_merged_model()
    for name, doc in SAMPLE_DOCS.items():
        raw, parsed = run_on_document(doc, model, tokenizer)
        print(f"\n=== {name} ===")
        print("Raw:", raw)
        print("Parsed:", parsed)
