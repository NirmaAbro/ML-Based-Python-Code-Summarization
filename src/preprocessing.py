import json
import os
from tqdm import tqdm
from vocab import Vocabulary

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

MAX_CODE_LEN = 200
MAX_SUM_LEN = 30

def clean_text(text):
    return text.replace("\n", " ").strip()

def load_raw_split(split_name):
    path = os.path.join(RAW_DIR, f"{split_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_split(split_name, vocab, max_samples=None):
    raw_data = load_raw_split(split_name)
    processed = []

    print(f"Processing {split_name}...")
    for item in tqdm(raw_data):
        code = clean_text(item.get("code", ""))
        summary = clean_text(item.get("docstring", ""))

        if not code or not summary:
            continue
        if len(code.split()) > MAX_CODE_LEN:
            continue
        if len(summary.split()) > MAX_SUM_LEN:
            continue

        processed.append([code, summary])

        if split_name == "train":
            vocab.add_sentence(code)
            vocab.add_sentence(summary)

        if max_samples and len(processed) >= max_samples:
            break

    out_path = os.path.join(PROCESSED_DIR, f"{split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"{split_name}: saved {len(processed)} samples")

def main():
    vocab = Vocabulary()
   
    save_split("train",  vocab, max_samples=500)
    save_split("validation",  vocab, max_samples=100)



    vocab_path = os.path.join(PROCESSED_DIR, "vocab.pkl")
    vocab.save(vocab_path)

    print(f"Vocabulary saved to {vocab_path}")
    print(f"Vocab size: {len(vocab)}")

if __name__ == "__main__":
    main()
