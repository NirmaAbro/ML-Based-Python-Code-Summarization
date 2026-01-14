# import os
# import json
# from datasets import load_dataset
# from tqdm import tqdm

# PROCESSED_DIR = "./data/processed"
# os.makedirs(PROCESSED_DIR, exist_ok=True)

# MAX_CODE_LEN = 200
# MAX_SUM_LEN = 30
# SUBSET = 7000  # small subset for testing

# print("Loading CodeSearchNet Python dataset...")
# dataset = load_dataset("kejian/codesearchnet-python-raw")["train"].shuffle(seed=42).select(range(SUBSET))

# # Split manually
# train_data = dataset[:5000]
# val_data   = dataset[5000:6000]
# test_data  = dataset[6000:7000]

# def clean_text(text):
#     if text is None:
#         return ""
#     return text.replace("\n", " ").strip()

# def save_json(data, filename):
#     out_list = []
#     for item in tqdm(data, desc=f"Processing {filename}"):
#         code = clean_text(item["code"])
#         summary = clean_text(item["docstring"])

#         if not code or not summary:
#             continue

#         if len(code.split()) > MAX_CODE_LEN or len(summary.split()) > MAX_SUM_LEN:
#             continue

#         out_list.append([code, summary])

#     with open(os.path.join(PROCESSED_DIR, filename), "w", encoding="utf-8") as f:
#         json.dump(out_list, f, ensure_ascii=False, indent=2)
#     print(f"{filename} saved with {len(out_list)} samples")

# save_json(train_data, "train.json")
# save_json(val_data, "validation.json")
# save_json(test_data, "test.json")

import os
import json
import logging
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# -------------------- Save split --------------------
def save_split(data_list, split_name):
    path = os.path.join(RAW_DATA_DIR, f"{split_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False)
    logger.info(f"Saved {len(data_list)} samples to {path}")

# -------------------- Main --------------------
def main():
    logger.info("Loading Hugging Face dataset: kejian/codesearchnet-python-raw")
    dataset = load_dataset("kejian/codesearchnet-python-raw")
    logger.info("Available splits: %s", dataset.keys())

    # Convert HF Dataset → Python list (CRITICAL FIX)
    all_data = list(dataset["train"])

    logger.info("Total samples loaded: %d", len(all_data))

    # Manual split
    train_split, valid_split = train_test_split(
        all_data, test_size=0.1, random_state=42
    )

    save_split(train_split, "train")
    save_split(valid_split, "validation")

    # Preview
    logger.info("Previewing first 3 samples:")
    for i in range(3):
        logger.info("Sample %d CODE:\n%s", i + 1, train_split[i]["code"][:200])
        logger.info("Sample %d SUMMARY:\n%s", i + 1, train_split[i]["docstring"][:200])

if __name__ == "__main__":
    main()

