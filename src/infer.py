# # src/infer.py
# import sys
# import os
# import torch
# from vocab import Vocabulary
# from model.seq2seq import Seq2Seq

# # -------------------- Device --------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------------------- Hyperparameters (must match train.py) --------------------
# EMBED_SIZE = 256
# HIDDEN_SIZE = 512
# MAX_SUM_LEN = 30

# # -------------------- Paths --------------------
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# VOCAB_PATH = os.path.join(BASE_DIR, "data", "processed", "vocab.pkl")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pt")

# # -------------------- Utility functions --------------------
# def encode_code(code, vocab):
#     """Convert code string to tensor of token IDs"""
#     ids = vocab.numericalize(code)
#     return torch.tensor(ids).unsqueeze(0)  # batch size = 1

# def decode_summary(ids, vocab):
#     """Convert list of token IDs back to a readable string"""
#     words = [vocab.idx2word[i] for i in ids if i not in (0, 1, 2)]  # remove <PAD>, <SOS>, <EOS>
#     return " ".join(words)

# # -------------------- Main function --------------------
# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python infer.py \"<python code>\"")
#         return

#     code_input = sys.argv[1]
#     print("Input code:\n", code_input)

#     # -------------------- Load Vocabulary --------------------
#     if not os.path.exists(VOCAB_PATH):
#         print(f"Error: vocab.pkl not found at {VOCAB_PATH}")
#         return
#     vocab = Vocabulary.load(VOCAB_PATH)
#     print(f"Vocabulary loaded. Size: {len(vocab)}")

#     # -------------------- Build Model --------------------
#     model = Seq2Seq(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)

#     # -------------------- Load trained weights --------------------
#     if not os.path.exists(MODEL_PATH):
#         print(f"Error: model.pt not found at {MODEL_PATH}")
#         return
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.eval()
#     print("Model loaded successfully.")

#     # -------------------- Encode and Decode --------------------
#     src_tensor = encode_code(code_input, vocab).to(DEVICE)

#     # Greedy decoding
#     with torch.no_grad():
#         output_ids = model.greedy_decode(src_tensor, max_len=MAX_SUM_LEN)

#     summary = decode_summary(output_ids[0].tolist(), vocab)
#     print("\nGenerated summary:\n", summary)


# if __name__ == "__main__":
#     main()

# # Generated summary:
# #  specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific specific
# # (venv) due to 500 sample of code 

# src/infer.py
import sys
import os
import torch
from vocab import Vocabulary
from model.seq2seq import Seq2Seq

# -------------------- Device --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Hyperparameters (must match train.py) --------------------
EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_SUM_LEN = 30  # maximum length of summary to generate

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_PATH = os.path.join(BASE_DIR, "data", "processed", "vocab.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pt")

# -------------------- Utility functions --------------------
def encode_code(code, vocab):
    """Convert code string to tensor of token IDs"""
    ids = vocab.numericalize(code)
    return torch.tensor(ids).unsqueeze(0)  # shape: [1, seq_len]

def decode_summary(ids, vocab):
    """Convert list of token IDs back to a readable string"""
    words = [vocab.idx2word[i] for i in ids if i not in (0, 1, 2)]  # ignore <PAD>, <SOS>, <EOS>
    return " ".join(words)

# -------------------- Main function --------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py \"<python code>\"")
        return

    code_input = sys.argv[1]
    print("Input code:\n", code_input)

    # -------------------- Load Vocabulary --------------------
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: vocab.pkl not found at {VOCAB_PATH}")
        return

    vocab = Vocabulary.load(VOCAB_PATH)
    print(f"Vocabulary loaded. Size: {len(vocab)}")

    # -------------------- Build Model --------------------
    model = Seq2Seq(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)

    # -------------------- Load trained weights --------------------
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model.pt not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # -------------------- Encode input --------------------
    src_tensor = encode_code(code_input, vocab).to(DEVICE)

    # -------------------- Greedy decoding --------------------
    with torch.no_grad():
        output_ids = model.greedy_decode(src_tensor, max_len=MAX_SUM_LEN)

    # -------------------- Decode summary --------------------
    summary = decode_summary(output_ids[0].tolist(), vocab)
    print("\nGenerated summary:\n", summary)


if __name__ == "__main__":
    main()
