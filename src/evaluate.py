# src/evaluate.py
import os
import json
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

from vocab import Vocabulary
from model.seq2seq import Seq2Seq

# -------------------- Device --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Hyperparameters (same as train.py) --------------------
EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_SUM_LEN = 30

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "validation.json")
VOCAB_PATH = os.path.join(BASE_DIR, "data", "processed", "vocab.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pt")


def generate_summary(model, vocab, code):
    ids = vocab.numericalize(code)
    src = torch.tensor(ids).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_ids = model.greedy_decode(src, max_len=MAX_SUM_LEN)

    summary = vocab.decode(output_ids[0].tolist())
    return summary


def main():
    print("Loading validation data...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("Loading vocabulary...")
    vocab = Vocabulary.load(VOCAB_PATH)

    print("Loading model...")
    model = Seq2Seq(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    references = []
    hypotheses = []

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []

    print("Evaluating...")
    for code, ref_summary in tqdm(dataset):
        pred_summary = generate_summary(model, vocab, code)

        references.append([ref_summary.split()])
        hypotheses.append(pred_summary.split())

        score = rouge.score(ref_summary, pred_summary)["rougeL"].fmeasure
        rouge_scores.append(score)

    bleu = corpus_bleu(references, hypotheses)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    print("\n===== Evaluation Results =====")
    print(f"BLEU-4 Score : {bleu:.4f}")
    print(f"ROUGE-L F1   : {avg_rouge:.4f}")


if __name__ == "__main__":
    main()
