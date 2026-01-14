ML-Based Python Code Summarization

Seq2Seq with Attention using PyTorch

📌 Project Overview

This project implements a machine learning–based system for automatic Python code summarization.
It generates natural language descriptions for Python functions using a Sequence-to-Sequence (Seq2Seq) neural network with an attention mechanism, built using PyTorch.

Automated code summarization helps developers understand, document, and maintain large codebases efficiently.

🎯 Motivation

Understanding source code manually is time-consuming, especially in large projects.
This project aims to:

Improve code readability

Assist in automatic documentation

Demonstrate practical NLP techniques applied to source code

✨ Features

Python code → English summary generation

Seq2Seq encoder–decoder architecture

Attention mechanism for better context understanding

Complete preprocessing pipeline

Training and inference scripts

Reproducible dataset download process

🧠 Model Architecture

Encoder: LSTM-based encoder for tokenized Python code

Decoder: LSTM-based decoder with attention

Attention: Bahdanau-style attention

Loss Function: Cross-Entropy Loss

Optimizer: Adam

Framework: PyTorch

📊 Dataset Information

Dataset Type: Python code and corresponding natural language summaries

Source: Hugging Face (e.g., CodeSearchNet-style dataset)

Format: JSON

Note:
The dataset is not included in this repository due to GitHub’s file size limitations (>100MB).
A script is provided to download and prepare the dataset automatically.

🗂 Project Structure
ML-Based-Python-Code-Summarization/
│
├── src/
│   ├── dataset.py          # Dataset loading logic
│   ├── preprocessing.py   # Tokenization and preprocessing
│   ├── vocab.py            # Vocabulary handling
│   ├── train.py            # Model training script
│   ├── infer.py            # Inference / prediction script
│   │
│   └── model/
│       ├── encoder.py      # Encoder implementation
│       ├── decoder.py      # Decoder implementation
│       ├── attention.py    # Attention mechanism
│       └── seq2seq.py      # Seq2Seq wrapper
│
├── download_data_hf.py     # Dataset download script
├── README.md               # Project documentation
├── .gitignore              # Ignored files (datasets, venv, models)
└── requirements.txt        # Python dependencies

💻 Requirements

Python 3.8+

Git

Internet connection (for dataset download)

🔧 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/NirmaAbro/ML-Based-Python-Code-Summarization.git
cd ML-Based-Python-Code-Summarization

2️⃣ Create and Activate Virtual Environment
Windows
python -m venv venv
source venv/Scripts/activate

Linux / macOS
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

📥 Dataset Download

Run the following command to download and prepare the dataset:

python download_data_hf.py


📌 The dataset will be stored locally inside the data/ directory.

🏋️ Training the Model

To train the model, run:

python src/train.py


Training progress will be displayed in the terminal

The trained model will be saved locally

Training time depends on dataset size and hardware

🔍 Running Inference (Generate Summary)

After training, generate summaries using:

python src/infer.py


You can modify infer.py to input your own Python code snippets.

📈 Evaluation

The model is evaluated using BLEU score

Qualitative evaluation is also performed by comparing generated summaries with ground-truth descriptions

🧪 Example Result

Input Code:

def multiply(a, b):
    return a * b


Generated Summary:

Returns the product of two numbers

⚠️ Limitations

Performance depends on dataset quality

Long and complex code snippets may reduce accuracy

Vocabulary size is limited

Seq2Seq models struggle with very large contexts

🚀 Future Improvements

Replace Seq2Seq with Transformer-based models (CodeBERT, T5)

Support multiple programming languages

Improve evaluation metrics

Add web-based UI for live inference

🛠 Technologies Used

Python

PyTorch

Hugging Face Datasets

NumPy

Git & GitHub

👤 Author

Nirma Abro
Machine Learning Project
Academic / Research-Oriented Implementation

📄 License

This project is intended for educational and research purposes.
