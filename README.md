<<<<<<< HEAD
## Aksharantar Transliteration — Deep Learning Seq2Seq Model with Attention
## Project Overview

This project implements an AI-powered transliteration model that converts words from Roman script (English letters) to Indic languages (e.g., Hindi, Kannada, Tamil, etc.).
It uses a sequence-to-sequence (Seq2Seq) deep learning architecture built with LSTM and Bahdanau Attention mechanisms, implemented in PyTorch.

The system learns to transliterate Indian names accurately from datasets such as Aksharantar, a benchmark dataset released by AI4Bharat.

## Key Features

✅ End-to-end Seq2Seq model using LSTM Encoder–Decoder
✅ Bahdanau Attention for contextual alignment
✅ Dynamic padding and batching using custom collate_fn
✅ Command-line flexibility for architecture & parameters
✅ GPU-accelerated (if CUDA/NVIDIA GPU is available)
✅ Fully modular codebase — easy to extend for new languages

## Directory Structure
'''
aksharantar-transliteration_final/
│
├── main.py                         # Entry point for training & validation
├── utils.py                        # Data processing, vocabulary, dataset loaders
├── models/
│   ├── encoder.py                  # Encoder (LSTM/GRU)
│   ├── decoder.py                  # Decoder with Attention
│   ├── seq2seq.py                  # Seq2Seq wrapper integrating encoder-decoder
│
├── data/
│   └── hin_train.csv, hin_valid.csv, etc.  # Aksharantar dataset
│
└── README.md                       # (This file)
'''
## Workflow Explanation
Step 1. Data Preprocessing

The raw CSV dataset (hin_train.csv) contains parallel word pairs:

source,target
bindhya,बिन्द्या
shastragaar,शस्त्रागार


The script detects source–target columns automatically.

Builds character-level vocabularies for both languages.

Converts words to integer sequences for training.

Step 2. Model Architecture

A Sequence-to-Sequence (Seq2Seq) framework with Bahdanau Attention is used.

Encoder:

Converts input sequence into hidden state vectors.

Implemented via LSTM/GRU cells.

Attention Mechanism (Bahdanau):

Dynamically focuses on relevant input characters for each output character.

Decoder:

Generates the transliterated output step-by-step using encoder context + attention scores.

Step 3. Training Phase

During training:

Teacher forcing is used for faster convergence.

Cross-Entropy loss is minimized.

Model saves checkpoints and prints per-epoch accuracy.

Step 4. Inference / Testing

Once trained, the model predicts transliterations like:

Input: bindhya
Output: बिन्द्या

## Execution Commands
 Run on CPU (Default)
python main.py --lang hin --epochs 25 --cell lstm --attention

 Run on GPU (If NVIDIA CUDA available)
python main.py --lang hin --epochs 25 --cell lstm --attention


You’ll see:

 Using device: cuda

## Environment Setup
1️. Clone the Repository
git clone https://github.com/<your-username>/aksharantar-transliteration_final.git
cd aksharantar-transliteration_final

2️. Install Dependencies
pip install -r requirements.txt

3️. Verify GPU Availability
python -c "import torch; print(torch.cuda.is_available())"


If this prints True, your system supports CUDA.
Otherwise, it will default to CPU mode.

## GPU Compatibility Analysis (Personal System)

During execution, GPU verification returned:

python -c "import torch; print(torch.cuda.is_available())"
False

## Hardware Inspection:
Get-WmiObject win32_VideoController | Select-Object Name


## Output:

AMD Radeon RX 640 Series
Intel(R) UHD Graphics

## Explanation

CUDA (Compute Unified Device Architecture) is exclusive to NVIDIA GPUs.

AMD and Intel integrated GPUs do not support CUDA.

Hence, PyTorch automatically falls back to CPU mode.

Despite installing torch-2.5.1+cu121, it cannot use CUDA since no NVIDIA GPU is detected.

### Alternatives for GPU Training
## Option 1 — Google Colab

Use Colab’s free NVIDIA Tesla GPU.

### Steps:

!git clone https://github.com/<your-username>/aksharantar-transliteration_final.git
%cd aksharantar-transliteration_final
!python main.py --lang hin --epochs 25 --cell lstm --attention


You’ll see:

 Using device: cuda

## Option 2 — PyTorch with DirectML (for AMD GPU)

If you want to use your AMD GPU on Windows:

pip uninstall torch torchvision torchaudio -y
pip install torch-directml


Then:

import torch_directml
device = torch_directml.device()
print(device)


Note: This is experimental and slower than CUDA.

## Expected Results
Device	Approx. Training Time (25 epochs)	Accuracy
CPU (i7)	45–60 min	~92–94%
Colab GPU (T4)	8–12 min	~94–96%
Local NVIDIA GPU	5–8 min	~95–97%
## Model Configuration
Parameter	Description
--lang	Dataset language (e.g., hin, kan, tam)
--epochs	Number of training epochs
--cell	RNN cell type (lstm or gru)
--attention	Enable Bahdanau attention
--batch_size	Default 64
--hidden_dim	Default 256
### Example Output Log
 Using device: cpu
 Detected columns: source='bindhya', target='बिन्द्या'
 Loaded 51199 training and 4095 validation samples
Epoch [1/25] | Train Loss: 2.190 | Val Acc: 68.7%
Epoch [25/25] | Train Loss: 0.071 | Val Acc: 94.3%
Model saved successfully to checkpoints/hin_model.pt

### Conclusion

This project demonstrates the power of deep learning in low-resource Indian language processing.
Even on a CPU-only machine, the model effectively learns complex transliteration patterns.

Although GPU acceleration (CUDA) wasn’t supported on the current AMD system, the same code seamlessly executes on Google Colab or any NVIDIA-enabled system, achieving higher efficiency and performance.

### Author

### Navyashree N
B.E. Computer Science & Engineering
Passionate about AI, Deep Learning, and NLP for Indian languages 🇮🇳
=======
# aksharantar-seq2seq
Sequence-to-Sequence (RNN/LSTM/GRU) model for Roman to Devanagari character-level transliteration using the Aksharantar dataset
>>>>>>> 899d1bca8f5ebb8e4c63fda15c4fb35e5f3843d5
