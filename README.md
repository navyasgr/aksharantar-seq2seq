

#  Character-Level Transliteration: A Sequence-to-Sequence (Seq2Seq) Model

## Project Goal

This project is a submission for the Technical Aptitude & Problem Solving Round. The objective is to implement a character-level **Sequence-to-Sequence (Seq2Seq)** model using a Recurrent Neural Network (RNN) to solve Roman-to-Devanagari transliteration.

The function $\mathbf{F}$ is trained to map an input sequence of Latin characters ($X_{\text{roman}}$) to the corresponding output sequence of Devanagari characters ($Y_{\text{devanagari}}$):

$$\mathbf{Y} = \mathbf{F}(\mathbf{X}) \quad \text{e.g., } \mathbf{\text{ajanabee}} \rightarrow \mathbf{\text{अजनबी}}$$

## 1. Problem-Solving and Engineering Narrative

My approach prioritizes robust software design and training efficiency, specifically tailored to overcome the constraint of **limited GPU/CPU resources** commonly found in cloud environments (like Colab/Kaggle). The entire model is built from scratch using PyTorch, adhering to the principles of modularity and flexibility.

### Architectural Overview

The model uses a standard Encoder-Decoder architecture:

1.  **Encoder RNN:** Processes the input Roman string and compresses it into a fixed-size **context vector** (the final hidden state).
2.  **Decoder RNN:** Uses the context vector as its initial state and generates the Devanagari output sequence one character at a time.
3.  **Flexibility:** The architecture is built to dynamically accept hyperparameters for the RNN `cell_type` (`RNN`, `LSTM`, `GRU`), `num_layers`, `hidden_size`, and `embedding_dim`.



### Engineering Trade-offs under Low-Resource Constraints

The following trade-offs were made to ensure stability and rapid convergence with minimal GPU memory:

| Challenge | Solution & Implementation | Justification (Self-Narrated Assumption) |
| :--- | :--- | :--- |
| **Computational Cost (FLOPs)** | Defaulted to a single-layer, unidirectional **GRU (Gated Recurrent Unit)** cell. | **Assumption:** GRU strikes the best balance between performance and computational efficiency (fewer parameters than LSTM) for sequence modeling tasks like this, which are sensitive to training time. |
| **Memory Inefficiency (Padding)** | Implemented **dynamic batching** where sequences of similar lengths are grouped and processed using PyTorch's `pack_padded_sequence`. | **Problem-Solving:** This is crucial. It minimizes the amount of 'dummy' computation performed on padding tokens, dramatically speeding up training and reducing memory pressure on the limited GPU. |
| **Vocabulary Size** | Used a **minimal character-level vocabulary** for both source and target. | **Justification:** Unlike a word-level model, a character vocabulary is orders of magnitude smaller, directly shrinking the $\mathbf{V} \times \mathbf{E}$ embedding layer and the $\mathbf{H} \times \mathbf{V}$ output layer, saving essential parameters. |
| **Exposure Bias / Generalization** | Utilized a **decaying Teacher Forcing schedule** during training. | **Justification:** While Teacher Forcing is fast, relying on ground truth too long leads to poor generalization. A planned decay forces the decoder to rely on its own predictions over time, making it more robust at inference. |

---

## 2. Technical Complexity Analysis (Required Questions)

The analysis below is provided for a single forward pass, assuming a **Vanilla RNN** cell ($L=1$), with the following parameters as requested:

* $E$: Input Embedding Size
* $H$: Hidden State Size
* $T$: Sequence Length (Input and Output)
* $V$: Vocabulary Size (Source and Target)

### A. Total Number of Parameters ($\Theta$)

This is the sum of trainable weights and biases across all components (Embedding, Encoder RNN, Decoder RNN, and Output Layer):

$$\mathbf{\text{Total Parameters} \approx V \cdot (E + 1) + 2H \cdot (E + H + 1)}$$

### B. Total Number of Computations (FLOPs)

This calculation represents the time complexity (number of Floating-Point Operations) for a single forward pass over $T$ time steps, focusing on dominant matrix multiplication terms:

$$\mathbf{\text{Total Computations} \approx 4T \cdot (E \cdot H + H^2 + H \cdot V)}$$

---

## 3. Project Structure

The project follows good software engineering practices with a modular file structure:
'''
aksharantar-seq2seq/ ├── data/ # Location for the extracted Aksharantar dataset sample ├── models/ # Saved model checkpoints ├── main.py # Main execution script, training loop, and evaluation ├── utils.py # Data preparation, tokenization, and custom DataLoader logic ├── encoder.py # Defines the configurable RNN Encoder (supports RNN, LSTM, GRU) ├── decoder.py # Defines the configurable RNN Decoder (supports RNN, LSTM, GRU) ├── model.py # Seq2Seq wrapper integrating Encoder and Decoder └── requirements.txt # Project dependencies (PyTorch, numpy, etc.)
'''
## 4. Execution

### Prerequisites

# Recommended environment: Python 3.x
pip install -r requirements.txt
That was my mistake; I focused on completing the technical code files while you were asking for the complete README.md content.

Here is the complete, final README.md content designed to be copy-pastable and to showcase your problem-solving skills and constraints-based engineering decisions, exactly as requested.

## Character-Level Transliteration: A Sequence-to-Sequence (Seq2Seq) Model

## Project Goal

This project is a submission for the Technical Aptitude & Problem Solving Round. The objective is to implement a character-level **Sequence-to-Sequence (Seq2Seq)** model using a Recurrent Neural Network (RNN) to solve Roman-to-Devanagari transliteration.

The function $\mathbf{F}$ is trained to map an input sequence of Latin characters ($X_{\text{roman}}$) to the corresponding output sequence of Devanagari characters ($Y_{\text{devanagari}}$):

$$\mathbf{Y} = \mathbf{F}(\mathbf{X}) \quad \text{e.g., } \mathbf{\text{ajanabee}} \rightarrow \mathbf{\text{अजनबी}}$$

## 1. Problem-Solving and Engineering Narrative

My approach prioritizes robust software design and training efficiency, specifically tailored to overcome the constraint of **limited GPU/CPU resources** commonly found in cloud environments (like Colab/Kaggle). The entire model is built from scratch using PyTorch, adhering to the principles of modularity and flexibility.

### Architectural Overview

The model uses a standard Encoder-Decoder architecture:

1.  **Encoder RNN:** Processes the input Roman string and compresses it into a fixed-size **context vector** (the final hidden state).
2.  **Decoder RNN:** Uses the context vector as its initial state and generates the Devanagari output sequence one character at a time.
3.  **Flexibility:** The architecture is built to dynamically accept hyperparameters for the RNN `cell_type` (`RNN`, `LSTM`, `GRU`), `num_layers`, `hidden_size`, and `embedding_dim`.



### Engineering Trade-offs under Low-Resource Constraints

The following trade-offs were made to ensure stability and rapid convergence with minimal GPU memory:

| Challenge | Solution & Implementation | Justification (Self-Narrated Assumption) |
| :--- | :--- | :--- |
| **Computational Cost (FLOPs)** | Defaulted to a single-layer, unidirectional **GRU (Gated Recurrent Unit)** cell. | **Assumption:** GRU strikes the best balance between performance and computational efficiency (fewer parameters than LSTM) for sequence modeling tasks like this, which are sensitive to training time. |
| **Memory Inefficiency (Padding)** | Implemented **dynamic batching** where sequences of similar lengths are grouped and processed using PyTorch's `pack_padded_sequence`. | **Problem-Solving:** This is crucial. It minimizes the amount of 'dummy' computation performed on padding tokens, dramatically speeding up training and reducing memory pressure on the limited GPU. |
| **Vocabulary Size** | Used a **minimal character-level vocabulary** for both source and target. | **Justification:** Unlike a word-level model, a character vocabulary is orders of magnitude smaller, directly shrinking the $\mathbf{V} \times \mathbf{E}$ embedding layer and the $\mathbf{H} \times \mathbf{V}$ output layer, saving essential parameters. |
| **Exposure Bias / Generalization** | Utilized a **decaying Teacher Forcing schedule** during training. | **Justification:** While Teacher Forcing is fast, relying on ground truth too long leads to poor generalization. A planned decay forces the decoder to rely on its own predictions over time, making it more robust at inference. |

---

## 2. Technical Complexity Analysis (Required Questions)

The analysis below is provided for a single forward pass, assuming a **Vanilla RNN** cell ($L=1$), with the following parameters as requested:

* $E$: Input Embedding Size
* $H$: Hidden State Size
* $T$: Sequence Length (Input and Output)
* $V$: Vocabulary Size (Source and Target)

### A. Total Number of Parameters ($\Theta$)

This is the sum of trainable weights and biases across all components (Embedding, Encoder RNN, Decoder RNN, and Output Layer):

$$\mathbf{\text{Total Parameters} \approx V \cdot (E + 1) + 2H \cdot (E + H + 1)}$$

### B. Total Number of Computations (FLOPs)

This calculation represents the time complexity (number of Floating-Point Operations) for a single forward pass over $T$ time steps, focusing on dominant matrix multiplication terms:

$$\mathbf{\text{Total Computations} \approx 4T \cdot (E \cdot H + H^2 + H \cdot V)}$$

---

## 3. Project Structure

The project follows good software engineering practices with a modular file structure:
'''
aksharantar-seq2seq/
├── data/
│   └── aksharantar_sampled/  # Extracted Aksharantar dataset files
├── models/                   # Saved model checkpoints and artifacts
├── main.py                   # Main execution script, training loop, and evaluation
├── utils.py                  # Data loading, preprocessing, tokenization, and custom DataLoader logic
├── encoder.py                # Defines the configurable RNN Encoder (supports RNN, LSTM, GRU)
├── decoder.py                # Defines the configurable RNN Decoder (supports RNN, LSTM, GRU)
├── model.py                  # Seq2Seq wrapper integrating Encoder and Decoder
└── requirements.txt          # Project dependencies (PyTorch, numpy, etc.)
'''
## 4. Execution

### Prerequisites

# Recommended environment: Python 3.x
pip install -r requirements.txt
Example Run
The model can be launched with custom configuration. The default settings reflect the efficient choices made for low-resource environments:
python main.py --hidden_size 256 --num_layers 1 --cell_type GRU --batch_size 64
