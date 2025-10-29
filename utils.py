import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter

# =====================================================
# Vocabulary Builder
# =====================================================
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(list(text.strip()))
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for idx, char in enumerate(counter.keys(), start=4):
        vocab[char] = idx
    return vocab

# =====================================================
# Convert sentence → tensor
# =====================================================
def text_to_tensor(text, vocab, max_len):
    ids = [vocab.get(ch, vocab["<UNK>"]) for ch in list(text.strip())]
    ids = [vocab["<SOS>"]] + ids + [vocab["<EOS>"]]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)

# =====================================================
# Dataset
# =====================================================
class TransliterationDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab, max_len):
        self.df = df
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.iloc[idx]["english"]
        tgt = self.df.iloc[idx]["native"]
        src_tensor = text_to_tensor(src, self.src_vocab, self.max_len)
        tgt_tensor = text_to_tensor(tgt, self.tgt_vocab, self.max_len)
        return src_tensor, tgt_tensor

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = torch.stack(srcs)
    tgts = torch.stack(tgts)
    return srcs, tgts

# =====================================================
# Load data + build vocabs
# =====================================================
def load_language_pairs(train_path, valid_path):
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    if "english" not in train_df.columns:
        cols = train_df.columns.tolist()
        train_df.columns = ["english", "native"]
        valid_df.columns = ["english", "native"]
        print(f"✅ Detected columns: source='{train_df.iloc[0,0]}', target='{train_df.iloc[0,1]}'")

    src_texts = train_df["english"].astype(str).tolist() + valid_df["english"].astype(str).tolist()
    tgt_texts = train_df["native"].astype(str).tolist() + valid_df["native"].astype(str).tolist()

    src_vocab = build_vocab(src_texts)
    tgt_vocab = build_vocab(tgt_texts)

    return train_df, valid_df, src_vocab, tgt_vocab, list(zip(src_texts, tgt_texts))
