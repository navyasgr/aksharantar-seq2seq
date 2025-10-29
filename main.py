import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import (
    build_vocab,
    load_language_pairs,
    TransliterationDataset,
    collate_fn
)
from models.seq2seq import Seq2Seq
import argparse
import os

# ===============================================
#  Argument Parsing
# ===============================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., hin)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--cell", type=str, choices=["rnn", "gru", "lstm"], default="lstm")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=30)
    return parser.parse_args()

# ===============================================
#  Training Loop
# ===============================================
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ===============================================
#  Validation Loop
# ===============================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0)
            loss = criterion(output[:, 1:].reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ===============================================
#  Main Entry
# ===============================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    train_path = os.path.join("data", f"aksharantar_sampled/{args.lang}/{args.lang}_train.csv")
    valid_path = os.path.join("data", f"aksharantar_sampled/{args.lang}/{args.lang}_valid.csv")

    train_df, valid_df, src_vocab, tgt_vocab, pairs = load_language_pairs(train_path, valid_path)
    print(f"✅ Loaded {len(train_df)} training and {len(valid_df)} validation samples")

    train_dataset = TransliterationDataset(train_df, src_vocab, tgt_vocab, args.max_len)
    valid_dataset = TransliterationDataset(valid_df, src_vocab, tgt_vocab, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Seq2Seq(len(src_vocab), len(tgt_vocab), args.hidden_size, cell_type=args.cell, use_attention=args.attention).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        print(f"📘 Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")

    torch.save(model.state_dict(), f"checkpoints/model_{args.lang}.pth")
    print("✅ Training complete. Model saved!")

if __name__ == "__main__":
    main()
