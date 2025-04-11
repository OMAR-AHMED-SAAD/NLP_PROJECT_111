import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from .metrics import evaluate_model
from typing import Dict, Tuple
import numpy as np

def train_seq2seq(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int
) -> None:
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            question = batch['question'].to(device)
            answer = batch['answer'].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(question, answer[:, :-1])  # Exclude <eos> token in target
            output_dim = output.shape[-1]

            # Reshape for loss calculation
            output = output.contiguous().view(-1, output_dim)
            trg = answer[:, 1:].contiguous().view(-1)  # Exclude <sos> token in target

            loss = criterion(output, trg)
            loss.backward()

            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

def evaluate_seq2seq(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:

    model.to(device)
    model.eval()

    total_loss = 0
    y_true = []
    y_true_mask = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            question = batch['question'].to(device)
            answer = batch['answer'].to(device)
            attention_mask_answer = batch['attention_mask_answer'].to(device)

            output = model(question, answer[:, :-1])  # Exclude <eos> token in target
            output_dim = output.shape[-1]

            # Reshape for loss calculation
            output = output.contiguous().view(-1, output_dim)
            trg = answer[:, 1:].contiguous().view(-1)  # Exclude <sos> token in target
            y_true.extend(trg.cpu().numpy())
            y_pred.extend(output.argmax(dim=-1).cpu().numpy())
            y_true_mask.extend(attention_mask_answer[:, 1:].contiguous().view(-1).cpu().numpy())

            loss = criterion(output, trg)
            total_loss += loss.item()

    total_loss /= len(dataloader)
    print(f"Validation Loss: {total_loss:.4f}")
    y_true_mask = np.array(y_true_mask).astype(bool)
    y_true = np.array(y_true)[y_true_mask]
    y_pred = np.array(y_pred)[y_true_mask]
    metrics = evaluate_model(y_true, y_pred)
    print(f"Validation Metrics: {metrics}")
    return total_loss, metrics