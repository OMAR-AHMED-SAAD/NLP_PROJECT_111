import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from .metrics import *
from typing import Dict, Tuple
import numpy as np
from .bpe_tokenizer import BPETokenizer
import pickle
import os
import copy

def train_qa_context_model_boilerplate(model: nn.Module,
                                      train_dataloader: DataLoader,
                                        criterion: nn.Module,
                                        optimizer: optim.Optimizer,
                                        device: torch.device,
                                        num_epochs: int,
                                        inputs: List[str],
                                        val_dataloader: DataLoader=None,
                                        evaluate_val_dataset: bool=False) -> Tuple[List[float], List[float]]:
    """
    Train and evaluate the QA context model.

    Args:

        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (CPU or GPU).
        num_epochs (int): Number of epochs to train for.
        inputs (List[str]): List of input tensor names.
        val_dataloader (DataLoader, optional): DataLoader for validation data.
        evaluate_val_dataset (bool): Whether to evaluate on the validation dataset after each epoch.

    Returns:
        Tuple[List[float], List[float]]: Training and validation losses.
        """
    model.to(device)
    train_losses = []
    val_losses = []

    # Early stopping & LR scheduling
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    scheduler = None
    if evaluate_val_dataset:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            inputs_tensors = {input_name: batch[input_name].to(device) for input_name in inputs}
            answer_start = batch['answer_start'].contiguous().view(-1).to(device)
            answer_end = batch['answer_end'].contiguous().view(-1).to(device)
            optimizer.zero_grad()

            # Forward pass
            start_logits, end_logits = model(**inputs_tensors)

            start_logits = start_logits.contiguous().view(-1, start_logits.size(-1))
            end_logits = end_logits.contiguous().view(-1, end_logits.size(-1))

            loss_start = criterion(start_logits, answer_start)
            loss_end = criterion(end_logits, answer_end)
            loss = loss_start + loss_end

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")
        
        train_loss, train_metrics = evaluate_qa_context_model_boilerplate(model, train_dataloader, criterion, device, inputs=inputs, prefix_str="Training")
        train_losses.append(avg_epoch_loss)
        if evaluate_val_dataset and val_dataloader is not None:
            val_loss, val_metrics = evaluate_qa_context_model_boilerplate(model, val_dataloader, criterion, device, inputs=inputs) 
            val_losses.append(val_loss)
            # update the learning rate scheduler
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"Validation loss improved to {best_val_loss:.4f}. Saving model state.")
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve. No improvement for {epochs_no_improve} epochs.")
                if epochs_no_improve >= 5:
                    print("Early stopping triggered.")
                    break
        else:
            val_losses.append(None) # if no validation data, append None
        
        print('-'*50)

    
    # Load the best model 
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model state.")
    
    return train_losses, val_losses

def evaluate_qa_context_model_boilerplate(model: nn.Module, 
                                        dataloader: DataLoader,
                                        criterion: nn.Module,
                                        device: torch.device,
                                        inputs: List[str],
                                        prefix_str: str="Validation") -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the QA context model.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on (CPU or GPU).
        inputs (List[str]): List of input tensor names.
        prefix_str (str): Prefix string for logging.

    Returns:
        Tuple[float, Dict[str, float]]: Validation loss and evaluation metrics.
    """
    model.eval()
    total_loss = 0
    y_true_start = []
    y_true_end = []
    y_pred_start = []
    y_pred_end = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs_tensors = {input_name: batch[input_name].to(device) for input_name in inputs}
            answer_start = batch['answer_start'].contiguous().view(-1).to(device)
            answer_end = batch['answer_end'].contiguous().view(-1).to(device)

            start_logits, end_logits = model(**inputs_tensors)

            start_logits = start_logits.contiguous().view(-1, start_logits.size(-1))
            end_logits = end_logits.contiguous().view(-1, end_logits.size(-1))
            
            loss_start = criterion(start_logits, answer_start)
            loss_end = criterion(end_logits, answer_end)
            loss = loss_start + loss_end

            total_loss += loss.item()

            # Collect predictions and true values
            y_true_start.extend(answer_start.cpu().numpy())
            y_true_end.extend(answer_end.cpu().numpy())
            y_pred_start.extend(start_logits.argmax(dim=-1).cpu().numpy())
            y_pred_end.extend(end_logits.argmax(dim=-1).cpu().numpy())

    total_loss /= len(dataloader)
    print(f"{prefix_str} Loss: {total_loss:.4f}")
    
    metrics = evaluate_qa_predictions(pred_start=np.array(y_pred_start),
                                      pred_end=np.array(y_pred_end),
                                      gt_start=np.array(y_true_start),
                                      gt_end=np.array(y_true_end))
    print(f"{prefix_str} Metrics: {metrics}")
    
    return total_loss, metrics

def predict_qa_context_model_boilerplate(model: nn.Module,
                             dataloader: DataLoader,
                             tokenizer: BPETokenizer,
                             inputs: List[str],
                             device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts and decodes the answer spans from the QA model.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        tokenizer (BPETokenizer): Tokenizer for the model.
        inputs (List[str]): List of input tensor names.
        device (torch.device): Device to evaluate on (CPU or GPU).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted and true labels.
    """
    model.eval()
    preds = []
    contexts = []
    questions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs_tensors = {input_name: batch[input_name].to(device) for input_name in inputs}
            context = batch['context'].to(device)
            question = batch['question'].to(device)
            answer = batch['answer'].to(device)

            start_logits, end_logits = model(**inputs_tensors)

            y_pred_start = start_logits.argmax(dim=-1).cpu().numpy()  # Shape: (batch_size, seq_len)
            y_pred_end = end_logits.argmax(dim=-1).cpu().numpy()        # Shape: (batch_size, seq_len)

            context = context.cpu().numpy()  # Shape: (batch_size, seq_len)
            question = question.cpu().numpy()

            for i, (start_idx, end_idx) in enumerate(zip(y_pred_start, y_pred_end)):
                if start_idx > end_idx:
                    answer_tokens = []
                else:
                    answer_tokens = context[i, start_idx:end_idx + 1].tolist()
                decoded_answer_pred = tokenizer.decode(answer_tokens)
                decoded_answer_true = tokenizer.decode(answer[i].cpu().numpy().tolist())
                decoded_context = tokenizer.decode(context[i].tolist())
                decoded_question = tokenizer.decode(question[i].tolist())
                contexts.append(decoded_context)
                questions.append(decoded_question)
                true_labels.append(decoded_answer_true)
                preds.append(decoded_answer_pred)
    return np.array(preds), np.array(true_labels), np.array(contexts), np.array(questions)
