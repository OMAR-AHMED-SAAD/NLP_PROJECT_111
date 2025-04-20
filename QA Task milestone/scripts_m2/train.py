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

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.4f}")
        
        train_loss, train_metrics = evaluate_qa_context_model_boilerplate(model, train_dataloader, criterion, device, inputs=inputs, prefix_str="Training")
        train_losses.append(train_loss)
        if evaluate_val_dataset and val_dataloader is not None:
            val_loss, val_metrics = evaluate_qa_context_model_boilerplate(model, val_dataloader, criterion, device, inputs=inputs) 
            val_losses.append(val_loss)
        print('-'*50)
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

def train_qa_context_model(model: nn.Module,
                           train_dataloader: DataLoader,
                           criterion: nn.Module,
                           optimizer: optim.Optimizer,
                           device: torch.device,
                           num_epochs: int,
                           val_dataloader: DataLoader=None,
                           evaluate_val_dataset: bool=False) -> None:
    """
    Train and evaluate the QA context model.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (CPU or GPU).
        num_epochs (int): Number of epochs to train for.
        val_dataloader (DataLoader, optional): DataLoader for validation data.
        evaluate_val_dataset (bool): Whether to evaluate on the validation dataset after each epoch.

    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            context_question = batch['context_question'].to(device)
            answer_start = batch['answer_start'].contiguous().view(-1).to(device)
            answer_end = batch['answer_end'].contiguous().view(-1).to(device)
            optimizer.zero_grad()

            # Forward pass
            start_logits, end_logits = model(context_question)

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

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.4f}")
        
        train_metrics = evaluate_qa_context_model(model, train_dataloader, criterion, device, prefix_str="Training")
        if evaluate_val_dataset and val_dataloader is not None:
            val_metrics = evaluate_qa_context_model(model, val_dataloader, criterion, device) 
        print('-'*50)

def evaluate_qa_context_model(model: nn.Module, 
                              dataloader: DataLoader,
                              criterion: nn.Module,
                              device: torch.device,
                              prefix_str: str="Validation") -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the QA context model.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on (CPU or GPU).
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
            context_question = batch['context_question'].to(device)
            answer_start = batch['answer_start'].contiguous().view(-1).to(device)
            answer_end = batch['answer_end'].contiguous().view(-1).to(device)

            start_logits, end_logits = model(context_question)

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
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs_tensors = {input_name: batch[input_name].to(device) for input_name in inputs}
            context = batch['context'].to(device)
            answer = batch['answer'].to(device)

            start_logits, end_logits = model(**inputs_tensors)

            y_pred_start = start_logits.argmax(dim=-1).cpu().numpy()  # Shape: (batch_size, seq_len)
            y_pred_end = end_logits.argmax(dim=-1).cpu().numpy()        # Shape: (batch_size, seq_len)

            context = context.cpu().numpy()  # Shape: (batch_size, seq_len)

            for i, (start_idx, end_idx) in enumerate(zip(y_pred_start, y_pred_end)):
                if start_idx > end_idx:
                    answer_tokens = []
                else:
                    answer_tokens = context[i, start_idx:end_idx + 1].tolist()
                decoded_answer_pred = tokenizer.decode(answer_tokens)
                decoded_answer_true = tokenizer.decode(answer[i].cpu().numpy().tolist())
                true_labels.append(decoded_answer_true)
                preds.append(decoded_answer_pred)
    return np.array(preds), np.array(true_labels)


def predict_qa_context_model(model: nn.Module,
                             dataloader: DataLoader,
                             tokenizer: BPETokenizer,
                             device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts and decodes the answer spans from the QA model.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        tokenizer (BPETokenizer): Tokenizer for the model.
        device (torch.device): Device to evaluate on (CPU or GPU).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted and true labels.
    """
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            context_question = batch['context_question'].to(device)
            context = batch['context'].to(device)
            answer = batch['answer'].to(device)

            start_logits, end_logits = model(context_question)

            y_pred_start = start_logits.argmax(dim=-1).cpu().numpy()  # Shape: (batch_size, seq_len)
            y_pred_end = end_logits.argmax(dim=-1).cpu().numpy()        # Shape: (batch_size, seq_len)

            context = context.cpu().numpy()  # Shape: (batch_size, seq_len)

            for i, (start_idx, end_idx) in enumerate(zip(y_pred_start, y_pred_end)):
                if start_idx > end_idx:
                    answer_tokens = []
                else:
                    answer_tokens = context[i, start_idx:end_idx + 1].tolist()
                decoded_answer_pred = tokenizer.decode(answer_tokens)
                decoded_answer_true = tokenizer.decode(answer[i].cpu().numpy().tolist())
                true_labels.append(decoded_answer_true)
                preds.append(decoded_answer_pred)
    return np.array(preds), np.array(true_labels)



def save_model(model: nn.Module, model_path: str, save_again: bool=False) -> None:
    """
    Save the model to the specified path.

    Args:
        model (nn.Module): The model to save.
        model_path (str): Path to save the model.
        save_again (bool): Whether to overwrite the existing model file.
    """
    if os.path.exists(model_path) and not save_again:
        print(f"Model file {model_path} already exists. Skipping save.")
    else:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            print(f"Model saved to {model_path}")


def load_model(model_path: str) -> nn.Module:
    """
    Load the model from the specified path.

    Args:
        model_path (str): Path to load the model from.

    Returns:
        nn.Module: The loaded model.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        print(f"Model loaded from {model_path}")
    return model

def train_transformer_qa_model(model: nn.Module,
                               train_dataloader: DataLoader,
                               criterion: nn.Module,
                               optimizer: optim.Optimizer,
                               device: torch.device,
                               num_epochs: int,
                               val_dataloader: DataLoader=None,
                               evaluate_val_dataset: bool=False,
                               scheduler=None) -> None:
    """
    Train and evaluate the Transformer QA model.

    Args:
        model (nn.Module): The Transformer QA model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (CPU or GPU).
        num_epochs (int): Number of epochs to train for.
        val_dataloader (DataLoader, optional): DataLoader for validation data.
        evaluate_val_dataset (bool): Whether to evaluate on the validation dataset after each epoch.
        scheduler: Learning rate scheduler (optional).
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            context_question = batch['context_question'].to(device)
            attention_mask = batch['attention_mask_context_question'].to(device)
            type_mask = batch['context_question_type_mask'].to(device)
            answer_start = batch['answer_start'].contiguous().view(-1).to(device)
            answer_end = batch['answer_end'].contiguous().view(-1).to(device)
            
            optimizer.zero_grad()

            # Forward pass
            start_logits, end_logits = model(
                input_ids=context_question,
                attention_mask=attention_mask,
                token_type_ids=type_mask
            )

            start_logits = start_logits.contiguous().view(-1, start_logits.size(-1))
            end_logits = end_logits.contiguous().view(-1, end_logits.size(-1))

            loss_start = criterion(start_logits, answer_start)
            loss_end = criterion(end_logits, answer_end)
            loss = loss_start + loss_end

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader):.4f}")
        
        train_metrics = evaluate_transformer_qa_model(model, train_dataloader, criterion, device, prefix_str="Training")
        
        if evaluate_val_dataset and val_dataloader is not None:
            val_metrics = evaluate_transformer_qa_model(model, val_dataloader, criterion, device) 
        print('-'*50)


def evaluate_transformer_qa_model(model: nn.Module, 
                                 dataloader: DataLoader,
                                 criterion: nn.Module,
                                 device: torch.device,
                                 prefix_str: str="Validation") -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the Transformer QA model.

    Args:
        model (nn.Module): The Transformer QA model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on (CPU or GPU).
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
        for batch in tqdm(dataloader, desc=f"{prefix_str} Evaluation"):
            context_question = batch['context_question'].to(device)
            attention_mask = batch['attention_mask_context_question'].to(device)
            type_mask = batch['context_question_type_mask'].to(device)
            answer_start = batch['answer_start'].contiguous().view(-1).to(device)
            answer_end = batch['answer_end'].contiguous().view(-1).to(device)

            # Forward pass
            start_logits, end_logits = model(
                input_ids=context_question,
                attention_mask=attention_mask,
                token_type_ids=type_mask
            )

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
    
    metrics = evaluate_qa_predictions(
        pred_start=np.array(y_pred_start),
        pred_end=np.array(y_pred_end),
        gt_start=np.array(y_true_start),
        gt_end=np.array(y_true_end)
    )
    print(f"{prefix_str} Metrics: {metrics}")
    
    return total_loss, metrics


def predict_transformer_qa_model(model: nn.Module,
                                dataloader: DataLoader,
                                tokenizer: BPETokenizer,
                                device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts and decodes the answer spans from the Transformer QA model.

    Args:
        model (nn.Module): The Transformer QA model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        tokenizer (BPETokenizer): Tokenizer for the model.
        device (torch.device): Device to evaluate on (CPU or GPU).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted and true labels.
    """
    model.eval()
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            context_question = batch['context_question'].to(device)
            attention_mask = batch['attention_mask_context_question'].to(device)
            type_mask = batch['context_question_type_mask'].to(device)
            context = batch['context'].to(device)
            answer = batch['answer'].to(device)

            # Forward pass
            start_logits, end_logits = model(
                input_ids=context_question,
                attention_mask=attention_mask,
                token_type_ids=type_mask
            )

            y_pred_start = start_logits.argmax(dim=-1).cpu().numpy()
            y_pred_end = end_logits.argmax(dim=-1).cpu().numpy()
            context = context.cpu().numpy()

            for i, (start_idx, end_idx) in enumerate(zip(y_pred_start, y_pred_end)):
                if start_idx > end_idx:
                    # Invalid prediction, start index is after end index
                    answer_tokens = []
                else:
                    # Get the predicted answer tokens from the context
                    answer_tokens = context_question[i, start_idx:end_idx + 1].cpu().numpy().tolist()
                
                decoded_answer_pred = tokenizer.decode(answer_tokens)
                decoded_answer_true = tokenizer.decode(answer[i].cpu().numpy().tolist())
                
                true_labels.append(decoded_answer_true)
                preds.append(decoded_answer_pred)
                
    return np.array(preds), np.array(true_labels)
