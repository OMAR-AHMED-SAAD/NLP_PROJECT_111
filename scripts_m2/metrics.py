from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from typing import List, Dict, Union

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Evaluate the performance of a model using common classification metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        pad_token (int): The token used for padding. Default is 1.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    eval_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return eval_metrics


def calculate_bleu_score(references: List[str], candidates: List[str]):
    """
    Calculate the BLEU score for a candidate sentence against a reference sentence.

    Args:
        references (list[str]): The reference sentence.
        candidates (list[str]): The candidate sentence.

    Returns:
        float: The BLEU score.
    """
    for i in range(len(references)):
        references[i] = references[i].split()
    for i in range(len(candidates)):
        candidates[i] = candidates[i].split()
    return sentence_bleu(references, candidates)


def span_f1(pred_starts: np.ndarray, pred_ends: np.ndarray, gt_starts: np.ndarray, gt_ends: np.ndarray):
    """
    Compute the average token-level F1 score between predicted and ground truth spans.
    
    Args:
        pred_starts (np.ndarray): Array of predicted start indices.
        pred_ends (np.ndarray): Array of predicted end indices.
        gt_starts (np.ndarray): Array of ground truth start indices.
        gt_ends (np.ndarray): Array of ground truth end indices.
    
    Returns:
        float: Average span F1 score across all samples.
    """
    f1_scores = []
    
    for p_start, p_end, g_start, g_end in zip(pred_starts, pred_ends, gt_starts, gt_ends):
        # Construct sets of token indices
        pred_range = set(range(p_start, p_end + 1))
        gt_range = set(range(g_start, g_end + 1))
        
        if len(pred_range) == 0 or len(gt_range) == 0:
            f1 = 0.0
        else:
            # Calculate intersection size
            intersection = len(pred_range.intersection(gt_range))
            precision = intersection / len(pred_range)
            recall = intersection / len(gt_range)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)



def evaluate_qa_predictions(pred_start: np.ndarray,
                            pred_end: np.ndarray,
                            gt_start: np.ndarray,
                            gt_end: np.ndarray) -> dict:
    """
    Calculate evaluation metrics (accuracy, precision, recall, F1) for QA predictions.

    Args:
        pred_start (np.ndarray): Predicted start indices.
        pred_end (np.ndarray): Predicted end indices.
        gt_start (np.ndarray): Ground truth start indices.
        gt_end (np.ndarray): Ground truth end indices.
    Returns:
        dict: A dictionary containing metrics for the start predictions, end predictions, 
              and a joint exact match metric (both start and end correct).
    """
    
    start_metrics = evaluate_model(gt_start, pred_start)

    end_metrics = evaluate_model(gt_end, pred_end)
    
    # Compute a joint exact match metric:
    # A sample is considered correct only if both predicted start and end match the ground truth.
    joint_correct = np.sum((pred_start == gt_start) & (pred_end == gt_end))
    joint_exact_match = joint_correct / len(gt_start)

    span_overlap_f1 = span_f1(pred_start, pred_end, gt_start, gt_end)
    
    metrics = {
        "start_accuracy": start_metrics["accuracy"],
        "start_precision": start_metrics["precision"],
        "start_recall": start_metrics["recall"],
        "start_f1_score": start_metrics["f1_score"],
        "end_accuracy": end_metrics["accuracy"],
        "end_precision": end_metrics["precision"],
        "end_recall": end_metrics["recall"],
        "end_f1_score": end_metrics["f1_score"],
        "joint_exact_match": joint_exact_match,
        "span_overlap_f1": span_overlap_f1
    }
    
    return metrics

