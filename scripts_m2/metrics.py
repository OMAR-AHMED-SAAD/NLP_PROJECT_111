from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def evaluate_model(y_true: np.array, y_pred: np.array):
    """
    Evaluate the performance of a model using common classification metrics.

    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
        pad_token (int): The token used for padding. Default is 1.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted')
    }
    return metrics


def calculate_bleu_score(references: list[str], candidates: list[str]):
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
