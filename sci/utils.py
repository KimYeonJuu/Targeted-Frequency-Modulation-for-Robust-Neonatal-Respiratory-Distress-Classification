import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve

def parse_hw(s: str):
    if s is None or s == '' or s.lower() == 'same':
        return None
    s = s.lower()
    if 'x' in s:
        a, b = s.split('x')
        return (int(a), int(b))
    v = int(s)
    return (v, v)

def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy.
    
    Args:
        y_true: Ground-truth labels (list or numpy array)
        y_pred: Predicted labels (list or numpy array)
    
    Returns:
        float: Accuracy value
    """
    return accuracy_score(y_true, y_pred)

def compute_f1(y_true, y_prob, eps=1e-8):
    """
    Compute F1 using the threshold that maximizes the precision-recall F1 score.
    
    Args:
        y_true: Ground-truth labels (list or numpy array)
        y_prob: Predicted probabilities for the positive class (list or numpy array)
    
    Returns:
        tuple:
            best_f1 (float): Best F1 score
            best_threshold (float): Threshold that achieves the best score
    """
    # Get precision, recall, and thresholds from the precision-recall curve.
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has one fewer element than precision/recall, so use precision[:-1] and recall[:-1].
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + eps)

    # Index of the maximum F1 score.
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # Generate predictions at the selected threshold and compute sklearn F1.
    y_pred_opt = (np.array(y_prob) >= best_threshold).astype(int)
    return f1_score(y_true, y_pred_opt)

def compute_auroc(y_true, y_prob):
    """
    Compute AUROC (area under the ROC curve).
    
    Args:
        y_true: Ground-truth labels (list or numpy array)
        y_prob: Predicted probabilities for the positive class (list or numpy array)
    
    Returns:
        float: AUROC value
    """
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError as e:
        # Handle degenerate cases such as a single observed class.
        print(f"Error while computing AUROC: {e}")
        return 0.5  # Return a conservative default.

def compute_auprc(y_true, y_prob):
    """
    Compute AUPRC (area under the precision-recall curve).
    
    Args:
        y_true: Ground-truth labels (list or numpy array)
        y_prob: Predicted probabilities for the positive class (list or numpy array)
    
    Returns:
        float: AUPRC value
    """
    try:
        return average_precision_score(y_true, y_prob)
    except ValueError as e:
        # Handle degenerate cases such as a single observed class.
        print(f"Error while computing AUPRC: {e}")
        return 0.0  # Return a conservative default.
