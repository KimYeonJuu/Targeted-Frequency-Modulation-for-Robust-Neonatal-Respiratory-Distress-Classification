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
    정확도(Accuracy)를 계산합니다.
    
    Args:
        y_true: 실제 레이블 (list or numpy array)
        y_pred: 예측 레이블 (list or numpy array)
    
    Returns:
        float: 정확도 값
    """
    return accuracy_score(y_true, y_pred)

def compute_f1(y_true, y_prob, eps=1e-8):
    """
    예측 확률에 대해 최적의 임계값(threshold)을 찾아 F1 스코어를 최대화합니다.
    
    Args:
        y_true: 실제 레이블 (list or numpy array)
        y_prob: 예측 확률 (list or numpy array) - positive class에 대한 확률
    
    Returns:
        tuple:
            best_f1 (float): 최적 F1 스코어
            best_threshold (float): 해당 스코어를 달성하는 임계값
    """
    # precision–recall 곡선에서 threshold, precision, recall 구하기
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds는 precision/recall의 마지막 값이 빠져 있으므로, [:–1]
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + eps)

    # 최댓값 인덱스
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # 해당 임계값으로 최종 예측 생성 → sklearn F1 사용
    y_pred_opt = (np.array(y_prob) >= best_threshold).astype(int)
    return f1_score(y_true, y_pred_opt)

def compute_auroc(y_true, y_prob):
    """
    AUROC (Area Under ROC Curve)를 계산합니다.
    
    Args:
        y_true: 실제 레이블 (list or numpy array)
        y_prob: 예측 확률 (list or numpy array) - positive class에 대한 확률
    
    Returns:
        float: AUROC 값
    """
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError as e:
        # 모든 레이블이 같은 클래스인 경우 등의 예외 처리
        print(f"AUROC 계산 중 오류 발생: {e}")
        return 0.5  # 기본값 반환

def compute_auprc(y_true, y_prob):
    """
    AUPRC (Area Under Precision-Recall Curve)를 계산합니다.
    
    Args:
        y_true: 실제 레이블 (list or numpy array)
        y_prob: 예측 확률 (list or numpy array) - positive class에 대한 확률
    
    Returns:
        float: AUPRC 값
    """
    try:
        return average_precision_score(y_true, y_prob)
    except ValueError as e:
        # 모든 레이블이 같은 클래스인 경우 등의 예외 처리
        print(f"AUPRC 계산 중 오류 발생: {e}")
        return 0.0  # 기본값 반환
