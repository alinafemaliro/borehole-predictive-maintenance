import numpy as np


def calculate_recall_at_k(y_true, y_score, k_percent: int) -> float:
    """
    Recall@K%: among the top K% highest predicted risk,
    what proportion of the true positives are captured?

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1)
    y_score : array-like
        Predicted probabilities/scores
    k_percent : intspython -c "from src.modeling.model_comparator import ModelComparator; print('OK import ModelComparator')"

        e.g. 10 means top 10%

    Returns
    -------
    float
        Recall@K% in [0, 1]
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n = len(y_true)
    if n == 0:
        return 0.0

    positives = int(y_true.sum())
    if positives == 0:
        return 0.0

    k = max(1, int(np.floor(n * (int(k_percent) / 100.0))))

    top_idx = np.argsort(y_score)[::-1][:k]
    captured = int(y_true[top_idx].sum())

    return float(captured / positives)
