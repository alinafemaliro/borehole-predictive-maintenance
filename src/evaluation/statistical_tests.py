import numpy as np


def perform_mcnemar_test(y_true, pred_a, pred_b, alpha: float = 0.05) -> dict:
    """
    McNemar's test for paired binary classifiers.

    Builds 2x2 table of disagreements:
        b = count(a correct, b wrong)
        c = count(a wrong, b correct)

    Uses exact binomial test when possible.
    """
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)

    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    b = int(np.sum(a_correct & (~b_correct)))
    c = int(np.sum((~a_correct) & b_correct))

    n = b + c
    if n == 0:
        return {
            "b": b, "c": c,
            "p_value": 1.0,
            "significant": False,
            "note": "No disagreements; models identical on this sample."
        }

    # Exact binomial test: under H0, b ~ Binomial(n, 0.5)
    p_value = None
    try:
        from scipy.stats import binomtest
        p_value = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Fallback approximate chi-square with continuity correction
        # X^2 = (|b-c|-1)^2 / (b+c)
        x2 = (abs(b - c) - 1.0) ** 2 / n
        try:
            from scipy.stats import chi2
            p_value = float(1.0 - chi2.cdf(x2, df=1))
        except Exception:
            p_value = 1.0  # last-resort fallback

    return {
        "b": b,
        "c": c,
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "alpha": alpha
    }


def perform_wilcoxon_test(scores_a, scores_b, alpha: float = 0.05) -> dict:
    """
    Wilcoxon signed-rank test for paired samples (non-parametric).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b) or len(a) == 0:
        return {
            "p_value": 1.0,
            "significant": False,
            "note": "Invalid paired samples"
        }

    # If all equal, nothing to test
    if np.allclose(a, b):
        return {
            "p_value": 1.0,
            "significant": False,
            "note": "All paired differences are zero."
        }

    p_value = None
    try:
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p_value = float(p_value)
    except Exception:
        # fallback: simple sign test approximation
        diffs = a - b
        pos = int(np.sum(diffs > 0))
        neg = int(np.sum(diffs < 0))
        n = pos + neg
        if n == 0:
            p_value = 1.0
        else:
            try:
                from scipy.stats import binomtest
                p_value = float(binomtest(min(pos, neg), n=n, p=0.5, alternative="two-sided").pvalue)
            except Exception:
                p_value = 1.0

    return {
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "alpha": alpha
    }
