import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def _ensure_dir(path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)


def plot_metric_comparison(results: dict, save_path: str):
    """Bar plots for PR-AUC and F2 across models for each horizon."""
    _ensure_dir(save_path)

    models = [m for m in results.keys() if m not in ("statistical_comparisons", "best_models")]
    if not models:
        return

    first = models[0]
    horizon_keys = [k for k in results[first].keys() if k.startswith("horizon_")]
    if not horizon_keys:
        return

    for hk in horizon_keys:
        names, pr, f2 = [], [], []
        for m in models:
            if hk not in results[m]:
                continue
            names.append(m)
            pr.append(results[m][hk].get("pr_auc", 0.0))
            f2.append(results[m][hk].get("f2_score", 0.0))

        if not names:
            continue

        x = np.arange(len(names))
        width = 0.35

        plt.figure()
        plt.bar(x - width / 2, pr, width, label="PR-AUC")
        plt.bar(x + width / 2, f2, width, label="F2")
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel("Score")
        plt.title(f"Model comparison ({hk})")
        plt.legend()
        plt.tight_layout()

        out = save_path.replace(".png", f"_{hk}.png")
        plt.savefig(out, dpi=150)
        plt.close()


def plot_learning_curves_comparison(learning_curves: dict, save_path: str):
    """Placeholder plot so imports never fail."""
    _ensure_dir(save_path)
    plt.figure()
    plt.title("Learning curves (placeholder)")
    plt.xlabel("Epoch / iteration")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_statistical_significance(stat_results: dict, save_path: str):
    """Minimal plot: which McNemar pairs are significant (1) vs not (0)."""
    _ensure_dir(save_path)

    mcn = stat_results.get("mcnemar_tests", {})
    if not mcn:
        return

    labels = list(mcn.keys())
    sig = [1 if mcn[k].get("significant") else 0 for k in labels]

    plt.figure()
    plt.bar(np.arange(len(labels)), sig)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks([0, 1], ["No", "Yes"])
    plt.title("McNemar significance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
