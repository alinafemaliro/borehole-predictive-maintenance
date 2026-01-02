"""
Comprehensive model comparison for borehole predictive maintenance.
Compares multiple models across different prediction horizons and metrics.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

from ..evaluation.metrics_calculator import calculate_recall_at_k
from ..evaluation.statistical_tests import perform_mcnemar_test, perform_wilcoxon_test
from ..visualization.model_comparison_plots import (
    plot_metric_comparison,
    plot_learning_curves_comparison,
    plot_statistical_significance,
)


class ModelComparator:
    """Compare multiple models for predictive maintenance."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config["model_comparison"]
        self.models_to_compare = self.model_config["models_to_compare"]
        self.prediction_horizons = self.model_config["prediction_horizons"]

        self.results: Dict = {}
        self.trained_models: Dict[str, Any] = {}

        # Optional validation set for early stopping (xgboost, etc.)
        self.X_val = None
        self.y_val = None

    def run_comparison(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        logger.info("Starting model comparison")

        if X_val is not None and y_val is not None:
            self.X_val = X_val
            self.y_val = y_val

        comparison_results: Dict[str, Any] = {}

        for model_name in self.models_to_compare:
            logger.info(f"Training and evaluating {model_name}")
            model_results: Dict[str, Any] = {}

            for horizon in self.prediction_horizons:
                y_train_h = self._create_horizon_target(y_train, horizon)
                y_test_h = self._create_horizon_target(y_test, horizon)

                model = self._train_model(model_name, X_train, y_train_h)
                self.trained_models[f"{model_name}_{horizon}d"] = model

                metrics = self._evaluate_model(
                    model=model,
                    model_name=model_name,
                    X_test=X_test,
                    y_test=y_test_h,
                    feature_names=feature_names,
                )

                # Cross-validation (skip prophet for now)
                if model_name != "prophet":
                    cv_scores = self._perform_cross_validation(model_name, X_train, y_train_h)
                    metrics["cv_scores"] = cv_scores

                model_results[f"horizon_{horizon}"] = metrics

            comparison_results[model_name] = model_results

        # Statistical comparisons (uses first horizon)
        comparison_results["statistical_comparisons"] = self._compare_models_statistically(
            comparison_results, X_test, y_test
        )

        # Best models
        comparison_results["best_models"] = self._identify_best_models(comparison_results)

        # Save + plots
        self._save_comparison_results(comparison_results)
        self._generate_comparison_plots(comparison_results)

        self.results = comparison_results
        return comparison_results

    def _create_horizon_target(self, y_original: np.ndarray, horizon_days: int) -> np.ndarray:
        """
        Placeholder: convert your target into horizon-specific labels.
        For now returns y_original.
        """
        return y_original

    def _train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        if model_name == "xgboost":
            return self._train_xgboost(X_train, y_train)
        if model_name == "random_forest":
            return self._train_random_forest(X_train, y_train)
        if model_name == "logistic_regression":
            return self._train_logistic_regression(X_train, y_train)
        if model_name == "gradient_boosting":
            return self._train_gradient_boosting(X_train, y_train)
        if model_name == "lightgbm":
            return self._train_lightgbm(X_train, y_train)
        if model_name == "lstm":
            return self._train_lstm(X_train, y_train)
        if model_name == "prophet":
            return self._train_prophet(X_train, y_train)
        raise ValueError(f"Unknown model: {model_name}")

    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray):
        params = self.config["models"]["xgboost"]
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            objective=params["objective"],
            eval_metric=params["eval_metric"],
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        if self.X_val is not None and self.y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(self.X_val, self.y_val)],
                early_stopping_rounds=params.get("early_stopping_rounds", 50),
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        return model

    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        params = self.config["models"]["random_forest"]
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            class_weight=params["class_weight"],
            bootstrap=params["bootstrap"],
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray):
        params = self.config["models"]["logistic_regression"]
        model = LogisticRegression(
            penalty=params["penalty"],
            C=params["C"],
            class_weight=params["class_weight"],
            solver=params["solver"],
            max_iter=params["max_iter"],
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    def _train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray):
        params = self.config["models"]["gradient_boosting"]
        model = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    def _train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray):
        params = self.config["models"]["lightgbm"]
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            feature_fraction=params["feature_fraction"],
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        return model

    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray):
        params = self.config["models"]["lstm"]
        from tensorflow import keras

        seq_len = params["sequence_length"]
        X_seq = self._reshape_for_lstm(X_train, seq_len)
        y_seq = y_train[seq_len - 1 :]

        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(
                units=params["units"][0],
                return_sequences=True,
                input_shape=(seq_len, X_seq.shape[2]),
                dropout=params["dropout"],
                recurrent_dropout=params["recurrent_dropout"],
            )
        )
        model.add(
            keras.layers.LSTM(
                units=params["units"][1],
                dropout=params["dropout"],
                recurrent_dropout=params["recurrent_dropout"],
            )
        )
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dropout(params["dropout"]))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=params["patience"], restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        ]

        model.fit(
            X_seq,
            y_seq,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )

        return model

    def _train_prophet(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Placeholder: Prophet needs a dataframe with ds/y (timestamps + value).
        We'll keep this as a stub until you structure the time series properly.
        """
        from prophet import Prophet

        params = self.config["models"]["prophet"]
        model = Prophet(
            seasonality_mode=params["seasonality_mode"],
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            holidays_prior_scale=params["holidays_prior_scale"],
        )
        return model

    def _reshape_for_lstm(self, X: np.ndarray, sequence_length: int) -> np.ndarray:
        n_samples = X.shape[0] - sequence_length + 1
        n_features = X.shape[1]
        X_out = np.zeros((n_samples, sequence_length, n_features))
        for i in range(n_samples):
            X_out[i] = X[i : i + sequence_length]
        return X_out

    def _evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        # predictions
        if model_name == "lstm":
            seq_len = self.config["models"]["lstm"]["sequence_length"]
            X_seq = self._reshape_for_lstm(X_test, seq_len)
            y_proba = model.predict(X_seq, verbose=0).flatten()
            y_true = y_test[seq_len - 1 :]
        elif model_name == "prophet":
            # placeholder (until Prophet is wired to ds/y)
            y_proba = np.random.random(len(y_test))
            y_true = y_test
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
            y_true = y_test

        y_true = np.asarray(y_true).astype(int)
        y_pred = (y_proba > 0.5).astype(int)

        metrics: Dict[str, Any] = {}

        # Recall@K%
        metrics["recall_at_k"] = {}
        for k in self.model_config["evaluation_metrics"]["k_values"]:
            metrics["recall_at_k"][f"recall@{k}%"] = calculate_recall_at_k(y_true, y_proba, k)

        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        metrics["pr_auc"] = float(auc(recall, precision))

        # F2
        metrics["f2_score"] = float(fbeta_score(y_true, y_pred, beta=2))

        # Secondary metrics
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))
        metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["confusion_matrix"] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        if feature_names is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            metrics["feature_importance"] = dict(zip(feature_names, importances))

        metrics["model_complexity"] = self._calculate_model_complexity(model, model_name)
        return metrics

    def _calculate_model_complexity(self, model: Any, model_name: str) -> Dict:
        complexity: Dict[str, Any] = {}
        if model_name in ("xgboost", "random_forest", "lightgbm", "gradient_boosting"):
            if hasattr(model, "n_estimators"):
                complexity["n_estimators"] = int(model.n_estimators)
            if hasattr(model, "n_features_in_"):
                complexity["n_features"] = int(model.n_features_in_)
            if hasattr(model, "max_depth") and model.max_depth is not None:
                complexity["max_depth"] = int(model.max_depth)
        elif model_name == "lstm":
            complexity["trainable_params"] = int(model.count_params())
            complexity["layers"] = int(len(model.layers))
        return complexity

    def _perform_cross_validation(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict:
        cv = self.model_config["cross_validation"]
        # NOTE: test_size_days is treated as number of rows/samples in TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=cv["n_splits"], test_size=cv["test_size_days"])

        scores = {"recall_at_k": [], "pr_auc": [], "f2_score": []}
        k_for_cv = 10  # use 10% for CV recall@k%

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = self._train_model(model_name, X_tr, y_tr)

            if model_name == "lstm":
                seq_len = self.config["models"]["lstm"]["sequence_length"]
                X_val_seq = self._reshape_for_lstm(X_val, seq_len)
                y_proba = model.predict(X_val_seq, verbose=0).flatten()
                y_true = y_val[seq_len - 1 :]
            else:
                y_proba = model.predict_proba(X_val)[:, 1]
                y_true = y_val

            y_true = np.asarray(y_true).astype(int)
            y_pred = (y_proba > 0.5).astype(int)

            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            scores["pr_auc"].append(float(auc(recall, precision)))
            scores["recall_at_k"].append(float(calculate_recall_at_k(y_true, y_proba, k_for_cv)))
            scores["f2_score"].append(float(fbeta_score(y_true, y_pred, beta=2)))

        cv_out: Dict[str, Any] = {}
        for m, arr in scores.items():
            arr = np.asarray(arr, dtype=float)
            cv_out[m] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        return cv_out

    def _compare_models_statistically(self, comparison_results: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        horizon = self.prediction_horizons[0]
        preds = {}
        y_true_aligned = y_test

        for model_name in self.models_to_compare:
            if model_name == "prophet":
                continue
            key = f"{model_name}_{horizon}d"
            if key not in self.trained_models:
                continue

            model = self.trained_models[key]

            if model_name == "lstm":
                seq_len = self.config["models"]["lstm"]["sequence_length"]
                X_seq = self._reshape_for_lstm(X_test, seq_len)
                proba = model.predict(X_seq, verbose=0).flatten()
                y_true_aligned = y_test[seq_len - 1 :]
            else:
                proba = model.predict_proba(X_test)[:, 1]
                y_true_aligned = y_test

            preds[model_name] = (proba > 0.5).astype(int)

        model_names = list(preds.keys())
        mcnemar = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a, b = model_names[i], model_names[j]
                mcnemar[f"{a}_vs_{b}"] = perform_mcnemar_test(y_true_aligned, preds[a], preds[b])

        # Wilcoxon on CV means (if present); if not present, skip
        wilcoxon = {}
        for metric in ("pr_auc", "f2_score"):
            pairs = {}
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    a, b = model_names[i], model_names[j]
                    try:
                        a_cv = comparison_results[a][f"horizon_{horizon}"]["cv_scores"][metric]["mean"]
                        b_cv = comparison_results[b][f"horizon_{horizon}"]["cv_scores"][metric]["mean"]
                        # create small paired arrays (placeholder) - later you can store fold-level values
                        pairs[f"{a}_vs_{b}"] = perform_wilcoxon_test([a_cv] * 5, [b_cv] * 5)
                    except Exception:
                        continue
            wilcoxon[metric] = pairs

        return {"mcnemar_tests": mcnemar, "wilcoxon_tests": wilcoxon}

    def _identify_best_models(self, comparison_results: Dict) -> Dict:
        best_models: Dict[str, Any] = {}
        model_names = [m for m in self.models_to_compare if m in comparison_results]

        for horizon in self.prediction_horizons:
            hk = f"horizon_{horizon}"
            best_models[horizon] = {}

            # pr_auc
            best_models[horizon]["pr_auc"] = self._best_by_metric(comparison_results, model_names, hk, "pr_auc")
            # f2
            best_models[horizon]["f2_score"] = self._best_by_metric(comparison_results, model_names, hk, "f2_score")
            # recall@10%
            best_models[horizon]["recall_at_k"] = self._best_by_recall10(comparison_results, model_names, hk)

            # overall = best by f2_score
            best_models[horizon]["overall"] = best_models[horizon]["f2_score"]

        return best_models

    def _best_by_metric(self, results: Dict, models: List[str], horizon_key: str, metric: str) -> Dict:
        best_model = None
        best_score = -np.inf
        for m in models:
            if horizon_key not in results[m]:
                continue
            score = results[m][horizon_key].get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_model = m
        return {"model": best_model, "score": float(best_score)}

    def _best_by_recall10(self, results: Dict, models: List[str], horizon_key: str) -> Dict:
        best_model = None
        best_score = -np.inf
        for m in models:
            if horizon_key not in results[m]:
                continue
            recall_dict = results[m][horizon_key].get("recall_at_k", {})
            score = recall_dict.get("recall@10%", -np.inf)
            if score > best_score:
                best_score = score
                best_model = m
        return {"model": best_model, "score": float(best_score)}

    def _save_comparison_results(self, results: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("models/model_artifacts/comparison_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"model_comparison_{timestamp}.json"

        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(convert(results), f, indent=2)

        logger.info(f"Comparison results saved to {results_file}")

    def _generate_comparison_plots(self, results: Dict):
        # Metric comparison
        plot_metric_comparison(results, save_path="models/model_artifacts/metric_comparison.png")

        # Learning curves placeholder
        if hasattr(self, "learning_curves"):
            plot_learning_curves_comparison(
                self.learning_curves, save_path="models/model_artifacts/learning_curves.png"
            )

        # Statistical significance plot
        if "statistical_comparisons" in results:
            plot_statistical_significance(
                results["statistical_comparisons"],
                save_path="models/model_artifacts/statistical_significance.png",
            )
