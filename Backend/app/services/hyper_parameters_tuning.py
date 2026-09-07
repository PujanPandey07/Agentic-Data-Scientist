import os
import time
import logging
import joblib
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


class HyperparameterTuningService:
    def tune(self, df, target_column, best_candidate, problem_type, dataset_id,
             max_trials=20, time_budget_seconds=120):
        logger.info(
            f"Starting tuning: algorithm={best_candidate.algorithm}, "
            f"max_trials={max_trials}, time_budget={time_budget_seconds}s"
        )

        X = df.drop(columns=[target_column])
        y = df[target_column]

        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Computed once, up front, rather than relying on each fold's fit()
        # to infer it — avoids the intermittent XGBoost "num_class should be
        # greater equal to 1" error seen across repeated Optuna trials.
        num_classes = int(pd.Series(y).nunique()
                          ) if problem_type == "classification" else None

        scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"

        study = optuna.create_study(direction="maximize")
        start_time = time.time()

        def objective(trial):
            model = self._suggest_model(
                trial, best_candidate.algorithm, problem_type, num_classes)
            scores = cross_val_score(
                model, X, y, cv=3, scoring=scoring, n_jobs=-1)
            return float(scores.mean())

        study.optimize(objective, n_trials=max_trials,
                       timeout=time_budget_seconds, show_progress_bar=False)
        elapsed = time.time() - start_time

        best_params = study.best_params
        final_model = self._build_model(
            best_candidate.algorithm, best_params, problem_type, num_classes)
        final_model.fit(X, y)

        os.makedirs("outputs/models", exist_ok=True)
        tuned_path = f"outputs/models/{dataset_id}_tuned_model.pkl"
        joblib.dump(final_model, tuned_path)

        report = {
            "original_algorithm": best_candidate.algorithm,
            "best_trial_score": round(study.best_value, 5) if study.best_trial else None,
            "best_params": best_params,
            "num_trials_completed": len(study.trials),
            "time_seconds": round(elapsed, 2),
            "tuned_model_path": tuned_path,
        }

        try:
            importances = optuna.importance.get_param_importances(study)
            report["param_importance"] = {
                k: round(float(v), 4) for k, v in importances.items()}
        except Exception as e:
            logger.warning(f"Could not compute param importances: {e}")
            report["param_importance"] = {}

        logger.info(
            f"Tuning done: {report['num_trials_completed']} trials in {report['time_seconds']}s, "
            f"best_score={report['best_trial_score']}"
        )

        return final_model, report

    def _suggest_model(self, trial, algorithm, problem_type, num_classes=None):
        if algorithm == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return RandomForestClassifier(**params) if problem_type == "classification" else RandomForestRegressor(**params)

        elif algorithm == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            from xgboost import XGBClassifier, XGBRegressor
            if problem_type == "classification":
                # Explicit objective/num_class for multiclass avoids
                # XGBoost's sklearn wrapper occasionally mis-inferring
                # num_class across repeated fresh instantiations in
                # Optuna's trial loop.
                if num_classes is not None and num_classes > 2:
                    params["objective"] = "multi:softprob"
                    params["num_class"] = num_classes
                else:
                    params["objective"] = "binary:logistic"
                return XGBClassifier(**params)
            return XGBRegressor(**params)

        elif algorithm == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 0.001, 100, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": "saga",
                "max_iter": 1000,
            }
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)

        elif algorithm == "svm_rbf":
            params = {
                "C": trial.suggest_float("C", 0.1, 100, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }
            from sklearn.svm import SVC, SVR
            return SVC(kernel="rbf", **params) if problem_type == "classification" else SVR(kernel="rbf", **params)

        elif algorithm == "svm_linear":
            params = {
                "C": trial.suggest_float("C", 0.01, 100, log=True),
            }
            from sklearn.svm import SVC, SVR
            return SVC(kernel="linear", **params) if problem_type == "classification" else SVR(kernel="linear", **params)

        elif algorithm == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 150),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            }
            from lightgbm import LGBMClassifier, LGBMRegressor
            if problem_type == "classification":
                if num_classes is not None and num_classes > 2:
                    params["objective"] = "multiclass"
                    params["num_class"] = num_classes
                else:
                    params["objective"] = "binary"
                return LGBMClassifier(verbose=-1, **params)
            return LGBMRegressor(verbose=-1, **params)

        elif algorithm == "neural_network_mlp":
            n_layers = trial.suggest_int("n_layers", 1, 3)
            layer_size = trial.suggest_categorical("layer_size", [32, 64, 128])
            params = {
                "hidden_layer_sizes": tuple([layer_size] * n_layers),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "max_iter": 500,
            }
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            return MLPClassifier(**params) if problem_type == "classification" else MLPRegressor(**params)

        logger.error(f"Tuning not implemented for algorithm: {algorithm}")
        raise ValueError(f"Tuning not implemented for: {algorithm}")

    def _build_model(self, algorithm, params, problem_type, num_classes=None):
        # Shim trial object that just replays the already-chosen params
        # instead of asking Optuna to suggest new ones. suggest_categorical
        # needs a fallback default even when the param isn't in `params`
        # (e.g. n_layers/layer_size not directly present as final params
        # for MLP — see note below).
        shim = type(
            "T", (),
            {
                "suggest_int": lambda self, n, l, h: params.get(n, l),
                "suggest_float": lambda self, n, l, h, log=False: params.get(n, l),
                "suggest_categorical": lambda self, n, c: params.get(n, c[0]),
            },
        )()
        return self._suggest_model(shim, algorithm, problem_type, num_classes)
