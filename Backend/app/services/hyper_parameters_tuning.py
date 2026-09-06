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

        scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"

        study = optuna.create_study(direction="maximize")
        start_time = time.time()

        def objective(trial):
            model = self._suggest_model(
                trial, best_candidate.algorithm, problem_type)
            scores = cross_val_score(
                model, X, y, cv=3, scoring=scoring, n_jobs=-1)
            return float(scores.mean())

        study.optimize(objective, n_trials=max_trials,
                       timeout=time_budget_seconds, show_progress_bar=False)
        elapsed = time.time() - start_time

        best_params = study.best_params
        final_model = self._build_model(
            best_candidate.algorithm, best_params, problem_type)
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

    def _suggest_model(self, trial, algorithm, problem_type):
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
            return XGBClassifier(**params) if problem_type == "classification" else XGBRegressor(**params)

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

        logger.error(f"Tuning not implemented for algorithm: {algorithm}")
        raise ValueError(f"Tuning not implemented for: {algorithm}")

    def _build_model(self, algorithm, params, problem_type):
        return self._suggest_model(type('T', (), {'suggest_int': lambda self, n, l, h: params.get(n, l),
                                                  'suggest_float': lambda self, n, l, h, log=False: params.get(n, l),
                                                  'suggest_categorical': lambda self, n, c: params.get(n, c[0])})(),
                                   algorithm, problem_type)
