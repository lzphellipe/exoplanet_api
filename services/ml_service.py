import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self, data_dir: str = "data"):
        """Initialize MLService with directories and logging.

        Args:
            data_dir (str): Directory for data storage. Defaults to 'data'.
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.model_dir = os.path.join(data_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def apply_lightgbm_model(self, data_train_final: pd.DataFrame, data_test_final: pd.DataFrame,
                             model_path: str) -> dict:
        """Apply a LightGBM classifier with nested CV and Bayesian optimization.

        Args:
            data_train_final (pd.DataFrame): Training dataset with features, label, and identifier.
            data_test_final (pd.DataFrame): Test dataset with features and identifier.
            model_path (str): Path to save the trained model.

        Returns:
            dict: Contains metrics, model path, and test predictions DataFrame.
        """
        self.logger.info(f"Training LightGBM model and predicting on test data")
        try:
            # Make a copy of training data
            data_input = data_train_final.copy()

            # Extract feature names
            feature_names = data_train_final.drop(columns=["label", "identifier"]).columns

            # Extract labels and features
            y = data_input["label"].values
            X = data_input[feature_names].values

            # Binarize labels
            lb = LabelBinarizer()
            y = lb.fit_transform(y).reshape(-1)

            # Define hyperparameter search space
            param_lightgbm = {
                'learning_rate': Real(1e-3, 1, prior='log-uniform'),
                'n_estimators': Integer(10, 500),
                'num_leaves': Integer(2, 100),
                'max_depth': Integer(3, 10)
            }

            # Outer CV for evaluation
            cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []

            # Nested CV
            for train_ix, validation_ix in cv_outer.split(X, y):
                X_train, X_validation = X[train_ix, :], X[validation_ix, :]
                y_train, y_validation = y[train_ix], y[validation_ix]

                # Inner CV for hyperparameter optimization
                cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
                base_model = LGBMClassifier(random_state=1)
                search = BayesSearchCV(
                    estimator=base_model,
                    search_spaces=param_lightgbm,
                    scoring='accuracy',
                    cv=cv_inner,
                    n_iter=5,
                    refit=True,
                    random_state=1,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                # Evaluate on validation fold
                y_validation_predict = best_model.predict(X_validation)
                accuracies.append(accuracy_score(y_validation, y_validation_predict))
                precisions.append(
                    precision_score(y_validation, y_validation_predict, average='binary', zero_division=1))
                recalls.append(recall_score(y_validation, y_validation_predict, average='binary'))
                f1_scores.append(f1_score(y_validation, y_validation_predict, average='binary'))

            # Aggregate metrics
            metrics = {
                'accuracy_mean': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies)),
                'precision_mean': float(np.mean(precisions)),
                'precision_std': float(np.std(precisions)),
                'recall_mean': float(np.mean(recalls)),
                'recall_std': float(np.std(recalls)),
                'f1_mean': float(np.mean(f1_scores)),
                'f1_std': float(np.std(f1_scores))
            }

            # Train final model
            best_final_model = LGBMClassifier(random_state=1, **search.best_params_)
            best_final_model.fit(X, y)

            # Save model
            joblib.dump(best_final_model, model_path)
            self.logger.info(f"Model saved to {model_path}")

            # Prepare test features
            X_test = data_test_final[feature_names].values
            test_ids = data_test_final['identifier'].values

            # Predict on test set
            y_test_predict = best_final_model.predict(X_test)
            y_test_prob = best_final_model.predict_proba(X_test)[:, 1]

            # Create results DataFrame
            results_tests = pd.DataFrame({
                'identifier': test_ids,
                'pred_label': y_test_predict,
                'prob_exoplanet': y_test_prob
            })

            return {
                "metrics": metrics,
                "model_path": model_path,
                "results_tests": results_tests
            }
        except Exception as e:
            self.logger.error(f"Error applying LightGBM model: {e}")
            raise

    def predict(self, data_test_final: pd.DataFrame, model_path: str) -> pd.DataFrame:
        """Predict exoplanet classifications using a trained LightGBM model.

        Args:
            data_test_final (pd.DataFrame): Test dataset with features and identifier.
            model_path (str): Path to the trained model.

        Returns:
            pd.DataFrame: Predictions with identifier, pred_label, and prob_exoplanet.
        """
        self.logger.info(f"Making predictions with model: {model_path}")
        try:
            # Load model
            model = joblib.load(model_path)

            # Prepare test features
            feature_names = data_test_final.drop(columns=["identifier"]).columns
            X_test = data_test_final[feature_names].values
            test_ids = data_test_final['identifier'].values

            # Predict
            y_test_predict = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1]

            # Create results DataFrame
            results_tests = pd.DataFrame({
                'identifier': test_ids,
                'pred_label': y_test_predict,
                'prob_exoplanet': y_test_prob
            })

            return results_tests
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise