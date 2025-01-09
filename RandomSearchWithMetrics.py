from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, brier_score_loss
import pandas as pd
import numpy as np
import random

class RandomSearchWithMetrics:
    def __init__(self, pipeline, params, X, y, n_iter=10, cv=5, random_state=42, n_repeats=5):
        """
        Initialize the RandomSearchWithMetrics class.

        Args:
            pipeline: The ML pipeline (e.g., sklearn Pipeline object).
            params: Dictionary of hyperparameter names and their possible values.
            X: Feature dataset (numpy array or Pandas DataFrame).
            y: Target dataset (numpy array or Pandas Series).
            n_iter: Number of iterations to perform random search.
            cv: Number of cross-validation splits.
            random_state: Random seed for reproducibility.
            n_repeats: Number of times to repeat cross-validation for stability.
        """
        self.pipeline = pipeline
        self.params = params
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.n_repeats = n_repeats  # For repeated cross-validation to ensure stable metrics
        self.history = pd.DataFrame()  # In-memory storage for the results

    def generate_random_params(self):
        """
        Randomly generate a set of hyperparameters from the provided parameter grid.

        Returns:
            A dictionary with randomly selected values for each parameter.
        """
        params = {}
        for key, values in self.params.items():
            if isinstance(values, list):  # Ensure values is a list to allow random selection
                params[key] = random.choice(values)
        return params

    def fit_and_evaluate(self):
        """
        Perform random search with cross-validation and store the results in `self.history`.
        """
        # Set seed for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        for i in range(self.n_iter):  # Perform `n_iter` random search iterations
            # Randomly generate a new set of hyperparameters
            params = self.generate_random_params()
            self.pipeline.set_params(**params)  # Apply the hyperparameters to the pipeline

            # Initialize lists to accumulate metrics across repeats
            
            # f1_scores, accuracies, brier_scores, roc_aucs = [], [], [], []

            f1_scores, accuracies, roc_aucs = [], [], []

            for j in range(self.n_repeats):  # Repeat cross-validation `n_repeats` times
                # Create KFold object for cross-validation
                kf = KFold(n_splits=self.cv, shuffle=True)

                # Perform cross-validation predictions for both labels and probabilities
                y_pred = cross_val_predict(self.pipeline, self.X, self.y, cv=kf, method='predict')
                y_probabilities = cross_val_predict(self.pipeline, self.X, self.y, cv=kf, method='predict_proba')

                # Check if the problem is binary or multiclass
                if len(np.unique(self.y)) == 2:  # Binary classification
                    y_probabilities = y_probabilities[:, 1]  # Get probabilities for the positive class

                    # Calculate metrics for binary classification
                    f1_scores.append(f1_score(self.y, y_pred, average='weighted'))
                    accuracies.append(accuracy_score(self.y, y_pred))
                    # brier_scores.append(brier_score_loss(self.y, y_probabilities))
                    roc_aucs.append(roc_auc_score(self.y, y_probabilities))

                else:  # Multiclass classification
                    # Calculate metrics for multiclass classification
                    f1_scores.append(f1_score(self.y, y_pred, average='weighted'))
                    accuracies.append(accuracy_score(self.y, y_pred))
                    # brier_scores.append(brier_score_loss(self.y, y_probabilities)) 
                    roc_aucs.append(roc_auc_score(self.y, y_probabilities, multi_class='ovr', average='weighted'))

            # Calculate average metrics across all repeats
            avg_metrics = {
                'f1': np.mean(f1_scores),
                'accuracy': np.mean(accuracies),
                # 'brier_score': np.mean(brier_scores),
                'roc_auc': np.mean(roc_aucs)
            }

            # Add the hyperparameter values to the metrics dictionary
            avg_metrics.update(params)

            # Append the results to the history DataFrame
            self.history = pd.concat([self.history, pd.DataFrame([avg_metrics])], ignore_index=True)

    def get_results(self):
        """
        Retrive all combinations of hyperparameters and their metric values (roc_auc, f1, accuracy)

        Returns a pandas dataframe with hyperparameter combinations and metrics
        """ 

        return self.history

   


   
