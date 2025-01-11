from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from .optimization_algorithms.random_search_with_metrics import RandomSearchWithMetrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

class RandomForestRandomSearch:
    def __init__(self, dataset, n_iter=10, cv=5, random_state=42, n_repeats=5):
        """
        Initialize the RandomForestRandomSearch class.

        Args:
            dataset: The preprocessed dataset (Pandas DataFrame).
            params: Dictionary of hyperparameter names and their possible values for the RandomForest model.
            n_iter: Number of iterations to perform random search.
            cv: Number of cross-validation splits.
            random_state: Random seed for reproducibility.
            n_repeats: Number of times to repeat cross-validation for stability.
        """
        # Split the dataset into features (X) and target (y)
        self.X = dataset.drop(columns=['target'])  # Assumes 'target' column is the label column
        self.y = dataset['target']  # Assumes 'target' column is the label
        self.history = None   
        self.random_state = random_state 

        # Define the pipeline with a random forest classifier and optional preprocessing (scaling)
        self.pipeline = Pipeline([
            ('clf', RandomForestClassifier(random_state=random_state))  # Classifier
        ])

        self.params = {
            'clf__n_estimators': [50, 100, 200, 500],
            'clf__criterion': ['gini', 'entropy', 'log_loss'],
            'clf__max_depth': [None, 10, 20, 30, 40],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1],
            'clf__max_features': [None, 'sqrt', 'log2'],
            'clf__bootstrap': [True, False]
        }
        
        # Initialize RandomSearchWithMetrics
        self.random_search = RandomSearchWithMetrics(
            pipeline=self.pipeline,
            params=self.params,
            X=self.X,
            y=self.y,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            n_repeats=n_repeats
        )
    

    def perform_random_search(self):
        """
        Perform the random search and hyperparameter tuning using RandomSearchWithMetrics.
        """
        self.random_search.fit_and_evaluate()

        self.history = self.random_search.get_results()

        return self.history

    def fit_and_evaluate_default(self):
        """
        Fit and evaluate the RandomForestClassifier using default parameters.
        """

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y)
        
         # Initialize the DecisionTreeClassifier with default parameters
        clf = RandomForestClassifier(random_state=42)
        
        # Fit the model to the training data
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba, average="weighted") if y_pred_proba is not None else None

        # Prepare results
        results = {
            'f1': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'clf__n_estimators': clf.n_estimators,
            'clf__criterion': clf.criterion,
            'clf__max_depth': clf.max_depth,
            'clf__min_samples_split': clf.min_samples_split,
            'clf__min_samples_leaf': clf.min_samples_leaf,
            'clf__min_weight_fraction_leaf': clf.min_weight_fraction_leaf,
            'clf__max_features': clf.max_features,
            'clf__bootstrap': clf.bootstrap
        }

        print("Default model results:", results)

        # Convert results to a DataFrame
        self.default_results = pd.DataFrame([results])

        return self.default_results
    
    def get_results(self):
        """
        Get all results of metrics for a model returned in Pandas dataframe

        """

        default_results = self.fit_and_evaluate_default()

        random_results = self.perform_random_search()

        res = pd.concat([default_results, random_results], ignore_index=True)

        return res