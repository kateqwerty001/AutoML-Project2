from .optimization.optimizer_all_models import OptimizerAllModels
from .preprocessing.data_preprocessor import DataPreprocessor

class Auto2Class:
    def __init__(self, dataframe, target_column_name, test_size=0.2, random_state=42, n_iter=10, cv=5, n_repeats=1, metric = 'roc_auc'):
        """
        Initialize the Auto2Class for automated binary classification model selection.

        Args:
            dataframe: The dataset to be used for training and evaluation.
            target_column_name: The name of the column containing the target variable.
            test_size: Fraction of the data to be used for testing (default is 0.2).
            random_state: Random seed for reproducibility.
            n_iter: Number of iterations for model optimization.
            cv: Number of cross-validation splits.
            n_repeats: Number of times to repeat cross-validation for stability.
            metric: The evaluation metric be optimized during model selection (default is 'roc_auc').
        """
        self.dataframe = dataframe
        self.target_column_name = target_column_name
        self.n_iter = n_iter
        self.n_repeats = n_repeats
        self.metric = metric
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.params_rf = None
        self.params_dt = None
        self.params_xgb = None
        

    def perform_model_selection(self):
        """
        Perform the model selection process by preprocessing the data and running optimization 
        on Random Forest, Decision Tree, and XGBoost models.

        This method preprocesses the data, performs model optimization, and stores the best 
        hyperparameters for each model.
        """
        # Preprocess the data
        preprocessed_data = DataPreprocessor(self.dataframe, self.target_column_name).preprocess()
        self.optimizer = OptimizerAllModels(preprocessed_data, self.test_size, self.random_state, self.n_iter, self.cv, self.n_repeats, self.metric)
        self.optimizer.perform_analysis()

        # Store the best hyperparameters for each model after optimization
        self.params_rf = self.optimizer.params_rf
        self.params_dt = self.optimizer.params_dt
        self.params_xgb = self.optimizer.params_xgb
        return 
