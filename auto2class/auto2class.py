from .optimization.optimizer_all_models import OptimizerAllModels
from .preprocessing.data_preprocessor import DataPreprocessor

class Auto2Class:
    def __init__(self, dataframe, target_column_name, test_size=0.2, random_state=42, n_iter=10, cv=5, n_repeats=1, metric_to_eval = 'roc_auc'):
        self.dataframe = dataframe
        self.target_column_name = target_column_name
        self.n_iter = n_iter
        self.n_repeats = n_repeats
        self.metric = metric_to_eval
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.params_rf = None
        self.params_dt = None
        self.params_xgb = None
        

    def perform_model_selection(self):
        preprocessed_data = DataPreprocessor(self.dataframe, self.target_column_name).preprocess()
        self.optimizer = OptimizerAllModels(preprocessed_data, self.test_size, self.random_state, self.n_iter, self.cv, self.n_repeats, self.metric)
        self.optimizer.perform_analysis()
        self.params_rf = self.optimizer.params_rf
        self.params_dt = self.optimizer.params_dt
        self.params_xgb = self.optimizer.params_xgb
        return 
