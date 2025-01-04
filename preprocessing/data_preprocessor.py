from .redundant_features_handler import RedundantFeaturesHandler
from .missing_values_handler import MissingValuesHandler
from .outliers_handler import OutliersHandler
from .feature_type_extractor import FeatureTypeExtractor
from .correlated_features_handler import CorrelationFeaturesHandler
from .class_balance_handler import ClassBalanceHandler
import warnings


class DataPreprocessor:
    '''
    A class where
    '''

    def __init__(self, dataset, target_column_name):
        self.dataset = dataset.copy()  # the whole dataset, with a target column
        self.target_column_name = target_column_name  # the name of the target column

    def preprocess(self):
        '''
        Preprocess the dataset function.
        returns X and y
        '''
        warnings.filterwarnings("ignore")

        print('---------------Preprocessing the dataset---------------')

        print('---------------Extracting Day, Month and Year--------------')
        self.dataset = FeatureTypeExtractor().separate_datetime(self.dataset)

        print('---------------Deleting redundant features--------------')
        self.dataset = RedundantFeaturesHandler(self.dataset).handle()

        print('---------------Handling missing values------------------')
        self.dataset = MissingValuesHandler(self.dataset).handle()

        print('---------------Handling outliers------------------------')
        self.dataset = OutliersHandler(self.dataset).handle()

        print('---------------1. Encoding categorical features ------')
        self.dataset = FeatureTypeExtractor().encode_categorical(self.dataset, self.target_column_name)

        print('---------------2. Min max scaling --------------------')
        self.dataset = FeatureTypeExtractor().min_max_scale(self.dataset, self.target_column_name)

        print('---------------3. Encoding of target------------')
        self.dataset = FeatureTypeExtractor().encode_target(self.dataset, self.target_column_name)

        X = self.dataset.drop(self.target_column_name, axis=1)
        y = self.dataset[self.target_column_name]

        print('---------------Removing highly correlated columns ------------')
        X = CorrelationFeaturesHandler().fit_transform(X)

        print('---------------Handling imbalanced classes------------')
        X, y = ClassBalanceHandler().fit_resample(X, y)

        print('---------------Dataset preprocessing is done------------')

        return X, y
