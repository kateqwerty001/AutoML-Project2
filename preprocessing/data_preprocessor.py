from .redundant_features_handler import RedundantFeaturesHandler
from .missing_values_handler import MissingValuesHandler
from .outliers_handler import OutliersHandler
from .feature_type_extractor import FeatureTypeExtractor

class DataPreprocessor:
    '''
    A class which preprocesses the dataset.
    '''
    def __init__(self, dataset, target_column_name):
        self.dataset = dataset # the whole dataset, with a target column
        self.target_column_name = target_column_name # the name of the target column

    def preprocess(self):
        '''
        Preprocess the dataset function.
        returns X and y
        '''
        print('---------------Preprocessing the dataset---------------')
        
        print('---------------Deleting redundant features--------------')
        # Remove redundant features
        self.dataset = RedundantFeaturesHandler(self.dataset).handle()
        print('---------------Extracting Day, Month and Year--------------')
        # a class must be created to extract day, month and year from the date
        print('---------------Handling missing values------------------')
        # Handle missing values
        self.dataset = MissingValuesHandler(self.dataset).handle()
        print('---------------Handling outliers------------------------')
        # Handle outliers
        self.dataset = OutliersHandler(self.dataset).handle()
        print('---------------One hot encoding of categorical features--')
        # One hot encoding of categorical features
        feature_type_extractor = FeatureTypeExtractor()
        for feature in self.dataset.columns:
            feature_type = feature_type_extractor.get_feature_type(self.dataset[feature], self.dataset)
            if feature_type == feature_type_extractor.CATEGORICAL and feature != self.target_column_name:
                self.dataset = feature_type_extractor.one_hot_encode(feature, self.dataset)
            elif feature_type == feature_type_extractor.CONTINIOUS or feature_type == feature_type_extractor.DISCRETE and feature != self.target_column_name:
                self.dataset[feature] = feature_type_extractor.min_max_scale(self.dataset[feature])

        print('---------------Dataset preprocessing is done------------')
        X = self.dataset.drop(self.target_column_name, axis=1)
        y = self.dataset[self.target_column_name]
        return X, y