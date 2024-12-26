import pandas as pd

class FeatureTypeExtractor:
    '''
    A class which extracts the types of feature (column) in the dataset, in order to understand, which transformations we shiould do.
    This class also has 
    - a method to one-hot encode the categorical features and 
    - min-max scale the continious features.
    '''
    def __init__(self):
        self.CATEGORICAL = 'categorical'
        self.TEXT = 'text'
        self.CONTINIOUS = 'continious'
        self.DISCRETE = 'discrete'
        self.DATETIME = 'datetime'
        self.INDEX = 'index' # If the feature is an index column of the dataset.
        self.UNKNOWN = 'unknown' # If the type of the feature was not recognized by the class.
        self.MAX_CATEGORIES = 15 # If the number of unique values in a column is greater than this value, then the column is considered as a text column, otherwise it is a categorical column.

    def get_feature_type(self, x, dataset):
        '''
        Returns the type of the feature.
        x - a vector, a column of the dataset.
        dataset - the dataset, which has a feature column x.
        '''
        first_column = dataset.columns[0] 

        # Check if the input is a vector, not a matrix.
        if (pd.DataFrame(x).shape[1]!=1):
            raise ValueError('The input must be a vector')
        
        x_type = str(x.dtype)

        if x_type.startswith("float"):
            return self.CONTINIOUS
        
        if x_type.startswith("int") or x_type.startswith("uint"):
            max_len = len(str(x.max()))
            if first_column == x.name and max_len < 10 and len(x.unique()) == len(x):
            # We assume that the index column (must be the 1st column)
            # has values with length < 10, if it is an integer value, 
            # otherwise it is just a DISCRETE column.
                return self.INDEX
            return self.DISCRETE
        
        if x_type.startswith("datetime"):
            return self.DATETIME
        
        if x_type == 'object':
            if len(x.unique()) > self.MAX_CATEGORIES:
                max_len = x.str.len().max()
                if len(x.unique()) == len(x) and max_len < 30 and first_column == x.name:
                # We assume that the index column (must be the 1st column)
                # has values with length < 15, if it is a string value, 
                # otherwise it is just a text column.
                    return self.INDEX
                else:
                    return self.TEXT
            else:
                return self.CATEGORICAL
        
        return self.UNKNOWN
    
    def one_hot_encode(self, feature_name, dataset):
        '''
        One hot encodes the categorical features.
        feature_name - the name of the feature, which should be one-hot encoded.
        dataset - the dataset, which has the feature.
        '''
        one_hot = pd.get_dummies(dataset[feature_name], prefix=feature_name)
        
        dataset = dataset.drop(columns=[feature_name])
        dataset = pd.concat([dataset, one_hot], axis=1)
        
        return dataset
        
    
    def min_max_scale(self, x):
        '''
        Min-max scales the continious features.
        x - a vector, a column of the dataset.
        '''
        return (x - x.min()) / (x.max() - x.min())
    