from .feature_type_extractor import FeatureTypeExtractor


class RedundantFeaturesHandler:
    '''
    This class is responsible for removing redundant features from the dataset.
    We consider a feature redundant if:
    - Its type is INDEX or UNKOWN or TEXT (our library doesn't support text columns (which were not considered as categorical)
    - It has more than 90% missing values
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.to_delete = set()

    def handle(self):
        for feature in self.dataset.columns:
            feature_type = FeatureTypeExtractor().get_feature_type(self.dataset[feature], self.dataset)
            if feature_type is 'index':
                self.to_delete.add(feature)
                print(f'Feature: {feature} was considered as {feature_type}. It will be removed.')
            elif feature_type is 'unknown':
                self.to_delete.add(feature)
                print(f'Feature: {feature} was not identified. It will be removed.')
            elif feature_type is 'text':
                self.to_delete.add(feature)
                print(f'Feature: {feature} is of type: {feature_type}, not categorical. It will be removed, as our library does not support text columns.')
            elif self.dataset[feature].isnull().sum() / len(self.dataset) > 0.9:
                self.to_delete.add(feature)
                print(f'Feature: {feature} has more than 90% missing values. It will be removed.')

        self.dataset.drop(self.to_delete, axis=1, inplace=True)
        return self.dataset