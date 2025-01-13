import dtreeviz
from sklearn.tree import DecisionTreeClassifier

class ExplainDecisionTree:
    '''
    This class contains methods to explain the model for DecisionTreeClassifier.
    '''
    def __init__(self, model):
        self.model = model

    def plot_tree(self, X_train, y_train):
        '''
        Plots the decision tree using the model's plot_tree method.

        Parameters:
        - X_train: Training data
        - y_train: Target labels
        '''
        # Visualize the decision tree
        viz_model = dtreeviz.model(self.model, X_train, y_train, target_name='target', feature_names=X_train.columns)
        
        # Open visualization in a pop-up window
        v = viz_model.view()     # render as SVG into internal object 
        v.show()

        return
