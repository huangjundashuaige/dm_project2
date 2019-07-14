
import numpy as np
from decimal import Decimal,getcontext
import json


class Decision_tree:
    
    def __init__(self, num_max_features=5, max_depth=10, min_samples=5, min_impurity=1e-6):
        self.num_max_features = num_max_features
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.min_samples = min_samples
        self.tree = {}
    
    def get_param(self):
        return json.dumps(self.tree)

    def quantization_tree(self,prec=8):
        self.tree = self.__quantization_tree_node(prec,self.tree,prec)

    def __quantization_tree_node(self,tree_node,prec=8):
        getcontext().prec = prec
        tree_node['value'] = Decimal(str(tree_node['value']))
        if tree_node['left'] == None:
            self.__quantization_tree_node(tree['left'],prec)
        if tree_node['right'] == None:
            self.__quantization_tree_node(tree['right'],prec)
        return tree_node
    def build(self, X, y):
        self.tree = self.__build(X, y, self.max_depth)
        
    
    def __build(self, X, y, height):
        bestfeature, bestValue = self.select_best_feature(X, y)
        if bestfeature == None:
            return bestValue
        
        tree = {}
        
        height -= 1
        if height < 0 or X.shape[0] < self.min_samples_split:
            return self.avg_data(y)
        
        tree['bestFeature'] = bestfeature
        tree['bestVal'] = bestValue
        
        left_X, left_y, right_X, right_y = self.split_data_set(X, y, bestfeature, bestValue)
        tree['right'] = self.__build(right_X, right_y, height)
        tree['left'] = self.__build(left_X, left_y, height)
        return tree

    
    def predict(self, X):
        y = [0] * X.shape[0]
        for i in range(X.shape[0]):
            y[i] = self.__predict(self.tree, X.iloc[i, :])
        return y


    def __predict(self, tree, data):
        if not isinstance(tree, dict):
            return float(tree)
        
        if data[tree['bestFeature']] > tree['bestVal']:
            if type(tree['left']) == 'float':
                return tree['left']
            else:
                return self.__predict(tree['left'], data)
        else:
            if type(tree['right'])=='float':
                return tree['right']
            else:
                return self.__predict(tree['right'], data)

    
    def mse_data(self, label):
        return (np.var(label) * np.shape(label)[0]).item()
    
    
    def avg_data(self, label):
        return np.mean(label)
    
    
    def split_data_set(self, X, y, feature, value):
        left_index = np.nonzero(X.iloc[:, feature] > value)[0]
        right_index = np.nonzero(X.iloc[:, feature] < value)[0]
        return X.iloc[left_index, :], y.iloc[left_index, :], X.iloc[right_index, :], y.iloc[right_index, :]
    
    
    def choose_best_feature(self, X, y):
        '''
        Parameters
        ----------
        X : matrix of shape = [n_samples, n_features]
            The matrix consisting of the input data
            
        y : array of shape = [n_samples]
            The label of X
        '''
        n_features = X.shape[1]
        features_index = []
        
        best_MSE = np.inf
        best_feature = 0
        best_value = 0
        
        MSE = self.mse_data(y)

        for i in range(self.max_features_num):
            features_index.append(np.random.randint(n_features))
        
        for feature in features_index:
            for value in set(X.iloc[:, feature]):
                left_X, left_y, right_X, right_y = self.split_data_set(X, y, feature, value)

                new_MSE = self.mse_data(left_y) + self.mse_data(right_y)
                if best_MSE > new_MSE:
                    best_feature = feature
                    best_value = value
                    best_MSE = new_MSE
        
        if (MSE - best_MSE) < self.min_impurity_split:
            return None, self.avg_data(y)
        
        return best_feature, best_value