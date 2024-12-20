import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, thresold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.thresold = thresold
        self.left = left
        self.right = right
        self.value=None

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_feature=None):
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.n_feature=n_feature
        self.root=None

    def fit(self, X, y):
        self.n_feature = X.shape[1] if not self.n_feature else min(X.shape[1], self.n_feature)
        self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feature, replace=False)

        # find best split
        best_feature, best_thresold = self._best_split(X, y, feat_idx)

        # creating child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresold, left, right)
    
    def _best_split(self, X, y, feat_idx):
        best_gain = 1
        split_idx, split_thresold = None, None

        for feat_idx, in feat_idxs:
            X_column = X[:, feat_idx]
            thresolds = np.unique(X_column)

            for thr in thresolds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresold = thr
        
        return split_idx, split_thresold
    
    def _infromation_gain(self, y, X_column, thresold):
        # parent entrop
        parent_entropy = self._entropy(y)

        # childern entropy
        left_idxs, right_idxs = self._split(X_column, thresold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted emtropy of childern
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entopy(y[left_idxs]), self._entopy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG

        informations_gain = parent_entropy - child_entropy
        return informations_gain
    

    def _split(self, X_column, split_thresold):
        left_idxs = np.argwhere(X_column <= split_thresold).flatten()
        right_idxs = np.argwhere(X_column > split_thresold).flatten()
        return left_idxs, right_idxs

    
    def _entopy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value 

    def predict(self, X):
        return np.array([self._traverse_tree(x) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value()
        
        if x[node.feature] <= node.thresold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
