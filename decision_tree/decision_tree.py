import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.tree = None
    
    
    def information_gain(self, y, subsets):
        y_encoded = y.map({"Yes": 1, "No": 0})
        total_entropy = entropy(np.bincount(y_encoded))
        total_samples = len(y_encoded)

        subset_entropy = sum(entropy(np.bincount(subset.map({"Yes": 1, "No": 0}))) * len(subset) for subset in subsets) / total_samples

        return total_entropy - subset_entropy
    
    def decision_tree_id3(self, X, y, features):
        # Convert 'Yes' to 1 and 'No' to 0
        y_encoded = y.map({"Yes": 1, "No": 0})

        # Base Cases:
        # 1. If all examples have the same label
        unique_labels = y_encoded.unique()
        if len(unique_labels) == 1:
            return "Yes" if unique_labels[0] == 1 else "No"


        # 2. If no more features to test, return the most frequent label
        if not features:
            return y.value_counts().idxmax()

        # Find the best feature to split on
        max_gain = -1
        best_feature = None
        for feature in features:
            subsets = [y[X[feature] == value] for value in X[feature].unique()]
            gain = self.information_gain(y, subsets)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature

        if max_gain == 0:
            return y.value_counts().idxmax()

        tree = {best_feature: {}}
        remaining_features = features - {best_feature}
        for value in X[best_feature].unique():
            subtree_X = X[X[best_feature] == value].drop(columns=[best_feature])
            subtree_y = y[X[best_feature] == value]
            tree[best_feature][value] = self.decision_tree_id3(subtree_X, subtree_y, remaining_features)

        return tree

    
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # TODO: Implement 
        self.tree = self.decision_tree_id3(X, y, set(X.columns))

    def predict_sample(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        for feature, subtree in tree.items():
            feature_value = sample[feature]
            if feature_value in subtree:
                return self.predict_sample(sample, subtree[feature_value])
        return None
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        return X.apply(self.predict_sample, axis=1, tree=self.tree)
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        rules = []

        def _get_rules_from_tree(tree, conditions):
            if not isinstance(tree, dict):
                rules.append((conditions, tree))
                return
            for feature, subtree in tree.items():
                for value, child_tree in subtree.items():
                    _get_rules_from_tree(child_tree, conditions + [(feature, value)])

        _get_rules_from_tree(self.tree, [])
        return rules


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))

def main():
    # Code from 1.1 in the notebook
    data_1 = pd.read_csv('data_1.csv')
    print(data_1)

    # Code from 1.2 in the notebook, removed dt (since I am just using the same file)
    # Separate independent (X) and dependent (y) variables
    X = data_1.drop(columns=['Play Tennis'])
    y = data_1['Play Tennis']

    # Create and fit a Decrision Tree classifier
    model_1 = DecisionTree()  # <-- Should work with default constructor
    model_1.fit(X, y)

    # Verify that it perfectly fits the training set
    print(f'Accuracy: {accuracy(y_true=y, y_pred=model_1.predict(X)) * 100 :.1f}%')
    for rules, label in model_1.get_rules():
        conjunction = ' ∩ '.join(f'{attr}={value}' for attr, value in rules)
        print(f'{"✅" if label == "Yes" else "❌"} {conjunction} => {label}')

main()