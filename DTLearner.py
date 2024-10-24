import numpy as np
import pandas as pd

class DTLearner:
    def __init__(self, leaf_size=1, max_depth=2, verbose=False):
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.tree = None
        self.features = None  # Store features for later use
        print(f"Initialized DTLearner with leaf_size={leaf_size} and max_depth={max_depth}")

    @staticmethod
    def select_best_feature(correlations, features, method='alphabetical'):
        # Ensure correlations only include features, not the target
        correlations = correlations[features]
        tied_features = correlations[correlations.abs() == correlations.abs().max()].index.tolist()
        if method == 'alphabetical':
            sorted_features = sorted(tied_features)
            print(f"\n***Tied features: {tied_features}")
            print(f"Sorted features: {sorted_features}")
            print(f"Selected feature: {sorted_features[0]}")
            return sorted_features[0]
        elif method == 'random':
            return np.random.choice(tied_features)
        elif method == 'appearance':
            for feature in features:
                if feature in tied_features:
                    return feature
        else:
            raise ValueError("Invalid method. Choose 'alphabetical', 'random', or 'appearance'.")

    def build_tree(self, data, features, target, depth=0):
        print(f"Building tree at depth {depth} with max_depth {self.max_depth}")

        # Base case: Stop if we exceed max_depth or have few samples
        if depth >= self.max_depth or len(data) <= self.leaf_size:
            print(f"Reached base case at depth {depth}, creating a leaf node with target mean.")
            return ["Leaf", target.mean()]  # Correct leaf node representation

        # Calculate correlations between features and target
        correlations = data.corrwith(target)
        print(f"Correlations: \n{correlations}")

        # Select the best feature based on correlations
        print(f"Selecting the best feature at depth {depth} from features: {features}")
        best_feature = self.select_best_feature(correlations, features, method='alphabetical')
        split_val = data[best_feature].median()
        print(f"Split value determined for feature {best_feature}: {split_val}")

        # Split data into left and right branches
        left_data = data[data[best_feature] <= split_val]
        right_data = data[data[best_feature] > split_val]
        left_target = target[left_data.index]
        right_target = target[right_data.index]

        # Check for empty branches
        if left_data.empty or right_data.empty:
            print(f"No further split possible at depth {depth}, creating a leaf node.")
            return ["Leaf", target.mean()]

        # Recursively build the left and right subtrees
        print(f"Building left subtree at depth {depth + 1}")
        left_tree = self.build_tree(left_data, features, left_target, depth + 1)

        print(f"Building right subtree at depth {depth + 1}")
        right_tree = self.build_tree(right_data, features, right_target, depth + 1)

        return [best_feature, split_val, left_tree, right_tree]

    def add_evidence(self, data, features, target):
        print(f"Training model with data size: {len(data)}")
        self.features = features  # Store features for query usage
        self.tree = self.build_tree(data, features, target)
        print(f"Decision tree built and stored in self.tree")
        
    def query(self, points):
        print(f"Querying the tree with input points")
        if isinstance(points, dict):
            points = [points]

        predictions = []
        for point in points:
            prediction = self.query_point(point, self.tree)
            predictions.append(prediction)
        return np.array(predictions)
    
    def query_point(self, point, tree):
    # Base case: if the node is a leaf, return its value
        if tree[0] == "Leaf":
            return tree[1]
        else:
            feature = tree[0]
            split_val = tree[1]
            # Decide whether to go left or right in the tree
            if point[feature] <= split_val:
                return self.query_point(point, tree[2])
            else:
                return self.query_point(point, tree[3])


# Example usage with minimal data
if __name__ == "__main__":
    # Create dummy data without the target variable in 'data'
    data = pd.DataFrame({
        'X2: Volatile Acidity': [0.56, 0.72, 0.89],
        'X10: Sulphates': [0.5, 0.6, 0.8],
        'X11: Alcohol': [9.4, 10.9, 9.1]
    })
    target = pd.Series([6, 5, 4], name='Y: Quality')  # Dummy target
    features = ['X2: Volatile Acidity', 'X10: Sulphates', 'X11: Alcohol']

    # Create and train the learner
    learner = DTLearner(leaf_size=1, max_depth=2)
    learner.add_evidence(data, features, target)

    # Dummy query with DataFrame
    query_points = pd.DataFrame([
        {'X2: Volatile Acidity': 0.6, 'X10: Sulphates': 0.55, 'X11: Alcohol': 9.5}
    ])
    predictions = learner.query(query_points)
    print(f"Predictions: {predictions}")
