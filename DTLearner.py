import numpy as np
import pandas as pd
from graphviz import Digraph


class DTLearner:
    def __init__(self, leaf_size=1, max_depth=5, verbose=False):
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


        if not tied_features:
            # If tied_features is empty, return None
            print("No valid features to select from. Returning None.")
            return None
        if method == 'alphabetical':
            sorted_features = sorted(tied_features)
            print(f"\n***Tied features: {tied_features}")
            print(f"Sorted features: {sorted_features}")
            if sorted_features[0] != None:
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

        if best_feature is None:
            print(f"No valid feature selected at depth {depth}. Creating a leaf node.")
            return ["Leaf", target.mean()]  # Create a leaf node if no feature is selected

        # Check if the feature is categorical
        if data[best_feature].dtype == 'object':  # Categorical feature
            unique_values = data[best_feature].unique()
            print(f"Splitting on categorical feature '{best_feature}' with values: {unique_values}")

            # Create a subtree for each unique value
            subtrees = {}
            for value in unique_values:
                subset = data[data[best_feature] == value]
                subset_target = target[subset.index]
                subtrees[value] = self.build_tree(subset, features, subset_target, depth + 1)

            return [best_feature, subtrees]  # Return the feature and the subtrees for each category

        else:  # Numerical feature (use median split as before)
            split_val = data[best_feature].median()
            print(f"Split value determined for feature {best_feature}: {split_val}")

            left_data = data[data[best_feature] <= split_val]
            right_data = data[data[best_feature] > split_val]
            left_target = target[left_data.index]
            right_target = target[right_data.index]

            if left_data.empty or right_data.empty:
                print(f"No further split possible at depth {depth}, creating a leaf node.")
                return ["Leaf", target.mean()]

            left_tree = self.build_tree(left_data, features, left_target, depth + 1)
            right_tree = self.build_tree(right_data, features, right_target, depth + 1)

            return [best_feature, split_val, left_tree, right_tree]

    def add_evidence(self, data, features, target):# varies from outline
        print(f"Training model with data size: {len(data)}")
        self.features = features  # Store features for query usage
        self.tree = self.build_tree(data, features, target)
        print(f"Decision tree built and stored in self.tree")
        self.visualize_tree(self.tree, 'decision_tree')
        
    def query(self, points):
        print(f"Querying the tree with input points")

        if isinstance(points, dict):
            points = [points]
        elif isinstance(points, pd.DataFrame):
            # Convert DataFrame to a list of dictionaries (records)
            points = points.to_dict(orient='records')
        elif isinstance(points, pd.Series):
            points = [points.to_dict()]
        else:
            # Assume points is already an iterable of dictionaries
            pass

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

    def visualize_tree(self, tree, file_path, feature_names=None):

        """
        Visualize the decision tree using Graphviz.

        Args:
            tree (list): The decision tree to visualize.
            file_path (str): The file path for saving the tree visualization.
            feature_names (list): List of feature names.
        """
        dot = Digraph(comment='Decision Tree')
        node_counter = [0]  # Use a list to have a mutable integer

        def add_nodes_edges(tree, parent=None):
            node_id = node_counter[0]
            if tree[0] == 'Leaf':
                label = f'Leaf: {tree[1]:.2f}'
                dot.node(str(node_id), label, shape='box', style='filled', color='lightgrey')
            else:
                feature = tree[0]
                split_val = tree[1]
                label = f'{feature}\n<= {split_val:.2f}'
                dot.node(str(node_id), label)
                # Left child
                node_counter[0] += 1
                left_child_id = node_counter[0]
                add_nodes_edges(tree[2], parent=node_id)
                dot.edge(str(node_id), str(left_child_id), label='True')
                # Right child
                node_counter[0] += 1
                right_child_id = node_counter[0]
                add_nodes_edges(tree[3], parent=node_id)
                dot.edge(str(node_id), str(right_child_id), label='False')
            return

        add_nodes_edges(tree)
        # Save and render the graph
        dot.render(file_path, format='png', cleanup=True)
        print(f"Decision tree visualization saved to {file_path}.png")

    def test_decision_tree(self, tree, features, expected_values):
        for i, data_point in enumerate(features):
            predicted_value = self.query_point(data_point, tree)
            expected_value = expected_values[i]
            print(f"Expected Value: {expected_value}, Predicted Value: {predicted_value:.2f}")



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
