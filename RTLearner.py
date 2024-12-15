from DTLearner import DTLearner
import numpy as np
class RTLearner(DTLearner):
    def __init__(self, leaf_size=1, max_depth=5, verbose=False):
        # Call the parent class (DTLearner) constructor
        super().__init__(leaf_size=leaf_size, max_depth=max_depth, verbose=verbose)
        print(f"Initialized RTLearner with leaf_size={leaf_size} and max_depth={max_depth}")

    # Override or add methods if you need different behavior
    # For example, you might want a different tree-building strategy:
    def build_tree(self, data, features, target, depth=0):
        print(f"Building tree at depth {depth} with max_depth {self.max_depth}")

        # Base case: Stop if we exceed max_depth or have few samples
        if depth >= self.max_depth or len(data) <= self.leaf_size:
            print(f"Reached base case at depth {depth}, creating a leaf node with target mean.")
            return ["Leaf", target.mean()]  # Correct leaf node representation

        # Select a random feature at each node
        print(f"Selecting the best feature at depth {depth} from features: {features}") # <------------------------
        best_feature = np.random.choice(features) # <--------------------------------------------------------------

        if best_feature is None:
            print(f"No valid feature selected at depth {depth}. Creating a leaf node.")
            return ["Leaf", target.mean()]  # Create a leaf node if no feature is selected 
        
            # Check if the feature is categorical
        if data[best_feature].dtype == 'object':  # Categorical feature <------------------------
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
    
        # Add any new methods specific to RTLearner here if needed