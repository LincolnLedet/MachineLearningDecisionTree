import numpy as np
from RTLearner import RTLearner
from DTLearner import DTLearner

class BagLearner:
    def __init__(self, learner_type, kwargs=None, num_learners=10, verbose=False):
        """
        Initialize the BagLearner.

        Args:
            learner_type (class): The learner class, e.g., RTLearner or DTLearner.
            kwargs (dict): Arguments to initialize the learner.
            num_learners (int): The number of learners to use in the ensemble.
            verbose (bool): If True, print detailed logs.
        """
        self.learners = []
        self.num_learners = num_learners
        self.verbose = verbose
        self.learner_type = learner_type

        # Initialize each learner with the provided kwargs
        for _ in range(num_learners):
            learner = learner_type(**(kwargs if kwargs else {}))
            self.learners.append(learner)
        if verbose:
            print(f"Initialized BagLearner with {num_learners} {learner_type.__name__}s")

    def add_evidence(self, data, features, target):
        """
        Train each learner on a random subset of the data.

        Args:
            data (DataFrame): The training data.
            features (list): The list of feature names.
            target (Series): The target values.
        """
        n = len(data)
        for i, learner in enumerate(self.learners):
            # Bootstrap sample (sample with replacement)
            indices = np.random.choice(n, n, replace=True)
            bootstrap_data = data.iloc[indices]
            bootstrap_target = target.iloc[indices]
            learner.add_evidence(bootstrap_data, features, bootstrap_target)
            if self.verbose:
                print(f"Trained learner {i+1} on bootstrap sample")

    def query(self, points):
        """
        Query each learner and average the results.

        Args:
            points (DataFrame): The data points to query.
        
        Returns:
            np.array: The averaged predictions from all learners.
        """
        predictions = []
        for i, learner in enumerate(self.learners):
            learner_predictions = learner.query(points)
            predictions.append(learner_predictions)
            if self.verbose:
                print(f"Learner {i+1} predictions: {learner_predictions}")

        # Average the predictions across all learners
        avg_predictions = np.mean(predictions, axis=0)
        return avg_predictions
    def visualize_trees(self):
        """
        Visualize all the trees in the ensemble.
        """
        for i, learner in enumerate(self.learners):
            file_path = f"tree_{i + 1}"
            print(f"Visualizing tree {i + 1}")
            learner.visualize_tree(learner.tree, file_path)


# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Create some dummy data
    data = pd.DataFrame({
        'X2: Volatile Acidity': [0.56, 0.72, 0.89, 0.47, 0.65],
        'X10: Sulphates': [0.5, 0.6, 0.8, 0.3, 0.55],
        'X11: Alcohol': [9.4, 10.9, 9.1, 8.7, 10.2]
    })
    target = pd.Series([6, 5, 4, 7, 6], name='Y: Quality')  # Dummy target
    features = ['X2: Volatile Acidity', 'X10: Sulphates', 'X11: Alcohol']

    # Initialize BagLearner with RTLearner
    bag_learner = BagLearner(learner_type=RTLearner, kwargs={'leaf_size': 1, 'max_depth': 3}, num_learners=5, verbose=True)
    bag_learner.add_evidence(data, features, target)

    # Query the BagLearner with some data points
    query_points = pd.DataFrame([
        {'X2: Volatile Acidity': 0.6, 'X10: Sulphates': 0.55, 'X11: Alcohol': 9.5},
        {'X2: Volatile Acidity': 0.4, 'X10: Sulphates': 0.35, 'X11: Alcohol': 8.3}
    ])
    predictions = bag_learner.query(query_points)
    print(f"Bagged Predictions: {predictions}")
