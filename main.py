from process_dataset import process_dataset
from DTLearner import DTLearner  # Assuming your decision tree learner class is in a file called DTLearner.py
from RTLearner import RTLearner
from BagLearner import BagLearner
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

def test_decision_tree(learner, test_features, expected_values, task_type):
    """
    Test the decision tree with both regression and classification metrics.
    
    Args:
        learner: The decision tree learner instance.
        test_features (DataFrame): The features for testing.
        expected_values (Series): The ground truth values.
        task_type (str): 'regression' or 'classification' indicating the task type.
    """
    predictions = learner.query(test_features)
    for i, predicted_value in enumerate(predictions):
        expected_value = expected_values.iloc[i]
        print(f"Expected Value: {expected_value}, Predicted Value: {predicted_value:.2f}")

    if task_type == 'regression':
        # Calculate and display regression metrics
        rmse = np.sqrt(mean_squared_error(expected_values, predictions))
        mae = mean_absolute_error(expected_values, predictions)
        print("\n------------Regression Metrics:------------")
        print(f"RMSE: {rmse:.2f}") # RMSE is the square root of the average of the squared differences between the predicted values and the actual values.
        print(f"MAE: {mae:.2f}\n") # MAE is the average of the absolute differences between the predicted values and the actual values.

    elif task_type == 'classification':
        # Convert predictions to integer classes if necessary
        predictions = np.round(predictions).astype(int)
        expected_values = expected_values.astype(int)

        # Calculate and display classification metrics
        accuracy = accuracy_score(expected_values, predictions)
        conf_matrix = confusion_matrix(expected_values, predictions)
        precision = precision_score(expected_values, predictions, average='weighted')
        recall = recall_score(expected_values, predictions, average='weighted')
        f1 = f1_score(expected_values, predictions, average='weighted')

        print("\n------------Classification Metrics:------------")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}\n")

    else:
        raise ValueError("Invalid task_type. Choose 'regression' or 'classification'.")

    # Display predictions for reference


def main():
    # Step 1: Process the dataset
    file_path = 'data/winequality-red.csv'
    #file_path = 'data/wine-simple.csv'
    data, features, target = process_dataset(file_path, ignore_first_column=True)
    print("\nSample of the loaded data:")
    print(data.head())

    if target.nunique() <= 10:  # Assuming 10 or fewer unique values is a classification problem
        task_type = 'classification'
    else:
        task_type = 'regression'

    # Step 2: Split the data into training and testing sets
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=0.2, random_state=42
    )

    # Step 3: Create and train the decision tree
    #learner
    learner = BagLearner(learner_type=DTLearner, kwargs={'leaf_size': 1, 'max_depth': 6}, num_learners=20, verbose=True)
    learner.add_evidence(data, features, target)
    learner.visualize_trees()

    # Step 4: Test the decision tree on the test set
    print("\nTesting the Decision Tree on the test set:")
    test_decision_tree(learner, test_data, test_target, task_type)

    # Step 5: Use query_points_wine for querying the model
    #query_points_wine = [
    #    {'alcohol': 9.8, 'sulphates': 0.74, 'volatile acidity': 0.72},
        #]
    query_points_wine = [
        {
            'fixed acidity': 7.4,
            'volatile acidity': 0.7,
            'citric acid': 0.0,
            'residual sugar': 1.9,
            'chlorides': 0.076,
            'free sulfur dioxide': 11,
            'total sulfur dioxide': 34,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4,
        }
    ]


    query_points_bike = [
    {'temperature': -10, 'humidity': 0, 'windspeed': 100000},
    ]

    # Convert the list of dictionaries to a pandas DataFrame
    query_df = pd.DataFrame(query_points_wine)


    # Step 6: Query the learner with query_points_wine
    print("\nPredictions for query_points_wine:")
    test_decision_tree(learner, test_data, test_target, task_type)

if __name__ == "__main__":
    main()
