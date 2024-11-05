from process_dataset import process_dataset
from DTLearner import DTLearner  # Assuming your decision tree learner class is in a file called DTLearner.py
from sklearn.model_selection import train_test_split
import pandas as pd

def test_decision_tree(learner, test_features, expected_values=None):
    predictions = learner.query(test_features)
    for i, predicted_value in enumerate(predictions):
        if expected_values is not None:
            expected_value = expected_values.iloc[i]
            print(f"Expected Value: {expected_value}, Predicted Value: {predicted_value:.2f}")
        else:
            print(f"Prediction for test point {i+1}: {predicted_value:.2f}")

def main():
    # Step 1: Process the dataset
    file_path = 'data/bikes.csv'
    #file_path = 'data/wine-simple.csv'
    data, features, target = process_dataset(file_path, ignore_first_column=True)
    print("\nSample of the loaded data:")
    print(data.head())

    # Step 2: Split the data into training and testing sets
    train_data, test_data, train_target, test_target = train_test_split(
        data, target, test_size=0.2, random_state=42
    )

    # Step 3: Create and train the decision tree
    #  learner
    learner = DTLearner(leaf_size=1, max_depth=4)  # Adjust as needed
    learner.add_evidence(train_data, features, train_target)

    # Step 4: Test the decision tree on the test set
    print("\nTesting the Decision Tree on the test set:")
    test_decision_tree(learner, test_data, test_target)

    # Step 5: Use query_points_wine for querying the model
    #query_points_wine = [
    #    {'alcohol': 9.8, 'sulphates': 0.74, 'volatile acidity': 0.72},
    #]
    query_points_wine = [
          {'X11: Alcohol': 12.8, 'X10 Sulphates': 0.74, 'X2: Volatile Acidity': .72},
          {'X11: Alcohol': 5.5,  'X10 Sulphates': 0.74, 'X2: Volatile Acidity': .72}
        ]

    query_points_bike = [
    {'temperature': 0.3, 'humidity': 0.65, 'windspeed': 0.15},
    {'temperature': 0.5, 'humidity': 0.55, 'windspeed': 0.25}
    ]

    # Convert the list of dictionaries to a pandas DataFrame
    query_df = pd.DataFrame(query_points_bike)

    # Step 6: Query the learner with query_points_wine
    print("\nPredictions for query_points_wine:")
    test_decision_tree(learner, query_df)

if __name__ == "__main__":
    main()
