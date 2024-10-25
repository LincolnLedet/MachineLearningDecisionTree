from process_dataset import process_dataset
from DTLearner import DTLearner # Assuming your decision tree learner class is in a file called dt_learner.py

def main():
    # Step 1: Process the dataset
    file_path = 'data/wine-simple.csv'
    data, features, target = process_dataset(file_path, ignore_first_column=True)
    print("\nSample of the loaded data:")
    print(data.head())

    # Step 2: Create and train the decision tree learner
    learner = DTLearner(leaf_size=1, max_depth=2)  # You can adjust the leaf_size and max_depth as needed
    learner.add_evidence(data, features, target)
    #learner.visualize_tree(learner.tree, 'decision_tree', features)


    # Step 3: Perform a dummy query (You can adjust this based on your data)
    query_data = {'X11: Alcohol': 12.8, 'X10 Sulphates': 0.74, 'X2: Volatile Acidity': .72},
    prediction = learner.query(query_data)
    print(f"Prediction for query {query_data}: {prediction}")

if __name__ == "__main__":
    main()