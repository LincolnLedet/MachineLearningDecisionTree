import pandas as pd


def process_dataset(file_path, ignore_first_column=False, target_column=None):
    """
    Loads and processes the dataset.

    Parameters:
    - file_path: str, the path to the dataset CSV file
    - ignore_first_column: bool, if True, the first column will be ignored (e.g., row indices)
    - target_column: str, the name of the target column (if None, assumes the last column)

    Returns:
    - data: pd.DataFrame, the loaded data
    - features: list, the list of feature column names
    - target: pd.Series, the target values
    """
    # Load the dataset from the CSV file
    data = pd.read_csv(file_path)

    # Optionally ignore the first column
    if ignore_first_column:
        data = data.iloc[:, 1:]

    # If no target column is specified, use the last column as the target
    if target_column is None:
        target_column = data.columns[-1]

    # Split the data into features and target
    features = data.drop(columns=[target_column]).columns.tolist()
    target = data[target_column]

    print(f"Loaded dataset with {len(data)} rows and {len(features)} features.")
    print(f"Target column: {target_column}")
    print(f"Feature columns: {features}")

    return data, features, target


# Main wrapper to test the function
if __name__ == "__main__":
    # Example usage: Reading the wine-simple.csv file
    file_path = 'data/wine-simple.csv'
    data, features, target = process_dataset(file_path, ignore_first_column=True)

    # Print the first few rows to verify
    print("\nSample of the loaded data:")
    print(data.head())

    print("\nFeatures:", features)
    print("\nTarget values sample:")
    print(target.head())
    