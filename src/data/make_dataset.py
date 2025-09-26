import os
import pandas as pd
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def import_dataset(raw_path):
    """Load the raw CSV file"""
    return pd.read_csv(raw_path)

def split_data(df):
    # Drop date column (not useful for modeling, causes errors with sklearn)
    if "date" in df.columns:
        df = df.drop(columns=["date"])    
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')        
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    raw_path = "data/raw_data/raw.csv"
    output_folderpath = "data/processed_data"
    df = import_dataset(raw_path)
    X_train, X_test, y_train, y_test = split_data(df)
    create_folder_if_necessary(output_folderpath)
    save_dataframes(X_train, X_test, y_train, y_test, output_folderpath)
