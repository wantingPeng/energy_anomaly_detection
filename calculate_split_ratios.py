import pandas as pd
from sklearn.model_selection import train_test_split

def calculate_anomaly_ratio_for_split(df, split_name):
    """Calculate anomaly ratio for a given dataframe split"""
    anomaly_count = (df['anomaly_label'] == 1).sum()
    normal_count = (df['anomaly_label'] == 0).sum()
    total_count = len(df)
    
    anomaly_ratio = anomaly_count / total_count
    normal_ratio = normal_count / total_count
    
    print(f"\n===== {split_name} Set Statistics =====")
    print(f"Total records: {total_count}")
    print(f"Anomaly count: {anomaly_count} ({anomaly_ratio:.4%})")
    print(f"Normal count: {normal_count} ({normal_ratio:.4%})")
    print(f"Anomaly to normal ratio: 1:{normal_count/anomaly_count:.2f}")
    
    return anomaly_count, normal_count, anomaly_ratio

def split_and_calculate_ratios(parquet_path, train_size=0.7, test_size=0.15, val_size=0.15):
    """
    Split the dataset into train, test, and validation sets and calculate anomaly ratios.
    
    Args:
        parquet_path (str): Path to the parquet file
        train_size (float): Proportion for training set (default: 0.7)
        test_size (float): Proportion for test set (default: 0.15)
        val_size (float): Proportion for validation set (default: 0.15)
    """
    # Check if proportions sum to 1
    if abs(train_size + test_size + val_size - 1.0) > 0.001:
        raise ValueError(f"Split proportions must sum to 1, got: {train_size + test_size + val_size}")
    
    # Load the parquet file
    print(f"Loading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Calculate overall statistics first
    print("\n===== Overall Dataset Statistics =====")
    total_records = len(df)
    anomaly_count = (df['anomaly_label'] == 1).sum()
    normal_count = (df['anomaly_label'] == 0).sum()
    print(f"Total records: {total_records}")
    print(f"Anomaly count: {anomaly_count} ({anomaly_count/total_records:.4%})")
    print(f"Normal count: {normal_count} ({normal_count/total_records:.4%})")
    print(f"Anomaly to normal ratio: 1:{normal_count/anomaly_count:.2f}")
    
    # Split the dataset
    # First split into train and temp (test+val)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(test_size + val_size),
        random_state=42,
        stratify=df['anomaly_label']  # Stratify to maintain the same class distribution
    )
    
    # Then split temp into test and val
    test_df, val_df = train_test_split(
        temp_df, 
        test_size=val_size/(test_size + val_size),  # Relative size
        random_state=42,
        stratify=temp_df['anomaly_label']
    )
    
    # Calculate ratios for each split
    train_stats = calculate_anomaly_ratio_for_split(train_df, "Training")
    test_stats = calculate_anomaly_ratio_for_split(test_df, "Test")
    val_stats = calculate_anomaly_ratio_for_split(val_df, "Validation")
    
    # Verify the split sizes
    print("\n===== Split Verification =====")
    print(f"Training set: {len(train_df)} records ({len(train_df)/total_records:.2%})")
    print(f"Test set: {len(test_df)} records ({len(test_df)/total_records:.2%})")
    print(f"Validation set: {len(val_df)} records ({len(val_df)/total_records:.2%})")

if __name__ == "__main__":
    parquet_file = "experiments/statistic_30_window_features_contact/filtered_window_features.parquet"
    split_and_calculate_ratios(parquet_file, train_size=0.7, test_size=0.15, val_size=0.15)

