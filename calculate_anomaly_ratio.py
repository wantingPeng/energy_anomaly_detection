import pandas as pd

def calculate_anomaly_ratio(parquet_path):
    """
    Calculate the ratio of anomalies to normal data in a parquet file.
    
    Args:
        parquet_path (str): Path to the parquet file containing the data
        
    Returns:
        tuple: (anomaly_count, normal_count, anomaly_ratio)
    """
    # Load the parquet file
    print(f"Loading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Count anomalies and normal data
    anomaly_count = (df['anomaly_label'] == 1).sum()
    normal_count = (df['anomaly_label'] == 0).sum()
    total_count = len(df)
    
    # Calculate ratios
    anomaly_ratio = anomaly_count / total_count
    normal_ratio = normal_count / total_count
    
    # Print results
    print(f"\n===== Anomaly Statistics =====")
    print(f"Total records: {total_count}")
    print(f"Anomaly count: {anomaly_count} ({anomaly_ratio:.4%})")
    print(f"Normal count: {normal_count} ({normal_ratio:.4%})")
    print(f"Anomaly to normal ratio: 1:{normal_count/anomaly_count:.2f}")
    
    return anomaly_count, normal_count, anomaly_ratio

if __name__ == "__main__":
    parquet_file = "experiments/statistic_40_window_features_ring/filtered_window_features_40.parquet"
    calculate_anomaly_ratio(parquet_file)
