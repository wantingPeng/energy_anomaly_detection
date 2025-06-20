from sys import intern
import pandas as pd
import os

def preview_parquet(parquet_path, sample_size=1000, output_csv_path=None):
    """
    Load and preview a Parquet file.
    
    Args:
        parquet_path (str): Path to the input Parquet file.
        sample_size (int, optional): Number of rows to sample for inspection. Defaults to 1000.
        output_csv_path (str, optional): If given, saves the sample to a CSV file.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # 1. Load the parquet file
    print(f"Loading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # 2. Show basic info
    print("\n=== Dataset Info ===")
    print(df.info())
    print("\n=== last 5 rows ===")
    print(df.tail())

    # 3. Sampling a small subset for manual checking
    if sample_size > 0:
        sample_df = df.head(sample_size)#df.sample(n=min(sample_size, len(df)), random_state=42)
        if output_csv_path:
            print(f"\nSaving a sample of {len(sample_df)} rows to {output_csv_path}")
            sample_df.to_csv(output_csv_path, index=False)
        else:
            print("\nSample data first 5 rows:")
            print(sample_df.head())
    else:
        print("\nSample size <= 0, no sample saved.")

if __name__ == "__main__":
    # ====== Customize below ======
    parquet_file = "Data/processed/lsmt_timeFeatures/add_timeFeatures/train/contact/batch_0/part.0.parquet"
    #sample_output_csv = "Data/data_preview/add_timeFeature/contact_batch_0.csv"
    # ====== ================== ======

    preview_parquet(parquet_file, sample_size=1000, output_csv_path=None)
