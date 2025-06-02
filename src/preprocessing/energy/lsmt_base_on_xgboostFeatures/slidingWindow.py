import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.logger import logger
import gc
import json

def apply_sliding_window(df, window_size=6, step_size=2):
    """
    Apply sliding window to create LSTM-ready sequences from features.
    
    Args:
        df (pandas.DataFrame): DataFrame containing statistical features from sliding windows
        window_size (int, optional): Size of the sliding window. Defaults to 6.
        step_size (int, optional): Step size for sliding. Defaults to 2.
    
    Returns:
        tuple: (X, y) where:
            - X is a numpy array of shape (num_samples, window_size, num_features)
            - y is a numpy array of shape (num_samples,)
    """
    logger.info(f"Starting sliding window process with window_size={window_size}, step_size={step_size}")
    
    # Columns to exclude from features
    exclude_cols = ['window_start', 'window_end', 'segment_id', 'anomaly_label', 'overlap_ratio', 'step_size']
    
    # Get all unique segment_ids
    segment_ids = df['segment_id'].unique()
    logger.info(f"Found {len(segment_ids)} unique segment_ids")
    
    X_list = []
    y_list = []
    
    total_sequences = 0
    skipped_segments = 0
    segments_with_anomalies = 0
    
    try:
        for segment_id in tqdm(segment_ids, desc="Processing segments"):
            try:
                # Get data for this segment and sort by window_start
                segment_df = df[df['segment_id'] == segment_id].sort_values('window_start')
                
                # Skip if there are not enough samples in this segment
                if len(segment_df) < window_size:
                    logger.info(f"Skipping segment_id {segment_id} with only {len(segment_df)} samples (less than window_size {window_size})")
                    skipped_segments += 1
                    continue
                
                # Create a list to store segment sequences
                segment_X = []
                segment_y = []
                
                # Check if segment has any anomalies
                has_anomalies = segment_df['anomaly_label'].sum() > 0
                if has_anomalies:
                    segments_with_anomalies += 1
                
                # Apply sliding window within this segment
                for i in range(0, len(segment_df) - window_size + 1, step_size):
                    window = segment_df.iloc[i:i+window_size]
                    
                    # Determine label for this window based on anomaly_label values
                    anomaly_count = window['anomaly_label'].sum()
                    window_label = 1 if anomaly_count >= 2 else 0
                    
                    # Extract features (excluding non-feature columns)
                    features = window.drop(columns=exclude_cols, errors='ignore')
                    
                    # Add to lists
                    segment_X.append(features.values)
                    segment_y.append(window_label)
                
                # Add segment sequences to main lists
                X_list.extend(segment_X)
                y_list.extend(segment_y)
                
                total_sequences += len(segment_X)
                
                # Clear segment data to free memory
                del segment_df, segment_X, segment_y
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing segment {segment_id}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(segment_ids)} segments, skipped {skipped_segments} segments with insufficient samples")
        logger.info(f"Segments with anomalies: {segments_with_anomalies} ({segments_with_anomalies/(len(segment_ids)-skipped_segments)*100:.2f}%)")
        
        # Convert lists to numpy arrays
        if not X_list:
            logger.warning("No valid sequences were generated")
            return np.array([]), np.array([])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        logger.info(f"Class distribution - Normal: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.2f}%), Anomaly: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.2f}%)")
        
        return X, y, {
            "total_segments": len(segment_ids),
            "processed_segments": len(segment_ids) - skipped_segments,
            "skipped_segments": skipped_segments,
            "segments_with_anomalies": segments_with_anomalies,
            "segments_with_anomalies_pct": segments_with_anomalies/(len(segment_ids)-skipped_segments)*100,
            "total_sequences": len(y),
            "normal_sequences": int(sum(y == 0)),
            "anomaly_sequences": int(sum(y == 1)),
            "normal_sequences_pct": float(sum(y == 0)/len(y)*100),
            "anomaly_sequences_pct": float(sum(y == 1)/len(y)*100)
        }
    
    except Exception as e:
        logger.error(f"Error during sliding window processing: {str(e)}")
        # Return what we've processed so far if any
        if X_list:
            return np.array(X_list), np.array(y_list), {}
        return np.array([]), np.array([]), {}

def main():
    """
    Main function to process the dataset and save results.
    """
    try:
        logger.info(f"Starting sliding window sequence creation for LSTM")
        
        # Input and output paths
        input_file = "Data/processed/lsmt_base_on_xgboostFeatures/standerlizes/contact/standardized_data.parquet"
        output_dir = "Data/processed/lsmt_base_on_xgboostFeatures/slidingWindow/contact"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded data with shape {df.shape}")
        
        # Apply sliding window
        X, y, stats = apply_sliding_window(df)
        
        # Free memory
        del df
        gc.collect()
        
        if len(X) == 0:
            logger.error("No sequences were generated. Exiting.")
            return
        
        # Save results
        X_file = os.path.join(output_dir, "X_sequences.npy")
        y_file = os.path.join(output_dir, "y_labels.npy")
        
        logger.info(f"Saving X with shape {X.shape} to {X_file}")
        np.save(X_file, X)
        
        logger.info(f"Saving y with shape {y.shape} to {y_file}")
        np.save(y_file, y)
        
        # Save metadata
        metadata = {
            'X_shape': X.shape,
            'y_shape': y.shape,
            'window_size': 6,
            'step_size': 2,
            'feature_count': X.shape[2],
            'processing_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stats': stats
        }
        
        # Save metadata as text file
        metadata_file = os.path.join(output_dir, "metadata.txt")
        with open(metadata_file, 'w') as f:
            for key, value in metadata.items():
                if key != 'stats':
                    f.write(f"{key}: {value}\n")
            
            if stats:
                f.write("\nClass Distribution:\n")
                f.write(f"Normal sequences (0): {stats['normal_sequences']} ({stats['normal_sequences_pct']:.2f}%)\n")
                f.write(f"Anomaly sequences (1): {stats['anomaly_sequences']} ({stats['anomaly_sequences_pct']:.2f}%)\n")
                
                f.write("\nSegment Statistics:\n")
                f.write(f"Total segments: {stats['total_segments']}\n")
                f.write(f"Processed segments: {stats['processed_segments']}\n")
                f.write(f"Skipped segments: {stats['skipped_segments']}\n")
                f.write(f"Segments with anomalies: {stats['segments_with_anomalies']} ({stats['segments_with_anomalies_pct']:.2f}%)\n")
        
        # Also save metadata as JSON for easier parsing
        json_metadata_file = os.path.join(output_dir, "metadata.json")
        with open(json_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Processing completed successfully")
        logger.info(f"Saved metadata to {metadata_file} and {json_metadata_file}")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
