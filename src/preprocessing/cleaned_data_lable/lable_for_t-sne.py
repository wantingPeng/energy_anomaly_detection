#!/usr/bin/env python3
"""
Label Contacting data based on anomaly time intervals for t-SNE analysis.

This script processes Contacting energy data and labels each timestamp as 1 if it falls
within any Kontaktieren anomaly time interval, otherwise 0.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from tqdm import tqdm
import gc

import sys
from pathlib import Path
# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.append(str(src_dir))

from utils.logger import logger


def create_anomaly_labels(
    contacting_file: str = "Data/machine/cleaning_utc/Contacting_cleaned.parquet",
    kontaktieren_times_file: str = "Data/cleaned_data_lable_t-sne/kontaktieren_times.parquet",
    output_dir: str = "Data/cleaned_data_lable_t-sne",
    output_filename: str = "contacting_labeled_data.parquet",
    chunk_size: int = 100000
) -> Tuple[str, dict]:
    """
    Create anomaly labels for Contacting data based on Kontaktieren time intervals.
    
    Args:
        contacting_file (str): Path to Contacting cleaned parquet file
        kontaktieren_times_file (str): Path to Kontaktieren times parquet file
        output_dir (str): Directory to save labeled output
        output_filename (str): Name of output file
        chunk_size (int): Size of chunks for processing large data
        
    Returns:
        Tuple[str, dict]: Output file path and statistics
    """
    try:
        logger.info("Starting anomaly labeling process for Contacting data")
        
        # Construct paths
        contacting_path = Path(contacting_file)
        kontaktieren_path = Path(kontaktieren_times_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        full_output_path = output_path / output_filename
        
        logger.info(f"Contacting data file: {contacting_path}")
        logger.info(f"Kontaktieren times file: {kontaktieren_path}")
        logger.info(f"Output file: {full_output_path}")
        
        # Load Kontaktieren anomaly time intervals
        logger.info("Loading Kontaktieren anomaly time intervals...")
        anomaly_intervals = pd.read_parquet(kontaktieren_path)
        logger.info(f"Loaded {len(anomaly_intervals)} anomaly time intervals")
        print(f"Loaded {len(anomaly_intervals)} anomaly time intervals")
        
        # Convert to numpy arrays for faster processing
        start_times = anomaly_intervals['StartTime'].values
        end_times = anomaly_intervals['EndTime'].values
        
        # Sort intervals by start time for efficient searching
        sort_idx = np.argsort(start_times)
        start_times = start_times[sort_idx]
        end_times = end_times[sort_idx]
        
        logger.info(f"Time intervals range: {start_times[0]} to {end_times[-1]}")
        print(f"Time intervals range: {pd.to_datetime(start_times[0])} to {pd.to_datetime(end_times[-1])}")
        
        # Get total number of rows for progress tracking
        logger.info("Getting total number of rows in Contacting data...")
        total_rows = len(pd.read_parquet(contacting_path, columns=['ID']))
        logger.info(f"Total rows to process: {total_rows:,}")
        print(f"Total rows to process: {total_rows:,}")
        
        # Process data in chunks
        logger.info(f"Processing data in chunks of {chunk_size:,} rows...")
        
        chunk_results = []
        stats = {
            'total_rows': 0,
            'anomaly_rows': 0,
            'normal_rows': 0,
            'chunks_processed': 0
        }
        
        # Read and process data in chunks using pyarrow
        parquet_file = pq.ParquetFile(contacting_path)
        
        # Calculate number of chunks
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        logger.info(f"Will process {num_chunks} chunks")
        
        chunk_idx = 0
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            chunk_df = batch.to_pandas()
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_df):,} rows)")
            
            # Extract timestamps
            timestamps = chunk_df['TimeStamp'].values
            
            # Create labels using vectorized operations
            labels = label_timestamps_vectorized(timestamps, start_times, end_times)
            
            # Add label column to chunk
            chunk_df['anomaly_label'] = labels
            
            # Update statistics
            anomaly_count = np.sum(labels)
            normal_count = len(labels) - anomaly_count
            
            stats['total_rows'] += len(chunk_df)
            stats['anomaly_rows'] += anomaly_count
            stats['normal_rows'] += normal_count
            stats['chunks_processed'] += 1
            
            # Store chunk result
            chunk_results.append(chunk_df)
            
            # Log progress
            anomaly_percentage = (anomaly_count / len(chunk_df)) * 100
            logger.info(f"  Chunk {chunk_idx + 1}: {anomaly_count:,} anomalies ({anomaly_percentage:.2f}%)")
            print(f"  Chunk {chunk_idx + 1}: {anomaly_count:,} anomalies ({anomaly_percentage:.2f}%)")
            
            # Save intermediate results every 10 chunks to manage memory
            if len(chunk_results) >= 10:
                logger.info("Saving intermediate results...")
                intermediate_df = pd.concat(chunk_results, ignore_index=True)
                
                if chunk_idx == 9:  # First save
                    intermediate_df.to_parquet(full_output_path, index=False)
                else:  # Append to existing file
                    existing_df = pd.read_parquet(full_output_path)
                    combined_df = pd.concat([existing_df, intermediate_df], ignore_index=True)
                    combined_df.to_parquet(full_output_path, index=False)
                    del existing_df, combined_df
                
                # Clear memory
                del chunk_results, intermediate_df
                chunk_results = []
                gc.collect()
            
            chunk_idx += 1
        
        # Save remaining chunks
        if chunk_results:
            logger.info("Saving final chunks...")
            final_df = pd.concat(chunk_results, ignore_index=True)
            
            if Path(full_output_path).exists():
                existing_df = pd.read_parquet(full_output_path)
                combined_df = pd.concat([existing_df, final_df], ignore_index=True)
                combined_df.to_parquet(full_output_path, index=False)
                del existing_df, combined_df
            else:
                final_df.to_parquet(full_output_path, index=False)
            
            del final_df, chunk_results
            gc.collect()
        
        # Verify final results
        logger.info("Verifying final results...")
        final_df = pd.read_parquet(full_output_path)
        actual_total = len(final_df)
        actual_anomalies = final_df['anomaly_label'].sum()
        actual_normal = actual_total - actual_anomalies
        
        if actual_total != stats['total_rows']:
            logger.warning(f"Row count mismatch: expected {stats['total_rows']}, got {actual_total}")
        
        # Update final statistics
        stats['total_rows'] = actual_total
        stats['anomaly_rows'] = actual_anomalies
        stats['normal_rows'] = actual_normal
        stats['anomaly_percentage'] = (actual_anomalies / actual_total) * 100
        
        # Log final statistics
        logger.info("Labeling process completed successfully")
        logger.info(f"Final statistics:")
        logger.info(f"  - Total rows: {stats['total_rows']:,}")
        logger.info(f"  - Anomaly rows (label=1): {stats['anomaly_rows']:,} ({stats['anomaly_percentage']:.3f}%)")
        logger.info(f"  - Normal rows (label=0): {stats['normal_rows']:,}")
        logger.info(f"  - Output file: {full_output_path}")
        
        print(f"\n{'='*70}")
        print("LABELING SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Total rows processed: {stats['total_rows']:,}")
        print(f"🔴 Anomaly rows (label=1): {stats['anomaly_rows']:,} ({stats['anomaly_percentage']:.3f}%)")
        print(f"🟢 Normal rows (label=0): {stats['normal_rows']:,}")
        print(f"📁 Saved to: {full_output_path}")
        print(f"{'='*70}")
        
        return str(full_output_path), stats
        
    except Exception as e:
        logger.error(f"Error in create_anomaly_labels: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise


def label_timestamps_vectorized(timestamps, start_times, end_times):
    """
    Efficiently label timestamps using vectorized operations.
    
    Args:
        timestamps: Array of timestamps to label
        start_times: Array of anomaly start times (sorted)
        end_times: Array of anomaly end times (corresponding to start_times)
        
    Returns:
        Array of labels (1 for anomaly, 0 for normal)
    """
    labels = np.zeros(len(timestamps), dtype=np.int8)
    
    with tqdm(total=len(timestamps), desc="Labeling timestamps", leave=False) as pbar:
        # For each timestamp, check if it falls in any interval
        for i, ts in enumerate(timestamps):
            # Use binary search to find potential intervals
            # Find intervals where start_time <= ts
            right_bound = np.searchsorted(start_times, ts, side='right')
            
            # Check intervals that could contain this timestamp
            for j in range(right_bound):
                if start_times[j] <= ts <= end_times[j]:
                    labels[i] = 1
                    break
            
            if i % 10000 == 0:
                pbar.update(10000)
        
        # Update remaining progress
        pbar.update(len(timestamps) % 10000)
    
    return labels


def create_sample_labeled_data(
    input_file: str = "Data/cleaned_data_lable_t-sne/contacting_labeled_data.parquet",
    output_dir: str = "Data/cleaned_data_lable_t-sne",
    sample_size: int = 100000,
    anomaly_ratio: float = 0.5
) -> str:
    """
    Create a balanced sample of labeled data for t-SNE analysis.
    
    Args:
        input_file (str): Path to full labeled dataset
        output_dir (str): Output directory
        sample_size (int): Total size of sample
        anomaly_ratio (float): Ratio of anomaly samples in the output
        
    Returns:
        str: Path to sample file
    """
    try:
        logger.info(f"Creating balanced sample of {sample_size:,} records")
        
        output_path = Path(output_dir)
        sample_file = output_path / f"contacting_sample_{sample_size}_balanced.parquet"
        
        # Load data
        df = pd.read_parquet(input_file)
        
        # Calculate sample sizes
        anomaly_sample_size = int(sample_size * anomaly_ratio)
        normal_sample_size = sample_size - anomaly_sample_size
        
        # Sample data
        anomaly_data = df[df['anomaly_label'] == 1].sample(n=min(anomaly_sample_size, (df['anomaly_label'] == 1).sum()), random_state=42)
        normal_data = df[df['anomaly_label'] == 0].sample(n=min(normal_sample_size, (df['anomaly_label'] == 0).sum()), random_state=42)
        
        # Combine and shuffle
        sample_df = pd.concat([anomaly_data, normal_data], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save sample
        sample_df.to_parquet(sample_file, index=False)
        
        logger.info(f"Sample created: {len(sample_df):,} records ({(sample_df['anomaly_label'] == 1).sum():,} anomalies)")
        print(f"✅ Sample created: {len(sample_df):,} records ({(sample_df['anomaly_label'] == 1).sum():,} anomalies)")
        print(f"📁 Saved to: {sample_file}")
        
        return str(sample_file)
        
    except Exception as e:
        logger.error(f"Error creating sample: {str(e)}")
        raise


def main():
    """Main function to run the labeling process."""
    try:
        # Create labels for full dataset
        output_path, stats = create_anomaly_labels()
        
        # Create a balanced sample for t-SNE
        if stats['anomaly_rows'] > 0:
            sample_path = create_sample_labeled_data(input_file=output_path)
            print(f"✅ Also created balanced sample for t-SNE analysis: {sample_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"❌ Execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
