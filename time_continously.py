import os
import numpy as np
import pandas as pd
from src.utils.logger import logger
import datetime
from pandas import Timestamp

def check_time_continuity_by_segment(npz_file_path):
    """
    Check if the timestamps with the same segment_id are ordered chronologically and continuous.
    
    Args:
        npz_file_path: Path to the NPZ file containing timestamps and segment_ids
    """
    logger.info(f"Checking time continuity by segment in: {npz_file_path}")
    
    try:
        # Load the NPZ file
        npz_data = np.load(npz_file_path, allow_pickle=True)
        
        # Extract timestamps and segment_ids
        timestamps = npz_data['timestamps']
        segment_ids = npz_data['segment_ids']
        
        logger.info(f"Loaded {len(timestamps)} timestamps and {len(segment_ids)} segment_ids")
        logger.info(f"Timestamp type: {type(timestamps[0])}")
        
        # Get unique segment IDs
        unique_segments = np.unique(segment_ids)
        logger.info(f"Found {len(unique_segments)} unique segment IDs")
        
        segments_with_unordered_timestamps = 0
        segments_with_discontinuities = 0
        
        # Check each segment
        for segment_id in unique_segments:
            # Get indices for this segment
            segment_indices = np.where(segment_ids == segment_id)[0]
            
            if len(segment_indices) <= 1:
                logger.info(f"Segment {segment_id} has only {len(segment_indices)} entries, skipping")
                continue
                
            # Get timestamps for this segment
            segment_timestamps = timestamps[segment_indices]
            
            # Convert Timestamp objects to nanoseconds for comparison
            if isinstance(segment_timestamps[0], (pd.Timestamp, Timestamp)):
                # Convert Timestamp to nanoseconds or Unix timestamp (seconds since epoch)
                segment_timestamps_ns = np.array([ts.timestamp() for ts in segment_timestamps])
                
                # Check if timestamps are ordered
                is_ordered = np.all(np.diff(segment_timestamps_ns) >= 0)
                if not is_ordered:
                    segments_with_unordered_timestamps += 1
                    logger.warning(f"Segment {segment_id} has timestamps that are not in ascending order")
                    # Print some examples of unordered timestamps
                    for i in range(len(segment_timestamps_ns)-1):
                        if segment_timestamps_ns[i+1] < segment_timestamps_ns[i]:
                            logger.warning(f"  Unordered pair at index {i}: {segment_timestamps[i]} > {segment_timestamps[i+1]}")
                            if i >= 2:  # Show a bit of context
                                break
                
                # Check if timestamps are continuous (no gaps)
                timestamp_diffs = np.diff(segment_timestamps_ns)
                unique_diffs = np.unique(timestamp_diffs)
                
                # If there's more than one unique difference, flag it
                if len(unique_diffs) > 1:
                    segments_with_discontinuities += 1
                    # Convert seconds back to timedeltas for readability
                    readable_diffs = [pd.Timedelta(seconds=diff) for diff in unique_diffs]
                    logger.warning(f"Segment {segment_id} has non-uniform timestamp differences. Unique diffs: {readable_diffs}")
                
                # Additional check for large gaps
                max_diff = np.max(timestamp_diffs) if len(timestamp_diffs) > 0 else 0
                expected_diff = np.median(timestamp_diffs) if len(timestamp_diffs) > 0 else 0
                
                # If the maximum difference is significantly larger than the expected difference
                if max_diff > 2 * expected_diff and expected_diff > 0:
                    max_diff_td = pd.Timedelta(seconds=max_diff)
                    expected_diff_td = pd.Timedelta(seconds=expected_diff)
                    logger.warning(f"Segment {segment_id} has large timestamp gaps. Max diff: {max_diff_td}, Expected diff: {expected_diff_td}")
            else:
                logger.warning(f"Unexpected timestamp type: {type(segment_timestamps[0])}")
        
        # Summary
        if segments_with_unordered_timestamps == 0 and segments_with_discontinuities == 0:
            logger.info(f"All segments in {os.path.basename(npz_file_path)} have chronologically ordered and continuous timestamps")
        else:
            logger.warning(f"Found {segments_with_unordered_timestamps} segments with unordered timestamps and {segments_with_discontinuities} segments with discontinuities")
            
    except Exception as e:
        logger.error(f"Error processing {npz_file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    # Set the directory containing NPZ files
    npz_dir = "Data/processed/lsmt/standerScaler_in_segment/spilt_after_sliding/sliding_window/contact"
    
    # Get all NPZ files in the directory
    npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('_windows.npz')]
    
    logger.info(f"Found {len(npz_files)} NPZ files to check")
    
    # Check each NPZ file
    for npz_file in npz_files:
        check_time_continuity_by_segment(npz_file)

if __name__ == "__main__":
    main()