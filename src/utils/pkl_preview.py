import pickle
import os

def load_anomaly_dict(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")

    try:
        with open(pkl_path, "rb") as f:
            anomaly_dict = pickle.load(f)
        return anomaly_dict
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file: {e}")

def preview_anomaly_dict(anomaly_dict, max_stations=3, max_entries=10):
    for station_idx, (station, intervals) in enumerate(anomaly_dict.items()):
        print(f"Station: {station}")
        for i, (start, end) in enumerate(intervals[:max_entries]):
            print(f"  [{i}] {start}  {end}")

if __name__ == "__main__":
    pkl_file_path = "experiments/random_forest/statistic_feature_1024s_256/random_forest_20251019_122442_contact/model/model_params.pkl"
    
    print(f"Loading anomaly dict from: {pkl_file_path}")
    anomaly_dict = load_anomaly_dict(pkl_file_path)

    print("\nSample preview of anomaly intervals:")
    preview_anomaly_dict(anomaly_dict)
