import dask.dataframe as dd
import pandas as pd
from datetime import datetime
from pathlib import Path
from glob import glob
from src.utils.logger import logger
from src.preprocessing import data_loader, data_save


def find_energy_file(station_name: str) -> str:
    """Glob match and return the latest energy parquet file for a station"""
    pattern = f"Data/interim/Energy_Data_cleaned/{station_name}_*.parquet"
    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No energy file found for station: {station_name}")
    return sorted(matches)[-1]


def check_single_anomaly(row, energy_data_cache, station_mapping):
    """Check energy data availability for a single anomaly row."""
    station = row['station']
    start_time = pd.TimeStamp(row['StartTime'])
    end_time = pd.TimeStamp(row['EndTime'])

    energy_station = station_mapping.get(station)
    if not energy_station:
        logger.warning(f"Unknown station mapping for {station}")
        return "unknown_station"

    if energy_station not in energy_data_cache:
        energy_file = find_energy_file(energy_station)
        energy_df = dd.read_parquet(energy_file).repartition(npartitions=10).persist()
        energy_data_cache[energy_station] = energy_df

    energy_df = energy_data_cache[energy_station]

    # 时间段筛选
    period_data = energy_df[(energy_df['TimeStamp'] >= start_time) & 
                            (energy_df['TimeStamp'] <= end_time)]
    count = period_data.shape[0].compute()

    if count == 0:
        return "none"

    expected = int((end_time - start_time).total_seconds()) + 1
    if count >= expected * 0.95:
        return "full"
    else:
        actual_range = period_data[['TimeStamp']].compute()
        logger.info(f"Partial data: {start_time} ~ {end_time}, available: {actual_range.min()[0]} ~ {actual_range.max()[0]}")
        return "partial"


def check_energy_data_availability():
    logger.info("🔍 Starting energy data availability check...")

    anomaly_path = "Data/interim/Anomaly_Data/Duration_of_Anomalies_cleaned.parquet"
    anomaly_df = data_loader(anomaly_path)
    logger.info(f"✅ Loaded {len(anomaly_df)} anomaly records")

    station_mapping = {
        "Kontaktieren": "contacting",
        "Pcb": "pcb",
        "Ringmontage": "ring"
    }

    energy_data_cache = {}
    anomaly_df['EnergyDataStatus'] = anomaly_df.apply(
        lambda row: check_single_anomaly(row, energy_data_cache, station_mapping),
        axis=1
    )

    output_path = "Data/interim/Anomaly_Data/Anomalies_EnergyDataStatus.parquet"
    data_save(anomaly_df, output_path)
    logger.info(f"💾 Saved updated anomalies to {output_path}")

    stats = anomaly_df['EnergyDataStatus'].value_counts()
    report_path = "experiments/reports/EnergyDataStatus.md"
    report_md = f"""# ⚡ Energy Data Status Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Statistics
{stats.to_markdown()}

## Summary
- Total: {len(anomaly_df)}
- Full: {stats.get('full', 0)}
- Partial: {stats.get('partial', 0)}
- None: {stats.get('none', 0)}
"""

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_md)

    logger.info(f"📄 Report written to {report_path}")
