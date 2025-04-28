# Duration of Anomalies Preprocessing Summary

## Overview
- Original file: Data/Anomaly_Data/Duration_of_Anomalies.csv
- Preprocessing date: 2025-04-27 11:42:23

## Original Data Statistics
- Total rows: 5893
- Columns: Condition, Comment, Station, Line, Date, Shift, Downtime
- Missing values: {
  "Condition": 0,
  "Comment": 0,
  "Station": 0,
  "Line": 0,
  "Date": 0,
  "Shift": 0,
  "Downtime": 0
}

## Preprocessing Steps
1. Text Field Standardization
   - Stripped whitespace from all text fields
   - Standardized case for categorical fields
   - Preserved original content in Comment field

2. Time Processing
   - Converted Date to timezone-aware timestamps (Europe/Berlin)
   - Generated StartTime and EndTime columns
   - Converted Downtime to minutes (DowntimeMinutes)

3. Data Validation
   - Checked for missing values
   - Validated time intervals
   - Verified data consistency

## Cleaned Data Statistics
- Total rows: 5893
- Missing values: {
  "Condition": 0,
  "Comment": 0,
  "Station": 0,
  "Line": 0,
  "Date": 0,
  "Shift": 0,
  "Downtime": 0,
  "StartTime": 0,
  "DowntimeMinutes": 0,
  "EndTime": 0
}
- Time range: {
  "start": "2023-12-30T11:47:58+00:00",
  "end": "2025-02-25T02:45:50.000000002+00:00"
}
- Downtime statistics: {
  "count": 5893.0,
  "mean": 17.4643842977544,
  "std": 13.691541641403205,
  "min": 0.0,
  "25%": 8.183333333333334,
  "50%": 13.183333333333334,
  "75%": 21.35,
  "max": 95.6
}

## Notes
- All timestamps are timezone-aware (UTC)
- Downtime is stored in minutes
- Original data preserved, cleaned version saved as Parquet
