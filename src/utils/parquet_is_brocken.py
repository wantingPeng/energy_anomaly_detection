from pyarrow.parquet import ParquetFile
from pathlib import Path

path = Path("Data/processed/lsmt/standerScaler/train.parquet")
for f in path.glob("*.parquet"):
    try:
        ParquetFile(str(f))
    except Exception as e:
        print(f"❌ {f.name} is corrupted: {e}")
