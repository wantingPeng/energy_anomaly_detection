from src.preprocessing.data_loader import data_loader
import matplotlib.pyplot as plt

def main():
    df = data_loader(
        "Data/interim/Energy_Data_cleaned/ring_20250503_075538/*.parquet"
    )
    segment_lengths = df.groupby("segment_id").size()
    summary = segment_lengths.describe().compute()
    print(df.head())
    print("Segment lengths summary:\n", summary)

    # 直方图可视化 segment 长度分布（限最大1000显示）
    lengths = segment_lengths.compute()
    lengths[lengths < 1000].plot.hist(bins=50)
    plt.xlabel("Segment Length (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Segment Lengths (<1000s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.savefig("segment_length_hist.png", dpi=300)
if __name__ == "__main__":
    main()