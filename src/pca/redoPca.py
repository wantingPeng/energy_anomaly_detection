# src/t-sne/pca_analysis.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pick_features(df: pd.DataFrame, label_col: str, ts_col: str) -> pd.DataFrame:
    """
    Select numeric features excluding TimeStamp and label.
    """
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c != label_col]
    if ts_col in cols:
        cols.remove(ts_col)
    return df[cols]


def plot_scatter(emb: np.ndarray, labels: np.ndarray, save_path: Path, title: str):
    plt.figure(figsize=(7, 6))
    for val, name in [(0, "normal"), (1, "anomaly")]:
        m = (labels == val)
        if m.any():
            plt.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.7, label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def silhouette_or_nan(emb: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return silhouette_score(emb, y, metric="euclidean")
    except Exception:
        return float("nan")


def run_pca_and_save(X: pd.DataFrame, y: np.ndarray, out_dir: Path, tag: str) -> dict:
    """
    Clean -> scale -> PCA(2) -> save embedding, figure, report. Return metrics.
    """
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = pd.Series(y, index=X.index).values

    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    Xs = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(Xs)

    sil = silhouette_or_nan(emb, y)
    evr_sum = float(pca.explained_variance_ratio_.sum())

    # save artifacts
    pd.DataFrame({"pc1": emb[:, 0], "pc2": emb[:, 1], "label": y}).to_csv(
        out_dir / f"pca_embedding{tag}.csv", index=False
    )
    plot_scatter(emb, y, out_dir / f"pca_scatter{tag}.png",
                 f"PCA{tag} on {len(X)} rows | EVR={evr_sum:.4f}")
    with open(out_dir / f"features_used{tag}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(X.columns))
    with open(out_dir / f"report{tag}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join([
            f"Rows used: {len(X)}",
            f"Features used: {X.shape[1]}",
            f"Removed constant cols: {const_cols}",
            f"Explained variance ratio (PC1+PC2): {evr_sum:.6f}",
            f"Silhouette score by labels: {sil:.6f}" if np.isfinite(sil) else "Silhouette unavailable",
            f"Outputs: pca_embedding{tag}.csv, pca_scatter{tag}.png, features_used{tag}.txt",
        ]))
    return {"silhouette": sil, "evr_sum": evr_sum, "const_cols": const_cols}


def main():
    ap = argparse.ArgumentParser(description="PCA baseline with optional balanced visualization")
    ap.add_argument("--data", default="Data/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet", type=str)
    ap.add_argument("--out", default="experiments/pca_analysis", type=str)
    ap.add_argument("--n_samples", default=10000, type=int)
    ap.add_argument("--label_col", default="anomaly_label", type=str)
    ap.add_argument("--timestamp_col", default="TimeStamp", type=str)
    ap.add_argument("--balance_vis", action="store_true",
                    help="Also run a class-balanced visualization by downsampling majority class")
    args = ap.parse_args()

    data_path, out_dir = Path(args.data), Path(args.out)
    ensure_dir(out_dir)

    df = pd.read_parquet(data_path).head(args.n_samples).copy()
    if args.label_col not in df.columns:
        raise ValueError(f"Label column {args.label_col} not found")
    if args.timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column {args.timestamp_col} not found")

    y_full = df[args.label_col].astype(int).values
    X_full = pick_features(df, label_col=args.label_col, ts_col=args.timestamp_col)

    # full data
    metrics_full = run_pca_and_save(X_full, y_full, out_dir, tag="")
    print(f"[PCA] EVR(full)={metrics_full['evr_sum']:.4f}  Silhouette(full)={metrics_full['silhouette']:.6f}")

    # optional balanced visualization
    if args.balance_vis:
        labels = df[args.label_col].astype(int)
        idx_pos = labels[labels == 1].index
        idx_neg = labels[labels == 0].index
        if len(idx_pos) == 0 or len(idx_neg) == 0:
            print("[PCA] Skip balanced run: only one class present.")
            return
        n_bal = min(len(idx_pos), len(idx_neg))
        idx_pos_s = idx_pos.to_series().sample(n=n_bal, random_state=42).index
        idx_neg_s = idx_neg.to_series().sample(n=n_bal, random_state=42).index
        keep = idx_pos_s.union(idx_neg_s)

        X_bal = X_full.loc[keep]
        y_bal = labels.loc[keep].values

        metrics_bal = run_pca_and_save(X_bal, y_bal, out_dir, tag="_balanced")
        print(f"[PCA] EVR(bal)={metrics_bal['evr_sum']:.4f}  Silhouette(bal)={metrics_bal['silhouette']:.6f}")


if __name__ == "__main__":
    main()
