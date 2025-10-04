# src/t-sne/tsne_analysis.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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


def plot_tsne(emb: np.ndarray, labels: np.ndarray, save_path: Path, title: str):
    plt.figure(figsize=(7, 6))
    for val, name in [(0, "normal"), (1, "anomaly")]:
        m = (labels == val)
        if m.any():
            plt.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.7, label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
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


def run_tsne_and_save(X: pd.DataFrame,
                      y: np.ndarray,
                      out_dir: Path,
                      tag: str,
                      perplexity: float,
                      learning_rate,
                      metric: str,
                      early_exaggeration: float,
                      max_iter: int,
                      random_state: int = 42) -> float:
    """
    Fit t-SNE on (X, y), save embedding, plot, and return silhouette.
    """
    # standardize
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = pd.Series(y, index=X.index).values

    # drop constants
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        metric=metric,
        early_exaggeration=early_exaggeration,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )
    emb = tsne.fit_transform(Xs)
    sil = silhouette_or_nan(emb, y)

    # save
    pd.DataFrame({"tsne_1": emb[:, 0], "tsne_2": emb[:, 1], "label": y}).to_csv(
        out_dir / f"tsne_embedding{tag}.csv", index=False
    )
    plot_tsne(emb, y, out_dir / f"tsne_scatter{tag}.png",
              f"t-SNE{tag} on {len(X)} rows | perp={perplexity} | metric={metric}")

    with open(out_dir / f"features_used{tag}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(X.columns))

    with open(out_dir / f"report{tag}.txt", "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"Rows used: {len(X)}",
                    f"Features used: {X.shape[1]}",
                    f"Removed constant cols: {const_cols}",
                    f"perplexity: {perplexity}",
                    f"learning_rate: {learning_rate}",
                    f"metric: {metric}",
                    f"early_exaggeration: {early_exaggeration}",
                    f"max_iter: {max_iter}",
                    f"Silhouette score by labels: {sil:.6f}" if np.isfinite(sil) else "Silhouette unavailable",
                    f"Outputs: tsne_embedding{tag}.csv, tsne_scatter{tag}.png, features_used{tag}.txt",
                ]
            )
        )
    return sil


def main():
    ap = argparse.ArgumentParser(description="t-SNE clustering analysis with optional balanced visualization")
    ap.add_argument("--data", default="Data/redoData/downsampleData_scratch_1minut_contact/cleaned_1minut_20251003_154524.parquet", type=str)
    ap.add_argument("--out", default="experiments/tsne_analysis", type=str)
    ap.add_argument("--n_samples", default=10000, type=int)
    ap.add_argument("--label_col", default="anomaly_label", type=str)
    ap.add_argument("--timestamp_col", default="TimeStamp", type=str)
    # t-SNE params
    ap.add_argument("--perplexity", default=30.0, type=float)
    ap.add_argument("--learning_rate", default="auto")   # or set numeric like 200
    ap.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--early_exaggeration", default=12.0, type=float)
    ap.add_argument("--max_iter", default=1000, type=int)
    # visualization balancing
    ap.add_argument("--balance_vis", action="store_true",
                    help="Additionally run a balanced version for visualization by downsampling the majority class")
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

    # run on full data
    sil_full = run_tsne_and_save(
        X=X_full,
        y=y_full,
        out_dir=out_dir,
        tag="",  # no suffix
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        metric=args.metric,
        early_exaggeration=args.early_exaggeration,
        max_iter=args.max_iter,
    )
    print(f"[t-SNE] Silhouette (full): {sil_full}")

    # optional: balanced visualization by downsampling the majority class
    if args.balance_vis:
        labels = df[args.label_col].astype(int)
        idx_pos = labels[labels == 1].index
        idx_neg = labels[labels == 0].index

        if len(idx_pos) == 0 or len(idx_neg) == 0:
            print("[t-SNE] Skip balanced run: only one class present.")
            return

        n_bal = min(len(idx_pos), len(idx_neg))
        idx_pos_s = idx_pos.to_series().sample(n=n_bal, random_state=42).index
        idx_neg_s = idx_neg.to_series().sample(n=n_bal, random_state=42).index
        keep = idx_pos_s.union(idx_neg_s)

        X_bal = X_full.loc[keep]
        y_bal = labels.loc[keep].values

        sil_bal = run_tsne_and_save(
            X=X_bal,
            y=y_bal,
            out_dir=out_dir,
            tag="_balanced",
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
            metric=args.metric,
            early_exaggeration=args.early_exaggeration,
            max_iter=args.max_iter,
        )
        print(f"[t-SNE] Silhouette (balanced): {sil_bal}")


if __name__ == "__main__":
    main()
