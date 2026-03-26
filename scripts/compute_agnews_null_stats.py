import numpy as np
import pickle
import os

EMBEDDINGS_DIR = r"C:\IP\greenmlops\airflow\include\data\embeddings\ag_news"
NULL_DIST_PAIRS = 500
SUBSAMPLE_N     = 256
SIGMA_MULTIPLIER = 2.0


def median_bandwidth(x):
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(x), size=min(500, len(x)), replace=False)
    samp = x[idx]
    diff = samp[:, None, :] - samp[None, :, :]
    sq   = (diff ** 2).sum(axis=-1)
    return float(np.median(sq[sq > 0]))


def rbf_kernel(a, b, bw):
    diff = a[:, None, :] - b[None, :, :]
    sq   = (diff ** 2).sum(axis=-1)
    return np.exp(-sq / (2.0 * bw))


def compute_mmd(x, y, bandwidth):
    kxx = rbf_kernel(x, x, bandwidth).mean()
    kyy = rbf_kernel(y, y, bandwidth).mean()
    kxy = rbf_kernel(x, y, bandwidth).mean()
    return float(kxx + kyy - 2.0 * kxy)


def main():
    ref_path = os.path.join(EMBEDDINGS_DIR, "ref_embeddings.npy")
    pca_path = os.path.join(EMBEDDINGS_DIR, "pca_model.pkl")
    out_path = os.path.join(EMBEDDINGS_DIR, "mmd_null_stats.npy")

    ref_embeddings = np.load(ref_path)
    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)

    ref_pca   = pca_model.transform(ref_embeddings)
    bandwidth = median_bandwidth(ref_pca)

    print(f"ref_pca shape : {ref_pca.shape}")
    print(f"bandwidth     : {bandwidth:.6f}")

    rng       = np.random.default_rng(42)
    null_mmds = []

    for i in range(NULL_DIST_PAIRS):
        idx_a = rng.choice(len(ref_pca), size=SUBSAMPLE_N, replace=True)
        idx_b = rng.choice(len(ref_pca), size=SUBSAMPLE_N, replace=True)
        mmd   = compute_mmd(ref_pca[idx_a], ref_pca[idx_b], bandwidth)
        null_mmds.append(mmd)
        if (i + 1) % 100 == 0:
            print(f"  computed {i + 1}/{NULL_DIST_PAIRS} null pairs")

    null_mmds = np.array(null_mmds)
    mmd_mean  = float(null_mmds.mean())
    mmd_sigma = float(null_mmds.std())
    threshold = mmd_mean + SIGMA_MULTIPLIER * mmd_sigma

    stats = np.array([mmd_mean, mmd_sigma, threshold, bandwidth])
    np.save(out_path, stats)

    print(f"\nnull MMD mean  : {mmd_mean:.6e}")
    print(f"null MMD sigma : {mmd_sigma:.6e}")
    print(f"threshold      : {threshold:.6e}")
    print(f"bandwidth      : {bandwidth:.6e}")
    print(f"saved to       : {out_path}")


if __name__ == "__main__":
    main()