# SONG `UMAP_dispersion` baseline reports

Source snapshot: `4c711c83502b235ae931f09ba785b86e21e769fd`. These are small, exploratory, single-run measurements on an Apple Silicon Mac with Python 3.11.15, NumPy 2.4.6 and two Numba/BLAS threads. They are **not** a reproduction of the [SONG paper](https://arxiv.org/abs/1912.04896): the paper uses MNIST and Fashion-MNIST, five repetitions, 20-component external PCA and adjusted mutual information (AMI). The notebooks record their exact inputs and parameters.

## Static 600-digit smoke benchmark

The same 600 scikit-learn digits (64 features, pixel values scaled to `[0,1]`) were embedded by each method. SONG uses 10 self-organization steps here for a quick baseline; its default is 100. Standalone UMAP uses its own default epoch budget. Timing excludes JIT warm-up, and compute budgets differ.

| Method | Elapsed seconds | Trustworthiness @10 | K-means AMI | 5-fold kNN label probe |
| --- | ---: | ---: | ---: | ---: |
| SONG core | 0.031 | 0.855 | 0.338 | 0.565 |
| SONG + UMAP dispersion | 0.126 | 0.856 | 0.338 | 0.647 |
| UMAP | 0.463 | 0.982 | 0.905 | 0.973 |
| PCA | 0.001 | 0.817 | 0.492 | 0.575 |

The kNN probe is cross-validation **on the completed 2D embedding**, so it describes label separability rather than out-of-sample prediction accuracy. See [the executed notebook](01_static_baseline.ipynb), [raw CSV](static_baseline.csv), and the [side-by-side SONG dispersion versus UMAP image](figures/umap_dispersion_vs_umap.png).

## PCA-path investigation

The 64-feature data trigger SONG's internal PCA at its default `pc_components=50`. In `fit`, samples are projected to PCA coordinates but coding vectors are projected only when the original feature count exceeds 100. A one-parameter comparison at 30 steps gave:

| Setting | Trustworthiness @10 | K-means AMI |
| --- | ---: | ---: |
| `pc_components=50` | 0.853 | 0.413 |
| `pc_components=64` (bypasses internal PCA) | 0.978 | 0.854 |

Changing `pc_components` also changes preprocessing, so this contrast indicates a high-priority investigation rather than proving the full effect of the code defect. See the [parameter notebook](03_tuning_candidates.ipynb), [CSV](tuning_sweep.csv), and [visual comparison](figures/pca_path_impact.png). The sweep has one random seed and cannot establish a new default.

## Incremental smoke check

A second `fit` on the same model fails because the branch uses the removed `np.int` alias. A process-local alias in [the incremental notebook](02_incremental_baseline.ipynb) permits an exploratory continuation, which yielded normalized, aligned old-point drift of about 0.94 on a synthetic heterogeneous increment. That run does not validate incremental stability. The repository's regression tests currently report **1 passed, 2 expected failures**; fixing them is part of the improvement work.

Run the notebooks and tests using the commands in the [root README](../README.md#reproducible-performance-reports). Figures are saved in [`figures/`](figures/), while CSV files retain unrounded values.
