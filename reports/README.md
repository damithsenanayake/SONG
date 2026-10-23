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

The initial baseline showed that a second `fit` failed because the branch used the removed `np.int` alias. The alias is now fixed and the incremental-fit regression passes. The older [incremental notebook](02_incremental_baseline.ipynb) used a temporary alias and reported normalized, aligned old-point drift of about 0.94 on a synthetic heterogeneous increment; it remains a historical baseline rather than evidence of stability. The current ordered-MNIST experiment is summarized below.

Run the notebooks and tests using the commands in the [root README](../README.md#reproducible-performance-reports). Figures are saved in [`figures/`](figures/), while CSV files retain unrounded values.

## PCA coordinate fix and visual check

The original baseline files above are saved outputs from commit `4c711c8`; rerunning their notebooks on the current branch will recompute them with the improved code. [The five-seed validation notebook](04_pca_fix_validation.ipynb) and [raw results](pca_fix_validation.csv) test the corrected 50-component path on the same 600 digit samples. Means across five seeds are:

| Method | Seconds | Trustworthiness @10 | K-means ARI | kNN label probe |
| --- | ---: | ---: | ---: | ---: |
| SONG, fixed PCA (11 dispersion epochs) | 0.189 | 0.975 | 0.755 | 0.940 |
| SONG, fixed PCA (100 dispersion epochs) | 0.229 | 0.983 | 0.815 | 0.965 |
| SONG, no PCA (11 dispersion epochs) | 0.186 | 0.977 | 0.805 | 0.959 |
| Standalone UMAP | 0.499 | 0.982 | 0.863 | 0.979 |

The pre-fix 30-step, 50-component run scored 0.853 trustworthiness and 0.303 ARI (one seed). The 100-epoch setting is now available through `SONG(um_epochs=100)`; the default remains 11 to preserve the existing runtime. The [comparison plot](figures/pca_fix_embeddings.png) shows the class structure and remaining overlap. See the [five-seed score plot](figures/pca_fix_scores.png) for run-to-run variation. These results are small exploratory checks, not paper-scale replication.

To rerun only the current validation notebook, use `PATH="$PWD/.venv/bin:$PATH" NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python reports/run.py 04_pca_fix_validation.ipynb` from the repository root.


## Full MNIST train/test profile

The [executed MNIST notebook](05_mnist_full_profile.ipynb) profiles the [Keras MNIST 60,000/10,000 split](https://keras.io/api/datasets/mnist/) with 784 float32 pixel features scaled to `[0,1]`. SONG fits only the training images; training labels are used afterward for prototype-majority labels and 2D probes. This is a single-seed run on an Apple Silicon Mac with four Numba and four BLAS threads after JIT warm-up. It is not a paper reproduction.

| SONG steps | Fit seconds | Prototypes | Prototype test accuracy | Core 2D test accuracy | UMAP train dispersion seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 1.75 | 1,049 | 91.58% | 90.90% | 8.01 |
| 100 (default) | 5.04 | 1,557 | 93.81% | 93.54% | 7.92 |

At 100 steps, the pairwise distance kernel uses **0.51 s** of fit time; sequential prototype and map updates in `train_for_batch_batch` use **4.20 s**. The current Numba distance kernel already parallelizes input-row blocks. On a 1,000 × 1,050 × 50D matrix, it took **6.62 ms** with one thread and **1.20 ms** with eight. The [thread-scaling chart](figures/mnist_distance_threads.png) and [full pipeline timing chart](figures/mnist_pipeline_profile.png) show the measurements. For larger batches, improving distance calculation alone has limited end-to-end impact; UMAP refinement and prototype updates dominate.

See the [100-step train/test scatter](figures/mnist_song_core_train_test_100steps_full.png), [dispersion scatter](figures/mnist_song_dispersion_train_test_100steps_full.png), and [prototype confusion matrix](figures/mnist_prototype_confusion_100steps_full.png). The dispersion train and test coordinates come from separate UMAP fits, so cross-frame accuracy is a diagnostic rather than a stable out-of-sample score. Raw measurements are in [`mnist_profile_100steps_full.json`](mnist_profile_100steps_full.json), [`mnist_stage_profile_100steps_full.csv`](mnist_stage_profile_100steps_full.csv), and [`mnist_distance_threads.csv`](mnist_distance_threads.csv). Reproduce them with the commands in the notebook. The downloaded archive and embedding arrays stay in ignored `data/`.


## SONG-initialized UMAP preprint experiments

The [draft preprint](../preprint/SONG_initialized_UMAP.md) and [executed image-rich notebook](06_song_umap_preprint_experiments.ipynb) combine three preliminary results. The reproducible [initialization comparison](song_umap_initialization.py) uses all 60,000 MNIST training images and the held-out 10,000 test images, with matched 50-component PCA input and UMAP parameters. SONG initialization raises 11-epoch held-out 2D 5-NN accuracy from 86.19% to 92.87%, but **total fit time rises from 3.06 s to 7.68 s** once SONG preparation is counted. At 50 epochs, accuracy is 96.09% for spectral and 96.04% for SONG initialization, with total fit times 5.55 s and 10.65 s. See the [comparison image](figures/song_umap_initialization_mnist.png), [CSV](song_umap_initialization.csv), and [metadata](song_umap_initialization_meta.json).

The [ordered heterogeneous-addition experiment](song_increment_mnist.py) fits digits 0–4, then adds digits 5–9. SONG's old-point neighborhood overlap at 15 is only **8.17%** after the second fit; the current implementation does not establish preservation of the prior structure. UMAP `transform` keeps the old points fixed by construction, while UMAP `update` raised `ZeroDivisionError` for this particular increment in the installed release. The [before/after image](figures/song_increment_mnist.png) and [JSON](song_increment_mnist.json) record the run. This is a SONG core test, not an incremental hybrid UMAP test.

For the user's prototype-search question, the [exact search benchmark](prototype_search_bench.py) uses 10,000 real MNIST PCA queries against 1,557 fitted prototypes. The current Numba lookup takes 40.27 ms, and scikit-learn ArgKmin takes 7.38 ms with identical assignments. See the [CSV](prototype_search_bench.csv). The training distance kernel already parallelizes input rows; prototype/map updates dominate fit time. [The neighbor-stage profile](mnist_neighbor_stages.py) separates SONG lookup from UMAP graph and layout stages.
