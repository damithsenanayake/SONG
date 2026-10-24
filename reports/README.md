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


## Incremental rehearsal and position KL

The [SONG rehearsal comparison](song_rehearsal_mnist.py) tests the user's hypothesis that presenting only new classes causes old structure to move. Initial old-digit 5-NN accuracy was 97.78%; after a no-rehearsal update it fell to 90.78%. Replaying old images progressively improved retention; the [aligned class plot](figures/song_rehearsal_mnist.png) shows the qualitative effect. This is a SONG-core study, not a hybrid SONG–UMAP update.

The [nested replay frontier](song_rehearsal_optimization.py) tests 0, 5, 10, 20, 35, 50, 75, and 100% of old training images. Under an illustrative target of at most a one-percentage-point old-class accuracy drop and normalized class-centroid drift ≤0.20, 75% is the smallest tested fraction that passes. At 75%, old-class accuracy is 97.37%, drift 0.175, old-anchor neighborhood KL 0.063 nats, and update time 2.73 s versus 1.90 s without replay. Retention is not perfectly monotonic across fractions, so the fraction should be selected from a measured frontier rather than by unvalidated binary search. See the [four-metric chart](figures/song_rehearsal_tradeoff.png) and [raw JSON](song_rehearsal_optimization.json).

The same script evaluates novelty on held-out MNIST before and after an update. Frozen old-map nearest-prototype squared distance detects digits 5–9 against 0–4 with AUROC 0.881; KL of prototype assignment against old occupancy scores 0.464. Gaussian position KL between old and updated coordinates scores 0.758, but requires an update first. The [preprint's KL section](../preprint/SONG_initialized_UMAP.md#42-rehearsal-budget-and-a-distributional-retention-objective) gives the proposed retention objective and the distinction between pre-update detection and post-update analysis. All are single-seed diagnostics, not calibrated thresholds.


## Implemented KL retention ablation

SONG now has opt-in `retention_kl_weight` and `retention_position_weight` parameters. On the first fit, it stores a small old-anchor sample. After an incremental fit, it minimizes a neighborhood-distribution KL plus Gaussian-position KL by adjusting the coordinates of prototypes selected by those anchors. This does not update input coding vectors, so old-class identity still depends on rehearsal. The analytic gradient is checked against finite differences in [`tests/test_retention.py`](../tests/test_retention.py).

The [matched MNIST ablation](song_retention_kl_mnist.py) compares copied models with identical initial state and nested old-data replay sets. At 35% replay, old-digit accuracy is 96.90% with and without KL, but normalized old-class centroid drift falls from **0.226 to 0.092** and old-anchor neighborhood KL falls from **0.089 to 0.002** nats. Update time rises from about 2.33 s to 2.45 s. Under the example accuracy-and-drift target, 35% replay plus KL passes in seed 42, while 35% replay without KL fails. A [three-seed check](song_retention_kl_seeds.csv) finds that KL lowers mean old-centroid drift from 0.224 to 0.112 at 35% replay, but the combined target passes only 1/3 KL runs. A smaller replay budget is therefore not established across seeds. With zero replay, KL improves position drift from 0.681 to 0.208 but old-digit accuracy remains near 90%, well below the 97.78% initial probe. See the [comparison image](figures/song_retention_kl_mnist.png), [raw JSON](song_retention_kl_mnist.json), and the [revised preprint](../preprint/SONG_initialized_UMAP.md#43-implemented-kl-coordinate-refinement). All results are one seed and should be repeated before choosing a default.


## Trained-prototype KL with no raw-data replay

The [prototype-only ablation](song_prototype_kl_mnist.py) is a distinct experiment from old-image rehearsal. Before the second `fit`, `retention_source='prototypes'` snapshots the 909 trained input and output prototypes (seed 42) and protects their indices from recycling. After fitting digits 5–9 **without any old image rows**, it minimizes neighborhood and/or Gaussian-position KL for corresponding old prototypes. The [plot](figures/song_prototype_kl_mnist.png) visually compares this with new-only fitting and 35% raw rehearsal; [seed-42 JSON](song_prototype_kl_mnist.json) includes input/output losses and optimizer diagnostics.

Across three seeds, new-only fitting loses 6.82 ± 0.37 percentage points of old-digit accuracy with centroid drift 0.597 ± 0.104. Prototype neighborhood KL alone still loses 5.64 ± 0.48 points. Gaussian-position KL loses only 0.50 ± 0.29 points with drift 0.030 ± 0.006 and passes the illustrative retention target in all three seeds. Both KL terms yield similar retention (0.46 ± 0.33 points lost, drift 0.030 ± 0.005). **Restoring the trained prototypes exactly matches position-only KL at lower cost**, showing that the gain is largely from keeping old prototype locations, not from a demonstrated advantage of KL itself. Mean new-digit accuracy falls from 95.88% with no retention to 94.93% with position KL or exact restoration. See the [individual-seed CSV](song_prototype_kl_seeds.csv), [script](song_prototype_kl_seeds.py), and [preprint discussion](../preprint/SONG_initialized_UMAP.md#44-trained-prototype-kl-without-old-image-replay). The dense KL objective is quadratic in old prototype count. Follow-on MNIST increments and graph consistency remain untested; only a small multi-increment smoke test is included.
