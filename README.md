# SONG
This Repository Contains the Source Code for the Python Implementation of Self Organizing Nebulous Growths.

*NOTE: This implementation has several improvements not described in the publication* 

## Installation
SONG is now available in the Python Packaging Index (PyPI). You can simply install the stable version (current release 1.0.0) via ``pip`` using the following command. 

``$ pip install song-vis``

If you would like the bleeding edge version, you can simply download it via the github source code. To install this package, download the zip file, extract it, and run the following command on the commandline

``$ cd <extracted_location>``

``$ python setup.py install``

Alternatively, you can use pip to install this package as follows

``$ cd <extracted_location>``

``$ pip install .``

## Running The Algorithm

In python, after installation, run the following. 

``from song.song import SONG``

This will import the libraries. To execute the algorithm run the following

`` Y = SONG().fit_transform(X)``

To retain a model after training it, use a variable to store the model

``model = SONG()``<br>
``model.fit(X)``

## Optional incremental layout retention

SONG can retain a fixed sample of old input anchors and refine prototype **coordinates** after an incremental fit using old-neighborhood KL and Gaussian-position KL:

```python
model = SONG(so_steps=30, dispersion_method='none',
             retention_kl_weight=1.0, retention_position_weight=1.0,
             retention_anchor_count=512)
model.fit(old_data)
model.fit(new_data)
print(model.retention_result_)
```

The weights default to zero. In the default `retention_source='samples'` mode, enable them before the first fit so old input anchors can be stored. That mode adjusts output coordinates after training; it does not prevent input coding vectors from drifting. The [matched MNIST ablation](reports/song_retention_kl_mnist.json) shows that rehearsal remains necessary for old-class separation in this mode.

To use **trained prototypes instead of raw-data replay**, set `retention_source='prototypes'` with the KL weights. On each incremental fit, SONG snapshots the previous input and output prototypes, keeps their indices available, and refines both spaces after fitting the new data:

```python
model = SONG(so_steps=30, dispersion_method='none',
             retention_source='prototypes',
             retention_kl_weight=1.0, retention_position_weight=1.0)
model.fit(old_data)
model.fit(new_data)  # only new rows
```

The [three-seed prototype experiment](reports/song_prototype_kl_seeds.csv) finds strong old-position retention without old-image replay. Its Gaussian-position KL result is matched by simply restoring the saved prototypes, so the [preprint draft](preprint/SONG_initialized_UMAP.md#44-trained-prototype-kl-without-old-image-replay) treats that restore as an essential control. This mode has been evaluated on one heterogeneous increment, not on later increments or UMAP graph refinement.

## Reproducible performance reports

The [`reports/`](reports/) directory contains six executed notebooks with plots and CSV results for the `UMAP_dispersion` branch: [static embeddings](reports/01_static_baseline.ipynb), [incremental behavior](reports/02_incremental_baseline.ipynb), [parameter candidates](reports/03_tuning_candidates.ipynb), [PCA-fix validation](reports/04_pca_fix_validation.ipynb), [full MNIST profiling](reports/05_mnist_full_profile.ipynb), and [SONG-initialized UMAP preprint experiments](reports/06_song_umap_preprint_experiments.ipynb). The [draft preprint](preprint/SONG_initialized_UMAP.md) reports both the early-epoch quality gain and the lack of an end-to-end speedup on MNIST. The [side-by-side SONG dispersion and UMAP plot](reports/figures/umap_dispersion_vs_umap.png) uses the same 600 labeled digits in both panels. The original baseline smoke benchmarks are not a reproduction of the [SONG paper](https://arxiv.org/abs/1912.04896).

To rerun all notebooks on the current code (overwriting the saved historical baseline outputs) with Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r reports/requirements.txt
PATH="$PWD/.venv/bin:$PATH" NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python reports/run.py
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

The current branch passes the static smoke test. Two regression tests are marked expected failures: a second `fit` uses the removed `np.int` alias, and nearest distances lose their fractional values. The incremental notebook first records the actual failure, then uses a process-local compatibility alias solely to inspect later behavior. Fixing those failures is tracked in GitHub issues.
