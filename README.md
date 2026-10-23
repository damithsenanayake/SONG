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

## Reproducible performance reports

The [`reports/`](reports/) directory contains three executed notebooks with plots and CSV results for the `UMAP_dispersion` branch: [static embeddings](reports/01_static_baseline.ipynb), [incremental behavior](reports/02_incremental_baseline.ipynb), and [parameter candidates](reports/03_tuning_candidates.ipynb). The [side-by-side SONG dispersion and UMAP plot](reports/figures/umap_dispersion_vs_umap.png) uses the same 600 labeled digits in both panels. These small, one-seed smoke benchmarks are not a reproduction of the [SONG paper](https://arxiv.org/abs/1912.04896).

To rerun them with Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r reports/requirements.txt
PATH="$PWD/.venv/bin:$PATH" NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python reports/run.py
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q
```

The current branch passes the static smoke test. Two regression tests are marked expected failures: a second `fit` uses the removed `np.int` alias, and nearest distances lose their fractional values. The incremental notebook first records the actual failure, then uses a process-local compatibility alias solely to inspect later behavior. Fixing those failures is tracked in GitHub issues.
