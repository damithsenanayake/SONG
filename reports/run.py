"""Execute report notebooks from the repository root."""

from pathlib import Path
import time
import sys
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / 'reports' / name for name in sys.argv[1:]] if len(sys.argv) > 1 else sorted((ROOT / 'reports').glob('[0-9][0-9]_*.ipynb'))
for path in paths:
    notebook = nbformat.read(path, as_version=4)
    start = time.perf_counter()
    print(f'Executing {path.name}', flush=True)
    NotebookClient(
        notebook,
        timeout=900,
        kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}},
    ).execute()
    nbformat.write(notebook, path)
    print(f'Finished {path.name} in {time.perf_counter() - start:.1f}s', flush=True)
