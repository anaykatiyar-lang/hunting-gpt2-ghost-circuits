# Analysis environment record

The full-record notebook records these package versions in its saved output. This records the environment visible in the notebook; it is not evidence that a clean environment has reproduced the complete notebook.

| Package | Recorded version |
|---|---|
| TransformerLens | 3.6.0 |
| PyTorch | 2.11.0+cpu |
| Transformers | 5.18.0 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 (install log) |
| requests | 2.32.4 (install log) |
| einops | 0.8.2 (install log) |
| tqdm | 4.67.3 (install log) |

The notebook prints `Python: 3.10+`. `requirements.txt` lists the recorded non-PyTorch packages. Install PyTorch separately with the index URL for the intended CPU or accelerator backend; for the recorded CPU build, use `python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`. The 552-target analysis script also requires SciPy, statsmodels, and tabulate; these are listed in `requirements-analysis.txt` because their exact original versions were not recorded. The installation instructions have not been validated in a clean environment, and no end-to-end notebook reproduction is claimed.
