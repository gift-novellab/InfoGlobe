# InfoGlobe_ICML

InfoGlobe is a dimension reduction method via statistical manifold learning for multinomial data. This repository includes the core `InfoGlobe` package, simulation data under `sim_data/`, and a demonstration notebook `test.ipynb` for a minimal end-to-end example.

## Repository structure

```text
.
├── InfoGlobe/
│   ├── __init__.py
│   ├── constants.py
│   ├── filter.py
│   ├── infoglobe.py
│   ├── metrics.py
│   ├── plot.py
│   └── utils.py
├── sim_data/
└── test.ipynb
```

## Environment setup

We recommend using a fresh conda environment.

```bash
conda create -n infoglobe python=3.10 -y
conda activate infoglobe
pip install -r requirements.txt
```

A minimal `requirements.txt` can look like:

```txt
numpy
scipy
anndata
scanpy
torch
matplotlib
seaborn
tqdm
```

If you prefer to export directly from a working environment, activate your environment first and run:

```bash
pip freeze > requirements.txt
```

## Running the demo notebook

The repository contains `test.ipynb`, which demonstrates a basic InfoGlobe workflow on simulated data.

1. Open Jupyter from the repository root:
   ```bash
   jupyter notebook
   ```
2. Open `test.ipynb`.
3. Run all cells from top to bottom.

## What the demo does

The notebook performs the following steps:

1. Imports the required packages and the local `InfoGlobe` module.
2. Loads the example dataset:
   ```python
   adata = sc.read_h5ad('sim_data/adata/adata_1.h5ad')
   ```
3. Extracts the expression matrix and normalizes each cell to sum to 1.
4. Initializes the model:
   ```python
   gb = InfoGlobe.infoglobe.GlobeEmbedding(A=[n, k], Q=[k, m], c=1)
   ```
5. Fits the model:
   ```python
   gb.fit(torch.Tensor(P_gd), max_iter=20000)
   ```
6. Visualizes the reconstructed data matrix and the learned Markov kernel.

## Notes

- Run the notebook from the repository root so that `import InfoGlobe` works correctly.
- GPU is optional. The code automatically uses CUDA if available; otherwise it falls back to CPU.
- The demo is intended as a minimal example for reviewers and users. For custom datasets, prepare a matrix with nonnegative values and normalize each sample to sum to 1 before fitting.

## Using your own data

If your data matrix is cell × gene, you can adapt it as follows:

```python
P = adata.X
P = P / P.sum(axis=1, keepdims=True)
P_gd = P.T

n, m = P_gd.shape
k = 9

gb = InfoGlobe.infoglobe.GlobeEmbedding(A=[n, k], Q=[k, m], c=1)
gb.fit(torch.Tensor(P_gd), max_iter=20000)
```

You may tune `k` and `max_iter` depending on dataset size and the desired latent complexity.
