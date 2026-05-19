import os
import time
import numpy as np
import scanpy as sc
import scvi
from scipy import sparse

INPUT_DIR = "/old_nfs/chengwang_data/ICML_data/sim_data/globe"
OUTPUT_ROOT = "/old_nfs/chengwang_data/ICML_data/benchmark_results/scvi"
DEPTHS = [10000, 5000, 2000, 200]
SEED = 0

os.makedirs(OUTPUT_ROOT, exist_ok=True)
scvi.settings.seed = SEED


def ensure_counts_layer(adata):
    if "counts" not in adata.layers:
        if sparse.issparse(adata.X):
            adata.layers["counts"] = adata.X.copy()
        else:
            adata.layers["counts"] = np.asarray(adata.X).copy()
    return adata


for depth in DEPTHS:
    infile = os.path.join(INPUT_DIR, f"adata_depth_{depth}.h5ad")
    outdir = os.path.join(OUTPUT_ROOT, f"depth_{depth}")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== scVI | depth={depth} =====")
    print("reading:", infile)

    adata = sc.read_h5ad(infile)
    adata_scvi = adata.copy()
    adata_scvi = ensure_counts_layer(adata_scvi)

    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        layer="counts",
    )

    model = scvi.model.SCVI(adata_scvi)
    model.view_anndata_setup()

    start = time.time()
    model.train()
    end = time.time()

    runtime_seconds = end - start
    emb = model.get_latent_representation()

    adata_scvi.obsm["X_scVI"] = emb
    adata_scvi.uns["scvi_runtime_seconds"] = float(runtime_seconds)
    adata_scvi.uns["scvi_runtime_minutes"] = float(runtime_seconds / 60)
    adata_scvi.uns["scvi_depth"] = int(depth)

    np.save(os.path.join(outdir, "X_scVI.npy"), emb)
    adata_scvi.write(os.path.join(outdir, f"adata_depth_{depth}_scvi.h5ad"))
    model.save(os.path.join(outdir, "scvi_model"), overwrite=True)

    with open(os.path.join(outdir, "runtime.txt"), "w") as f:
        f.write(f"runtime_seconds\t{runtime_seconds:.6f}\n")
        f.write(f"runtime_minutes\t{runtime_seconds / 60:.6f}\n")

    print(f"embedding shape: {emb.shape}")
    print(f"runtime_seconds: {runtime_seconds:.2f}")
    print("saved to:", outdir)

print("\nAll scVI jobs finished.")