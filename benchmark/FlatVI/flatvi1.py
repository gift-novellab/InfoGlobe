import os
import numpy as np
import scanpy as sc
from scipy import sparse

INPUT_DIR = "/old_nfs/chengwang_data/ICML_data/sim_data/globe"
OUTPUT_ROOT = "/old_nfs/chengwang_data/ICML_data/benchmark_results/flatvi"
DEPTHS = [10000, 5000, 2000, 200]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for depth in DEPTHS:
    infile = os.path.join(INPUT_DIR, f"adata_depth_{depth}.h5ad")
    outdir = os.path.join(OUTPUT_ROOT, f"depth_{depth}")
    os.makedirs(outdir, exist_ok=True)

    outfile = os.path.join(outdir, f"adata_depth_{depth}_with_Xcounts_dense.h5ad")

    print(f"\n===== FlatVI prepare | depth={depth} =====")
    print("reading:", infile)

    adata = sc.read_h5ad(infile)

    if sparse.issparse(adata.X):
        X_dense = adata.X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(adata.X, dtype=np.float32)

    adata.layers["X_counts"] = X_dense

    print(type(adata.layers["X_counts"]))
    print(adata.layers["X_counts"].shape)
    print(adata.layers["X_counts"].dtype)

    adata.write(outfile)
    print("saved to:", outfile)

print("\nAll FlatVI prepare jobs finished.")