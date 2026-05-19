import os
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse

from scphere.util.trainer import Trainer
from scphere.model.vae import SCPHERE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scphere.util.plot import plot_trace

INPUT_DIR = "/old_nfs/chengwang_data/ICML_data/sim_data/globe"
OUTPUT_ROOT = "/old_nfs/chengwang_data/ICML_data/benchmark_results/scphere_multi_seed"
DEPTHS = [10000, 5000, 2000, 200]
SEEDS = [0, 1, 2, 3, 4]

N_TOP_GENES = 2000
Z_DIM = 16
MAX_EPOCH = 250
MB_SIZE = 128
LR = 1e-3

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --------------------------------------------------
# Step 1: 用最高深度数据选择统一 HVG
# --------------------------------------------------
ref_depth = 10000
ref_file = os.path.join(INPUT_DIR, f"adata_depth_{ref_depth}.h5ad")
ref_adata = sc.read_h5ad(ref_file)

ref_tmp = ad.AnnData(
    X=ref_adata.X.copy(),
    obs=ref_adata.obs.copy(),
    var=ref_adata.var.copy()
)
sc.pp.normalize_total(ref_tmp, target_sum=1e4)
sc.pp.log1p(ref_tmp)
sc.pp.highly_variable_genes(ref_tmp, n_top_genes=N_TOP_GENES, flavor="seurat")

shared_hvg_mask = ref_tmp.var["highly_variable"].to_numpy()
shared_hvg_names = ref_tmp.var_names[shared_hvg_mask].copy()

print("Reference HVG selected from depth=10000")
print("Number of HVGs:", len(shared_hvg_names))

# --------------------------------------------------
# Step 2: 逐个 depth × seed 训练
# --------------------------------------------------
for depth in DEPTHS:
    infile = os.path.join(INPUT_DIR, f"adata_depth_{depth}.h5ad")

    print(f"\n==============================")
    print(f"Processing depth = {depth}")
    print(f"reading: {infile}")

    adata = sc.read_h5ad(infile)

    # 对齐统一 HVG
    keep_mask = adata.var_names.isin(shared_hvg_names)
    if keep_mask.sum() != len(shared_hvg_names):
        print(f"Warning: only {keep_mask.sum()} / {len(shared_hvg_names)} HVGs found in depth={depth}")

    X = adata[:, keep_mask].X

    if sparse.issparse(X):
        x = X.toarray()
    else:
        x = np.asarray(X)

    print("Input shape after shared HVG:", x.shape)

    # 无 batch 信息
    batch = np.zeros(x.shape[0], dtype=int) - 1

    for seed in SEEDS:
        outdir = os.path.join(OUTPUT_ROOT, f"depth_{depth}", f"seed_{seed}")
        os.makedirs(outdir, exist_ok=True)

        print(f"\n===== scPhere | depth={depth} | seed={seed} =====")

        model = SCPHERE(
            n_gene=x.shape[1],
            n_batch=0,
            z_dim=Z_DIM,
            latent_dist="vmf",
            batch_invariant=False,
            observation_dist="nb",
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            x=x,
            batch_id=batch,
            max_epoch=MAX_EPOCH,
            mb_size=MB_SIZE,
            learning_rate=LR,
        )

        trainer.train()

        model_prefix = os.path.join(outdir, "scphere_model_250epoch")
        model.save_sess(model_prefix)

        z_mean = model.encode(x, batch)
        np.savetxt(os.path.join(outdir, "scphere_latent.tsv"), z_mean, delimiter="\t")

        ll = model.get_log_likelihood(x, batch)
        np.savetxt(os.path.join(outdir, "scphere_loglik.tsv"), ll, delimiter="\t")

        plot_trace(
            [np.arange(len(trainer.status["kl_divergence"])) * 50] * 2,
            [trainer.status["log_likelihood"], trainer.status["kl_divergence"]],
            ["log_likelihood", "kl_divergence"],
        )
        plt.savefig(os.path.join(outdir, "scphere_training.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # 写回 h5ad
        adata_out = adata.copy()
        adata_out.obsm["X_scPhere"] = z_mean
        adata_out.uns["scphere_depth"] = int(depth)
        adata_out.uns["scphere_seed"] = int(seed)
        adata_out.uns["scphere_n_top_genes"] = int(N_TOP_GENES)
        adata_out.uns["scphere_z_dim"] = int(Z_DIM)
        adata_out.uns["scphere_hvg_source_depth"] = int(ref_depth)
        adata_out.write(os.path.join(outdir, f"adata_depth_{depth}_scphere_seed_{seed}.h5ad"))

        print("embedding shape:", z_mean.shape)
        print("saved to:", outdir)

print("\nAll scPhere jobs finished.")