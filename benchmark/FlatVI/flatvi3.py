import os
import sys
import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

REPO_DIR = os.path.expanduser("~/projects/InfoGlobe/methods_src/FlatVI")
sys.path.insert(0, REPO_DIR)

from flatvi.datamodules.sc_datamodule import scDataModule
from flatvi.models.base.geometric_vae import GeometricNBVAE

INPUT_ROOT = "/old_nfs/chengwang_data/ICML_data/benchmark_results/flatvi"
EXP_ROOT = os.path.join(REPO_DIR, "project_folder", "experiments")
DEPTHS = [10000, 5000, 2000, 200]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

CONFIG_DIR = os.path.join(REPO_DIR, "configs")

for depth in DEPTHS:
    task_name = f"flatvi_depth_{depth}"
    adata_path = os.path.join(
        INPUT_ROOT,
        f"depth_{depth}",
        f"adata_depth_{depth}_with_Xcounts_dense.h5ad"
    )
    ckpt_path = os.path.join(
        EXP_ROOT,
        task_name,
        "checkpoints",
        "last.ckpt"
    )
    outdir = os.path.join(INPUT_ROOT, f"depth_{depth}")
    out_npy = os.path.join(outdir, "flatvi_latent.npy")
    out_h5ad = os.path.join(outdir, f"adata_depth_{depth}_flatvi.h5ad")

    print(f"\n===== FlatVI export | depth={depth} =====")
    print("adata_path:", adata_path)
    print("ckpt_path:", ckpt_path)
    print("config_dir:", CONFIG_DIR)

    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"adata not found: {adata_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not os.path.exists(CONFIG_DIR):
        raise FileNotFoundError(f"config dir not found: {CONFIG_DIR}")

    # 关键修复：绝对路径配置目录必须用 initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="train_vae",
            overrides=[
                "hydra/launcher=basic",
                "~launcher",
                f"train.task_name={task_name}",
                f"datamodule.path={adata_path}",
                "datamodule.x_layer=X_counts",
                "datamodule.cond_keys=[]",
                "datamodule.use_pca=False",
                "trainer.accelerator=gpu" if device == "cuda" else "trainer.accelerator=cpu",
                "trainer.devices=1",
                "logger.offline=True",
                "logger.log_model=False",
            ],
        )

    OmegaConf.resolve(cfg)

    datamodule = scDataModule(
        path=cfg.datamodule.path,
        x_layer=cfg.datamodule.x_layer,
        cond_keys=cfg.datamodule.cond_keys,
        use_pca=cfg.datamodule.use_pca,
        n_dimensions=cfg.datamodule.n_dimensions,
        train_val_test_split=cfg.datamodule.train_val_test_split,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
    )

    vae_kwargs = dict(
        in_dim=datamodule.in_dim,
        hidden_dims=cfg.model.hidden_dims,
        batch_norm=cfg.model.batch_norm,
        dropout=cfg.model.dropout,
        dropout_p=cfg.model.dropout_p,
        n_epochs_anneal_kl=cfg.model.n_epochs_anneal_kl,
        kl_warmup_fraction=cfg.model.kl_warmup_fraction,
        kl_weight=cfg.model.kl_weight,
        likelihood=cfg.model.likelihood,
        learning_rate=cfg.model.learning_rate,
        model_library_size=cfg.model.model_library_size,
    )

    model = GeometricNBVAE(
        l2=cfg.geometric_vae.l2,
        interpolate_z=cfg.geometric_vae.interpolate_z,
        eta_interp=cfg.geometric_vae.eta_interp,
        compute_metrics_every=cfg.geometric_vae.compute_metrics_every,
        start_jac_after=cfg.geometric_vae.start_jac_after,
        vae_kwargs=vae_kwargs,
        use_c=cfg.geometric_vae.use_c,
        detach_theta=cfg.geometric_vae.detach_theta,
        fl_weight=cfg.geometric_vae.fl_weight,
        trainable_c=cfg.geometric_vae.trainable_c,
        anneal_fl_weight=cfg.geometric_vae.anneal_fl_weight,
        max_fl_weight=cfg.geometric_vae.max_fl_weight,
        n_epochs_anneal_fl=cfg.geometric_vae.n_epochs_anneal_fl,
        fl_anneal_fraction=cfg.geometric_vae.fl_anneal_fraction,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    model = model.to(device)
    model.eval()

    adata = sc.read_h5ad(adata_path)
    X = adata.layers["X_counts"]
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    X = X.astype(np.float32)

    batch_size = 512
    mus = []

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            enc = model.encode(xb)
            mus.append(enc["mu"].detach().cpu().numpy())

    emb = np.concatenate(mus, axis=0)
    print("embedding shape:", emb.shape)

    np.save(out_npy, emb)

    adata_out = adata.copy()
    adata_out.obsm["X_flatvi"] = emb
    adata_out.uns["flatvi_depth"] = int(depth)
    adata_out.write(out_h5ad)

    print("saved latent to:", out_npy)
    print("saved h5ad to:", out_h5ad)

print("\nAll FlatVI export jobs finished.")