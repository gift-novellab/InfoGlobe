import sys
import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from hydra import initialize, compose
from omegaconf import OmegaConf

sys.path.insert(0, "../..")

from flatvi.datamodules.sc_datamodule import scDataModule
from flatvi.models.base.geometric_vae import GeometricNBVAE


TASK_NAME = "flatvi_t_cell"
ADATA_PATH = "/home/chongxiao/projects/InfoGlobe/t_cell_depleted_bm_rna_with_Xcounts_dense.h5ad"
CKPT_PATH = "/home/chongxiao/projects/InfoGlobe/methods_src/FlatVI/project_folder/experiments/flatvi_t_cell/checkpoints/last.ckpt"
OUT_PATH = "/home/chongxiao/projects/InfoGlobe/methods_src/FlatVI/project_folder/experiments/flatvi_t_cell/flatvi_latent.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# 组合与训练一致的配置
with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(
        config_name="train_vae",
        overrides=[
            "hydra/launcher=basic",
            "~launcher",
            f"train.task_name={TASK_NAME}",
            f"datamodule.path={ADATA_PATH}",
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

# 与训练一致地构造 datamodule
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

# 与 train_vae.py 一致地构造 vae_kwargs
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

# 与训练一致地构造 GeometricNBVAE
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

# 加载权重
ckpt = torch.load(CKPT_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("missing keys:", missing)
print("unexpected keys:", unexpected)

model = model.to(device)
model.eval()

# 读取输入
adata = sc.read_h5ad(ADATA_PATH)
X = adata.layers["X_counts"]
if sparse.issparse(X):
    X = X.toarray()
else:
    X = np.asarray(X)
X = X.astype(np.float32)

# 导出 latent mean（mu），用于 benchmark 更稳定
batch_size = 512
mus = []

with torch.no_grad():
    for i in range(0, X.shape[0], batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        enc = model.encode(xb)
        mus.append(enc["mu"].detach().cpu().numpy())

emb = np.concatenate(mus, axis=0)
print("embedding shape:", emb.shape)

np.save(OUT_PATH, emb)
print("saved to:", OUT_PATH)