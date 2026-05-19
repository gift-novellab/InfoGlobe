import os
import time
import numpy as np
import scanpy as sc
from scipy import sparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

INPUT_DIR = "/old_nfs/chengwang_data/ICML_data/sim_data/globe"
OUTPUT_ROOT = "/old_nfs/chengwang_data/ICML_data/benchmark_results/irvae"
DEPTHS = [10000, 5000, 2000, 200]

N_TOP_GENES = 2000
LATENT_DIM = 16
BATCH_SIZE = 128
EPOCHS = 250
LR = 1e-3
WEIGHT_DECAY = 1e-5
ISO_REG = 0.1
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("DEVICE =", DEVICE)
print("SEED =", SEED)


class ExprDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class IRVAEAdapted(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, z_dim)
        self.decoder = Decoder(z_dim, in_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z


def relaxed_distortion_measure(func, z, eta=0.2, create_graph=True):
    bs = len(z)
    z_perm = z[torch.randperm(bs, device=z.device)]
    alpha = (torch.rand(bs, device=z.device) * (1 + 2 * eta) - eta).unsqueeze(1)
    z_aug = alpha * z + (1 - alpha) * z_perm
    v = torch.randn_like(z)

    Jv = torch.autograd.functional.jvp(
        func, z_aug, v=v, create_graph=create_graph
    )[1]
    TrG = torch.sum(Jv.view(bs, -1) ** 2, dim=1).mean()

    JTJv = torch.autograd.functional.vjp(
        func, z_aug, v=Jv, create_graph=create_graph
    )[1]
    JTJv = JTJv.view(bs, -1)
    TrG2 = torch.sum(JTJv ** 2, dim=1).mean()

    return TrG2 / (TrG ** 2 + 1e-8)


def loss_fn(x, recon, mu, logvar, z, decoder, iso_reg):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    iso = relaxed_distortion_measure(decoder, z, eta=0.2, create_graph=True)
    total = recon_loss + kl + iso_reg * iso
    return total, recon_loss, kl, iso


for depth in DEPTHS:
    infile = os.path.join(INPUT_DIR, f"adata_depth_{depth}.h5ad")
    outdir = os.path.join(OUTPUT_ROOT, f"depth_{depth}")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== IRVAE | depth={depth} =====")
    print("reading:", infile)

    adata_raw = sc.read_h5ad(infile)
    adata_out = adata_raw.copy()

    tmp = adata_raw.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=N_TOP_GENES, flavor="seurat")

    X = tmp[:, tmp.var["highly_variable"]].X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    X = X.astype(np.float32)
    print("Input matrix after HVG:", X.shape)

    dataset = ExprDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = IRVAEAdapted(in_dim=X.shape[1], z_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    t0 = time.perf_counter()

    model.train()
    for epoch in range(EPOCHS):
        total_epoch = 0.0
        rec_epoch = 0.0
        kl_epoch = 0.0
        iso_epoch = 0.0

        for xb in loader:
            xb = xb.to(DEVICE)

            recon, mu, logvar, z = model(xb)
            loss, rec, kl, iso = loss_fn(
                xb, recon, mu, logvar, z, model.decoder, ISO_REG
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch += loss.item()
            rec_epoch += rec.item()
            kl_epoch += kl.item()
            iso_epoch += iso.item()

        n_batches = len(loader)
        total_epoch /= n_batches
        rec_epoch /= n_batches
        kl_epoch /= n_batches
        iso_epoch /= n_batches

        history.append([total_epoch, rec_epoch, kl_epoch, iso_epoch])

        print(
            f"Epoch {epoch+1:03d} | "
            f"total={total_epoch:.6f} "
            f"rec={rec_epoch:.6f} "
            f"kl={kl_epoch:.6f} "
            f"iso={iso_epoch:.6f}"
        )

    t1 = time.perf_counter()
    runtime_seconds = t1 - t0

    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(DEVICE)
        mu, logvar = model.encoder(X_tensor)
        emb = mu.detach().cpu().numpy()

    per_dim_std = emb.std(axis=0)
    print("embedding shape:", emb.shape)
    print("per-dim std:", np.round(per_dim_std, 6))
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print(f"runtime_minutes={runtime_seconds / 60:.2f}")

    np.save(os.path.join(outdir, "irvae_latent.npy"), emb)
    np.savetxt(
        os.path.join(outdir, "irvae_training.tsv"),
        np.asarray(history),
        delimiter="\t",
    )

    with open(os.path.join(outdir, "runtime.txt"), "w") as f:
        f.write(f"runtime_seconds\t{runtime_seconds:.6f}\n")
        f.write(f"runtime_minutes\t{runtime_seconds / 60:.6f}\n")

    adata_out.obsm["X_irvae"] = emb
    adata_out.uns["irvae_runtime_seconds"] = float(runtime_seconds)
    adata_out.uns["irvae_runtime_minutes"] = float(runtime_seconds / 60)
    adata_out.uns["irvae_depth"] = int(depth)
    adata_out.write(os.path.join(outdir, f"adata_depth_{depth}_irvae.h5ad"))

    print("saved to:", outdir)

print("\nAll IRVAE jobs finished.")