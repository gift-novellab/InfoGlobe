#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flatvi

export WANDB_MODE=offline
export WANDB_SILENT=true

REPO_DIR="$HOME/projects/InfoGlobe/methods_src/FlatVI"
INPUT_ROOT="/old_nfs/chengwang_data/ICML_data/benchmark_results/flatvi"
DEPTHS=(10000 5000 2000 200)

mkdir -p "$REPO_DIR/logs"
mkdir -p "$REPO_DIR/project_folder/experiments"
mkdir -p "$REPO_DIR/project_folder/data"

cd "$REPO_DIR/flatvi/train_hydra"

for depth in "${DEPTHS[@]}"; do
  TASK_NAME="flatvi_depth_${depth}"
  ADATA_PATH="${INPUT_ROOT}/depth_${depth}/adata_depth_${depth}_with_Xcounts_dense.h5ad"

  echo "===== FlatVI train | depth=${depth} ====="
  echo "ADATA_PATH=${ADATA_PATH}"

  /usr/bin/time -v \
  python train_vae.py \
    hydra/launcher=basic \
    '~launcher' \
    train.task_name="${TASK_NAME}" \
    datamodule.path="${ADATA_PATH}" \
    datamodule.x_layer=X_counts \
    datamodule.cond_keys=[] \
    datamodule.use_pca=False \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    logger.offline=True \
    logger.log_model=False \
  2>&1 | tee "$REPO_DIR/logs/${TASK_NAME}_$(date +%F_%H-%M-%S).log"

done