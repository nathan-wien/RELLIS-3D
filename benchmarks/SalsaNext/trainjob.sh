#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2

# Activate conda environment
source ~/.bashrc;
conda activate salsanext;

# Directories
DATASET="RELLIS-3D"
DATASET_DIR="$STOREDIR/datasets/$DATASET/dataset"
LOG_DIR="$STOREDIR/logs"

echo "Training started...";

cd ./train/tasks/semantic;  \
    ./train.py \
    -d $DATASET_DIR \
    -ac ./config/arch/salsanext_ouster.yml \
    -dc ./config/labels/rellis.yaml \
    -n rellis -l $LOG_DIR \
    -p ""
