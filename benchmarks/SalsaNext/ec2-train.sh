# Directories
STOREDIR="/data"
DATASET="Rellis-3D"
DATASET_DIR="$STOREDIR/datasets/$DATASET"
LOG_DIR="$STOREDIR/logs"

export CUDA_VISIBLE_DEVICES="0"

echo "Training started...";

cd ./train/tasks/semantic;  \
    ./train.py \
    -d $DATASET_DIR \
    -ac ./config/arch/salsanext_ouster.yml \
    -dc ./config/labels/rellis.yaml \
    -n rellis -l $LOG_DIR \
    -p ""
