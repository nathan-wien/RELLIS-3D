# Directories
STOREDIR="/data"
DATASET="Rellis-3D"
DATASET_DIR="$STOREDIR/datasets/$DATASET"
LOG_DIR="$STOREDIR/logs/eval"
MODEL_DIR="$STOREDIR/logs/logs/2021-10-12-01:12rellis"

export CUDA_VISIBLE_DEVICES="0"

echo "Evaluation started...";

cd ./train/tasks/semantic;  \
    python infer2.py \
    -d $DATASET_DIR \
    -l $LOG_DIR \
    -s test \
    -m $MODEL_DIR
