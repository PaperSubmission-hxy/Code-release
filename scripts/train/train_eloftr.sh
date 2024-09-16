#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

DATACFG_NAME=eloftr_data_config
CFG_NAME=eloftr_model

EXP_NAME="ELoFTR-$DATACFG_NAME@-@$CFG_NAME"
echo $EXP_NAME

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

data_cfg_path="configs/data/${DATACFG_NAME}.py"
main_cfg_path="configs/loftr/matchanything/exps/${CFG_NAME}.py"

n_nodes=2
n_gpus_per_node=8
torch_num_workers=4
batch_size=4
pin_memory=true
exp_name="${EXP_NAME}-bs$(($n_gpus_per_node * $n_nodes * $batch_size))"

cd $PROJECT_DIR
python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --n_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=80 \
    --flush_logs_every_n_steps=1 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=30 \
    --resume_from_latest \
    --method=loftr