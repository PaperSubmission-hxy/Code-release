#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

cd $PROJECT_DIR
sa1b_path=data/train_data/SA_1B

# Use depth estimation to obtain sky masks and depth maps:
python tools/prepare_largescale_dataset_transform.py --data_path $sa1b_path --transform_method depthanything --ray_n_workers 16 --ray_enable --n_split_scenes 500

# Obtain synthetic night-time images to construct pairs: 
python tools/prepare_largescale_dataset_transform.py --data_path $sa1b_path --transform_method style_day2night --ray_n_workers 16 --ray_enable --n_split_scenes 200