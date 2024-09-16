#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

cd $PROJECT_DIR
google_landmark_path=data/train_data/google-landmark

# Use depth estimation to obtain sky masks and depth maps:
python tools/prepare_largescale_dataset_transform.py --data_path $google_landmark_path --transform_method depthanything --ray_n_workers 16 --ray_enable --n_split_scenes 500

# Obtain synthetic thermal images to construct pairs: 
python tools/prepare_largescale_dataset_transform.py --data_path $google_landmark_path --transform_method style_rgb2if --ray_n_workers 16 --ray_enable --n_split_scenes 200