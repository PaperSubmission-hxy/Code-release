#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

cd $PROJECT_DIR
blendedmvs_path=data/train_data/BlendedMVS_matching

python tools/prepare_largescale_dataset_transform.py --data_path $blendedmvs_path --transform_method style_rgb2if --ray_n_workers 8 --pure_process_no_save_scene --ray_enable

python tools/prepare_largescale_dataset_transform.py --data_path $blendedmvs_path --transform_method style_day2night --ray_n_workers 8 --pure_process_no_save_scene --ray_enable