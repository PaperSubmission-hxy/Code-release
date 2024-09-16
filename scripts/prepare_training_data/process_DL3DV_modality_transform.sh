#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

DL3DV_path=data/train_data/DL3DV-10K

cd $PROJECT_DIR
python tools/prepare_largescale_dataset_transform.py --data_path $DL3DV_path  --transform_method depthanything --ray_n_workers 8 --pure_process_no_save_scene --sample_seq_interval 4 --ray_enable

python tools/prepare_largescale_dataset_transform.py --data_path $DL3DV_path --transform_method style_rgb2if --ray_n_workers 8 --pure_process_no_save_scene --sample_seq_interval 4 --ray_enable

python tools/prepare_largescale_dataset_transform.py --data_path $DL3DV_path --transform_method style_day2night --ray_n_workers 8 --pure_process_no_save_scene --sample_seq_interval 4 --ray_enable