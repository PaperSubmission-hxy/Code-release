#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

cd $PROJECT_DIR

DEVICE_ID='0'
NPZ_ROOT=data/test_data/visible_vectorized_map/scene_indices
NPZ_LIST_PATH=data/test_data/visible_vectorized_map/scene_indices/val_list.txt
OUTPUT_PATH=results/visible_vectorized_map

# ELoFTR pretrained:
CUDA_VISIBLE_DEVICES=$DEVICE_ID python tools/evaluate_datasets.py configs/models/eloftr_model.py --ckpt_path weights/eloftr_pretrained.ckpt --method loftr@-@ransac_affine --imgresize 832 --npe --npz_root $NPZ_ROOT --npz_list_path $NPZ_LIST_PATH --output_path $OUTPUT_PATH

# ROMA pretrained:
CUDA_VISIBLE_DEVICES=$DEVICE_ID python tools/evaluate_datasets.py  configs/models/roma_model.py --ckpt_path weights/roma_pretrained.ckpt --method ROMA_SELF_TRAIN@-@ransac_affine --imresize 832 --npe --npz_root $NPZ_ROOT --npz_list_path $NPZ_LIST_PATH --output_path $OUTPUT_PATH