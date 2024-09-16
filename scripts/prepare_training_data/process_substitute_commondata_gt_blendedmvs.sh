#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

blendedmvs_path=data/train_data/BlendedMVS_matching
npz_root_path=$blendedmvs_path/matching_indices_0.1_0.7_0.0/scene_info
npz_list_path=$blendedmvs_path/matching_indices_0.1_0.7_0.0/train_list.txt

cd $PROJECT_DIR
python tools/substitute_crossmodal_img_for_commondataset.py --dataset_path $blendedmvs_path --npz_root $npz_root_path --npz_list_path $npz_list_path --ray_enable --ray_n_workers 24
