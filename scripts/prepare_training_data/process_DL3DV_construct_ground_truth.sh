#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

cd $PROJECT_DIR
DL3DV_path=data/train_data/DL3DV-10K

# Stage1: Pair-wise matching (require GPUs)
python tools/prepare_largescale_dataset_video_matching.py --data_path $DL3DV_path --match_ray_n_workers 4 --match_ray_enable 

# Stage2: Merge and construct point trajectories (not require GPUs)
python tools/prepare_largescale_dataset_video_matching.py --data_path $DL3DV_path --prop_ray_n_workers 64 --merge_and_prop --prop_avg_motion_thr 30 --prop_min_n_matches 300 --enable_multiview_refinement --merge_and_find_tracks_only

# Stage3: Multi-view refinement (require GPUs)
python tools/prepare_largescale_dataset_video_matching.py --data_path $DL3DV_path --match_ray_n_workers 4 --merge_and_prop --prop_avg_motion_thr 30 --prop_min_n_matches 300 --enable_multiview_refinement --multiview_refine_only --match_ray_enable 

# Stage4: Construct training pairs, and save (not require GPUs)
python tools/prepare_largescale_dataset_video_matching.py --data_path $DL3DV_path --prop_ray_n_workers 64 --merge_and_prop --prop_avg_motion_thr 30 --prop_min_n_matches 300 --enable_multiview_refinement --save_final_npz --save_final_match_type in_h5