import os
import os.path as osp

from loguru import logger
from tqdm import tqdm
from shutil import rmtree
from pathlib import Path

from .dataset.coarse_sfm_refinement_dataset import CoarseColmapDataset
from .matcher_model import *

cfgs = {
    "coarse_colmap_data": {
        "img_resize": 1200,
        "df": None,
        "feature_track_assignment_strategy": "greedy",
        "img_preload": False,
    },
    "fine_match_debug": True,
    "multiview_matcher_data": {
        "max_track_length": 16,
        "chunk": 3000,
    },
    "fine_matcher": {
        "model": {
            "cfg_path": ['pairs_match_and_propogation/hydra_training_configs/experiment/multiview_refinement_matching.yaml'],
            "weight_path": ['weights/multiview_refinement_matching.ckpt'],
            "seed": 666,
        },
        "visualize": False,
        "extract_feature_method": "fine_match_backbone",
        "ray": {
            "slurm": False,
            "n_workers": 1,
            "n_cpus_per_worker": 1,
            "n_gpus_per_worker": 1,
            "local_mode": False,
        },
    },
}
def post_optimization(id2name_dict, keypoints_renamed, tracks, visible_tracks, visible_keypoints, ray_cfg=None, verbose=True):
    # Construct scene data
    colmap_image_dataset = CoarseColmapDataset(
        cfgs["coarse_colmap_data"],
        id2name_dict,
        keypoints_renamed,
        tracks,
        visible_tracks,
        visible_keypoints,
        verbose=verbose
    )

    fine_match_results = multiview_matcher(
        cfgs["fine_matcher"],
        cfgs["multiview_matcher_data"],
        colmap_image_dataset,
        use_ray=ray_cfg,
        ray_cfg=ray_cfg,
        verbose=verbose
    )
    colmap_images = colmap_image_dataset.get_refined_kpts_to_colmap_multiview(fine_match_results)
    return {id: image['xys'] for id, image in colmap_images.items()}