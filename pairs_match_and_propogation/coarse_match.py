import os
import ray
import math
import h5py
import os.path as osp
from loguru import logger
from typing import ChainMap
from pathlib import Path
from time import time

from .utils.ray_utils import ProgressBar, chunks, chunk_index, split_dict
from .compute_tracks import compute_tracks
from .utils.data_io import save_h5, load_h5, save_obj, load_obj
from .propogate_matches import propogate_matches
from .coarse_matcher_utils import Match2Kpts
from .coarse_match_worker import *
from .utils.matches_list import MatchesList

cfgs = {
    "data": {
             "img_resize": 1200, 
             "df": 8, 
             "pad_to": None,
             "img_type": "grayscale", # ['grayscale', 'rgb']
             "img_preload": False,
             'reload_gray_scale': True
            },
    "matcher": {
        "semi_dense_model": {
            "enable": True,
            "matcher": 'ROMA',
            "seed": 666
        },
        "pair_name_split": " ",
        "semi_dense_ransac": {
            "enable": True,
            "ransac_on_resized_space": True,
            "geo_model": "F",
            "ransac_method": "MAGSAC",
            "pixel_thr": 0.5,
            "max_iters": 10000,
            "conf_thr": 0.99999,
        },
    },
    "merge_cfgs":{
        "merge_on_resized_space": True,
        "kpt_score_agg_method": 'avg',
        'select_top_method': 'first_n',
        "nms_radius_max": 3, # 7*7 window
        "n_kpts": None, # Preserve all merged kpts

    },
    "ray": {
        "slurm": False,
        "n_workers": 8, # 16
        "n_cpus_per_worker": 2,
        "n_gpus_per_worker": 0.5,
        "local_mode": False,
    },
}

def pair_wise_matching(
    image_lists,
    image_mask_path,
    pair_list,
    output_dir,
    prop_save_name,
    dataset_name,
    n_seq_matches=5,
    prop_stop_ratio=0.03,
    prop_merge_radius=2, # window size: 2 * merge_radius + 1
    prop_stop_after_n_fail=5,
    prop_min_n_matches=None,
    prop_avg_motion_thr=40,
    regenerate_match=False,
    regenerate_prop=False,
    merge_and_prop=False,
    img_resize=None,
    img_preload=False,
    reload_gray_scale=True,
    matcher='ROMA',
    semi_dense_enable=True,
    match_round_ratio=-1,
    ray_cfg=None,
    semi_dense_ransac_cfg=None,
    enable_multiview_refinement=False,
    merge_and_find_tracks_only=False,
    multiview_refine_only=False,
    merge_cfg=None,
    verbose=True
):
    """
    Parameters:
    --------------
    run_sfm_later:
        if True: save keypoints and matches as later sfm wanted format
        else: save keypoints and matches for you repo wanted format
    """

    # Cfg overwrite:
    cfgs['matcher']['semi_dense_model']['enable'] = semi_dense_enable

    if isinstance(matcher, list):
        if len(matcher) != 1:
            logger.warning(f"Using combined dense matching methods, the loaded images may not satisify all methods!")
    else:
        matcher = [matcher]
    matcher_single = matcher[0]

    if 'eloftr' in matcher_single:
        cfgs['data']['df'] = 32
        cfgs['data']['pad_to'] = None
    elif "ROMA" in matcher_single:
        cfgs["data"]["img_type"] = 'rgb'

    cfgs['matcher']['semi_dense_model']['matcher'] = matcher
    cfgs['merge_cfgs']['nms_radius_max'] = prop_merge_radius
    if semi_dense_ransac_cfg is not None:
        cfgs['matcher']['semi_dense_ransac'] = semi_dense_ransac_cfg
    
    if merge_cfg is not None:
        cfgs["merge_cfgs"] = merge_cfg

    cfgs['data']['img_resize'] = img_resize
    cfgs['data']['img_preload'] = img_preload
    cfgs['data']['reload_gray_scale'] = reload_gray_scale

    # Construct directory
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    match_cache_path = str(output_dir / f"raw_matches.h5")
    img_scales_cache_path = str(output_dir / f"raw_img_scales.h5")
    track_cache_path = str(output_dir / f"tracks_{prop_save_name}.h5")
    kpts_cache_path = str(output_dir / f"kpts_{prop_save_name}.h5")

    refined_kpts_cache_path = str(output_dir / f"refined_kpts_{prop_save_name}.h5")
    prop_matches_cache_path = str(output_dir / f"{prop_save_name}.h5")

    if merge_and_prop and Path(prop_matches_cache_path).exists() and not regenerate_prop:
        merge_and_prop = False
        return prop_matches_cache_path

    if isinstance(pair_list, list):
        pair_list = pair_list
    else:
        assert osp.exists(pair_list)
        # Load pairs: 
        with open(pair_list, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')

    n_imgs = len(image_lists)
    if not Path(track_cache_path).exists():
        if ray_cfg is not None:
            # Matcher runner
            if osp.exists(match_cache_path) and not regenerate_match:
                if merge_and_prop:
                    # Initial ray:
                    cfg_ray = ray_cfg
                    if cfg_ray["slurm"]:
                        ray.init(address=os.environ["ip_head"])
                    else:
                        ray.init(
                            num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                            local_mode=cfg_ray["local_mode"],
                            ignore_reinit_error=True,
                        )

                    matches = load_h5(match_cache_path, transform_slash=True)
                    img_scales = load_h5(img_scales_cache_path, transform_slash=True)
                    logger.info("Caches raw matches loaded!")
            else:
                # Initial ray:
                cfg_ray = ray_cfg
                if cfg_ray["slurm"]:
                    ray.init(address=os.environ["ip_head"])
                else:
                    ray.init(
                        num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                        num_gpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_gpus_per_worker"]),
                        local_mode=cfg_ray["local_mode"],
                        ignore_reinit_error=True,
                    )

                pb = ProgressBar(len(pair_list), "Matching image pairs...") if verbose else None
                all_subset_ids = chunk_index(
                    len(pair_list), math.ceil(len(pair_list) / cfg_ray["n_workers"])
                )
                remote_func = match_worker_ray_wrapper.options(num_cpus=cfg_ray["n_cpus_per_worker"], num_gpus=cfg_ray["n_gpus_per_worker"])
                obj_refs = [
                    remote_func.remote(
                        subset_ids, image_lists, image_mask_path, pair_list, cfgs, pb.actor if pb is not None else None, verbose
                    )
                    for subset_ids in all_subset_ids
                ]
                pb.print_until_done() if pb is not None else None
                results = ray.get(obj_refs)
                matches = dict(ChainMap(*[m for m, _ in results])) # matches in original res
                img_scales = dict(ChainMap(*[s for _, s in results])) # scales = origin / resized
                logger.info("Matcher finish!")

                # over write anyway
                logger.info(f"Raw matches cach begin")
                save_h5(matches, match_cache_path, verbose=verbose, as_half=False)
                save_h5(img_scales, img_scales_cache_path, verbose=verbose, as_half=True)
                logger.info(f"Raw matches cach finish: {match_cache_path}")

            if merge_and_prop:
                # Change names to relative:
                image_basename_lists = [str(img_path).split(f'/{dataset_name}/')[1] for img_path in image_lists]
                img_scales = {str(img_path).split(f'/{dataset_name}/')[1] : scale for img_path, scale in img_scales.items()}
                matches_renamed = {}
                for pair, match in matches.items():
                    img0_path, img1_path = pair.split(' ')
                    # Filter wired matches may caused by previous half save:
                    valid_mask = ~((match[:, :4] < 0).sum(-1) > 0) | ((match[:, :4] > 10000).sum(-1) > 0)
                    if valid_mask.sum() < valid_mask.shape[0]:
                        logger.warning(f"Filted {valid_mask.shape[0] - valid_mask.sum()} wired matches")
                    matches_renamed[' '.join([str(img0_path).split(f'/{dataset_name}/')[1], str(img1_path).split(f'/{dataset_name}/')[1]])] = match[valid_mask]
                matches = matches_renamed

                # Combine keypoints
                pb = ProgressBar(n_imgs, "Combine keypoints") if verbose is not None else None
                all_kpts = Match2Kpts(
                    matches, image_basename_lists, img_scales, on_resized_space=cfgs['merge_cfgs']['merge_on_resized_space'] if match_round_ratio == -1 else False, name_split=cfgs["matcher"]["pair_name_split"]
                )
                sub_kpts = chunks(all_kpts, math.ceil(n_imgs / cfg_ray["n_workers"]))
                obj_refs = [
                    keypoints_worker_ray_wrapper.remote(sub_kpt, pb.actor if pb is not None else None, merge_cfg=cfgs['merge_cfgs'] if match_round_ratio == -1 else None, verbose=verbose)
                    for sub_kpt in sub_kpts
                ]
                pb.print_until_done() if pb is not None else None
                keypoints = dict(ChainMap(*ray.get(obj_refs)))
                logger.info("Combine keypoints finish!")

                # Convert keypoints match to keypoints indexs
                logger.info("Update matches")
                obj_refs = [
                    update_matches(
                        sub_matches,
                        keypoints,
                        img_scales,
                        on_resized_space=cfgs['merge_cfgs']['merge_on_resized_space'] if match_round_ratio == -1 else False,
                        merge=True if match_round_ratio == -1 else False,
                        verbose=verbose,
                        pair_name_split=cfgs["matcher"]["pair_name_split"],
                    )
                    for sub_matches in split_dict(matches, math.ceil(len(matches) / 1))
                ]
                updated_matches = dict(ChainMap(*obj_refs))

                # Post process keypoints:
                keypoints = {k: v for k, v in keypoints.items() if isinstance(v, dict)}
                pb = ProgressBar(len(keypoints), "Post-processing keypoints...") if verbose is not None else None
                obj_refs = [
                    transform_keypoints_ray_wrapper.remote(sub_kpts, img_scales=img_scales, on_resized_space=cfgs['merge_cfgs']['merge_on_resized_space'] if match_round_ratio == -1 else False, pba=pb.actor if pb is not None else None, verbose=verbose)
                    for sub_kpts in split_dict(
                        keypoints, math.ceil(len(keypoints) / cfg_ray["n_workers"])
                    )
                ]
                pb.print_until_done() if pb is not None else None
                kpts_scores = ray.get(obj_refs)
                final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
                final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))
        else: 
            # Matcher runner
            if osp.exists(match_cache_path) and not regenerate_match:
                if merge_and_prop:
                    matches = load_h5(match_cache_path, transform_slash=True)
                    img_scales = load_h5(img_scales_cache_path, transform_slash=True)
                    logger.info("Caches raw matches loaded!")
            else:
                all_ids = np.arange(0, len(pair_list))

                matches, img_scales = match_worker(all_ids, image_lists, image_mask_path, pair_list, cfgs, verbose=verbose)
                logger.info("Matcher finish!")

                # over write anyway
                logger.info(f"Raw matches cach begin: {match_cache_path}")
                save_h5(matches, match_cache_path, verbose=verbose, as_half=False)
                save_h5(img_scales, img_scales_cache_path, verbose=verbose, as_half=True)
                logger.info(f"Raw matches cached: {match_cache_path}")

            if merge_and_prop:
                # Change names to relative:
                image_basename_lists = [str(img_path).split(f'/{dataset_name}/')[1] for img_path in image_lists]
                img_scales = {str(img_path).split(f'/{dataset_name}/')[1] : scale for img_path, scale in img_scales.items()}
                matches_renamed = {}
                for pair, match in matches.items():
                    img0_path, img1_path = pair.split(' ')
                    # Filter wired matches may caused by previous half save:
                    valid_mask = ~((match[:, :4] < 0).sum(-1) > 0) | ((match[:, :4] > 10000).sum(-1) > 0)
                    if valid_mask.sum() < valid_mask.shape[0]:
                        logger.warning(f"Filted {valid_mask.shape[0] - valid_mask.sum()} wired matches")
                    matches_renamed[' '.join([str(img0_path).split(f'/{dataset_name}/')[1], str(img1_path).split(f'/{dataset_name}/')[1]])] = match[valid_mask]
                matches = matches_renamed

                # Combine keypoints
                logger.info("Combine keypoints!")
                all_kpts = Match2Kpts(
                    matches, image_basename_lists, img_scales, on_resized_space=cfgs['merge_cfgs']['merge_on_resized_space'] if match_round_ratio == -1 else False, name_split=cfgs["matcher"]["pair_name_split"]
                )
                sub_kpts = chunks(all_kpts, math.ceil(n_imgs / 1))  # equal to only 1 worker
                obj_refs = [keypoint_worker(sub_kpt, merge_cfg=cfgs['merge_cfgs'] if match_round_ratio == -1 else None, verbose=verbose) for sub_kpt in sub_kpts]
                keypoints = obj_refs[0]

                # Convert keypoints match to keypoints indexs
                logger.info("Update matches")
                obj_refs = [
                    update_matches(
                        sub_matches,
                        keypoints,
                        img_scales,
                        on_resized_space=cfgs['merge_cfgs']['merge_on_resized_space'] if match_round_ratio == -1 else False,
                        merge=True if match_round_ratio == -1 else False,
                        verbose=verbose,
                        pair_name_split=cfgs["matcher"]["pair_name_split"],
                    )
                    for sub_matches in split_dict(matches, math.ceil(len(matches) / 1))
                ]
                updated_matches = obj_refs[0]

                # Post process keypoints:
                keypoints = {
                    k: v for k, v in keypoints.items() if isinstance(v, dict)
                }
                logger.info("Post-processing keypoints...")
                kpts_scores = [
                    transform_keypoints(sub_kpts, img_scales=img_scales, on_resized_space=cfgs['merge_cfgs']['merge_on_resized_space'] if match_round_ratio == -1 else False, verbose=verbose)
                    for sub_kpts in split_dict(keypoints, math.ceil(len(keypoints) / 1))
                ]
                final_keypoints = [k for k, _ in kpts_scores][0]

    if merge_and_prop:
        # Reformat keypoints_dict and matches_dict
        if Path(track_cache_path).exists():
            # Pre-cached keypoints and tracks:
            final_keypoints = load_h5(kpts_cache_path, transform_slash=True)
            keypoints_renamed = {}
            name2id_dict = {key: id+1 for id, key in enumerate(final_keypoints.keys())} # imgid start from 1
            id2name_dict = {id+1: key for id, key in enumerate(final_keypoints.keys())}

            num_kpts_list = [len(value) for value in final_keypoints.values()]
            max_num_kpts = max(num_kpts_list)
            for id, (key, value) in enumerate(final_keypoints.items()):
                keypoints_renamed[id+1] = value

            t0 = time()
            track_infos = load_obj(track_cache_path)
            tracks, visible_tracks, visible_keypoints = track_infos['tracks'], track_infos['visible_tracks'], track_infos['visible_keypoints']
            t1 = time()
        else:
            keypoints_renamed = {}
            name2id_dict = {key: id+1 for id, key in enumerate(final_keypoints.keys())} # imgid start from 1
            id2name_dict = {id+1: key for id, key in enumerate(final_keypoints.keys())}

            num_kpts_list = [len(value) for value in final_keypoints.values()]
            max_num_kpts = max(num_kpts_list)
            for id, (key, value) in enumerate(final_keypoints.items()):
                keypoints_renamed[id+1] = value

            matches_renamed = {}
            for key, value in updated_matches.items():
                name0, name1 = key.split(cfgs["matcher"]["pair_name_split"])
                new_pair_name = cfgs["matcher"]["pair_name_split"].join(
                    [str(name2id_dict[name0]), str(name2id_dict[name1])]
                )
                matches_renamed[new_pair_name] = value

            t0 = time()
            matches_list = MatchesList(len(final_keypoints), max_num_kpts, 'largearray', matches_renamed)
            t1 = time()

            tracks, visible_tracks, visible_keypoints = compute_tracks(
                len(final_keypoints), num_kpts_list, matches_list, track_degree=3, ds='largearray')
            t2 = time()
            save_h5(final_keypoints, kpts_cache_path, verbose=verbose, as_half=False)
            save_obj({'tracks': tracks, 'visible_tracks': visible_tracks, 'visible_keypoints': visible_keypoints}, track_cache_path)
        
        if merge_and_find_tracks_only:
            logger.info(f"Only perform merge and construct tracks!")
            return prop_matches_cache_path
        
        if enable_multiview_refinement:
            from .post_optimization import post_optimization
            if Path(refined_kpts_cache_path).exists():
                t0 = time()
                final_keypoints = load_h5(refined_kpts_cache_path, transform_slash=True)
                for id, (key, value) in enumerate(final_keypoints.items()):
                    keypoints_renamed[id+1] = value
                t1 = time()
            else:
                id2abs_img_path_dict = {id:str(Path(image_lists[0].split(f'/{dataset_name}/')[0])/dataset_name/img_path) for id, img_path in id2name_dict.items()}
                keypoints_renamed = post_optimization(id2abs_img_path_dict, keypoints_renamed, tracks, visible_tracks, visible_keypoints, ray_cfg=ray_cfg)
                save_h5({id2abs_img_path_dict[id] : kpts for id, kpts in keypoints_renamed.items()}, refined_kpts_cache_path, verbose=verbose, as_half=False)

            if multiview_refine_only:
                logger.info(f"Only perform track refinement!")
                return prop_matches_cache_path

        # Propogate:
        prop_matches_dict = propogate_matches(id2name_dict, n_imgs, keypoints_renamed, tracks, visible_tracks, visible_keypoints, start_interval=n_seq_matches, stop_ratio=prop_stop_ratio, bear_n_fail=prop_stop_after_n_fail, min_n_matches=prop_min_n_matches, avg_motion_thr=prop_avg_motion_thr)

        t3 = time()

        save_h5(prop_matches_dict, prop_matches_cache_path, verbose=verbose, as_half=False)
    return prop_matches_cache_path