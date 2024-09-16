import argparse
import numpy as np
import os
import ray
import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import pickle
import math
import random
from loguru import logger
import natsort
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.ray_utils import ProgressBar, chunks_balance, chunks
from src.utils.utils import check_img_ok
from pairs_match_and_propogation.utils.data_io import load_h5
from pairs_match_and_propogation.coarse_match import pair_wise_matching
from pairs_match_and_propogation.utils.pairs_from_seq import pairs_from_seq

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input:
    parser.add_argument(
        '--data_path', type=str, default='data/train_data/DL3DV-10K') 
        
    parser.add_argument(
        '--verbose', action='store_true')

    parser.add_argument(
        '--regenerate_match', action='store_true')

    parser.add_argument(
        '--regenerate_prop', action='store_true')

    parser.add_argument(
        '--merge_and_prop', action='store_true')

    parser.add_argument(
        '--save_final_npz', action='store_true')

    parser.add_argument(
        '--save_final_match_type', choices=['seperate', 'in_h5'], default='in_h5')

    parser.add_argument(
        '--prop_merge_radius', type=int, default=3, help="Control the trade-off between accuracy & track-length (i.e., prop length)")

    parser.add_argument(
        '--prop_stop_ratio', type=float, default=0.03, help="Determine fail pair when prop_matches/n_kpts < ratio")

    parser.add_argument(
        '--prop_min_n_matches', type=int, default=300, help="Determine fail pair when prop_matches < thr")

    parser.add_argument(
        '--prop_avg_motion_thr', type=int, default=30, help="Determine fail pair when avg pix motion < thr pixs")

    parser.add_argument(
        '--prop_stop_after_n_fail', type=int, default=5, help="Stop when fail n times")

    parser.add_argument(
        '--enable_multiview_refinement', action='store_true')
    
    # Depth related params:
    parser.add_argument(
        '--match_method', default='ROMA', choices=['ROMA'])

    parser.add_argument(
        '--sample_seq_interval', type=int, default=4)

    parser.add_argument(
        '--match_seq_n', type=int, default=10)
    
    # Rays Related:
    # Conduct parallel matching for each scene
    parser.add_argument(
        '--match_ray_enable', action='store_true')
    parser.add_argument(
        '--match_ray_n_workers', type=int, default=4)

    parser.add_argument(
        '--select_n_imgs', type=int, default=8000)

    parser.add_argument(
        '--chunk_n_imgs_for_prop_and_save', type=int, default=None)

    parser.add_argument(
        '--merge_and_find_tracks_only', action='store_true', help='assume set merge_and_prop=True, save_final_npz=False (default) and matching has already finished, and only perform merge and return')
    parser.add_argument(
        '--multiview_refine_only', action='store_true', help="assume set merge_and_prop=True, save_final_npz=False (default), matching as well as merge and track finding, and only perform refinement and return")

    parser.add_argument(
        '--match_n_split', type=int, default=1)
    parser.add_argument(
        '--match_split_aim_id', type=int, default=0)

    # Conduct parallel process (merge & propogation) for scenes
    parser.add_argument(
        '--prop_ray_enable', action='store_true')
    parser.add_argument(
        '--prop_ray_n_workers', type=int, default=16)

    return parser.parse_args()

transforms_name = ['origin_rgb', 'depthanything', 'style_rgb2if', 'style_day2night']
probablity = [0.3, 0.2, 0.3, 0.2]

def save_result(_o_dir, _scene_id, _subset_id, _image_paths, _pos_pair_infos, dataset_name, gt_matches, _mask_paths=None, _min_pairs=20):
    if len(_pos_pair_infos) < _min_pairs:
        logger.warning(f"num of pairs is not enough {_min_pairs}, total {len(_pos_pair_infos)} pairs, skip!")
        return
    save_name = f'{_scene_id}-{_subset_id}-dataset.npz' if _subset_id is not None else f'{_scene_id}-dataset.npz'
    o_scene_path = Path(_o_dir) / save_name
    o_scene_info = {
        "dataset_name": dataset_name,
        'image_paths': _image_paths,
        "pair_infos": _pos_pair_infos,
        'gt_matches': gt_matches,
    }

    if _mask_paths is not None:
        print(f"Mask paths saved")
        o_scene_info.update({'mask_paths': _mask_paths})
    with open(o_scene_path, 'wb') as f:
        pickle.dump(o_scene_info, f)

def generate_random_id(probabilities, names):
    if not probabilities:
        raise ValueError("Probabilities list is empty.")
    
    total_probability = round(sum(probabilities), 3)
    if total_probability != 1:
        raise ValueError("Total probability should be 1.")
    
    rand_num = random.random()
    cumulative_prob = 0
    for idx, prob in enumerate(probabilities):
        cumulative_prob += prob
        if rand_num < cumulative_prob:
            return names[idx]
    
    raise ValueError("Failed to generate random ID.")

@ray.remote(num_cpus=1, max_calls=1)
def process_video_seq_ray_wrapper(*args, **kwargs):
    try:
        return process_video_seq(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error captured:{e}")

def process_video_seq(video_seq_paths, dataset_name, args, ray_config, match_output_path, npz_output_path, prop_save_name, pba=None):
    disable_tqdm = False if pba is None else True
    for video_seq_path in tqdm(video_seq_paths, total=len(video_seq_paths), desc="Process Video Seq...", disable=disable_tqdm):
        if not video_seq_path.is_dir():
            continue
        if dataset_name in ['DL3DV-10K']:
            seq_name = str(video_seq_path).split('/')[-2]
            suffix = "*.png"
        else:
            raise NotImplementedError
        image_paths = list(video_seq_path.glob(suffix))
        image_paths = [str(img_path) for img_path in image_paths]
        image_paths = natsort.natsorted(image_paths)[::args.sample_seq_interval][:args.select_n_imgs]

        if args.merge_and_prop and (args.chunk_n_imgs_for_prop_and_save is not None):
            image_paths_subsets = list(chunks(image_paths, args.chunk_n_imgs_for_prop_and_save))
        else:
            image_paths_subsets = [image_paths]
        
        enable_subset = True if len(image_paths_subsets) > 1 else False

        for subset_id, image_paths in enumerate(image_paths_subsets):
            # Construct pairs:
            pairs = pairs_from_seq(image_paths, args.match_seq_n)
            if len(pairs) == 0:
                logger.error(f"{seq_name}'s image pairs is 0")
                continue

            prop_save_name_subset = prop_save_name + f"_{subset_id}" if enable_subset else prop_save_name
            # Matching and prop:
            scene_matches_output_dir = match_output_path / seq_name
            try:
                cache_path = pair_wise_matching(image_paths, dataset_name=dataset_name, prop_save_name=prop_save_name_subset, image_mask_path=None, pair_list=pairs, matcher=args.match_method, output_dir=scene_matches_output_dir, ray_cfg=ray_config, merge_and_prop=args.merge_and_prop, regenerate_match=args.regenerate_match, regenerate_prop=args.regenerate_prop, prop_merge_radius=args.prop_merge_radius, prop_stop_ratio=args.prop_stop_ratio, prop_stop_after_n_fail=args.prop_stop_after_n_fail, prop_min_n_matches=args.prop_min_n_matches, prop_avg_motion_thr=args.prop_avg_motion_thr, enable_multiview_refinement=args.enable_multiview_refinement, merge_and_find_tracks_only=args.merge_and_find_tracks_only, multiview_refine_only=args.multiview_refine_only)
            except Exception as e:
                logger.error(f"Error captured:{e}, skip scene {seq_name}")
                continue

            if args.save_final_npz:
                save_name = f'{seq_name}-{subset_id}-dataset.npz' if enable_subset else f'{seq_name}-dataset.npz'
                scene_info_output_path = Path(npz_output_path) / save_name
                if scene_info_output_path.exists():
                    logger.info(f"{save_name} exist, skip final save!")
                    continue

                # Load propogated matches:
                try:
                    all_matches = load_h5(str(cache_path), transform_slash=True)
                except Exception as e:
                    logger.error(f"Error captured:{e}, skip scene {seq_name}")
                    continue

                all_img_paths = []
                all_mask_paths = []
                img_pairs_ids = []
                gt_matches_list = []

                if args.save_final_match_type == 'seperate':
                    match_save_dir = str(video_seq_path).replace('/scene_images/', f'/seperate_matches_{Path(npz_base_output_path).name}/')
                    if enable_subset:
                        match_save_dir = Path(match_save_dir) / f"{subset_id}"
                    Path(match_save_dir).mkdir(parents=True, exist_ok=True)
                for pair_id, (pair_name, matches) in enumerate(all_matches.items()):
                    img_path0, img_path1 = pair_name.split(' ')
                    img_path0 = str(Path(args.data_path)/img_path0)
                    img_path1 = str(Path(args.data_path)/img_path1)
                    rgb_img_path = img_path1

                    random_method = generate_random_id(probablity, transforms_name)

                    if random_method == 'origin_rgb':
                        pass
                    else:
                        img_path1 = str(img_path1).replace(f'scene_images/', f'train_{random_method}_transform/')
                    if not check_img_ok(img_path1):
                        logger.error(f"Image: {img_path1} not exist, may not generated! Use Rgb as substitution.")
                        img_path1 = rgb_img_path
                        assert Path(img_path1).exists(), f'{img_path1} not exist!'

                    swap = generate_random_id([0.5, 0.5], [True, False])
                    if swap:
                        temp_path = img_path1
                        img_path1 = img_path0
                        img_path0 = temp_path
                        matches = matches[:, [2,3,0,1]]

                    img_pairs_ids.append(((len(all_img_paths), len(all_img_paths)+1), 0.5))
                    all_img_paths += [str(img_path0).split(f'/{dataset_name}/')[1], str(img_path1).split(f'/{dataset_name}/')[1]]
                    if args.save_final_match_type == 'in_h5':
                        gt_matches_list += [matches]
                    elif args.save_final_match_type == 'seperate':
                        match_save_path = Path(match_save_dir) / f"{pair_id}.npy"
                        np.save(match_save_path, matches)
                        gt_matches_list += [str(match_save_path).split(f'/{dataset_name}/')[1]]
                save_result(str(npz_output_path), seq_name, subset_id if enable_subset else None, all_img_paths, img_pairs_ids, dataset_name, gt_matches_list)

        if pba is not None:
            pba.update.remote(1)

if __name__ == "__main__":
    args = parse_args()
    '''
    1. Matching (Need GPU)
    2. Merge and construct tracks (No GPU)
    3. Multiview refinement (Need GPU)
    4. Propogation and save results (No GPU)
    '''

    # Make output path:
    match_output_path = Path(args.data_path) / f'matches_{args.match_method}_sample_{args.sample_seq_interval}_match_{args.match_seq_n}'
    match_output_path.mkdir(parents=True, exist_ok=True)
    prop_save_name = f"prop_matches_stop_ratio_{args.prop_stop_ratio}_min_{args.prop_min_n_matches}_matches_merge_r_{args.prop_merge_radius}_bear_{args.prop_stop_after_n_fail}_fail_avgmotion_{args.prop_avg_motion_thr}"

    if args.enable_multiview_refinement:
        prop_save_name += '_mtvrefine'

    t_name = '_'.join([transform_name + '-' + str(pro) for transform_name, pro in zip(transforms_name, probablity)])
    npz_base_output_path = Path(args.data_path) / f'scene_info_matches_{args.match_method}_sample_{args.sample_seq_interval}_match_{args.match_seq_n}_{prop_save_name}_{t_name}_{args.save_final_match_type}'   # / --> concatenate the path
    npz_output_path = (npz_base_output_path / f"scene_info")   
    npz_output_path.mkdir(parents=True, exist_ok=True)

    # Read image list:
    dataset_name = Path(args.data_path).name
    if dataset_name in ['DL3DV-10K']:
        img_dir_name = 'scene_images'
        video_seq_paths = [path / 'images_4' for path in list((Path(args.data_path) / img_dir_name).glob('*'))]
    else:
        raise NotImplementedError
    
    if args.match_ray_enable:
        n_gpus = torch.cuda.device_count()
        n_cpus = multiprocessing.cpu_count()
        ray_config = {
            "slurm": False,
            "n_workers": args.match_ray_n_workers,
            "n_cpus_per_worker": max(4, int(n_cpus / args.match_ray_n_workers)),
            "n_gpus_per_worker": n_gpus / args.match_ray_n_workers,
            "local_mode": False,
        }
    else:
        ray_config = None

    assert not (args.match_ray_enable and args.prop_ray_enable), "Cannot simutaneously process parallel match and prop"

    if args.prop_ray_enable:
        # No cpu mode, and open a large number of workers:
        n_cpus = multiprocessing.cpu_count()
        cfg_ray = {
            "slurm": False,
            "n_workers": args.prop_ray_n_workers,
            "n_cpus_per_worker": max(1, int(n_cpus / args.prop_ray_n_workers)),
            "local_mode": False,
        }

        # Init ray:
        if cfg_ray["slurm"]:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                local_mode=cfg_ray["local_mode"],
                ignore_reinit_error=True,
            )

        pb = ProgressBar(len(video_seq_paths), "Process splits")
        subset_splits = chunks_balance(
            video_seq_paths, cfg_ray["n_workers"]
        )
        remote_func = process_video_seq_ray_wrapper.options(num_cpus=cfg_ray["n_cpus_per_worker"])
        obj_refs = [
            remote_func.remote(
                subset_split, dataset_name, args, ray_config, match_output_path, npz_output_path, prop_save_name, pb.actor if pb is not None else None
            )
            for subset_split in subset_splits
        ]
        pb.print_until_done() if pb is not None else None
        results = ray.get(obj_refs)
    else:
        # Image matching or multiview refinement phase
        if args.match_n_split > 1:
            # Split to multiple submit tasks in cluster:
            subset_splits = chunks_balance(
                video_seq_paths, args.match_n_split
            )
            video_subset_seq = subset_splits[args.match_split_aim_id]
        else:
            video_subset_seq = video_seq_paths
        process_video_seq(video_subset_seq, dataset_name, args, ray_config, match_output_path, npz_output_path, prop_save_name)
    
    if args.save_final_npz:
        saved_scene_list = list(npz_output_path.glob('*.npz'))
        train_list = []
        train_debug_list = []
        val_list = []

        for name in saved_scene_list:
            train_list.append(name.stem)
        
        train_debug_list = train_list[:10]

        with open(npz_base_output_path/'train_list.txt', 'w') as f:
            train_list = map(lambda x: x + '\n', train_list)  # unnecessary + '\n' but just to keep compatible with data loading.
            f.writelines(train_list)
        with open(npz_base_output_path/'train_debug_list.txt', 'w') as f:
            train_debug_list = map(lambda x: x + '\n', train_debug_list)  # unnecessary + '\n' but just to keep compatible with data loading.
            f.writelines(train_debug_list)
        with open(npz_base_output_path/'val_list.txt', 'w') as f:
            val_list = map(lambda x: x + '\n', val_list)
            f.writelines(val_list)