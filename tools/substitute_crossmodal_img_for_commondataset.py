import os
import argparse
import pprint
import pickle
import random
import ray
import math
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import multiprocessing
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils.utils import check_img_ok
from src.utils.ray_utils import ProgressBar, chunks_balance, chunks


transforms_name = ['origin_rgb', 'style_rgb2if', 'style_day2night']
probablity = [0.5, 0.25, 0.25]

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset_path', type=str, default='datasets/train_data/BlendedMVS_matching')
    parser.add_argument(
        '--npz_root', type=str, default='datasets/train_data/BlendedMVS_matching/matching_indices_0.1_0.2_0.0/scene_info')
    parser.add_argument(
        '--npz_list_path', type=str, default='datasets/train_data/BlendedMVS_matching/matching_indices_0.1_0.2_0.0/train_list.txt')

    # Conduct parallel process (merge & propogation) for scenes
    parser.add_argument(
        '--ray_enable', action='store_true')
    parser.add_argument(
        '--ray_n_workers', type=int, default=16)
    
    return parser.parse_args()

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

def swap_pair(a, b):
    temp = b
    b = a
    a = temp
    return a, b

@ray.remote(num_cpus=1, max_calls=1)
def process_n_split_ray_wrapper(*args, **kwargs):
    try:
        return process_n_split(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error captured:{e}")

def process_n_split(scene_ids, npz_output_dir, args, pba=None):
    for scene_id in scene_ids:
        npz_path = Path(args.npz_root) / f'{scene_id}.npz'
        with open(npz_path, 'rb') as f:
            scene_info = pickle.load(f)
        image_paths = scene_info['image_paths']
        depth_paths = scene_info['depth_paths']
        pair_infos = scene_info['pair_infos']
        poses = scene_info['poses']
        intrins = scene_info['intrinsics']

        image_paths_new = []
        depth_paths_new = []
        pair_infos_new = []
        poses_new = []
        intrins_new = []
        for pair_info in pair_infos:
            (idx0, idx1), _ = pair_info
            img_path0, img_path1 = Path(args.dataset_path) / image_paths[idx0], Path(args.dataset_path) / image_paths[idx1]
            rgb_img_path1 = img_path1
            depth_path0, depth_path1 = Path(args.dataset_path) / depth_paths[idx0], Path(args.dataset_path) / depth_paths[idx1]
            poses0, poses1 = poses[idx0], poses[idx1]
            intrins0, intrins1 = intrins[idx0], intrins[idx1]

            random_method = generate_random_id(probablity, transforms_name)

            if random_method == 'origin_rgb':
                pass
            else:
                img_path1 = str(img_path1).replace(f'source_dataset/', f'train_{random_method}_transform/')

            if not check_img_ok(img_path1):
                logger.error(f"Image: {img_path1} not exist, may not generated! Use Rgb as substitution.")
                img_path1 = rgb_img_path1
                assert Path(img_path1).exists(), f'{img_path1} not exist!'

            swap = generate_random_id([0.5, 0.5], [True, False])
            if swap:
                img_path0, img_path1 = swap_pair(img_path0, img_path1)
                depth_path0, depth_path1 = swap_pair(depth_path0, depth_path1)
                poses0, poses1 = swap_pair(poses0, poses1)
                intrsins0, intrins1 = swap_pair(intrins0, intrins1)
            
            pair_infos_new.append(((len(image_paths_new), len(image_paths_new)+1), 0.5))
            image_paths_new += [str(img_path0).split(f'/{dataset_name}/')[1], str(img_path1).split(f'/{dataset_name}/')[1]]
            depth_paths_new += [str(depth_path0).split(f'/{dataset_name}/')[1], str(depth_path1).split(f'/{dataset_name}/')[1]]
            poses_new += [poses0, poses1]
            intrins_new += [intrins0, intrins1]

        scene_info['image_paths'] = image_paths_new
        scene_info['depth_paths'] = depth_paths_new
        scene_info['pair_infos'] = pair_infos_new

        scene_info['poses'] = poses_new
        scene_info['intrinsics'] = intrins_new


        with open(Path(npz_output_dir) / f'{scene_id}.npz', 'wb') as f:
            pickle.dump(scene_info, f)

        if pba is not None:
            pba.update.remote(1)

if __name__ == '__main__':
    args = parse_args()
    pprint.pprint(args)
    
    with open(args.npz_list_path, 'r') as f:
        scene_ids = f.readlines()
        scene_ids = list(map(lambda x: x.split()[0], scene_ids))

    eval_scene_names = []
    dataset_name = Path(args.dataset_path).name
    original_indices_name = str(args.npz_root).split(f'/{dataset_name}/')[1].split('/')[0]
    aim_indices_name = '_'.join([original_indices_name] + [transform_name + '-' + str(pro) for transform_name, pro in zip(transforms_name, probablity)])
    npz_output_base_dir = Path(args.dataset_path) / aim_indices_name
    npz_output_dir = npz_output_base_dir / 'scene_info'
    npz_output_dir.mkdir(parents=True, exist_ok=True)

    if args.ray_enable:
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

        pb = ProgressBar(len(scene_ids), "Process splits")
        subset_splits = chunks_balance(
            scene_ids, cfg_ray["n_workers"]
        )
        remote_func = process_n_split_ray_wrapper.options(num_cpus=cfg_ray["n_cpus_per_worker"])
        obj_refs = [
            remote_func.remote(
                subset_split, npz_output_dir, args, pb.actor if pb is not None else None
            )
            for subset_split in subset_splits
        ]
        pb.print_until_done() if pb is not None else None
        results = ray.get(obj_refs)
    else:
        process_n_split(scene_ids, npz_output_dir, args)

    saved_scene_list = list(npz_output_dir.glob('*.npz'))
    train_list = []
    train_debug_list = []
    val_list = []

    for name in saved_scene_list:
        train_list.append(name.stem)

    with open(npz_output_base_dir/'train_list.txt', 'w') as f:
        train_list = map(lambda x: x + '\n', train_list)  # unnecessary + '\n' but just to keep compatible with data loading.
        f.writelines(train_list)