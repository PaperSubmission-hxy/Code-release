import argparse
import numpy as np
import os
import ray
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import multiprocessing
from torch.utils.data import DataLoader
import pickle
import math
import natsort
import threading
from loguru import logger
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.ray_utils import ProgressBar, chunks_balance

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input:
    parser.add_argument(
        '--data_path', type=str, default='data/train_data/google-landmark')
        
    parser.add_argument(
        '--n_split_scenes', type=int, default=500)

    parser.add_argument(
        '--n_workers', type=int, default=1)

    parser.add_argument(
        '--verbose', action='store_true')

    parser.add_argument(
        '--pure_process_no_save_scene', action='store_true', help='Only run stage1, no stage 2')

    parser.add_argument(
        '--regeneration', action='store_true')
    
    # Depth related params:
    parser.add_argument(
        '--depth_model_type', default='zoedepth_nk')

    parser.add_argument(
        # '--transform_method', default='depthanything')
        '--transform_method', default='rgb')
        # '--transform_method', default='style_rgb2if')
        # '--transform_method', default='style_day2night')

    parser.add_argument(
        '--sample_seq_interval', type=int, default=1)

    parser.add_argument(
        '--mask_method', default='depthanything')
    
    parser.add_argument(
        '--save_mask', default=True  # when using transform_method:"depthanything", we save sky mask
        )
        
    parser.add_argument(
        '--batch_size', default=1)
    
    # Rays Related:
    parser.add_argument(
        '--ray_enable', action='store_true')
    parser.add_argument(
        '--ray_n_workers', type=int, default=4)

    return parser.parse_args()

def process_n_split(params_list, worker_id=0, pba=None, transform_method='zoedepth', mask_method='depthanything', disable_gpu_use=False):
    logger.info(f"**********Worker {worker_id} running **********")
    aim_gpu_id = 0
    if not disable_gpu_use:
        if transform_method == 'depthanything':
            from image_transform_modules.depth_anything.dpt import DepthAnything
            model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
            }
            encoder = 'vitl' # or 'vitb', 'vits'
            model = DepthAnything(model_configs[encoder]).to(f"cuda:{int(aim_gpu_id)}")
            model.load_state_dict(torch.load(f'weights/depth_anything_{encoder}14.pth'))
        elif transform_method == 'style_rgb2if':
            from image_transform_modules.cyclegan.models.gen_style import img2style
            model = img2style(ckpt='weights/rgb2infrared_G_A.pth', device_id=aim_gpu_id)
            model.load_networks()
        elif transform_method == 'style_day2night':
            from image_transform_modules.cyclegan.models.gen_style import img2style
            model = img2style(ckpt='weights/day2night_G_A.pth', device_id=aim_gpu_id)
            model.load_networks()
        elif transform_method == 'rgb':
            model = None
        else:
            raise NotImplementedError
    else:
        model = None
    
    iterator = tqdm(params_list, total=len(params_list)) if pba is None else params_list
    for params in iterator:
        process_one_split(params, model, transform_method, mask_method)

        if pba is not None:
            pba.update.remote(1)

@ray.remote(num_cpus=1, max_calls=1)  # release gpu after finishing
def process_n_split_ray_wrapper(*args, **kwargs):
    try:
        return process_n_split(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error captured:{e}")

def process_one_split(params, model, transform_method='depthanything', mask_method='depthanything'):
    def save_result(_o_dir, _scene_id, _image_paths, _pos_pair_infos, dataset_name, _mask_paths=None, _min_pairs=20):
        if len(_pos_pair_infos) < _min_pairs:
            logger.warning(f"num of pairs is not enough {_min_pairs}, total {len(_pos_pair_infos)} pairs, skip!")
            return
        o_scene_path = Path(_o_dir) / f'{_scene_id}-dataset.npz'
        o_scene_info = {
            "dataset_name": dataset_name,
            'image_paths': _image_paths,
            "pair_infos": _pos_pair_infos,
        }

        if _mask_paths is not None:
            print(f"Mask paths saved")
            o_scene_info.update({'mask_paths': _mask_paths})
        with open(o_scene_path, 'wb') as f:
            pickle.dump(o_scene_info, f)

    image_paths, scene_id, args, output_path, dataset_name = params

    # Make threads to save image:
    image_save_threads = []
    if dataset_name == 'google-landmark':
        source_img_dir_name = 'train'
    elif dataset_name == "SA_1B":
        source_img_dir_name = 'images'
    elif dataset_name == "DL3DV-10K":
        source_img_dir_name = "scene_images"
    elif dataset_name in ['BlendedMVS_matching']:
        source_img_dir_name = 'source_dataset'
    else:
        raise NotImplementedError

    if model is None:
        pass
    else:
        # Check existing paths and skip:
        logger.info(f"{len(image_paths)} in total!")
        if (not args.regeneration) and args.pure_process_no_save_scene:
            process_img_paths = []
            for img_path in tqdm(image_paths, total=len(image_paths), desc="Find Un-processed Images..."):
                if 'style' in transform_method:
                    style_img_output_path = str(img_path).replace(f'/{source_img_dir_name}/', f'/train_{transform_method}_transform/')
                    if not Path(style_img_output_path).exists():
                        process_img_paths.append(img_path)
                else:
                    depth_output_path = str(img_path).replace(f'/{source_img_dir_name}/', f'/train_{transform_method}_transform/')
                    mask_output_path = str(img_path).replace(f'/{source_img_dir_name}/', f'/train_{transform_method}_mask/')
                    if not (Path(depth_output_path).exists() and Path(mask_output_path).exists()):
                        process_img_paths.append(img_path)
            image_paths = process_img_paths

        # Create model and data:
        aim_gpu_id = 0
        if transform_method == 'depthanything':
            from image_transform_modules.depth_anything.depthanything_dataset import DepthAnythingDataset
            dataset = DepthAnythingDataset(args.data_path, image_paths)
        elif transform_method == 'style_rgb2if' or transform_method == 'style_day2night':
            from image_transform_modules.cyclegan.cyclegan_dataset import CycleganDataset
            dataset = CycleganDataset(args.data_path, image_paths)
        else:
            raise NotImplementedError

        dataloader = DataLoader(dataset, num_workers=0, pin_memory=True, batch_size=args.batch_size, shuffle=False)  # num_workers=8

        if 'style' in transform_method:  # style transfer
            for data in tqdm(dataloader, total=len(dataset)):
                # NOTE: cannot batchlized since image sizes are not unique
                img_path = data['img_path'][0]
                style_img_output_path = img_path.replace(f'/{source_img_dir_name}/', f'/train_{transform_method}_transform/')
                    
                if not Path(style_img_output_path).exists() or args.regeneration:
                    data.update({'image':data['image'].cuda(aim_gpu_id)})
                    style_img = model.style_transfer(data)
                    output_dir = Path(style_img_output_path).parent
                    output_dir.mkdir(parents=True, exist_ok=True)

                    thread = threading.Thread(target=model.save_style_img, args=(style_img, style_img_output_path))
                    thread.start()
                    image_save_threads.append(thread)
            
            for thread in image_save_threads:
                thread.join()

        else:  # depth transform
            for data in tqdm(dataloader, total=len(dataset)):
                img_path = data['img_path'][0]
                depth_output_path = img_path.replace(f'/{source_img_dir_name}/', f'/train_{transform_method}_transform/')
                mask_output_path = img_path.replace(f'/{source_img_dir_name}/', f'/train_{transform_method}_mask/')
                
                if args.save_mask:  # save depth mask for depthanything
                    if not (Path(depth_output_path).exists() and Path(mask_output_path).exists()) or args.regeneration:
                        data.update({'image':data['image'].cuda(aim_gpu_id)})
                        depth = model.estimate_depth(data)
                        depth_output_dir = Path(depth_output_path).parent
                        depth_output_dir.mkdir(parents=True, exist_ok=True)
                        mask_output_dir = Path(mask_output_path).parent
                        mask_output_dir.mkdir(parents=True, exist_ok=True)
                        model.save_depth_img(depth, depth_output_path, mask_output_path)
                else:
                    if not Path(depth_output_path).exists() or args.regeneration:
                        data.update({'image':data['image'].cuda(aim_gpu_id)})
                        depth = model.estimate_depth(data)
                        output_dir = Path(depth_output_path).parent
                        output_dir.mkdir(parents=True, exist_ok=True)
                        model.save_depth_img(depth, depth_output_path, mask_output_path)

    if not args.pure_process_no_save_scene:
        all_img_paths = []
        all_mask_paths = [] if mask_method is not None else None
        img_pairs_ids = []
        num_filtered = 0
        
        for image_name in tqdm(image_paths, disable=not args.verbose):
            # img_path = Path(args.data_path) / image_name
            img_path = image_name

            try:
                img = Image.open(img_path)
            except:
                continue

            # Check image size:
            min_edge_size = min(img.size)

            if min_edge_size < 300:
                num_filtered += 1
                continue

            depth_path = str(img_path).replace(f'/{source_img_dir_name}/', f'/train_{transform_method}_transform/') if transform_method != 'rgb' else str(img_path)
            mask_path = str(img_path).replace(f'/{source_img_dir_name}/', f'/train_{mask_method}_mask/')
            style_img_path = str(img_path).replace(f'/{source_img_dir_name}/', f'/train_{transform_method}_transform/')
            if not Path(depth_path).exists():
                continue

            if Path(mask_path).exists():
                mask = np.array(Image.open(mask_path))
                valid_ratio = (mask>0).sum() / (mask.shape[0] * mask.shape[1] + 1e-4)
                if valid_ratio < 0.15:
                    num_filtered += 1
                    continue
            
            # img_pairs_ids --> (idx0, idx1, overlap_score)
            img_pairs_ids.append(((len(all_img_paths), len(all_img_paths)+1), 0.5))
            if 'style' in transform_method:   # style_transfer
                all_img_paths+=[str(img_path).split(f'/{dataset_name}/')[1], str(style_img_path).split(f'/{dataset_name}/')[1]]
            else:  # depth transform
                all_img_paths+=[str(img_path).split(f'/{dataset_name}/')[1], str(depth_path).split(f'/{dataset_name}/')[1]]
            
            if all_mask_paths is not None:
                all_mask_paths+=[str(mask_path).split(f'/{dataset_name}/')[1], str(mask_path).split(f'/{dataset_name}/')[1]]

        logger.info(f"{num_filtered} Images are filtered")
        save_result(output_path, str(scene_id), all_img_paths, img_pairs_ids, dataset_name, _mask_paths=all_mask_paths)

if __name__ == "__main__":
    args = parse_args()

    # Make output path:
    base_output_path = Path(args.data_path) / f'scene_info_{args.transform_method}_{args.n_split_scenes}'   # / --> concatenate the path
    output_path = (base_output_path / f"scene_info")   
    output_path.mkdir(parents=True, exist_ok=True)

    # Read image list:
    dataset_name = Path(args.data_path).name
    print(f"datasetname is: {dataset_name}")
    if dataset_name in ['google-landmark', 'SA_1B']:
        if dataset_name == 'google-landmark':
            img_txt_path = str(Path(args.data_path) / "train_gldv2.txt")
            with open(img_txt_path) as fid:
                image_infos = fid.read().splitlines()
            # Split image infos
            image_paths = []
            for image_info in image_infos:
                splitted = image_info.split(',')
                image_path, label = splitted[0], splitted[1]
                full_img_path = Path(args.data_path) / image_path
                image_paths.append(full_img_path)

        elif dataset_name == 'SA_1B':
            image_paths = list((Path(args.data_path) / 'images').glob("*.jpg"))
        else:
            raise NotImplementedError

        # Split and assign:
        n_imgs = len(image_paths)
        step = n_imgs // args.n_split_scenes + 1
        args_iter = []

        for assigned_scene_id, start_id in enumerate(range(0, n_imgs, step)):
            args_iter.append((image_paths[start_id:min(start_id+step, n_imgs)], assigned_scene_id, args, output_path, dataset_name))
    elif dataset_name in ['DL3DV-10K']:
        img_dir_name = 'scene_images'
        video_seq_paths = [path / 'images_4' for path in list((Path(args.data_path) / img_dir_name).glob('*'))]

        args_iter = []
        for video_seq_path in tqdm(video_seq_paths, total=len(video_seq_paths), desc="Parse images for each seq..."):
            if not video_seq_path.is_dir():
                continue
            if dataset_name in ['DL3DV-10K']:
                seq_name = str(video_seq_path).split('/')[-2]
                suffix = "*.png"
            else:
                raise NotImplementedError
            image_paths = list(video_seq_path.glob(suffix))
            image_paths = [str(img_path) for img_path in image_paths]
            image_paths = natsort.natsorted(image_paths)[::args.sample_seq_interval][:8000]

            args_iter.append((image_paths, str(seq_name), args, output_path, dataset_name))
    elif dataset_name == 'BlendedMVS_matching':
        scene_paths = list((Path(args.data_path)/"source_dataset").glob("*"))
        args_iter = []
        for scene_path in tqdm(scene_paths):
            scene_name = scene_path.stem
            img_dir_name = 'blended_images'
            # Use masked images:
            image_paths = natsort.natsorted(list((scene_path / img_dir_name).glob('*_masked.jpg')))
            args_iter.append((image_paths, str(scene_name), args, output_path, dataset_name))
    else:
        logger.error(f"Dataset name is {dataset_name}, maybe incorrenct!")
        raise NotImplementedError
    
    if args.ray_enable:
        n_cpus = multiprocessing.cpu_count()
        cfg_ray = {
            "slurm": False,
            "n_workers": args.ray_n_workers,
            "n_cpus_per_worker": max(1, int(n_cpus / args.ray_n_workers)),
            "local_mode": False,
        }

        # Init ray:
        if cfg_ray["slurm"]:
            ray.init(address=os.environ["ip_head"])
        else:
            if not args.pure_process_no_save_scene:
                ray.init(
                    num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                    local_mode=cfg_ray["local_mode"],
                    ignore_reinit_error=True,
                )
                remote_func = process_n_split_ray_wrapper.options(num_cpus=cfg_ray["n_cpus_per_worker"])
            else:
                n_gpus = (
                    torch.cuda.device_count()
                )
                cfg_ray.update({"n_gpus_per_worker": n_gpus / args.ray_n_workers,})

                ray.init(
                    num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                    num_gpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_gpus_per_worker"]),
                    local_mode=cfg_ray["local_mode"],
                    ignore_reinit_error=True,
                )
                remote_func = process_n_split_ray_wrapper.options(num_cpus=cfg_ray["n_cpus_per_worker"], num_gpus=cfg_ray["n_gpus_per_worker"])

        pb = ProgressBar(len(args_iter), "Process splits")
        subset_splits = chunks_balance(
            args_iter, cfg_ray["n_workers"]
        )
        obj_refs = [
            remote_func.remote(
                subset_split, id, pb.actor if pb is not None else None, transform_method=args.transform_method, mask_method=args.mask_method, disable_gpu_use=not args.pure_process_no_save_scene
            )
            for id, subset_split in enumerate(subset_splits)
        ]
        pb.print_until_done() if pb is not None else None
        results = ray.get(obj_refs)
    else:
        process_n_split(args_iter, transform_method=args.transform_method, mask_method=args.mask_method, disable_gpu_use=not args.pure_process_no_save_scene)
    
    if not args.pure_process_no_save_scene:
        saved_scene_list = list(output_path.glob('*.npz'))
        train_list = []
        train_debug_list = []
        val_list = []

        for name in saved_scene_list:
            train_list.append(name.stem)
        
        train_debug_list = train_list[:10]

        with open(base_output_path/'train_list.txt', 'w') as f:
            train_list = map(lambda x: x + '\n', train_list)  # unnecessary + '\n' but just to keep compatible with data loading.
            f.writelines(train_list)
        with open(base_output_path/'train_debug_list.txt', 'w') as f:
            train_debug_list = map(lambda x: x + '\n', train_debug_list)  # unnecessary + '\n' but just to keep compatible with data loading.
            f.writelines(train_debug_list)
        with open(base_output_path/'val_list.txt', 'w') as f:
            val_list = map(lambda x: x + '\n', val_list)
            f.writelines(val_list)