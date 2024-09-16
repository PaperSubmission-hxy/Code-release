import os
import math
from loguru import logger
from collections import abc
import numpy as np
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed
import random

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
)

from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.utils import comm
from src.datasets.megadepth import MegaDepthDataset
from src.datasets.megadepth_cross_modal import MegaDepthDatasetCrossModal
from src.datasets.common_data_pair_warp import CommonDatasetHomoWarp
from src.datasets.common_data_pair import CommonDataset
from src.datasets.sampler import RandomConcatSampler


class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.train_data_source = config.DATASET.TRAIN_DATA_SOURCE
        self.val_data_source = config.DATASET.VAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_data_sample_ratio = config.DATASET.TRAIN_DATA_SAMPLE_RATIO
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH
        self.train_gt_matches_padding_n = config.DATASET.TRAIN_GT_MATCHES_PADDING_N

        # 2. dataset config
        # general options
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # ScanNet options
        self.scan_img_resizeX = config.DATASET.SCAN_IMG_RESIZEX  # 640
        self.scan_img_resizeY = config.DATASET.SCAN_IMG_RESIZEY  # 480


        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD   # True
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD   # True
        self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
        self.coarse_scale = 1 / config.LOFTR.RESOLUTION[0]  # 0.125. for training loftr.

        self.load_origin_rgb = config.DATASET.LOAD_ORIGIN_RGB
        self.read_gray = config.DATASET.READ_GRAY
        self.normalize_img = config.DATASET.NORMALIZE_IMG
        self.resize_by_stretch = config.DATASET.RESIZE_BY_STRETCH
        self.homo_warp_use_mask = config.DATASET.HOMO_WARP_USE_MASK

        self.testNpairs = config.DATASET.TEST_N_PAIRS  # 1500
        self.fp16 = config.DATASET.FP16
        self.fix_bias = config.LOFTR.FIX_BIAS
        self.read_depth_test = config.LOFTR.MATCH_COARSE.USE_GT_COARSE or config.LOFTR.MATCH_FINE.USE_GT_FINE or config.LOFTR.MATCH_COARSE.PLOT_ORIGIN_SCORES or config.LOFTR.MATCH_COARSE.CAL_PER_OF_GT# read depth while testing for using gt coarse/fine
        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.n_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.n_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.n_workers,
            'pin_memory': True
        }
        
        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT
        
        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'validate', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            # logger.warning(" (set wolrd_size=1 and rank=0)")
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_data_source,
                self.train_npz_root,
                self.train_list_path,
                self.train_intrinsic_path,
                mode='train',
                min_overlap_score=self.min_overlap_score_train,
                pose_dir=self.train_pose_root)
            # setup multiple (optional) validation subsets
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                if not isinstance(self.val_data_source, (list, tuple)):
                    self.val_data_source = [self.val_data_source for _ in range(len(self.val_list_path))]
                if not isinstance(self.val_data_root, (list, tuple)):
                    self.val_data_root = [self.val_data_root for _ in range(len(self.val_list_path))]
                for data_root, data_source, npz_list, npz_root in zip(self.val_data_root, self.val_data_source, self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        data_root,
                        data_source,
                        npz_root,
                        npz_list,
                        self.val_intrinsic_path,
                        mode='val',
                        min_overlap_score=self.min_overlap_score_test,
                        pose_dir=self.val_pose_root))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_data_source,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        elif stage == 'validate':
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                if not isinstance(self.val_data_source, (list, tuple)):
                    self.val_data_source = [self.val_data_source for _ in range(len(self.val_list_path))]

                for data_source, npz_list, npz_root in zip(self.val_data_source, self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        data_source,
                        npz_root,
                        npz_list,
                        self.val_intrinsic_path,
                        mode='val',
                        min_overlap_score=self.min_overlap_score_test,
                        pose_dir=self.val_pose_root))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_data_source,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root)
            logger.info(f'[rank:{self.rank}] Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_data_source,
                self.test_npz_root,
                self.test_list_path,
                self.test_intrinsic_path,
                mode='test',
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.test_pose_root)
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_roots,
                       data_sources,
                       split_npz_roots,
                       scene_list_paths,
                       intri_path,
                       mode='train',
                       min_overlap_score=0.,
                       pose_dir=None):
        """ Setup train / val / test set"""
        if not isinstance(split_npz_roots, (list, tuple)): # For multiple dataset training
            data_sources = [data_sources]
            split_npz_roots = [split_npz_roots]
            scene_list_paths = [scene_list_paths]

        if not isinstance(data_roots, (list, tuple)): # For multiple dataset training
            data_roots = [data_roots] * len(scene_list_paths)
        
        all_dataset_list = []
        for idx, (data_root, data_source, scene_list_path, split_npz_root) in enumerate(zip(data_roots, data_sources, scene_list_paths, split_npz_roots)):
            with open(scene_list_path, 'r') as f:
                npz_names = [name.split()[0] for name in f.readlines()]

            if mode == 'train':
                local_npz_names = get_local_split(npz_names, self.world_size, self.rank, self.seed)
                sample_ratio = self.train_data_sample_ratio[idx] if len(self.train_data_sample_ratio) != 1 else self.train_data_sample_ratio[0]
            else:
                local_npz_names = npz_names
                sample_ratio = 1.0
            logger.info(f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')
            
            dataset_list = self._build_concat_dataset(data_root, data_source, local_npz_names, split_npz_root, sample_ratio, intri_path,
                                    mode=mode, min_overlap_score=min_overlap_score, pose_dir=pose_dir)
            all_dataset_list += dataset_list
        if mode == 'train':
            random.shuffle(all_dataset_list)
        return ConcatDataset(all_dataset_list)

    def _build_concat_dataset(
        self,
        data_root,
        data_source,
        npz_names,
        npz_dir,
        sample_ratio,
        intrinsic_path,
        mode,
        min_overlap_score=0.,
        pose_dir=None
    ):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None
        npz_names = [f'{n}.npz' for n in npz_names]
        for npz_name in tqdm(npz_names,
                             desc=f'[rank:{self.rank}] loading {mode} datasets',
                             disable=int(self.rank) != 0):
            npz_path = osp.join(npz_dir, npz_name)
            try:
                np.load(npz_path, allow_pickle=True)
            except:
                logger.info(f"{npz_path} cannot be opened!")
                continue
            if data_source == 'MegaDepth':
                datasets.append(
                    MegaDepthDataset(data_root,
                                     npz_path,
                                     mode=mode,
                                     min_overlap_score=min_overlap_score,
                                     img_resize=self.mgdpt_img_resize,
                                     df=self.mgdpt_df,
                                     img_padding=self.mgdpt_img_pad,
                                     depth_padding=self.mgdpt_depth_pad,
                                     augment_fn=augment_fn,
                                     coarse_scale=self.coarse_scale,
                                     testNpairs=self.testNpairs,
                                     fp16 = self.fp16,
                                     load_origin_rgb=self.load_origin_rgb,
                                     read_gray=self.read_gray,
                                     normalize_img=self.normalize_img,
                                     resize_by_stretch=self.resize_by_stretch,
                                     gt_matches_padding_n=self.train_gt_matches_padding_n,
                                     fix_bias = self.fix_bias,
                                     read_depth=self.read_depth_test,
                                     sample_ratio=sample_ratio,
                                     ))
            elif "Megadepth_cross_modal" in data_source:
                probability = data_source.split('@')[-1].split('-')
                datasets.append(
                    MegaDepthDatasetCrossModal(data_root,
                                     npz_path,
                                     mode=mode,
                                     min_overlap_score=min_overlap_score,
                                     img_resize=self.mgdpt_img_resize,
                                     df=self.mgdpt_df,
                                     img_padding=self.mgdpt_img_pad,
                                     depth_padding=self.mgdpt_depth_pad,
                                     augment_fn=augment_fn,
                                     coarse_scale=self.coarse_scale,
                                     testNpairs=self.testNpairs,
                                     probability=[float(prob) for prob in probability] if len(probability) == 3 else [0.0, 0.4, 0.8],
                                     load_origin_rgb=self.load_origin_rgb,
                                     read_gray=self.read_gray,
                                     normalize_img=self.normalize_img,
                                     resize_by_stretch=self.resize_by_stretch,
                                     gt_matches_padding_n=self.train_gt_matches_padding_n,
                                     fp16 = self.fp16,
                                     fix_bias = self.fix_bias,
                                     sample_ratio=sample_ratio
                                     ))
            elif data_source == "Common_dataset":
                datasets.append(
                    CommonDataset(data_root,
                                     npz_path,
                                     mode=mode,
                                     min_overlap_score=min_overlap_score,
                                     img_resize=self.mgdpt_img_resize,
                                     df=self.mgdpt_df,
                                     img_padding=self.mgdpt_img_pad,
                                     depth_padding=self.mgdpt_depth_pad,
                                     augment_fn=augment_fn,
                                     coarse_scale=self.coarse_scale,
                                     testNpairs=self.testNpairs,
                                     load_origin_rgb=self.load_origin_rgb,
                                     read_gray=self.read_gray,
                                     normalize_img=self.normalize_img,
                                     resize_by_stretch=self.resize_by_stretch,
                                     gt_matches_padding_n=self.train_gt_matches_padding_n,
                                     fp16 = self.fp16,
                                     fix_bias = self.fix_bias,
                                     sample_ratio=sample_ratio
                                     ))
            elif data_source == "Common_dataset_homo_warp":
                datasets.append(
                    CommonDatasetHomoWarp(data_root,
                                     npz_path,
                                     mode=mode,
                                     min_overlap_score=min_overlap_score,
                                     img_resize=self.mgdpt_img_resize,
                                     df=self.mgdpt_df,
                                     img_padding=self.mgdpt_img_pad,
                                     depth_padding=self.mgdpt_depth_pad,
                                     augment_fn=augment_fn,
                                     coarse_scale=self.coarse_scale,
                                     testNpairs=self.testNpairs,
                                     load_origin_rgb=self.load_origin_rgb,
                                     read_gray=self.read_gray,
                                     normalize_img=self.normalize_img,
                                     resize_by_stretch=self.resize_by_stretch,
                                     gt_matches_padding_n=self.train_gt_matches_padding_n,
                                     fp16 = self.fp16,
                                     fix_bias = self.fix_bias,
                                     sample_ratio=sample_ratio,
                                     homo_warp_use_mask=self.homo_warp_use_mask,
                                     ))
            else:
                raise NotImplementedError()
        return datasets

    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        assert self.data_sampler in ['scene_balance']
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.data_sampler == 'scene_balance':
            sampler = RandomConcatSampler(self.train_dataset,
                                          self.n_samples_per_subset,
                                          self.subset_replacement,
                                          self.shuffle, self.repeat, self.seed)
        else:
            sampler = None
        dataloader = DataLoader(self.train_dataset, sampler=sampler, **self.train_loader_params)
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        if not isinstance(self.val_dataset, abc.Sequence):
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
            return DataLoader(self.val_dataset, sampler=sampler, **self.val_loader_params)
        else:
            dataloaders = []
            for dataset in self.val_dataset:
                sampler = DistributedSampler(dataset, shuffle=False)
                dataloaders.append(DataLoader(dataset, sampler=sampler, **self.val_loader_params))
            return dataloaders

    def test_dataloader(self, *args, **kwargs):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
