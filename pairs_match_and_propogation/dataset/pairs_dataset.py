import os
import numpy as np
import random
import os.path as osp
import torch
from PIL import Image

from torch.utils.data import Dataset
from .utils import (
    read_grayscale, read_rgb
)

class PairMatchesDataset(Dataset):
    """Build image Matching Challenge Dataset image pair (val & test)"""

    def __init__(
        self,
        args,
        image_lists,
        covis_pairs,
        subset_ids,
        image_mask_path=None,
        rotation_adapt_cfg=None,
        left_kpts=None,
    ):
        """
        Parameters:
        ---------------
        """
        super().__init__()
        self.args = args
        self.img_dir = image_lists
        self.img_resize = args['img_resize']
        self.df = args['df']
        self.pad_to = args['pad_to']
        self.left_kpts = left_kpts
        self.img_dict = {}
        self.preload = args['img_preload']
        self.subset_ids = subset_ids # List

        if isinstance(covis_pairs, list):
            self.pair_list = covis_pairs
        else:
            assert osp.exists(covis_pairs)
            # Load pairs: 
            with open(covis_pairs, 'r') as f:
                self.pair_list = f.read().rstrip('\n').split('\n')
        
        self.rotation_interval = None
        self.rot_shake_mode = False
        if rotation_adapt_cfg is not None:
            if rotation_adapt_cfg['enable']:
                self.rotation_interval = rotation_adapt_cfg['rot_interval'] # degree
                if rotation_adapt_cfg['shake_mode']:
                    self.rot_shake_mode = True
                    self.rot_n_shake = rotation_adapt_cfg['n_shake'] # n_shake for each direction, totally 2 * n_shake
                else:
                    assert 360 % self.rotation_interval == 0

        self.img_read_func = read_grayscale if args['img_type'] == 'grayscale' else read_rgb

        if image_mask_path is not None:
            self.img_mask = np.array(Image.open(image_mask_path)) # H * W, Global mask used on all images (assume all images are the same size, i.e., video sequence)
            if len(self.img_mask.shape) == 3 and self.img_mask.shape[-1] == 4:
                self.img_mask = self.img_mask[..., :3].sum(-1).astype(bool)
        else:
            self.img_mask = None

    def __len__(self):
        return len(self.subset_ids)

    def __getitem__(self, idx):
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        pair_idx = self.subset_ids[idx]
        img_path0, img_path1 = self.pair_list[pair_idx].split(' ')
        
        img_scale0 = self.img_read_func(
            img_path0,
            self.img_mask,
            (self.img_resize,) if self.img_resize is not None else None,
            df=self.df,
            pad_to=self.pad_to,
            ret_scales=True,
        )
        img_scale1 = self.img_read_func(
            img_path1,
            self.img_mask,
            (self.img_resize,) if self.img_resize is not None else None,
            pad_to=self.pad_to,
            df=self.df,
            ret_scales=True,
        )

        img0, scale0, original_hw0 = img_scale0
        img1, scale1, original_hw1 = img_scale1

        data = {
            "image0": img0,
            "image1": img1,
            "scale0": scale0,  # 1*2
            "scale1": scale1,
            "f_name0": osp.basename(img_path0).rsplit('.', 1)[0],
            "f_name1": osp.basename(img_path1).rsplit('.', 1)[0],
            "frameID": pair_idx,
            # "img_path": [osp.join(self.img_dir, img_name)]
            "pair_key": (img_path0, img_path1),
        }

        if self.args['img_type'] != 'grayscale':
            data.update({
                "image0_rgb": img0,
                "image1_rgb": img1,
            })

        return data