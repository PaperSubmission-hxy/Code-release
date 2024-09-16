import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
from PIL import Image

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth

class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 testNpairs=300,
                 fp16=False,
                 fix_bias=False,
                 read_depth=False,
                 sample_ratio=1.0,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]
        self.sample_ratio = sample_ratio

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score > 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = -3.0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        if mode == 'test':
            self.pair_infos = self.scene_info['pair_infos'][:testNpairs].copy()
        else:
            self.pair_infos = self.scene_info['pair_infos'].copy()

        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 4000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.
        self.dataset_name = self.scene_info['dataset_name'] if "dataset_name" in self.scene_info else 'megadepth'
        self.load_origin_rgb = kwargs["load_origin_rgb"]
        self.read_gray = kwargs["read_gray"]
        self.normalize_img = kwargs["normalize_img"]
        self.resize_by_stretch = kwargs["resize_by_stretch"]
        self.gt_matches_padding_n = kwargs["gt_matches_padding_n"]

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        
        self.fp16 = fp16
        self.fix_bias = fix_bias
        if self.fix_bias:
            self.df = 1          
        self.read_depth = read_depth
        
    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        if len(self.pair_infos[idx]) == 3:
            (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
        elif len(self.pair_infos[idx]) == 2:
            (idx0, idx1), overlap_score = self.pair_infos[idx]
        else:
            raise NotImplementedError

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.dataset_name, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.dataset_name, self.scene_info['image_paths'][idx1])

        image0, mask0, scale0, origin_img_size0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None, read_gray=self.read_gray, normalize_img=self.normalize_img, resize_by_stretch=self.resize_by_stretch)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1, origin_img_size1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None, read_gray=self.read_gray, normalize_img=self.normalize_img, resize_by_stretch=self.resize_by_stretch)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.dataset_name, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.dataset_name, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        homo_mask0 = torch.zeros((1, image0.shape[-2], image0.shape[-1]))
        homo_mask1 = torch.zeros((1, image1.shape[-2], image1.shape[-1]))
        gt_matches = torch.zeros((self.gt_matches_padding_n, 4), dtype=torch.float)

        if self.fp16:
            data = {
                'image0': image0.half(),  # (1, h, w)
                'depth0': depth0.half(),  # (h, w)
                'image1': image1.half(),
                'depth1': depth1.half(),
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'homo_mask0': homo_mask0,
                'homo_mask1': homo_mask1,
                'homography': torch.zeros((3,3), dtype=torch.float),
                'gt_matches': gt_matches,
                'gt_matches_mask': torch.zeros((1,), dtype=torch.bool),
                'norm_pixel_mat': torch.zeros((3,3), dtype=torch.float),
                'homo_sample_normed': torch.zeros((3,3), dtype=torch.float),
                'origin_img_size0': origin_img_size0,
                'origin_img_size1': origin_img_size1,
                'scale0': scale0.half(),  # [scale_w, scale_h]
                'scale1': scale1.half(),
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (img_name0, img_name1),
                'rel_pair_names': (self.scene_info['image_paths'][idx0],
                                self.scene_info['image_paths'][idx1])
            }
        else:
            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'homo_mask0': homo_mask0,
                'homo_mask1': homo_mask1,
                'homography': torch.zeros((3,3), dtype=torch.float),
                'gt_matches': gt_matches,
                'gt_matches_mask': torch.zeros((1,), dtype=torch.bool),
                'norm_pixel_mat': torch.zeros((3,3), dtype=torch.float),
                'homo_sample_normed': torch.zeros((3,3), dtype=torch.float),
                'origin_img_size0': origin_img_size0,
                'origin_img_size1': origin_img_size1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (img_name0, img_name1),
                'rel_pair_names': (self.scene_info['image_paths'][idx0],
                                self.scene_info['image_paths'][idx1])
            }
        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                if self.fix_bias:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                            size=((image0.shape[1]-1)//8+1, (image0.shape[2]-1)//8+1),
                                                            mode='nearest',
                                                            recompute_scale_factor=False)[0].bool()
                else:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                            scale_factor=self.coarse_scale,
                                                            mode='nearest',
                                                            recompute_scale_factor=False)[0].bool()
            if self.fp16:
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
            else:
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        if self.load_origin_rgb:
            # For test time DKM or ROMA
            data.update({"image0_rgb_origin": torch.from_numpy(np.array(Image.open(img_name0).convert("RGB"))).permute(2,0,1) / 255., "image1_rgb_origin": torch.from_numpy(np.array(Image.open(img_name1).convert("RGB"))).permute(2,0,1) / 255.})

        return data
