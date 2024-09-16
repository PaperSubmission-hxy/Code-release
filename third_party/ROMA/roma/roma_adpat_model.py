import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from .models import roma_outdoor

class ROMA_Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = roma_outdoor(device=torch.device("cpu"), upsample_preds=True, symmetric=True, attenuate_cert=True)
        self.config = config
    
    def forward(self, data):
        if self.config['load_img_in_model']:
            img0, img1 = data['image0_path'][0], data['image1_path'][0]
        else:
            img0, img1 = data['image0_rgb_origin'][0], data['image1_rgb_origin'][0] # unbatch

        H_A, W_A = data["origin_img_size0"][0][0], data["origin_img_size0"][0][1]
        H_B, W_B = data["origin_img_size1"][0][0], data["origin_img_size1"][0][1]
        warp, dense_certainty = self.model.match(img0, img1)
        # Sample matches for estimation
        matches, certainty = self.model.sample(warp, dense_certainty, num=self.config['n_sample'])
        kpts0, kpts1 = self.model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

        mask = (kpts0[:, 0] <= img0.shape[-1]-1) * (kpts0[:, 1] <= img0.shape[-2]-1) * (kpts1[:, 0] <= img1.shape[-1]-1) * (kpts1[:, 1] <= img1.shape[-2]-1)
        data.update({'m_bids': torch.zeros_like(kpts0[:, 0])[mask], "mkpts0_f": kpts0[mask], "mkpts1_f": kpts1[mask], "mconf": certainty[mask]})
        # data.update({'m_bids': torch.zeros_like(kpts0[:, 0]), "mkpts0_f": kpts0, "mkpts1_f": kpts1, "mconf": certainty})

        # Warp query points:
        if 'query_points' in data:
            detector_kpts0 = data['query_points'].to(torch.float32) # B * N * 2
            within_mask = (detector_kpts0[..., 0] >= 0) & (detector_kpts0[..., 0] <= (W_A - 1)) & (detector_kpts0[..., 1] >= 0) & (detector_kpts0[..., 1] <= (H_A - 1))
            internal_detector_kpts0 = detector_kpts0[within_mask] 
            warped_detector_kpts0, cert_A_to_B = self.model.warp_keypoints(internal_detector_kpts0, warp, dense_certainty, H_A, W_A, H_B, W_B)
            data.update({"query_points_warpped": warped_detector_kpts0})
        return data