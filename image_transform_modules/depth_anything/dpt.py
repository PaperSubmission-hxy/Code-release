import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from PIL import Image
from image_transform_modules.depth_anything.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out
        
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('image_transform_modules/depth_anything/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)


class DepthAnything(DPT_DINOv2):
    def __init__(self, config):
        super().__init__(**config)
        
    def estimate_depth(self, data):
        with torch.no_grad():
            image = data['image']
            h, w, C = data["original_size"]
            depth_pred = self.forward(image)
            
            # # Resize back to original resolution by F.interpolate
            # depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            # depth = (depth- depth.min()) / (depth.max() - depth.min() + 1e-4)
            # # depth value fo distant objects will become small(close to 0), while the depth value of closer objects will become larger(closer to 1).
            # # depth_pred = 1.0 - depth
            # depth_pred = depth
            # depth_pred = depth_pred.cpu().numpy().astype(np.uint8)
            # # the grayscale depth map
            # depth_pred = np.repeat(depth_pred[..., np.newaxis], 3, axis=-1)
            # # apply a color palette to the depth map
            # # depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)
            
            # Scale prediction to [0, 1]
            depth_pred = depth_pred[0].cpu().numpy().astype(np.float64)

            # depth_non_zero = depth[depth!=0]
            # vmin = np.percentile(depth_non_zero, 2)
            # vmax = np.percentile(depth_non_zero, 85)
            # depth -= vmin
            # depth /= (vmax - vmin + 1e-4)
            # # depth_pred = 1.0 - depth
            # depth_pred = depth

            # Resize back to original resolution
            depth_pred = Image.fromarray(depth_pred)
            depth_pred = np.asarray(depth_pred.resize((w, h)))

        return depth_pred

    
    def save_depth_img(self, depth, save_path, save_mask_path=None):
        """
        depth: np.ndarray, H * W
        """
        depth = depth.astype(np.float64)
        non_zero_mask = ((depth!=0)*255).astype(np.uint8)
        depth_non_zero = depth[depth!=0]
        vmin = np.percentile(depth_non_zero, 2)
        vmax = np.percentile(depth_non_zero, 85)
        depth -= vmin
        depth /= (vmax - vmin + 1e-4)
        image = (depth * 255.0).clip(0, 255).astype(np.uint8)
        # non_zero_mask = ((depth>=0)*255).astype(np.uint8)

        Image.fromarray(image).save(save_path)

        if save_mask_path:
            Image.fromarray(non_zero_mask).save(save_mask_path)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--encoder",
#         default="vits",
#         type=str,
#         choices=["vits", "vitb", "vitl"],
#     )
#     args = parser.parse_args()
    
#     model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
#     print(model)
    