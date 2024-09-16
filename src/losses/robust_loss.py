from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RobustLosses(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask = False,
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
        ignore_empty_in_sparse_match_spv = False,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        self.ignore_empty_in_sparse_match_spv = ignore_empty_in_sparse_match_spv

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale, gt_match_b_mask=None):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices

            if gt_match_b_mask is not None:
                mask = torch.zeros_like(prob, dtype=torch.bool)
                if gt_match_b_mask.sum() != 0:
                    mask[gt_match_b_mask] = prob[gt_match_b_mask] > 0.99
                    # all_certainity_loss.append(F.binary_cross_entropy_with_logits(gm_certainty[:,0][gt_match_b_mask][mask], prob[gt_match_b_mask][mask]))
                
                if (~gt_match_b_mask).sum() != 0:
                    # Full supervision:
                    mask[~gt_match_b_mask] = prob[~gt_match_b_mask] > -1
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        if gt_match_b_mask is None:
            certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0], prob)
        else:
            certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:,0][mask], prob[mask])

        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:,0], prob)
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta", gt_match_b_mask=None):
        with torch.no_grad():
            if gt_match_b_mask is not None:
                mask = torch.zeros_like(prob, dtype=torch.bool)
                if gt_match_b_mask.sum() != 0:
                    mask[gt_match_b_mask] = prob[gt_match_b_mask] > 0.99
                    # all_certainity_loss.append(F.binary_cross_entropy_with_logits(gm_certainty[:,0][gt_match_b_mask][mask], prob[gt_match_b_mask][mask]))
                
                if (~gt_match_b_mask).sum() != 0:
                    # Full supervision:
                    mask[~gt_match_b_mask] = prob[~gt_match_b_mask] > -1

        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()

        if gt_match_b_mask is None:
            ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        else:
            ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0][mask], prob[mask])

        a = self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        return losses

    def forward(self, batch):
        loss_scalars = {}
        corresps = batch["corresps"]
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        gt_match_b_mask = batch['gt_matches_mask'][:, 0] if self.ignore_empty_in_sparse_match_spv else None
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),

            )
            h, w= flow.shape[-2:]
            gt_warp, gt_prob = batch['gt'][scale]['gt_warp'], batch['gt'][scale]['gt_prob']

            x2 = gt_warp.float()
            prob = gt_prob
            
            if self.local_largest_scale >= scale:
                prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale, gt_match_b_mask)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
                # loss_scalars.update({"gm_cls": gm_loss.clone().detach().cpu()})
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
                # loss_scalars.update({"gm_flow": gm_loss.clone().detach().cpu()})
            
            if delta_cls is not None:
                # flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
                # loss_scalars.update({"refine_loss": delta_cls_loss.clone().detach().cpu()})
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale, gt_match_b_mask)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
                # loss_scalars.update({"refine_loss": reg_loss.clone().detach().cpu()})
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()

        loss_scalars.update({'loss': tot_loss.clone().detach().cpu()})
        batch.update({"loss": tot_loss, "loss_scalars": loss_scalars})
