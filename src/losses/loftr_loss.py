from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace

        method = 'matchformer' if config['method'] == 'matchformer' else 'loftr' 
        self.loss_config = config['loftr']['loss']
        self.match_type = self.config[method]['match_coarse']['match_type']
        self.sparse_spvs = self.config[method]['match_coarse']['sparse_spvs']
        self.fine_sparse_spvs = self.config[method]['match_fine']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        self.correct_neg_w = self.loss_config['correct_neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']
        self.fine_mtd_spvs = self.config['loftr']['fine']['mtd_spvs']
        # binary
        self.binary = self.config['loftr']['match_coarse']['binary']
        self.binary_spv = self.config['loftr']['match_coarse']['binary_spv']
        # COARSE_OVERLAP_WEIGHT
        self.overlap_weightc = self.config['loftr']['loss']['coarse_overlap_weight']
        self.overlap_weightf = self.config['loftr']['loss']['fine_overlap_weight']
        # add sigmoid
        self.add_sigmoid = self.config['loftr']['match_coarse']['add_sigmoid']
        # local regress
        self.local_regress = self.config['loftr']['match_fine']['local_regress']
        self.local_regressw = self.config['loftr']['fine_window_size']
        self.local_regress_nomask = self.config['loftr']['match_fine']['local_regress_nomask']
        self.local_regress_temperature = self.config['loftr']['match_fine']['local_regress_temperature']
        self.local_regress_padone = self.config['loftr']['match_fine']['local_regress_padone']
        self.local_regress_slice = self.config['loftr']['match_fine']['local_regress_slice']
        self.local_regress_inner = self.config['loftr']['match_fine']['local_regress_inner']
        # multi regress
        self.multi_regress = self.config['loftr']['match_fine']['multi_regress']
        

    def compute_coarse_loss(self, conf, conf_gt, weight=None, overlap_weight=None, force_dense=False):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        del conf_gt
        # logger.info(f'real sum of conf_matrix_c_gt: {pos_mask.sum().item()}')
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            if self.overlap_weightc:
                loss_pos = loss_pos * overlap_weight # already been masked slice in supervision
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs and not force_dense:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                # logger.info("conf_pos_c: {loss_pos}".format(loss_pos=pos_conf.mean()))
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                if self.overlap_weightc:
                    loss_pos = loss_pos * overlap_weight # already been masked slice in supervision
                if self.config['loftr']['fp16log']:
                    logger.info(f'loss_pos_f_max: {loss_pos.max()}')

                loss =  c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                # logger.info("conf_pos_c: {loss_pos}".format(loss_pos=conf[pos_mask].mean()))
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                logger.info("conf_pos_c: {loss_pos}, conf_neg_c: {loss_neg}".format(loss_pos=conf[pos_mask].mean(), loss_neg=conf[neg_mask].mean()))
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                if self.overlap_weightc and not force_dense:
                    loss_pos = loss_pos * overlap_weight # already been masked slice in supervision

                loss_pos_mean, loss_neg_mean = loss_pos.mean(), loss_neg.mean()
                if self.correct_neg_w and weight is not None:
                    return c_pos_w * loss_pos_mean + c_neg_w * loss_neg_mean / weight[neg_mask].mean()
                elif not self.correct_neg_w:
                    return c_pos_w * loss_pos_mean + c_neg_w * loss_neg_mean
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_fine_loss(self, expec_f, expec_f_gt, overlap_weight=None):
        if self.fine_mtd_spvs and not self.multi_regress:
            return self._compute_fine_loss_mtd(expec_f, expec_f_gt, overlap_weight=overlap_weight)
        elif self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_mtd(self, conf_matrix_f, conf_matrix_f_gt, overlap_weight=None):
        """
        Args:
            conf_matrix_f (torch.Tensor): [m, WW, WW] <x, y>
            conf_matrix_f_gt (torch.Tensor): [m, WW, WW] <x, y>
        """
        if conf_matrix_f_gt.shape[0] == 0:
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                            # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                pass
            else:
                return None
        pos_mask, neg_mask = conf_matrix_f_gt == 1, conf_matrix_f_gt == 0
        del conf_matrix_f_gt
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            c_neg_w = 0.

        conf = torch.clamp(conf_matrix_f, 1e-6, 1-1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']
        
        if self.fine_sparse_spvs:
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            # logger.info("conf_pos_f: {loss_pos}".format(loss_pos=conf[pos_mask].mean()))
            if self.config['loftr']['fp16log']:
                logger.info(f'loss_pos_f_max: {loss_pos.max()}')

            if self.overlap_weightf:
                loss_pos = loss_pos * overlap_weight # already been masked slice in supervision
            return c_pos_w * loss_pos.mean()
        else:
            loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
            loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
            # logger.info("conf_pos_f: {loss_pos}, conf_neg_f: {loss_neg}".format(loss_pos=conf[pos_mask].mean(), loss_neg=conf[neg_mask].mean()))
            if self.overlap_weightf:
                loss_pos = loss_pos * overlap_weight # already been masked slice in supervision

            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

    
    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        logger.info(f'correct_mask.sum(): {correct_mask.sum()}')
        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        logger.info(f'mean abs of local gt delta: {offset_l2.sqrt().mean()}')
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None])
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        if self.overlap_weightc:
            if self.binary:
                loss_c = self.compute_coarse_loss(data['spv_matrix'], data['conf_matrix_gt'], weight=c_weight, overlap_weight=data['conf_matrix_error_gt'])
            else:
                loss_c = self.compute_coarse_loss(
                    data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                        else data['conf_matrix'],
                    data['conf_matrix_gt'],
                    weight=c_weight, overlap_weight=data['conf_matrix_error_gt'])
        
        else:
            if self.binary:
                loss_c = self.compute_coarse_loss(data['spv_matrix'], data['conf_matrix_gt'], weight=c_weight)
            else:
                loss_c = self.compute_coarse_loss(
                    data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                        else data['conf_matrix'],
                    data['conf_matrix_gt'],
                    weight=c_weight)

        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 1.5 coarse-level sigmoid loss
        if self.add_sigmoid:
            loss_s = self.compute_coarse_loss(
                data['sigmoid_matrix'],
                data['conf_matrix_gt'],
                weight=c_weight, overlap_weight=data['conf_matrix_error_gt'],
                force_dense=True)

            loss += loss_s * self.loss_config['coarse_sigmoid_weight']
            loss_scalars.update({"loss_s": loss_s.clone().detach().cpu()})
            logger.info(f'loss_s: {loss_s.clone().detach().cpu() * self.loss_config["coarse_sigmoid_weight"]}')
        
        # 2. fine-level loss
        if not self.fine_mtd_spvs or self.multi_regress:
            if self.multi_regress and 'm_ids_d' in data:
                data['expec_f'] = data['expec_f'][data['m_ids_d'], data['i_ids_d']] # select mutual nearest gt & depth val
                del data['m_ids_d'], data['i_ids_d']
            loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        else:
            if self.overlap_weightf:
                loss_f = self.compute_fine_loss(data['conf_matrix_f'], data['conf_matrix_f_gt'], data['conf_matrix_f_error_gt'])
            else:
                loss_f = self.compute_fine_loss(data['conf_matrix_f'], data['conf_matrix_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        if self.local_regress:
            if 'expec_f' not in data:
                if self.local_regress_slice:
                    sim_matrix_f, m_ids_dl, i_ids_dl, j_ids_d_il, j_ids_d_jl = data['sim_matrix_ff'], data['m_ids_dl'], data['i_ids_dl'], data['j_ids_d_il'], data['j_ids_d_jl']
                    del data['sim_matrix_ff'], data['m_ids_dl'], data['i_ids_dl'], data['j_ids_d_il'], data['j_ids_d_jl']
                else:
                    sim_matrix_f, m_ids_dl, i_ids_dl, j_ids_d_il, j_ids_d_jl = data['sim_matrix_f'], data['m_ids_dl'], data['i_ids_dl'], data['j_ids_d_il'], data['j_ids_d_jl']
                    del data['sim_matrix_f'], data['m_ids_dl'], data['i_ids_dl'], data['j_ids_d_il'], data['j_ids_d_jl']
                delta = create_meshgrid(3, 3, True, sim_matrix_f.device).to(torch.long) # [1, 3, 3, 2]
                if self.local_regress_nomask and not self.local_regress_inner:
                    delta = delta + torch.tensor([1], dtype=torch.long, device=sim_matrix_f.device) # [1, 3, 3, 2]
                m_ids_dl = m_ids_dl[...,None,None].expand(-1, 3, 3)
                i_ids_dl = i_ids_dl[...,None,None].expand(-1, 3, 3)
                # delta in (x, y) format
                j_ids_d_il = j_ids_d_il[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
                j_ids_d_jl = j_ids_d_jl[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]

                if self.local_regress_inner:
                    sim_matrix_f = sim_matrix_f.reshape(-1, self.local_regressw*self.local_regressw, self.local_regressw+2, self.local_regressw+2) # [M, WW, W+2, W+2]
                else:
                    sim_matrix_f = sim_matrix_f.reshape(-1, self.local_regressw*self.local_regressw, self.local_regressw, self.local_regressw) # [M, WW, W, W]
                if self.local_regress_nomask and not self.local_regress_inner:
                    sim_matrix_f = F.pad(sim_matrix_f, (1,1,1,1)) # [M, WW, W+2, W+2]
                sim_matrix_f = sim_matrix_f[m_ids_dl, i_ids_dl, j_ids_d_il, j_ids_d_jl]
                sim_matrix_f = sim_matrix_f.reshape(-1, 9)

                if self.local_regress_padone: # detach the gradient of center
                    sim_matrix_f[:,4] = -1e4  # inplace due to no need result for matmul's backward
                    softmax = F.softmax(sim_matrix_f / self.local_regress_temperature, -1)
                    heatmap = torch.ones_like(softmax) # ones_like for detach the gradient of center
                    heatmap[:,:4], heatmap[:,5:] = softmax[:,:4], softmax[:,5:]
                    heatmap = heatmap.reshape(-1, 3, 3)
                else:
                    sim_matrix_f = F.softmax(sim_matrix_f / self.local_regress_temperature, dim=-1)
                    heatmap = sim_matrix_f.reshape(-1, 3, 3)
                
                # compute coordinates from heatmap
                coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
                grid_normalized = create_meshgrid(3, 3, True, heatmap.device).reshape(1, -1, 2)
                
                # compute std over <x, y>
                var = torch.sum(grid_normalized**2 * heatmap.view(-1, 9, 1), dim=1) - coords_normalized**2
                std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)
                
                # for fine-level supervision
                # data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})
                data.update({'expec_f': coords_normalized})

            
            # loss_l = self._compute_fine_loss_l2_std(data['expec_f'], data['expec_f_gt'])
            loss_l = self._compute_fine_loss_l2(data['expec_f'], data['expec_f_gt'])
            loss += loss_l * self.loss_config['local_weight']
            loss_scalars.update({"loss_l":  loss_l.clone().detach().cpu()})
            # logger.info(f'loss_l: {loss_l.clone().detach().cpu() * self.loss_config["local_weight"]}')

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
