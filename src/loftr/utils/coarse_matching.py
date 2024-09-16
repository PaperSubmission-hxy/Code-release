import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from loguru import logger
import numpy as np

INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand

# def cross_softmax(x):
#     x_exp = torch.max(x, 1, keepdim=True)[0]
#     max2 = torch.max(x, 2, keepdim=True)[0]
#     x_exp = torch.maximum(x_exp, max2)
#     del max2
#     x_exp = torch.exp(x-x_exp)
#     del x
#     x_exp_sum = torch.sum(x_exp, 1, keepdim=True) + torch.sum(x_exp, 2, keepdim=True) - x_exp
#     return x_exp/x_exp_sum
def cross_softmax(sim_matrix, noninf_mask):
    out = torch.zeros_like(sim_matrix, device=sim_matrix.device, dtype=torch.float32)
    # conditional norm for numeracal stability
    a, b = sim_matrix[noninf_mask].min(), sim_matrix.max()
    if b-a > 500:
        a, b = a.detach(), b.detach()
        logger.info(f'sim_matrix numerical range overflow: min:{a}, max:{b}, norm range to 500 for stability')
        sim_matrix = (sim_matrix - a) / (b - a) * 500
    max00 = torch.max(sim_matrix.view(sim_matrix.shape[0], -1), -1)[0]
    x_exp = torch.exp(sim_matrix-max00[:,None,None].double())
    del sim_matrix
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True) + torch.sum(x_exp, 2, keepdim=True) - x_exp
    out[noninf_mask] = (x_exp[noninf_mask] / x_exp_sum[noninf_mask]).float()
    del x_exp_sum
    # x_exp[~noninf_mask] = 0
    del noninf_mask
    # x_exp = x_exp.float()
    return out

def block_cross_softmax(sim_matrix):
    out = torch.zeros_like(sim_matrix, device=sim_matrix.device)
def temp():
    max1 = torch.max(x, 1, keepdim=True)[0]
    max2 = torch.max(x, 2, keepdim=True)[0]
    max0 = torch.maximum(max1, max2)
    max00 = torch.max(x.reshape(x.shape[0], -1), -1)[0]
    x_exp = torch.exp(x-max00[:,None,None])
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True) + torch.sum(x_exp, 2, keepdim=True) - x_exp
    return x_exp/x_exp_sum
from torch.autograd import Function

class diffsign_function(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = torch.where(x > 0, torch.ones(x.shape, dtype=x.dtype, device=x.device), torch.full(x.shape, -1, dtype=x.dtype, device=x.device))
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = torch.where(x.abs() <= 1, grad_output, torch.zeros(x.shape, dtype=x.dtype, device=x.device))
        return grad_x

def diffsign(x):
    return diffsign_function.apply(x)

class diffsign2_function(Function):
    @staticmethod
    def forward(ctx, x):
        z = torch.where(x > 0, torch.ones(x.shape, dtype=x.dtype, device=x.device), torch.full(x.shape, -1, dtype=x.dtype, device=x.device))
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def diffsign2(x):
    return diffsign2_function.apply(x)

class diffsign3_function(Function):
    @staticmethod
    def forward(ctx, x):
        z = torch.where(x > 0, torch.ones(x.shape, dtype=x.dtype, device=x.device), torch.full(x.shape, -1, dtype=x.dtype, device=x.device))
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def diffsign3(x):
    return diffsign3_function.apply(x)

class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()
        
        self.mtd = config['mtd_spvs']
        self.fix_bias = config['fix_bias']
        self.binary = config['binary']
        self.force_nearest = config['force_nearest']
        self.binary_spv = config['binary_spv']
        self.normfeat = config['normfeat']
        self.normfeatmul = config['normfeatmul']
        self.diffsign2 = config['diffsign2'] # bp gradient regardless of |x| <= 1 or not
        self.diffsign3 = config['diffsign3'] # bp gradient with sim_matrix
        self.classify = config['classify']
        if self.classify:
            self.d_classify = config['d_classify']
            self.classifier = nn.Linear(256, self.d_classify)
            nn.init.eye_(self.classifier.weight)
        self.skip_softmax = config['skip_softmax']
        
        self.fp16matmul = config['fp16matmul']
        self.seqsoftmax = config['seqsoftmax']
        self.seqsoftmax2 = config['seqsoftmax2']
        self.ratio_test = config['ratio_test']
        self.ratio_test_val = config['ratio_test_val']
        self.use_gt_coarse = config['use_gt_coarse']
        self.cross_softmax = config['cross_softmax']
        self.plot_origin_scores = config['plot_origin_scores']

        self.use_percent_thr = config['use_percent_thr']
        self.percent_thr = config['percent_thr']
        self.fine_topk = config['fine_topk']
        
        self.add_sigmoid = config['add_sigmoid']
        self.sigmoid_bias = config['sigmoid_bias']
        self.sigmoid_sigma = config['sigmoid_sigma']
        
        self.cal_per_of_gt = config['cal_per_of_gt']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        if self.match_type == 'dual_softmax':
            if self.binary:
                if self.normfeatmul:
                    with torch.autocast(enabled=False, device_type='cuda'):
                        # norm_matrix = (torch.linalg.vector_norm(feat_c0, dim=-1)**2)[..., None] * (torch.linalg.vector_norm(feat_c1, dim=-1)**2)[:, None, :]
                        norm_matrix = feat_c0.abs().mean(-1)[:,:,None] * feat_c1.abs().mean(-1)[:,None,:]
                if self.normfeat:
                    with torch.autocast(enabled=False, device_type='cuda'):
                        feat_c0, feat_c1 = torch.nn.functional.normalize(feat_c0.float(), p=2, dim=-1), torch.nn.functional.normalize(feat_c1.float(), p=2, dim=-1)
                if self.classify:
                    feat_c0, feat_c1 = self.classifier(feat_c0), self.classifier(feat_c1)
                
                if self.diffsign3:
                    feat_b0, feat_b1 = torch.sigmoid(feat_c0) - 0.5, torch.sigmoid(feat_c1) - 0.5
                    feat_b0, feat_b1 = diffsign2(feat_b0), diffsign2(feat_b1)
                else:
                    if self.diffsign2:
                        feat_b0, feat_b1 = diffsign2(feat_c0), diffsign2(feat_c1)
                    else:
                        feat_b0, feat_b1 = diffsign(feat_c0), diffsign(feat_c1)
                
                sim_matrix = torch.einsum("nlc,nsc->nls", feat_b0,
                                            feat_b1)
                assert sim_matrix.dtype == torch.float16
                if self.normfeatmul:
                    sim_matrix = sim_matrix * norm_matrix / norm_matrix.max()
                # assert torch.all(sim_matrix >= -256)
                if mask_c0 is not None:
                    sim_matrix = sim_matrix.masked_fill_(
                        ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                        -1e4
                        # float("-inf") if sim_matrix.dtype == torch.float16 else -INF
                        )
                sim_matrix = (sim_matrix + torch.tensor([256], dtype=torch.float16, device=sim_matrix.device)) / torch.tensor([512], dtype=torch.float16, device=sim_matrix.device)
                if self.binary_spv == 'l2':
                    # spv_matrix = - torch.linalg.vector_norm(feat_c0, dim=-1)[..., None]**2 - torch.linalg.vector_norm(feat_c1, dim=-1)[:, None, :]**2 + 2*torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
                    spv_matrix = sim_matrix
                elif self.binary_spv == 'l2_add':
                    spv_matrix = ((torch.linalg.vector_norm(feat_c0, dim=-1)**2)[..., None] + (torch.linalg.vector_norm(feat_c1, dim=-1)**2)[:, None, :] + 2*sim_matrix)/4
                    spv_matrix = ((torch.linalg.vector_norm(feat_c0, dim=-1)**2)[..., None] + (torch.linalg.vector_norm(feat_c1, dim=-1)**2)[:, None, :] + 2*sim_matrix)/4
                    # spv_matrix = ((feat_n0[:, :, None, :] + feat_n1[:, None, :, :]) ** 2) / 4
                elif self.binary_spv == 'l2_softmax':
                    if mask_c0 is not None:
                        sim_matrix = sim_matrix.float().masked_fill_(
                            ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                            -1e4
                            # float("-inf") if sim_matrix.dtype == torch.float16 else -INF
                            )

                    with torch.autocast(enabled=False, device_type='cuda'):
                        spv_matrix = sim_matrix / self.temperature
                        spv_matrix = F.softmax(spv_matrix, 1) * F.softmax(spv_matrix, 2)
                        if self.training:
                            logger.info(f'num spv: {(spv_matrix>0.1).sum()}')
                if self.training:
                    sim_matrix = sim_matrix.float()
            else:
                if self.normfeatmul:
                    pass
                elif self.normfeat:
                    with torch.autocast(enabled=False, device_type='cuda'):
                        # norm1, norm2 = torch.linalg.vector_norm(feat_c0, dim=-1).mean(), torch.linalg.vector_norm(feat_c1, dim=-1).mean(n)
                        feat_c0, feat_c1 = torch.nn.functional.normalize(feat_c0.float(), p=2, dim=-1), torch.nn.functional.normalize(feat_c1.float(), p=2, dim=-1)
                if self.fp16matmul:
                    sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                            feat_c1) / self.temperature

                    if not self.config['fp16log']:
                        del feat_c0, feat_c1
                    if mask_c0 is not None:
                        sim_matrix = sim_matrix.masked_fill(
                            ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                            -1e4
                            )
                    if self.add_sigmoid:
                        sigmoid_matrix = torch.sigmoid((sim_matrix - self.sigmoid_bias) / self.sigmoid_sigma)
                        data.update({'sigmoid_matrix':sigmoid_matrix})
                else:
                    with torch.autocast(enabled=False, device_type='cuda'):
                        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0.float(),
                                                feat_c1.float()) / self.temperature
                        if not self.config['fp16log']:
                            del feat_c0, feat_c1
                        else:
                            logger.info(f'sim_matrix: {sim_matrix.dtype}')
                            logger.info(f'sim_matrixabsmax: {sim_matrix.abs().max()}')
                        if mask_c0 is not None:
                            sim_matrix = sim_matrix.masked_fill(
                                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                                -1e4
                                )
                        if self.add_sigmoid:
                            sigmoid_matrix = torch.sigmoid((sim_matrix - self.sigmoid_bias) / self.sigmoid_sigma)
                            data.update({'sigmoid_matrix':sigmoid_matrix})
                if not self.training and not self.skip_softmax and self.plot_origin_scores:
                    axes_lengths = {
                        'h0c': data['hw0_c'][0],
                        'w0c': data['hw0_c'][1],
                        'h1c': data['hw1_c'][0],
                        'w1c': data['hw1_c'][1]
                    }
                    mask = torch.ones_like(sim_matrix, device=sim_matrix.device, dtype=torch.bool)
                    mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                                    **axes_lengths)
                    if 'mask0' not in data:
                        mask_border(mask, self.border_rm, False)
                    else:
                        mask_border_with_padding(mask, self.border_rm, False,
                                                data['mask0'], data['mask1'])
                    mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                                    **axes_lengths)

                    mask = mask * (sim_matrix == sim_matrix.max(dim=2, keepdim=True)[0]) \
                    * (sim_matrix == sim_matrix.max(dim=1, keepdim=True)[0])
                    mask_v, all_j_ids = mask.max(dim=2)
                    b_ids, i_ids = torch.where(mask_v)
                    del mask, mask_v
                    j_ids = all_j_ids[b_ids, i_ids]
                    del all_j_ids
                if self.skip_softmax:
                    sim_matrix = sim_matrix
                    if not self.training and self.plot_origin_scores:
                        histc = torch.histc(sim_matrix.reshape(-1), bins=100, min=-30, max=70)
                        data.update({'histc_skipmn_in_softmax': [histc]})
                        if 'conf_matrix_gt' in data:
                            gt_mask = data['conf_matrix_gt'] == 1
                            histc = torch.histc(sim_matrix[gt_mask].reshape(-1), bins=100, min=-30, max=70)
                            data.update({'histc_skipmn_in_softmax_gt': [histc]})
                elif self.config['fp16log']:
                    logger.info(f'feat_c0absmax: {feat_c0.abs().max()}')
                    logger.info(f'feat_c1absmax: {feat_c1.abs().max()}')
                    t1 = F.softmax(sim_matrix, 1)
                    t2 = F.softmax(sim_matrix, 2)
                    sim_matrix = t1*t2
                    del t1, t2, feat_c0, feat_c1
                else:
                    assert not (self.cross_softmax or self.seqsoftmax or self.seqsoftmax2)
                    if self.cross_softmax:
                        with torch.autocast(enabled=False, device_type='cuda'):
                            sim_matrix = sim_matrix.double()
                            sim_matrix = cross_softmax(sim_matrix, (mask_c0[..., None] * mask_c1[:, None]).bool())
                    elif self.seqsoftmax:
                        sim_matrix = F.softmax(sim_matrix, 1)
                        sim_matrix = F.softmax(sim_matrix, 2)
                    elif self.seqsoftmax2:
                        sim_matrix = F.softmax(sim_matrix, 2)
                        sim_matrix = F.softmax(sim_matrix, 1)
                    else:
                        sim_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.match_type == 'sinkhorn':
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    float("-inf") if sim_matrix.dtype == torch.float16 else -INF
                    # float("-inf")
                    )

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            if self.config['sparse_spvs']:
                data.update({'conf_matrix_with_bin': assign_matrix.clone()})
        
        if self.binary:
            data.update({'spv_matrix': spv_matrix})
        else:
            data.update({'conf_matrix': sim_matrix})

        if not self.training and not self.skip_softmax and self.plot_origin_scores:
            histc_list = []
            for b_id in range(sim_matrix.shape[0]):
                mconf = sim_matrix[b_ids, i_ids, j_ids]
                histc = torch.histc(mconf.reshape(-1), bins=20, min=0, max=1)
                histc_list.append(histc)
            # histc = torch.histc(sim_matrix[mask].reshape(-1), bins=20, min=0, max=1)
            data.update({'histc_skipmn_in_softmax': histc_list})
        # predict coarse matches from conf_matrix
        if self.add_sigmoid:
            data.update(**self.get_coarse_match(sigmoid_matrix, data))
        else:
            data.update(**self.get_coarse_match(sim_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        if self.use_percent_thr:
            # conf_matrix = (conf_matrix - conf_matrix.min()) / (conf_matrix.max() - conf_matrix.min()) * 100
            # tmp = conf_matrix.clone().reshape(-1).cpu()
            # self.thr = np.percentile(tmp, self.percent_thr)
            # del tmp
            # logger.info(f'percent_thr: {self.percent_thr}, new thr: {self.thr}')
            mask = torch.ones_like(conf_matrix, device=conf_matrix.device, dtype=torch.bool)
            # mask = conf_matrix > 0
            # val, idx = torch.sort(conf_matrix.reshape(-1), descending=False)
            # self.thr = val[int(conf_matrix.numel() * self.percent_thr)]
            # logger.info(f'percent_thr: {self.percent_thr}, new thr: {self.thr}')
        else:
            mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        if self.ratio_test:
            assert not self.mtd or self.force_nearest
            val, idx = torch.topk(conf_matrix, 2, dim=2)
            mask = mask * (val[:,:,0] * self.ratio_test_val > val[:,:,1] )[:,:,None]
            val, idx = torch.topk(conf_matrix, 2, dim=1)
            mask = mask * (val[:,0,:] * self.ratio_test_val > val[:,1,:] )[:,None,:]
            
        # 2. mutual nearest
        if self.mtd and not self.force_nearest:
            b_ids, i_ids, j_ids = torch.where(mask)
            mconf = conf_matrix[b_ids, i_ids, j_ids]
        else:
            if self.binary:
                mask = mask \
                    * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                    * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
            else:
                mask = mask \
                    * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                    * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

            # 3. find all valid coarse matches
            # this only works when at most one `True` in each row
            if not self.training and self.cal_per_of_gt: # logger for gt percent in coarse matches
                gt_mask = mask * (data['conf_matrix_gt'] == 1)
                mask_v, all_j_ids = gt_mask.max(dim=2)
                b_ids, i_ids = torch.where(mask_v)
                j_ids = all_j_ids[b_ids, i_ids]
                mconf = conf_matrix[b_ids, i_ids, j_ids]
                logger.info(f'real sum of gt in c_predict: {b_ids.shape[0]}')
                if b_ids.shape[0] != 0:
                    logger.info(f'mean of gt conf in c_predict: {mconf.mean()}')
                del gt_mask
                
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix[b_ids, i_ids, j_ids]
            
        if self.use_percent_thr and mconf.numel() > 0:
            logger.info(f'num conf_matrix_c_predict: {b_ids.shape[0]}')
            val, idx = torch.sort(mconf.reshape(-1), descending=False)
            self.thr = val[int(mconf.numel() * self.percent_thr)]
            logger.info(f'percent_thr: {self.percent_thr}, new thr: {self.thr}')
            mask = mconf > self.thr
            b_ids, i_ids, j_ids, mconf = b_ids[mask], i_ids[mask], j_ids[mask], mconf[mask]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min, ),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                    len(data['spv_b_ids']),
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),
                    device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        if self.use_gt_coarse:
            assert not self.training
            logger.info(f'using gt coarse')
            logger.info(f'num gt coarse: {len(data["spv_b_ids"])}')
            b_ids, i_ids, j_ids, mconf = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids'], torch.ones(len(data['spv_b_ids']), device=_device)
        
        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        if self.fix_bias:
            scale = 8
        else:
            scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
            dim=1) * scale1
        
        m_bids = b_ids[mconf != 0]
        
        m_bids_f = repeat(m_bids, 'b -> b k', k = self.fine_topk).reshape(-1)
        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            # 'gt_mask': mconf == 0,
            'm_bids': m_bids,  # mconf == 0 => gt matches
            'm_bids_f': m_bids_f,
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches


    def pro(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None, profiler=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        if self.match_type == 'dual_softmax':
            if self.binary:
                if self.normfeatmul:
                    with torch.autocast(enabled=False, device_type='cuda'):
                        # norm_matrix = (torch.linalg.vector_norm(feat_c0, dim=-1)**2)[..., None] * (torch.linalg.vector_norm(feat_c1, dim=-1)**2)[:, None, :]
                        norm_matrix = feat_c0.abs().mean(-1)[:,:,None] * feat_c1.abs().mean(-1)[:,None,:]
                if self.normfeat:
                    with torch.autocast(enabled=False, device_type='cuda'):
                        feat_c0, feat_c1 = torch.nn.functional.normalize(feat_c0.float(), p=2, dim=-1), torch.nn.functional.normalize(feat_c1.float(), p=2, dim=-1)
                if self.classify:
                    feat_c0, feat_c1 = self.classifier(feat_c0), self.classifier(feat_c1)
                
                if self.diffsign3:
                    feat_b0, feat_b1 = torch.sigmoid(feat_c0) - 0.5, torch.sigmoid(feat_c1) - 0.5
                    feat_b0, feat_b1 = diffsign2(feat_b0), diffsign2(feat_b1)
                else:
                    if self.diffsign2:
                        feat_b0, feat_b1 = diffsign2(feat_c0), diffsign2(feat_c1)
                    else:
                        feat_b0, feat_b1 = diffsign(feat_c0), diffsign(feat_c1)
                
                sim_matrix = torch.einsum("nlc,nsc->nls", feat_b0,
                                            feat_b1)
                assert sim_matrix.dtype == torch.float16
                if self.normfeatmul:
                    sim_matrix = sim_matrix * norm_matrix / norm_matrix.max()
                # assert torch.all(sim_matrix >= -256)
                if mask_c0 is not None:
                    sim_matrix = sim_matrix.masked_fill_(
                        ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                        -256.0
                        # float("-inf") if sim_matrix.dtype == torch.float16 else -INF
                        )
                sim_matrix = (sim_matrix + torch.tensor([256], dtype=torch.float16, device=sim_matrix.device)) / torch.tensor([512], dtype=torch.float16, device=sim_matrix.device)
                if self.binary_spv == 'l2':
                    # spv_matrix = - torch.linalg.vector_norm(feat_c0, dim=-1)[..., None]**2 - torch.linalg.vector_norm(feat_c1, dim=-1)[:, None, :]**2 + 2*torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
                    spv_matrix = sim_matrix
                elif self.binary_spv == 'l2_add':
                    spv_matrix = ((torch.linalg.vector_norm(feat_c0, dim=-1)**2)[..., None] + (torch.linalg.vector_norm(feat_c1, dim=-1)**2)[:, None, :] + 2*sim_matrix)/4
                    spv_matrix = ((torch.linalg.vector_norm(feat_c0, dim=-1)**2)[..., None] + (torch.linalg.vector_norm(feat_c1, dim=-1)**2)[:, None, :] + 2*sim_matrix)/4
                    # spv_matrix = ((feat_n0[:, :, None, :] + feat_n1[:, None, :, :]) ** 2) / 4
                elif self.binary_spv == 'l2_softmax':
                    if mask_c0 is not None:
                        sim_matrix = sim_matrix.float().masked_fill_(
                            ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                            -INF
                            # float("-inf") if sim_matrix.dtype == torch.float16 else -INF
                            )

                    with torch.autocast(enabled=False, device_type='cuda'):
                        spv_matrix = sim_matrix / self.temperature
                        spv_matrix = F.softmax(spv_matrix, 1) * F.softmax(spv_matrix, 2)
                        if self.training:
                            logger.info(f'num spv: {(spv_matrix>0.1).sum()}')
                if self.training:
                    sim_matrix = sim_matrix.float()
            else:
                if self.normfeatmul:
                    pass
                elif self.normfeat:
                    with torch.autocast(enabled=False, device_type='cuda'):
                        feat_c0, feat_c1 = torch.nn.functional.normalize(feat_c0.float(), p=2, dim=-1), torch.nn.functional.normalize(feat_c1.float(), p=2, dim=-1)
                with profiler.profile("cmatch_mul"):
                    if self.fp16matmul:
                        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                                feat_c1) / self.temperature
                        if not self.config['fp16log']:
                            del feat_c0, feat_c1
                        if mask_c0 is not None:
                            sim_matrix = sim_matrix.float().masked_fill(
                                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                                -256
                                # float("-inf") if sim_matrix.dtype == torch.float16 else -INF
                                )
                    else:
                        with torch.autocast(enabled=False, device_type='cuda'):
                            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0.float(),
                                                    feat_c1.float()) / self.temperature
                            if not self.config['fp16log']:
                                del feat_c0, feat_c1
                            else:
                                logger.info(f'sim_matrix: {sim_matrix.dtype}')
                                logger.info(f'sim_matrixabsmax: {sim_matrix.abs().max()}')
                            if mask_c0 is not None:
                                sim_matrix = sim_matrix.masked_fill(
                                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                                    -256
                                    )
                if not self.training and not self.skip_softmax and self.plot_origin_scores:
                    axes_lengths = {
                        'h0c': data['hw0_c'][0],
                        'w0c': data['hw0_c'][1],
                        'h1c': data['hw1_c'][0],
                        'w1c': data['hw1_c'][1]
                    }
                    mask = torch.ones_like(sim_matrix, device=sim_matrix.device, dtype=torch.bool)
                    mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                                    **axes_lengths)
                    if 'mask0' not in data:
                        mask_border(mask, self.border_rm, False)
                    else:
                        mask_border_with_padding(mask, self.border_rm, False,
                                                data['mask0'], data['mask1'])
                    mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                                    **axes_lengths)

                    mask = mask * (sim_matrix == sim_matrix.max(dim=2, keepdim=True)[0]) \
                    * (sim_matrix == sim_matrix.max(dim=1, keepdim=True)[0])
                    mask_v, all_j_ids = mask.max(dim=2)
                    b_ids, i_ids = torch.where(mask_v)
                    del mask, mask_v
                    j_ids = all_j_ids[b_ids, i_ids]
                    del all_j_ids
                with profiler.profile("softmax"):

                    if self.skip_softmax:
                        sim_matrix = sim_matrix
                        if not self.training and self.plot_origin_scores:
                            histc = torch.histc(sim_matrix.reshape(-1), bins=20, min=-30, max=570)
                            data.update({'histc_skipmn_in_softmax': [histc]})
                    elif self.config['fp16log']:
                        logger.info(f'feat_c0absmax: {feat_c0.abs().max()}')
                        logger.info(f'feat_c1absmax: {feat_c1.abs().max()}')
                        t1 = F.softmax(sim_matrix, 1)
                        t2 = F.softmax(sim_matrix, 2)
                        sim_matrix = t1*t2
                        del t1, t2, feat_c0, feat_c1
                    else:
                        if self.cross_softmax:
                            with torch.autocast(enabled=False, device_type='cuda'):
                                sim_matrix = sim_matrix.double()
                                sim_matrix = cross_softmax(sim_matrix, (mask_c0[..., None] * mask_c1[:, None]).bool())
                        elif self.seqsoftmax:
                            sim_matrix = F.softmax(sim_matrix, 1)
                            sim_matrix = F.softmax(sim_matrix, 2)
                        elif self.seqsoftmax2:
                            sim_matrix = F.softmax(sim_matrix, 2)
                            sim_matrix = F.softmax(sim_matrix, 1)
                        else:
                            sim_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.match_type == 'sinkhorn':
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    float("-inf") if sim_matrix.dtype == torch.float16 else -INF
                    # float("-inf")
                    )

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            if self.config['sparse_spvs']:
                data.update({'conf_matrix_with_bin': assign_matrix.clone()})
        
        if self.binary:
            data.update({'spv_matrix': spv_matrix})
        else:
            data.update({'conf_matrix': sim_matrix})

        if not self.training and not self.skip_softmax and self.plot_origin_scores:
            histc_list = []
            for b_id in range(sim_matrix.shape[0]):
                mconf = sim_matrix[b_ids, i_ids, j_ids]
                histc = torch.histc(mconf.reshape(-1), bins=20, min=0, max=1)
                histc_list.append(histc)
            # histc = torch.histc(sim_matrix[mask].reshape(-1), bins=20, min=0, max=1)
            data.update({'histc_skipmn_in_softmax': histc_list})
        # predict coarse matches from conf_matrix
        with profiler.profile("get_cmatch"):
            # data.update(**self.get_coarse_match(sim_matrix, data))
            data.update(**self.get_coarse_match_pro(sim_matrix, data, profiler))

    @torch.no_grad()
    def get_coarse_match_pro(self, conf_matrix, data, profiler):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        with profiler.profile("confidence thresholding"):
            if self.use_percent_thr:
                # conf_matrix = (conf_matrix - conf_matrix.min()) / (conf_matrix.max() - conf_matrix.min()) * 100
                # tmp = conf_matrix.clone().reshape(-1).cpu()
                # self.thr = np.percentile(tmp, self.percent_thr)
                # del tmp
                mask = torch.ones_like(conf_matrix, device=conf_matrix.device, dtype=torch.bool)
                # val, idx = torch.sort(conf_matrix.reshape(-1), descending=False)
                # self.thr = val[int(conf_matrix.numel() * self.percent_thr)]
                # logger.info(f'percent_thr: {self.percent_thr}, new thr: {self.thr}')
            else:
                mask = conf_matrix > self.thr
            mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                            **axes_lengths)
            if 'mask0' not in data:
                mask_border(mask, self.border_rm, False)
            else:
                mask_border_with_padding(mask, self.border_rm, False,
                                        data['mask0'], data['mask1'])
            mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                            **axes_lengths)

        with profiler.profile("ratio test"):
            if self.ratio_test:
                assert not self.mtd or self.force_nearest
                val, idx = torch.topk(conf_matrix, 2, dim=2)
                mask = mask * (val[:,:,0] * self.ratio_test_val > val[:,:,1] )[:,:,None]
                val, idx = torch.topk(conf_matrix, 2, dim=1)
                mask = mask * (val[:,0,:] * self.ratio_test_val > val[:,1,:] )[:,None,:]

        # 2. mutual nearest
        if self.mtd and not self.force_nearest:
            with profiler.profile("m2d mutual nearest mask"):
                b_ids, i_ids, j_ids = torch.where(mask)
                mconf = conf_matrix[b_ids, i_ids, j_ids]
        else:
            with profiler.profile("mutual nearest mask"):
                if self.binary:
                    mask = mask \
                        * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                        * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
                else:
                    mask = mask \
                        * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                        * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

            # 3. find all valid coarse matches
            # this only works when at most one `True` in each row
            with profiler.profile("mask2conf"):
                mask_v, all_j_ids = mask.max(dim=2)
                b_ids, i_ids = torch.where(mask_v)
                j_ids = all_j_ids[b_ids, i_ids]
                mconf = conf_matrix[b_ids, i_ids, j_ids]

        if self.use_percent_thr and mconf.numel() > 0:
            val, idx = torch.sort(mconf.reshape(-1), descending=False)
            self.thr = val[int(mconf.numel() * self.percent_thr)]
            logger.info(f'percent_thr: {self.percent_thr}, new thr: {self.thr}')
            mask = mconf > self.thr
            b_ids, i_ids, j_ids, mconf = b_ids[mask], i_ids[mask], j_ids[mask], mconf[mask]

        logger.info(f'real sum of conf_matrix_c_predict: {b_ids.shape[0]}')

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min, ),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                    len(data['spv_b_ids']),
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),
                    device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        with profiler.profile("index2kpts"):
            if self.fix_bias:
                scale = 8
            else:
                scale = data['hw0_i'][0] / data['hw0_c'][0]
            scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
            mkpts0_c = torch.stack(
                [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
                dim=1) * scale0
            mkpts1_c = torch.stack(
                [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
                dim=1) * scale1
        
        with profiler.profile("repeat fine conf"):
            m_bids = b_ids[mconf != 0]
            
        m_bids_f = repeat(m_bids, 'b -> b k', k = self.fine_topk).reshape(-1)
        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            # 'gt_mask': mconf == 0,
            'm_bids': m_bids,  # mconf == 0 => gt matches
            'm_bids_f': m_bids_f,
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches
