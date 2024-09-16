import ray
import os
import copy
from loguru import logger

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from functools import partial
import cv2
from time import time
from pathlib import Path

from .utils.data_io import dict_to_cuda

from .coarse_matcher_utils import agg_groupby_2d, extract_geo_model_inliers, warp_kpts, nms_fast, bisect_nms_fast
from .utils.detector_wrapper import DetectorWrapper
from .dataset.pairs_dataset import PairMatchesDataset

def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))

def build_model(args):
    pl.seed_everything(args['seed'])
    logger.info(f"Use {args['matcher']} as coarse matcher")
    match_thr = args['match_thr']
    logger.info(f'Coarse match thr is:{match_thr}')
    
    if args['matcher'] == 'ROMA':
        from third_party.ROMA.roma.roma_sfm_adpat_model import ROMA_Model
        matcher = ROMA_Model(MAX_MATCHES=10000, SAMPLE_THRESH=0.8, MATCH_THRESH=match_thr)
    else:
        raise NotImplementedError
    detector = DetectorWrapper()

    detector.eval().cuda()
    matcher.eval().cuda()
    return detector, matcher

def extract_preds(data):
    """extract predictions assuming bs==1"""
    if "m_bids" in data:
        m_bids = data["m_bids"].cpu().numpy()
        assert (np.unique(m_bids) == 0).all()
    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()
    mconfs = data["mconf"].cpu().numpy()

    return mkpts0, mkpts1, mconfs

def extract_inliers(data, args):
    """extract inlier matches assume bs==1.
    NOTE: If no inliers found, keep all matches.
    """
    mkpts0, mkpts1, mconfs = extract_preds(data)
    t0 = time()
    K0 = data["K0"][0].cpu().numpy() if args['geo_model'] == "E" else None
    K1 = data["K1"][0].cpu().numpy() if args['geo_model'] == "E" else None

    mkpts0_temp = mkpts0
    mkpts1_temp = mkpts1
    if args['ransac_on_resized_space']:
        # NOTE: rescale manner vary between loftr and sparse methodsk
        if 'scale0' in data:
            mkpts0_temp = mkpts0 / data['scale0'][:, [1,0]].cpu().numpy()
        if 'scale1' in data:
            mkpts1_temp = mkpts1 / data['scale1'][:, [1,0]].cpu().numpy()

    if len(mkpts0_temp) >= 8:
        try:
            inliers = extract_geo_model_inliers(
                mkpts0_temp,
                mkpts1_temp,
                mconfs,
                args['geo_model'],
                args['ransac_method'],
                args['pixel_thr'],
                args['max_iters'],
                args['conf_thr'],
                K0=K0,
                K1=K1,
            )
            mkpts0, mkpts1, mconfs = map(
                lambda x: x[inliers], [mkpts0, mkpts1, mconfs]
            )
        except:
            logger.error(f"RANCAC Failed!!!")
            pass
    return mkpts0, mkpts1, mconfs

@torch.no_grad()
def extract_matches(data, detector=None, matcher=None, ransac_args=None, inlier_only=True):
    # 1. inference
    detector(data)
    matcher(data)

    # 2. run RANSAC and extract inliers
    mkpts0, mkpts1, mconfs = (
        extract_inliers(data, ransac_args) if inlier_only else extract_preds(data)
    )

    return (torch.from_numpy(mkpts0), torch.from_numpy(mkpts1), torch.ones((mkpts0.shape[0],)), torch.ones((mkpts1.shape[0],)), torch.from_numpy(mkpts0), torch.from_numpy(mkpts1), mkpts0, mkpts1, mconfs)

def scale_pair_back(mkpts0, mkpts1, scale0, scale1):
    """
    Rescale mkpts back to original image scale.
    mkpts0: np.array or torch.Tensor [N*2]
    scale0: np.array or torch.Tensor, [1*2] or [2]
    """
    mkpts0 = mkpts0 * scale0
    mkpts1 = mkpts1 * scale1
    return mkpts0, mkpts1


@torch.no_grad()
def match_worker(subset_ids, image_lists, image_mask_path, covis_pairs_out, cfgs, pba=None, verbose=True):
    """extract matches from part of the possible image pair permutations"""
    args = cfgs['matcher']

    if args['semi_dense_model']['enable']:
        semi_dense_det_matchers = []
        for semi_dense_matcher_name in args['semi_dense_model']['matcher']:
            tmp_args = copy.deepcopy(args['semi_dense_model'])
            tmp_args['matcher'] = semi_dense_matcher_name
            semi_dense_det_fake, semi_dense_matcher = build_model(tmp_args)
            semi_dense_det_matchers.append([semi_dense_det_fake, semi_dense_matcher])

    matches = {}
    img_scales = {} # scales = origin_size / resized_size
    # Build dataset:
    dataset = PairMatchesDataset(cfgs["data"], image_lists, covis_pairs_out, subset_ids, image_mask_path, rotation_adapt_cfg=args["rotation_det"])
    dataloader = DataLoader(dataset, num_workers=2, pin_memory=True)

    tqdm_disable = True
    if not verbose:
        assert pba is None
    else:
        if pba is None:
            tqdm_disable = False

    # match all permutations
    for data in tqdm(dataloader, disable=tqdm_disable):
        f_name0, f_name1 = data['pair_key'][0][0], data['pair_key'][1][0]
        data_c = dict_to_cuda(data)


        data_input_for_matcher = {"image0": data_c['image0'], "image1": data_c['image1'],
                                "image0_rgb": data_c["image0_rgb"], "image1_rgb": data_c["image1_rgb"],
                                }
        if args['semi_dense_model']['enable']:
            mkpts0_all, mkpts1_all, mconfs_all = [], [], []
            for semi_dense_det_fake, semi_dense_matcher in semi_dense_det_matchers:
                _, _, _, _, _, _, mkpts0, mkpts1, mconfs = extract_matches(
                    data_input_for_matcher,
                    detector=semi_dense_det_fake,
                    matcher=semi_dense_matcher,
                    ransac_args=args['semi_dense_ransac'],
                    inlier_only=args['semi_dense_ransac']['enable'],
                )

                # Scale back to original resolution:
                mkpts0, mkpts1 = scale_pair_back(mkpts0, mkpts1, scale0=data['scale0'][:, [1,0]].numpy(), scale1=data['scale1'][:, [1,0]].numpy())

                mkpts0_all.append(mkpts0)
                mkpts1_all.append(mkpts1)
                mconfs_all.append(mconfs)

        final_matches = np.concatenate(
            [np.concatenate(
                    [mkpts0, mkpts1, mconfs[:, None]], -1
             
               ),  # (N, 5)
            ],
            0 # axis=0
        )

        # Remove Duplicated matches:
        mkpts0mkpts1 = final_matches[:, :4]
        unique_mkpts0mkpts1, index = np.unique(np.round(mkpts0mkpts1), axis=0, return_index=True)
        final_matches = final_matches[index]
        matches[args['pair_name_split'].join([f_name0, f_name1])] = final_matches

        if f_name0 not in img_scales:
            img_scales[f_name0] = data['scale0'][:, [1,0]].numpy() if 'scale0' in data else np.array([1., 1.])
        if f_name1 not in img_scales:
            img_scales[f_name1] = data['scale1'][:, [1,0]].numpy() if 'scale1' in data else np.array([1., 1.])

        if pba is not None:
            pba.update.remote(1)
    return matches, img_scales

@ray.remote(num_cpus=1, num_gpus=0.5, max_calls=1)  # release gpu after finishing
def match_worker_ray_wrapper(*args, **kwargs):
    try:
        return match_worker(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error captured in match worker:{e}")

def select_n_pts(pts, confs, n_pts, method='first_n'):
    """
    pts: np.array(N*2 or N*3), coordinates of 2D points in (x,y) format.
    confs: np.array(n), point confs of a 2D point.
    n_pts: int, num of aim points.
    """
    total_pts  = pts.shape[0]
    if n_pts is None:
        method == 'first_n'

    if method == 'first_n':
        good_matches = pts[:n_pts, :]
    else:
        raise NotImplementedError
    final_pts = good_matches.shape[0]

    return good_matches

def keypoint_worker(name_kpts, pba=None, merge_cfg=None, verbose=True):
    """merge keypoints associated with one image.
    python >= 3.7 only.
    """
    keypoints = {}

    if verbose:
        name_kpts = tqdm(name_kpts) if pba is None else name_kpts
    else:
        assert pba is None

    for name, kpts in name_kpts:
        # filtering
        if merge_cfg is None:
            kpt2score = agg_groupby_2d(kpts[:, :2].astype(int), kpts[:, -1], agg="sum")
            kpt2id_score = {
                k: (i, v)
                for i, (k, v) in enumerate(
                    sorted(kpt2score.items(), key=lambda kv: kv[1], reverse=True)
                )
            }
            keypoints[name] = kpt2id_score
        else:
            # Merge points:
            kpt2score = agg_groupby_2d(kpts[:, :2].astype(int), kpts[:, -1], agg=merge_cfg["kpt_score_agg_method"])
            # kpt2score = agg_groupby_2d(kpts[:, :2], kpts[:, -1], agg=merge_cfg["kpt_score_agg_method"])
            # 1. kpt2score dict ==> (3, N) ndarray
            kpts_scores = np.array([[k[0], k[1], v] for k, v in kpt2score.items()]).astype(float).T
            # 2. run nms
            if (kpts_scores[:2] > 1500).sum() > 0 or (kpts_scores[:2] < 0).sum() > 0:
                logger.error(f"Wired match value (>1500) exist in merge, clip them")
                kpts_scores[:2] = np.clip(kpts_scores[:2], 0, 1500)
                kpts_scores[2] = np.clip(kpts_scores[2], 0, 1.1)
            nmsed_kpts, nmsed_idxs, sprsd_kpts, pillar_kpts = nms_fast(kpts_scores, merge_cfg["nms_radius_max"]) # nmsed_kpts: N*3
            select_method = merge_cfg['select_top_method']
            selected_kpts = select_n_pts(nmsed_kpts.T, confs=kpts_scores[2][nmsed_idxs], n_pts=merge_cfg["n_kpts"], method=select_method) # N*2
            kpt2id_score = {tuple(k_s[:2].astype(int)): (i, k_s[-1]) for i, k_s in enumerate(selected_kpts)}
            keypoints[name] = kpt2id_score
            inf_dist = np.abs(sprsd_kpts - pillar_kpts).max(-1)

            _merged = inf_dist <= 5
            for _sprsd_kpt, _pillar_kpt in zip(sprsd_kpts[_merged], pillar_kpts[_merged]):
                _sprsd_kpt, _pillar_kpt = tuple(_sprsd_kpt.astype(int)), tuple(_pillar_kpt.astype(int))
                if _pillar_kpt not in kpt2id_score:  # exceed args.n_kpts, truncated.
                    continue
                _pillar_id_score = keypoints[name][_pillar_kpt]
                keypoints[name][_pillar_kpt] = (_pillar_id_score[0], _pillar_id_score[1]+kpt2score[_sprsd_kpt])
                keypoints[name][(*_sprsd_kpt, -2)] = (_pillar_id_score[0], _pillar_id_score[1]+kpt2score[_sprsd_kpt])  # -2 for multiview identification

        if pba is not None:
            pba.update.remote(1)
    return keypoints

@ray.remote(num_cpus=1)
def keypoints_worker_ray_wrapper(*args, **kwargs):
    try:
        return keypoint_worker(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error captured in kpts worker:{e}")


def update_matches(matches, keypoints, img_scales, on_resized_space=False, merge=False, pba=None, verbose=True, **kwargs):
    # convert match to indices
    ret_matches = {}

    if verbose:
        matches_items = tqdm(matches.items()) if pba is None else matches.items()
    else:
        assert pba is None
        matches_items = matches.items()

    for k, v in matches_items:
        name0, name1 = k.split(kwargs['pair_name_split'])
        v = np.copy(v)
        if on_resized_space:
            scale0, scale1 = img_scales[name0], img_scales[name1]
            v[:, :2] /= scale0
            v[:, 2:4] /= scale1

        mkpts0, mkpts1, mconfs = v[:, :2], v[:, 2:4], v[:, 4]

        mkpts0, mkpts1 = (
            map(tuple, mkpts0.astype(int)),
            map(tuple, mkpts1.astype(int)),
        )
        if (name0 not in keypoints) or (name1 not in keypoints):
            continue
        _kpts0, _kpts1 = keypoints[name0], keypoints[name1]

        mids = []
        new_confs = []
        for p0, p1, mconf in zip(mkpts0, mkpts1, mconfs):
            if p0 in _kpts0 and p1 in _kpts1:
                mids.append([_kpts0[p0][0], _kpts1[p1][0]])
                new_confs.append(mconf)
        mids = np.array(mids)
        new_confs = np.array(new_confs)

        if len(mids) == 0:
            mids = np.empty((0, 2))
            new_confs = np.empty((0,))

        def _merge_possible(name):  # only merge after dynamic nms (for now)
            return f'{name}_no-merge' not in keypoints

        if merge and _merge_possible(name0) and _merge_possible(name1):
            merge_ids = []
            merge_confs = []
            mkpts0, mkpts1 = map(tuple, v[:, :2].astype(int)), map(tuple, v[:, 2:4].astype(int))
            for p0, p1, mconf in zip(mkpts0, mkpts1, mconfs): 
                if (*p0, -2) in _kpts0 and (*p1, -2) in _kpts1:
                    merge_ids.append([_kpts0[(*p0, -2)][0], _kpts1[(*p1, -2)][0]])
                    merge_confs.append(mconf)
                elif p0 in _kpts0 and (*p1, -2) in _kpts1:
                    merge_ids.append([_kpts0[p0][0], _kpts1[(*p1, -2)][0]])
                    merge_confs.append(mconf)
                elif (*p0, -2) in _kpts0 and p1 in _kpts1:
                    merge_ids.append([_kpts0[(*p0, -2)][0], _kpts1[p1][0]]) 
                    merge_confs.append(mconf)
            merge_ids = np.array(merge_ids)
            merge_confs = np.array(merge_confs)

            if len(merge_ids) == 0:
                merge_ids = np.empty((0, 2))
                merge_confs = np.empty((0,))
                logger.warning("merge failed! No matches have been merged!")
            else:
                logger.info(f'merge successful! Merge {len(merge_ids)} matches')
            
            mids_multiview = np.concatenate([mids, merge_ids], axis=0)
            merge_confs = np.concatenate([new_confs, merge_confs], axis=0)
        else:
            assert len(mids) == v.shape[0]
            mids_multiview, merge_confs = mids, new_confs
        
        mids, idxs = np.unique(mids_multiview, axis=0, return_index=True)
        merge_confs = merge_confs[idxs]

        """
        Remove duplicated corres problem as follows by conf:
        NOTE: this may lead to the matching with direction property, i.e., a<->b != b<->a
        x------o_2
        \
        \
        \
            \ o_1
        """
        img0_pts_unique, index, inverse_idxs = np.unique(mids[:,0], axis=0, return_index=True, return_inverse=True)
        if len(index) != len(inverse_idxs):
            for idx, unique_idx in enumerate(inverse_idxs):
                selected_idx = index[unique_idx]
                if merge_confs[idx] > merge_confs[selected_idx]:
                    index[unique_idx] = idx
        mids = mids[index]
        merge_confs = merge_confs[index]

        img1_pts_unique, index, inverse_idxs = np.unique(mids[:, 1].round(), axis=0, return_index=True, return_inverse=True)
        if len(index) != len(inverse_idxs):
            for idx, unique_idx in enumerate(inverse_idxs):
                selected_idx = index[unique_idx]
                if merge_confs[idx] > merge_confs[selected_idx]:
                    index[unique_idx] = idx
        mids = mids[index]
        ret_matches[k] = mids.astype(int)  # (N,2)

        if pba is not None:
            pba.update.remote(1)

    return ret_matches

@ray.remote(num_cpus=1)
def update_matches_ray_wrapper(*args, **kwargs):
    return update_matches(*args, **kwargs)


def transform_keypoints(keypoints, img_scales, on_resized_space=False, pba=None, verbose=True):
    """assume keypoints sorted w.r.t. score"""
    ret_kpts = {}
    ret_scores = {}

    if verbose:
        keypoints_items = tqdm(keypoints.items()) if pba is None else keypoints.items()
    else:
        assert pba is None
        keypoints_items = keypoints.items()

    for k, v in keypoints_items:
        v = {_k: _v for _k, _v in v.items() if len(_k) == 2}
        kpts = np.array([list(kpt) for kpt in v.keys()]).astype(np.float32)
        scores = np.array([s[-1] for s in v.values()]).astype(np.float32)
        if len(kpts) == 0:
            logger.warning("corner-case n_kpts=0 exists!")
            kpts = np.empty((0,2))
        
        if on_resized_space:
            img_scale = img_scales[k]
            kpts *= img_scale
        ret_kpts[k] = kpts
        ret_scores[k] = scores
        if pba is not None:
            pba.update.remote(1)
    return ret_kpts, ret_scores

@ray.remote(num_cpus=1)
def transform_keypoints_ray_wrapper(*args, **kwargs):
    try:
        return transform_keypoints(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error captured in transform kpts:{e}")
