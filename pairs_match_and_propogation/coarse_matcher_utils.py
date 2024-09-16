import numpy as np
import cv2
from loguru import logger
import copy
import torch.nn.functional as F
import torch

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result, np.concatenate([rot_mat, np.array([[0,0,1]])], axis=0)

def warp_pts_by_affine(kpts, transform_matrix, reverse):
    if isinstance(kpts, torch.Tensor):
        if isinstance(transform_matrix, np.ndarray):
            transform_matrix = torch.from_numpy(transform_matrix)
        transform_matrix = transform_matrix.to(kpts.device).to(kpts.dtype)

        if reverse:
            transform_matrix = torch.linalg.inv(transform_matrix)
        kpts_homo = torch.cat([kpts, torch.ones((kpts.shape[0],1))], dim=1).transpose(1,0) # 3 * N
        kpts_homo_transformed = transform_matrix @ kpts_homo # 3 * N
        return kpts_homo_transformed[:2].transpose(1,0) # N * 2
    elif isinstance(kpts, np.ndarray):
        if isinstance(transform_matrix, torch.Tensor):
            transform_matrix = transform_matrix.cpu().numpy()

        if reverse:
            transform_matrix = np.linalg.inv(transform_matrix)
        kpts_homo = np.concatenate([kpts, np.ones((kpts.shape[0],1))], axis=1).T # 3 * N
        kpts_homo_transformed = transform_matrix @ kpts_homo # 3 * N
        return kpts_homo_transformed[:2].T # N * 2

DEGREE_REVERSE_DICT = {0:0, 180:180, 90:270, 270:90}
def warp_pts_by_degree(kpts, transform_matrix, h, w, reverse=False):
    """
    Coordinate transform by rotation degrees:
    kpts: N*2, x,y
    """
    degree = transform_matrix[0,0]
    if reverse:
        degree = DEGREE_REVERSE_DICT[degree]

    if degree == 0:
        pass
    elif degree == 90:
        x_new = kpts[:, 1]
        y_new = w - kpts[:, 0]
    elif degree == 180:
        x_new = w - kpts[:, 0]
        y_new = h - kpts[:, 1]
    elif degree == 270:
        x_new = h - kpts[:, 1]
        y_new = kpts[:, 0]
    else:
        raise NotImplementedError
    if isinstance(x_new, np.ndarray):
        kpts = np.stack([x_new, y_new], axis=1) # N*2
    elif isinstance(x_new, torch.Tensor):
        kpts = torch.stack([x_new, y_new], dim=1) # N*2
    return kpts

def warp_kpts(kpts, transform_matrix, h=None, w=None, reverse=False):
    """
    kpts: N * 2
    transform_matrix: 3 * 3 or degree (90, 180, 270)
    """
    if transform_matrix.shape == (3,3):
        # Affine matrix mode:
        return warp_pts_by_affine(kpts, transform_matrix, reverse)
    elif transform_matrix.shape == (1,1):
        assert (h is not None) and (w is not None)
        # Degree Mode: only support 90, 180, 270
        return warp_pts_by_degree(kpts, transform_matrix, h, w, reverse)
    else:
        raise NotImplementedError

def multi_scale_image(data, scale, image_id, df=8):
    assert image_id in [0,1], "error image id!"
    data = copy.deepcopy(data)

    # import ipdb; ipdb.set_trace()
    image = data[f"image{image_id}"]
    h, w = image.shape[2:]
    resize = (w // scale, h // scale)
    w_new, h_new = process_resize(w, h, resize, df=df)
    data[f"scale{image_id}"][:,0] *= (h / h_new)
    data[f"scale{image_id}"][:,1] *= (w / w_new)
    data[f"image{image_id}"] = F.interpolate(image, size=(h_new, w_new), mode='bilinear', align_corners=False)

    return data

def rotate_adapt_image(data, angle, image_id, df=8):
    assert image_id in [0,1], "error image id!"
    data = copy.deepcopy(data)

    # import ipdb; ipdb.set_trace()
    # assert data[f"scale{image_id}"].sum() == 2
    image = data[f"image{image_id}"]
    h, w = image.shape[2:]
    device = image.device

    # FIXME: rotate resized image, problem exists when rotated back matches which are in original image coordinate
    image, transform_matrix = rotate_image(image[0].permute(1,2,0).cpu().numpy(), angle)
    image = torch.from_numpy(image).to(device)[None, None] # b*c*h*w
    data[f"image{image_id}"] = image

    return data, transform_matrix

def extract_geo_model_inliers(mkpts0, mkpts1, mconfs,
                              geo_model, ransac_method, pixel_thr, max_iters, conf_thr,
                              K0=None, K1=None):
    # TODO: early return if len(mkpts) < min_candidates
    
    if geo_model == 'E':
        f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
        pixel_thr = pixel_thr / f_mean

        mkpts0, mkpts1 = map(lambda x: normalize_ketpoints(*x), [(mkpts0, K0), (mkpts1, K1)])
    
    if ransac_method == 'RANSAC':
        if geo_model == 'E':
            E, mask = cv2.findEssentialMat(mkpts0, 
                                           mkpts1,
                                           np.eye(3),
                                           threshold=pixel_thr, 
                                           prob=conf_thr, 
                                           method=cv2.RANSAC)
        elif geo_model == 'F':
            F, mask = cv2.findFundamentalMat(mkpts0,
                                             mkpts1,
                                             method=cv2.FM_RANSAC,
                                             ransacReprojThreshold=pixel_thr,
                                             confidence=conf_thr,
                                             maxIters=max_iters)
    elif ransac_method == 'DEGENSAC':
        assert geo_model == 'F'
        F, mask = pydegensac.findFundamentalMatrix(mkpts0,
                                                   mkpts1,
                                                   px_th=pixel_thr,
                                                   conf=conf_thr,
                                                   max_iters=max_iters)
    elif ransac_method == 'MAGSAC':
        F, mask = cv2.findFundamentalMat(mkpts0,
                                            mkpts1,
                                            method=cv2.USAC_MAGSAC,
                                            ransacReprojThreshold=pixel_thr,
                                            confidence=conf_thr,
                                            maxIters=max_iters)
    elif ransac_method == 'GCRANSAC':
        import pygcransac
        # https://github.com/danini/graph-cut-ransac
        w0, h0 = np.max(mkpts0, axis=0) - np.min(mkpts0, axis=0)
        w1, h1 = np.max(mkpts1, axis=0) - np.min(mkpts1, axis=0)
        F, mask = pygcransac.findFundamentalMatrix(np.concatenate([mkpts0.astype(np.float64), mkpts1.astype(np.float64)], axis=1), int(h0), int(w0), int(h1), int(w1), pixel_thr)
        # F, mask = pygcransac.findFundamentalMatrix(np.concatenate([mkpts0.astype(np.float64), mkpts1.astype(np.float64)], axis=1), int(h0), int(w0), int(h1), int(w1), pixel_thr)
    else:
        raise ValueError()
    
    if mask is not None:
        mask = mask.astype(bool).flatten()
    else:
        mask = np.full_like(mconfs, True, dtype=np.bool)
    return mask

def agg_groupby_2d(keys, vals, agg='avg'):
    """
    Args:
        keys: (N, 2) 2d keys
        vals: (N,) values to average over
        agg: aggregation method
    Returns:
        dict: {key: agg_val}
    """
    assert agg in ['avg', 'sum']
    unique_keys, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group_sums = np.bincount(group, weights=vals)
    values = group_sums if agg == 'sum' else group_sums / counts
    return dict(zip(map(tuple, unique_keys), values))


class Match2Kpts(object):
    """extract all possible keypoints for each image from all image-pair matches"""
    def __init__(self, matches, names, img_scales, on_resized_space=False, name_split='-', cov_threshold=0):
        self.names = names
        self.matches = matches
        self.cov_threshold = cov_threshold
        self.img_scales = img_scales
        self.work_on_resized_space=on_resized_space # rescale pts to resized space (which are resolution that matching perfroms), to avoid pix threshold adaption
        self.name2matches = {str(name): [] for name in names}
        for k in matches.keys():
            try:
                name0, name1 = k.split(name_split)
            except ValueError as _:
                name0, name1 = k.split('-')
            if (name0 not in self.name2matches) or (name1 not in self.name2matches):
                continue
            self.name2matches[name0].append((k, 0))
            self.name2matches[name1].append((k, 1))
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            name = self.names[idx]
            kpts = np.concatenate([self.matches[k][:, [2*id, 2*id+1, 4]]
                        for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold], 0)
            if self.work_on_resized_space:
                img_scale = self.img_scales[name]
                kpts /= img_scale # scale assumes in [x, y] formulation
            return name, kpts
        elif isinstance(idx, slice):
            names = self.names[idx]
            kpts = []
            for name in names:
                kpt = [self.matches[k][:, [2*id, 2*id+1, 4]]
                        for k, id in self.name2matches[name] if self.matches[k].shape[0] >= self.cov_threshold]
                if len(kpt) != 0:
                    kpt = np.concatenate(kpt,0)

                    if self.work_on_resized_space:
                        img_scale = self.img_scales[name]
                        kpt[:, :2] /= img_scale # scale assumes in [x, y] formulation

                    kpts.append(kpt)
                else:
                    kpts.append(np.empty((0,3)))
                    logger.warning(f"no keypoints in image:{name}")

            return list(zip(names, kpts))
        else:
            raise TypeError(f'{type(self).__name__} indices must be integers')

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points
    
    Think of a special condition: (radius=1)
        7 6 5 4 3 2 1
        7 6 5 4 3 2 1
        7 6 5 4 3 2 1
        7 6 5 4 3 2 1
        7 6 5 4 3 2 1
        7 6 5 4 3 2 1
        7 6 5 4 3 2 1
    Just running 2 iterations cannot retain all local-maximum points. (over-suppressed!)
    """
    assert(nms_radius >= 0)

    def max_pool(x):  # keep the `scores` shape unchanged
        """ Suppress points whose score isn't the maximum within the local patch.
        """
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)  # max: 1, non-max: 0 (over-kill!)
    for _ in range(2):  # restore some local-maximum points
        supp_mask = max_pool(max_mask.float()) > 0  # if there is a pillar point in the local region
        # if there is no pillar point in the local neighborhood
        # ==> neighborhood are all non-pillar points / are all pillar points (share the same score)
        #     (note that a kpts is not a pillar point does not mean there is a pillar point in its neighborhood)
        supp_scores = torch.where(supp_mask, zeros, scores)  # scores of non-suppressed points
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def nms_fast(in_corners, dist_thresh, H=None, W=None, distance_prior=False):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
    3xN [x_i,y_i,conf_i]^T

    Algo summary: 
        Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

    Grid Value Legend:
        -1 : Kept.
        s : Empty or suppressed.
        1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
        in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
        H - Image height.
        W - Image width.
        dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
        nmsed_corners - 3xM numpy matrix with surviving corners.
        nmsed_inds - M length numpy vector with surviving corner indices.
        corners - 3xN length numpy vectors of sorted in_corners (w.r.t. confidence)
        parent_inds - N length numpy vector recording parent id (in corners) of each kpt.
            kept_pts := corners[:, parent_inds] == corners
"""
    if H is None or W is None:
        H, W = map(lambda x: np.ceil(x.max()).astype(int)+1, [in_corners[1, :], in_corners[0, :]])
    grid = np.zeros((H, W), dtype=np.int32)  # Track NMS data.
    inds = np.full((H, W), -1, dtype=np.int32)  # Store indices of points.
    parent_inds = np.full((H, W), -1, dtype=np.int32)  # -1: not processed
    parent_distance = np.full((H, W), 100, dtype=np.float32) # 100: not processed
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int), np.full((1), -1).astype(int)
    # Initialize the grid.  # TODO: vectorization
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid, parent_inds, parent_distance = map(lambda x: np.pad(x, ((pad, pad), (pad, pad)), mode='constant'), [grid, parent_inds, parent_distance])
    # build distance grid to find nearest parent points
    x = np.linspace(0, 2*pad, 2*pad+1)
    y = np.linspace(0, 2*pad, 2*pad+1)
    xv , yv =np.meshgrid(x,y)
    distance_grid = abs(xv-pad) + abs(yv-pad)
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            # record which pt each pt is suppressed by.
            pt_ind = inds[rc[1], rc[0]]
            if distance_prior:
                distance_smaller_mask = parent_distance[pt[1] - pad: pt[1] + pad + 1, pt[0] - pad : pt[0] + pad + 1] > distance_grid 
                parent_inds[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1][distance_smaller_mask] = pt_ind
                parent_distance[pt[1] - pad: pt[1] + pad + 1, pt[0] - pad : pt[0] + pad + 1][distance_smaller_mask] = distance_grid[distance_smaller_mask]
            else:
                parent_inds[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = pt_ind
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    # Get all kpts' parent kpt
    out_parent_inds = parent_inds[rcorners[1, :]+pad, rcorners[0, :]+pad]
    # out_corners = corners
    _corners = corners.T[:, :2]
    suppressed = out_parent_inds != np.arange(len(out_parent_inds))
    # suppressed_mapping = {tuple(sprsd_pt[:2]): tuple(_corners[par_id][:2]) for sprsd_pt, par_id in \
    #                         zip(_corners[suppressed], out_parent_inds[suppressed])}
    sprsd_kpts, pillar_kpts = _corners[suppressed], _corners[out_parent_inds[suppressed]]
   
    return out, out_inds, sprsd_kpts, pillar_kpts  # out_corners, out_parent_inds

def bisect_nms_fast(in_kpts, min_radius, max_radius, max_n_kpts, rel_exceed_limit=0.2, below_limit=0.0, distance_prior=False):
    """
    Try NMS with radius in [min_rad, max_rad], and find
    the one leading to max(#kpts) < max_n_kpts through bisection
    
    FIXME: There are conditions using radius=x leads to about 1/2 max_n_kpts,
           but using radius=x-1 leads to max_n_kpts + epsilon (epsilon a small number),
           then nms with radius=x would be choosed which is "over-suprression".
    """
    assert min_radius < max_radius
    cur_max_n_kpts, cur_best_kpts = 0, None
    cur_best_sprsd_kpts, cur_best_pillar_kpts = None, None
    
    def _nms(_kpts, _r):
        nonlocal cur_max_n_kpts, cur_best_kpts, cur_best_sprsd_kpts, cur_best_pillar_kpts
        out_kpts, out_inds, sprsd_kpts, pillar_kpts = nms_fast(_kpts, _r, distance_prior=distance_prior)
        _n_kpts = out_kpts.shape[-1]
        
        if _n_kpts > cur_max_n_kpts and _n_kpts <= max_n_kpts * (1 + rel_exceed_limit):
            cur_max_n_kpts, cur_best_kpts = _n_kpts, out_kpts
            cur_best_sprsd_kpts, cur_best_pillar_kpts = sprsd_kpts, pillar_kpts
            return True
        elif _n_kpts <= max_n_kpts * (1 + rel_exceed_limit):
            return True
        else:
            out_kpts_soft,*_ = nms_fast(_kpts, _r+1) 
            _n_kpts_soft = out_kpts_soft.shape[-1]
            if _n_kpts_soft < below_limit * max_n_kpts and _n_kpts < max_n_kpts * (1 + rel_exceed_limit + below_limit):
                '''
                There are conditions using radius=x leads to about 1/2 max_n_kpts,
                but using radius=x-1 leads to max_n_kpts + epsilon (epsilon a small number),
                then nms with radius=x would be choosed which is "over-suprression".
                '''
                cur_max_n_kpts, cur_best_kpts = _n_kpts, out_kpts
                cur_best_sprsd_kpts, cur_best_pillar_kpts = sprsd_kpts, pillar_kpts
                logger.info('dynamic below limit applied!')
                return False 
            else:
                return False
            
    radius = list(range(min_radius, max_radius+1))[::-1]
    mid, low, high = 0, 0, len(radius)-1
    impossible, smaller_possible = False, False
    
    while low <= high:
        mid = (high + low) // 2 + (high + low) % 2
        if _nms(in_kpts, radius[mid]):
            if low == high == len(radius)-1:
                smaller_possible = True
            low = mid + 1
        else:
            if low == high:
                if mid == 0:
                    impossible = True
                else:
                    mid = mid - 1; break
            high = mid - 1
    
    if impossible:
        raise RuntimeError('The max_radius given is not big enough.')
    else:
        if smaller_possible:
            if cur_best_kpts.shape[-1] > max_n_kpts:
                logger.debug(f'r={radius[mid]} | n_kpts={cur_best_kpts.shape[-1]}')
            else:
                logger.warning(f'corrent number of kpts is:{cur_best_kpts.shape[-1]} | It might be possible to set min_radius smaller.')
        try:
            cur_best_kpts.shape[-1]
        except AttributeError:
            import ipdb; ipdb.set_trace()

            radius = list(range(min_radius, max_radius+1))[::-1]
            mid, low, high = 0, 0, len(radius)-1
            impossible, smaller_possible = False, False
            
            while low <= high:
                mid = (high + low) // 2 + (high + low) % 2
                if _nms(in_kpts, radius[mid]):
                    if low == high == len(radius)-1:
                        smaller_possible = True
                    low = mid + 1
                else:
                    if low == high:
                        if mid == 0:
                            impossible = True
                        else:
                            mid = mid - 1; break
                    high = mid - 1
                    import ipdb; ipdb.set_trace()
            import ipdb; ipdb.set_trace()

        return cur_best_kpts, radius[mid], cur_best_sprsd_kpts, cur_best_pillar_kpts  # (3, M)