import numpy as np
from pathlib import Path
import cv2
from loguru import logger
from tqdm import tqdm

class Propogation:
    def __init__(self, n_imgs, kpts_dict, tracks, visible_tracks, visible_keypoints, stop_ratio, min_n_matches) -> None:
        self.stop_ratio = stop_ratio
        self.n_imgs = n_imgs
        self.kpts_dict = kpts_dict # id from 1
        self.tracks = tracks
        self.visible_tracks = visible_tracks
        self.visible_keypoints = visible_keypoints
        self.min_n_matches = min_n_matches

    def get_cov_idxs(self, id0, id1):
        visible_tracks0, visible_tracks1 = set(self.visible_tracks[id0]), set(self.visible_tracks[id1])
        # visible_kpts_id0, visible_kpts_id1 = self.visible_keypoints[id0], self.visible_keypoints[id1]

        cov_track_ids = visible_tracks0.intersection(visible_tracks1)

        img0_kpt_id = []
        img1_kpt_id = []
        for track_id in cov_track_ids:
            track = self.tracks[track_id]
            img0_done = False
            img1_done = False
            for img_id, kpt_id in track: # img_id: start from 1
                if img_id == id0+1:
                    img0_kpt_id.append(kpt_id)
                    img0_done = True
                elif img_id == id1+1:
                    img1_kpt_id.append(kpt_id)
                    img1_done = True

                if img0_done and img1_done:
                    break
        
        # Return: matches np.array, N*2, indexes of kpts
        if len(img1_kpt_id) == 0:
            return np.empty((0, 2))
        else:
            return np.stack([np.array(img0_kpt_id), np.array(img1_kpt_id)], -1) # N * 2

    def get_matched_coords(self, id0, id1):
        match_idxs = self.get_cov_idxs(id0, id1) # N*2
        if len(match_idxs) != 0:
            mkpts0 = self.kpts_dict[id0+1][match_idxs[:, 0]] # N*2
            mkpts1 = self.kpts_dict[id1+1][match_idxs[:, 1]] # N*2

            return np.concatenate([mkpts0, mkpts1], axis=-1) # N*4
        else:
            return np.empty((0, 4))
    
    def continue_prop(self, id, matches):
        overlap_ratio = (len(matches) / len(self.kpts_dict[id+1])) if len(self.kpts_dict[id+1]) != 0 else 0
        average_motion = np.mean(np.linalg.norm(matches[:, :2] - matches[:, 2:4], axis=-1)) # pix, too small means very small motion
        if ((overlap_ratio > self.stop_ratio) if self.stop_ratio is not None else True) and ((len(matches) > self.min_n_matches) if self.min_n_matches is not None else True):
            return True, overlap_ratio, average_motion
        else:
            return False, overlap_ratio, average_motion

def plot_matches(img_path0, img_path1, matches):
    from .utils.plotting import make_matching_figure
    import matplotlib.cm as cm

    PLOT_SAVE_DIR = Path("vis_roma_prop")
    PLOT_SAVE_DIR.mkdir(exist_ok=True)

    # if rot_best_idx == 2:
    rot_best_idx = 0

    plot_mconfs = np.zeros((matches.shape[0],))
    plot_mkpts0 = matches[:, :2]
    plot_mkpts1 = matches[:, 2:4]
    plot_kpts0 = matches[:, :2]
    plot_kpts1 = matches[:, 2:4]
    txt = f"Nm: {plot_mkpts0.shape[0]}"
    color = cm.jet(plot_mconfs, alpha=0.05)

    # plot_mconfs = sparse_mconfs
    # plot_mkpts0 = sparse_mkpts0
    # plot_mkpts1 = sparse_mkpts1
    # plot_kpts0 = sparse_kpts0
    # plot_kpts1 = sparse_kpts1
    # txt = f"Nm: {plot_mkpts0.shape[0]}, rot:{90*rot_best_idx}"
    # color = cm.jet(plot_mconfs, alpha=0.2)

    img0 = cv2.cvtColor(cv2.imread(img_path0, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(cv2.imread(img_path1, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fig = make_matching_figure(img0, img1, plot_mkpts0, plot_mkpts1, color, kpts0=plot_kpts0, kpts1=plot_kpts1, text=txt, draw_detection=True, draw_match_type="corres", path=PLOT_SAVE_DIR / f"{Path(img_path0).name}-{Path(img_path1).name}.jpg", vertical=False, plot_size_factor=1)

def propogate_matches(id2imgname, n_imgs, keypoints_dict, tracks, visible_tracks, visible_keypoints, start_interval, stop_ratio=0.03, bear_n_fail=5, min_n_matches=None, avg_motion_thr=40):
    plot=False
    propogator = Propogation(n_imgs, keypoints_dict, tracks, visible_tracks, visible_keypoints, stop_ratio, min_n_matches)
    propogation_length_for_each_img_list = []
    propogated_mathces_dict = {}
    prop_pair_name = []
    prop_pair_matches = []
    processed_set = []
    prop_n_fail = 0
    logger.info("----------Propogation Matches begin----------")
    for i in tqdm(range(n_imgs), total=n_imgs):
        aim_id = i + start_interval
        propogation_length = start_interval
        if (i, aim_id) in processed_set or (aim_id, i) in processed_set:
            continue
        while True:
            if aim_id >= n_imgs:
                break
            matches = propogator.get_matched_coords(i, aim_id)
            continue_prop, overlap_ratio, average_motion = propogator.continue_prop(i, matches)
            propogation_length += 1
            if continue_prop:
                # pixs, to avoid small motion pairs that are too easy. However, for object centric scenes, overall pixel motion may small but view-point change is large
                if average_motion > avg_motion_thr: 
                    img_path0 = id2imgname[i+1]
                    img_path1 = id2imgname[aim_id+1]
                    prop_pair_name.append(' '.join([img_path0, img_path1]))
                    prop_pair_matches.append(matches)
                    if plot and overlap_ratio < 0.3:
                        plot_matches(img_path0, img_path1, matches)
                aim_id += 1
            else:
                prop_n_fail += 1
                aim_id += 1
                if prop_n_fail >= bear_n_fail:
                    break
        
        for pair_name, matches in zip(prop_pair_name[-20:], prop_pair_matches[-20:]):
            propogated_mathces_dict[pair_name] = matches

        propogation_length_for_each_img_list.append(propogation_length)
        processed_set.append((i, aim_id))
    prop_length_list = np.array(propogation_length_for_each_img_list)
    logger.info(f"{len(propogated_mathces_dict)} Pairs generated, mean/median/max propogation length: ${np.mean(prop_length_list)}/{np.median(prop_length_list)}/{np.max(prop_length_list)}$")

    logger.info("----------Propogation Matches Finish----------")
    return propogated_mathces_dict