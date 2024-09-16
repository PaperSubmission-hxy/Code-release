import os
import os.path as osp
from loguru import logger
import numpy as np
from copy import deepcopy

from torch.utils.data import Dataset
from .utils import (
    read_rgb,
)

from ..utils.geometry_utils import *


class CoarseColmapDataset(Dataset):

    def __init__(
        self,
        args,
        id2name_dict,
        keypoints2D,
        tracks,
        visible_tracks_by_imgs,
        visible_tracks_pt2d_idxs,
        verbose=True
    ):
        """
        Parameters:
        ---------------
        image_lists: ['path/to/image/0.png', 'path/to/image/1.png]
        covis_pairs: List or path
        colmap_results_dir: The directory contains images.bin(.txt) point3D.bin(.txt)...
        """
        super().__init__()

        self.img_resize = args['img_resize']
        self.df = args['df'] # 8
        self.feature_track_assignment_strategy = args['feature_track_assignment_strategy']
        self.verbose = verbose
        self.state = True
        self.preload = args['img_preload']

        self.frame_ids = list(range(len(id2name_dict)))

        self.frameId2colmapID_dict = id2name_dict
        self.colmap_images = {id: {"xys": kpts, 'point3D_ids': np.array(visible_tracks_by_imgs[id-1]), 'point2D_idxs': np.array(visible_tracks_pt2d_idxs[id-1])} for id, kpts in keypoints2D.items()}

        self.colmap_3ds = {}
        for id, track in enumerate(tracks):
            img_ids = []
            pt3d_idxs = []
            for img_id, pt_idx in track:
                img_ids.append(img_id)
                pt3d_idxs.append(pt_idx)
            self.colmap_3ds[id] = {'image_ids': np.array(img_ids), 'point2D_idxs': np.array(pt3d_idxs)} 

        # Verification:
        if (
            len(self.colmap_3ds) == 0
            or len(self.colmap_images) == 0
        ):
            self.state = False

        # Get keyframes and feature track(3D points) assignment
        logger.info("Building keyframes begin....")
        if self.feature_track_assignment_strategy == "greedy":
            self.keyframe_dict, self.point_cloud_assigned_imgID_kptID = self.get_keyframes_greedy(
                self.colmap_images, self.colmap_3ds, verbose=self.verbose
            )
        else:
            raise NotImplementedError

    def extract_corresponding_frames(self, colmap_frame_dict):
        """
        Update: {related_frameID: list}
        """
        for colmap_frameID, frame_info in colmap_frame_dict.items():
            related_frameID = []
            if not frame_info["is_keyframe"]:
                continue
            all_kpt_status = frame_info["all_kpt_status"]
            point_cloud_idxs = all_kpt_status[all_kpt_status >= 0]
            for point_cloud_idx in point_cloud_idxs:
                # Get related feature track
                image_ids = self.colmap_3ds[point_cloud_idx]['image_ids'].to_list()
                point2D_idxs = self.colmap_3ds[point_cloud_idx]['point2D_idxs']

                related_frameID.append(image_ids)

            all_related_frameID = np.concatenate(related_frameID)
            unique_frameID, counts = np.unique(all_related_frameID, return_counts=True)

            self_idx = np.squeeze(
                np.argwhere(unique_frameID == colmap_frameID)
            ).tolist()  # int
            unique_frameID = unique_frameID.tolist()
            unique_frameID.pop(self_idx)  # pop self index
            frame_info.update({"related_frameID": unique_frameID})

    def get_frameID2colmapID(self, frame_IDs, frame_names, colmap_images, only_basename_in_colmap=False):
        # frame_id equal to frame_idx
        frameID2colmapID_dict = {}
        colmapID2frameID_dict = {}
        for frame_ID in frame_IDs:
            frame_name = frame_names[frame_ID]
            frame_name = osp.basename(frame_name) if only_basename_in_colmap else frame_name

            for colmap_image in colmap_images.values():
                if frame_name == colmap_image.name:
                    # Registrated scenario
                    frameID2colmapID_dict[frame_ID] = colmap_image.id
                    colmapID2frameID_dict[colmap_image.id] = frame_ID
                    break
            if frame_ID not in frameID2colmapID_dict:
                # -1 means not registrated
                frameID2colmapID_dict[frame_ID] = -1
        return frameID2colmapID_dict, colmapID2frameID_dict

    def get_keyframes_greedy(self, colmap_images, colmap_3ds, verbose=True):
        # Get keyframes by sorting num of keypoints and tracks of a frame.
        # Get each 3D point's correspondence image index and keypoint index

        # Build keypoints state and colmap state. -3 means robbed, -2 means unoccupied, -1 unregisted by colmap, -2 means robbed, >=0 means index of the 3D point(feature track)
        colmap_images_state = (
            {}
        )  # {colmap_imageID:{state: np.array [N], unoccupied_num: int n}}
        for id, colmap_image in colmap_images.items():
            colmap_images_state[id] = {}
            colmap_images_state[id]["state"] = -2 * np.ones(
                (colmap_image['xys'].shape[0],)
            )  # [N], initial as all -2
            colmap_unregisted_mask = colmap_image['point3D_ids'] == -1
            colmap_unregisted_idxs = colmap_image['point2D_idxs'][colmap_unregisted_mask]
            colmap_images_state[id]["state"][
                colmap_unregisted_idxs
            ] = -1  # set unregistred keypoints to -1
            colmap_images_state[id]["point2D_idxs"] = colmap_image['point2D_idxs']
            colmap_images_state[id]["unoccupied_num"] = (
                (colmap_images_state[id]["state"][colmap_image['point2D_idxs']] == -2)
            ).sum()
        colmap_3d_states = {}
        for point_cloudID, point_cloud in colmap_3ds.items():
            colmap_3d_states[point_cloudID] = (
                -1,
            )  # (-1,): unoccupied, (imageid, pointidx): occupied

        # Iterate to find keyframes!
        keyframe_dict = {colmap_img_id: np.empty((0,)) for colmap_img_id in self.colmap_images.keys()}
        while not self._is_colmap_3d_empty(colmap_3d_states):
            assert len(colmap_images_state) != 0
            # Sort colmap images state:
            colmap_images_state = self._sort_colmap_images_state(colmap_images_state)

            # Set current frame with most keypoints to keyframe:
            current_keyframeID = list(colmap_images_state.keys())[0]
            current_selected_keyframe_state = colmap_images_state.pop(
                current_keyframeID
            )  # pop the first element of state dict
            # update current keyframe state
            occupied_keypoints_mask = current_selected_keyframe_state["state"][current_selected_keyframe_state["point2D_idxs"]] == -2
            current_selected_keyframe_state["state"][current_selected_keyframe_state["point2D_idxs"][
                occupied_keypoints_mask
            ]] = colmap_images[current_keyframeID]['point3D_ids'][occupied_keypoints_mask]
            keyframe_dict[current_keyframeID] = (current_selected_keyframe_state['state'][current_selected_keyframe_state['state'] >= 0]).astype(np.int32)

            # Update colmap_3d_state
            occupied_3d_ids = colmap_images[current_keyframeID]['point3D_ids'][
                occupied_keypoints_mask
            ]  # N'
            occupied_kpt_idx = colmap_images[current_keyframeID]['point2D_idxs'][
                occupied_keypoints_mask
            ]
            for i, occupied_3d_id in enumerate(occupied_3d_ids):
                colmap_3d_states[occupied_3d_id] = (
                    current_keyframeID,
                    occupied_kpt_idx[i],
                )
                # Get feature track of this 3D point
                img_ids = colmap_3ds[occupied_3d_id]['image_ids']
                point2d_idxs = colmap_3ds[occupied_3d_id]['point2D_idxs']
                related_track = zip(img_ids, point2d_idxs)  # [[img_id, point2d_idx]]

                # Update other points' state in a track as robbed: -3
                for node in related_track:
                    img_id, point2d_idx = node
                    if img_id == current_keyframeID:
                        continue
                    original_point_state = colmap_images_state[img_id]["state"][
                        point2d_idx
                    ]
                    assert (
                        original_point_state != -1
                    ), "The state of the point in the track shouldn't be -1, bug exists!"
                    # update state
                    colmap_images_state[img_id]["state"][point2d_idx] = -3

            colmap_images_state = self._update_colmap_images_state_unoccupied_number(
                colmap_images_state
            )
        return keyframe_dict, colmap_3d_states

    def _is_colmap_3d_empty(self, colmap_3d_state):
        num_non_empty = 0
        for state in colmap_3d_state.values():
            if len(state) == 1:
                num_non_empty += 1

        return num_non_empty == 0

    def _sort_colmap_images_state(self, colmap_images_state):
        # Sort colmap images state by "unoccupied_num"
        colmap_images_state_sorted = {
            k: v
            for k, v in sorted(
                colmap_images_state.items(),
                key=lambda item: item[1]["unoccupied_num"],
                reverse=True,
            )
        }
        return colmap_images_state_sorted

    def _update_colmap_images_state_unoccupied_number(self, colmap_images_state):
        # Update colmap image state's occupied
        for key in colmap_images_state.keys():
            colmap_images_state[key]["unoccupied_num"] = (
                colmap_images_state[key]["state"][colmap_images_state[key]["point2D_idxs"]] == -2
            ).sum()
        return colmap_images_state
    
    def get_refined_kpts_to_colmap_multiview(self, fine_match_results):
        colmap_images = deepcopy(self.colmap_images)
        for bag_results in fine_match_results:
            for refined_pts in bag_results:
                location, image_id, pt2d_idx = refined_pts[:2], int(refined_pts[2]), int(refined_pts[3])

                colmap_images[image_id]['xys'][pt2d_idx, :] = location
        return colmap_images

    def __len__(self):
        return len(self.frameId2colmapID_dict)

    def __getitem__(self, idx):
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        img_name = self.frameId2colmapID_dict[idx]
        img_scale = read_rgb(
            img_name,
            mask=None,
            resize=(self.img_resize,) if self.img_resize is not None else None,
            resize_no_larger_than=True,
            pad_to=None,
            df=self.df,
            ret_scales=True,
        )

        img, scale, original_hw = map(lambda x: x, img_scale)  # with dataloader operation
        data = {
            "image": img,  # 1*H*W because no dataloader operation, if batch: 1*H*W
            "scale": scale,  # 2
            "f_name": img_name,
            "img_name": img_name,
            "frameID": idx,
            "img_path": [img_name],
        }
        return data

