from configs.data.base import cfg

TRAIN_BASE_PATH = TEST_BASE_PATH = "data/train_data"
cfg.DATASET.TRAIN_DATA_SOURCE = ["MegaDepth", "Megadepth_cross_modal", "MegaDepth", "Megadepth_cross_modal", "MegaDepth", "Common_dataset_homo_warp", "Common_dataset_homo_warp", "Common_dataset_homo_warp", "Common_dataset_homo_warp"]
cfg.DATASET.TRAIN_DATA_ROOT = TRAIN_BASE_PATH
cfg.DATASET.TRAIN_NPZ_ROOT = [f"{TRAIN_BASE_PATH}/scannet_plus/matching_indices_0.1_0.7_0.0/scene_info", f"{TRAIN_BASE_PATH}/scannet_plus/matching_indices_0.1_0.7_0.0/scene_info", f"{TRAIN_BASE_PATH}/megadepth/indexs/megadepth_indices/scene_info_0.1_0.7", f"{TRAIN_BASE_PATH}/megadepth/indexs/megadepth_indices/scene_info_0.1_0.7", f"{TRAIN_BASE_PATH}/BlendedMVS_matching/matching_indices_0.1_0.7_0.0_origin_rgb-0.5_style_rgb2if-0.25_style_day2night-0.25/scene_info", f"{TRAIN_BASE_PATH}/SA_1B/scene_info_depthanything_500/scene_info", f"{TRAIN_BASE_PATH}/google-landmark/scene_info_style_rgb2if_200/scene_info", f"{TRAIN_BASE_PATH}/SA_1B/scene_info_style_day2night_200/scene_info", f"{TRAIN_BASE_PATH}/DL3DV-10K/scene_info_matches_ROMA_sample_4_match_10_prop_matches_stop_ratio_0.03_min_400_matches_merge_r_3_bear_5_fail_avgmotion_30_mtvrefine_origin_rgb-0.3_depthanything-0.2_style_rgb2if-0.3_style_day2night-0.2_in_h5/scene_info"]
cfg.DATASET.TRAIN_LIST_PATH = [f"{TRAIN_BASE_PATH}/scannet_plus/matching_indices_0.1_0.7_0.0/train_list.txt", f"{TRAIN_BASE_PATH}/scannet_plus/matching_indices_0.1_0.7_0.0/train_list.txt", f"{TRAIN_BASE_PATH}/megadepth/indexs/megadepth_indices/trainvaltest_list/train_list.txt", f"{TRAIN_BASE_PATH}/megadepth/indexs/megadepth_indices/trainvaltest_list/train_list.txt", f"{TRAIN_BASE_PATH}/BlendedMVS_matching/matching_indices_0.1_0.7_0.0_origin_rgb-0.5_style_rgb2if-0.25_style_day2night-0.25/train_list.txt", f"{TRAIN_BASE_PATH}/SA_1B/scene_info_depthanything_500/train_list.txt", f"{TRAIN_BASE_PATH}/google-landmark/scene_info_style_rgb2if_200/train_list.txt", f"{TRAIN_BASE_PATH}/SA_1B/scene_info_style_day2night_200/train_list.txt", f"{TRAIN_BASE_PATH}/DL3DV-10K/scene_info_matches_ROMA_sample_4_match_10_prop_matches_stop_ratio_0.03_min_400_matches_merge_r_3_bear_5_fail_avgmotion_30_mtvrefine_origin_rgb-0.3_depthanything-0.2_style_rgb2if-0.3_style_day2night-0.2_in_h5/train_list.txt"]
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = -5.0 
cfg.DATASET.TRAIN_DATA_SAMPLE_RATIO = [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.25]

cfg.DATASET.TEST_DATA_SOURCE = cfg.DATASET.VAL_DATA_SOURCE = ["MegaDepth"]
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = TEST_BASE_PATH
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = [f"{TEST_BASE_PATH}/indexs/megadepth_indices/scene_info_0.1_0.7"]
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = [f"{TEST_BASE_PATH}/indexs/megadepth_indices/trainvaltest_list/val_list.txt"]
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = -1.0

cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100

cfg.DATASET.READ_GRAY = True
cfg.DATASET.HOMO_WARP_USE_MASK = True

cfg.DATASET.MGDPT_IMG_RESIZE = (832, 832)