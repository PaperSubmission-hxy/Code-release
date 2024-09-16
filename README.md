# Installation:
Create the python environment by:
```
conda env create -f environment.yaml
conda activate env
```
We have tested our code on the device with CUDA 11.7.

Download pretrained weights from [here](https://drive.google.com/file/d/1ZKpc8qeJdxIjEVOO87rE8EmZgieOBQkx/view?usp=sharing) and place it under repo directory. Then unzip it by running the following command:
```
unzip weights.zip
rm -rf weights.zip
```

# Train:
## Data Preparing
There are three parts of data used in our framework, including multi-view datasets with geometry, video sequences, singe image datasets.
In general, these datasets are stored by original images and corresponding scene indices, including image pairs, and other metadata.
All datasets should be stored in `data/train_data`.
The total storage space required for all datasets and generated images is approximately 25TB.

### Multi-View Datasets with Geometry
For this type of dataset, we use MegaDepth, ScanNet++, and BlendedMVS datasets.
#### MegaDepth
We use depth maps provided in the [original MegaDepth dataset](https://www.cs.cornell.edu/projects/megadepth/) as well as undistorted images, corresponding camera intrinsics and extrinsics preprocessed by [D2-Net](https://github.com/mihaidusmanu/d2-net#downloading-and-preprocessing-the-megadepth-dataset). You can download them separately from the following links. 
- [MegaDepth undistorted images and processed depths](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz)

    - Note that we only use depth maps.

- [D2-Net preprocessed images](https://drive.google.com/drive/folders/1hxpOsqOZefdrba_BqnW490XpNX_LgXPB)

    - Images are undistorted manually in D2-Net since the undistorted images from MegaDepth do not come with corresponding intrinsics.

    - You can preprocess the images by the [official code](https://github.com/mihaidusmanu/d2-net/blob/master/megadepth_utils/preprocess_undistorted_megadepth.sh) in D2-Net.

Then, download the scene indexs from [here](https://drive.google.com/file/d/1YMAAqCQLmwMLqAkuRIJLDZ4dlsQiOiNA/view?usp=drive_link), and then unzip the file:
```shell
tar xf megadepth_indices.tar
```

Finally, the data structure should be like:
```
data/train_data/megadepth
    - Undistorted_SfM
        - 0000
        - 0001
        - ....
    - phoenix/S6/zl548/MegaDepth_v1
        - 0000
        - 0001
        - ....
    - indexs
        - megadepth_indices
            - scene_info_0.1_0.7
            - trainvaltest_list
```
For MegaDepth, we use visible-visible pairs and visible-depth map pairs for training, where the depth maps are directly from dataset.

#### ScanNet++

First, download the ScanNet++ data from its [official webset](https://kaldir.vc.in.tum.de/scannetpp/), where only iphone data is used in our framework. Extract images from video, render depth map from the reconstruction, and undistort images using its [tool box](https://github.com/scannetpp/scannetpp).

Then, the scene indexs construct by depth warping can be downloaded from [here](https://drive.google.com/file/d/1L0RX3C_PXiK4xnTrLckGjYOFHmuoerDC/view?usp=sharing).
The ScanNet++ dataset should be constructed like:
```
data/train_data/scannet_plus
    - data
        - 02455b3d20
            - iphone
            - scans
        - 02a980c994
    - matching_indices_0.1_0.7_0.0
        - scene_info
        - train_list.txt
        - val_list.txt
```
For ScanNet++, we use visible-visible pairs and visible-depth map pairs for training, where the depth maps are directly from dataset.

#### BlendedMVS
Download the dataset from its [official repo](https://github.com/YoYo000/BlendedMVS), where the dataset is composed of `BlendedMVS`, `BlendedMVS+`, `BlendedMVS++`.
Place the all the scenes of these subset into a same directory named `source_dataset`.
The sampled scene indices by us can be downloaded from [here](https://drive.google.com/file/d/1-LqQzfsNIKF4TtRGnUVLIePWitVIhHd-/view?usp=sharing).
The data structure should be like:

```
data/train_data/BlendedMVS_matching
    - source_dataset
        - 57f8d9bbe73f6760f10e916a
            - blended_images
            - cams
            - rendered_depth_maps
    - matching_indices_0.1_0.7_0.0
        - scene_info
        - train_list.txt
```

For BlendedMVS, we generate cross-modality pairs including visible-syn. thermal and visible-syn. night-time pairs by:
```shell
cd $REPO_DIR
# Generate images in different modalities:
sh scripts/prepare_training_data/process_blendedmvs_modality_transform.sh

# Substitute images in original pairs:
sh scripts/prepare_training_data/process_substitute_commondata_gt_blendedmvs.sh
```

#### Video Sequences
We use DL3DV dataset as video sequences dataset with contains large-scale high-quality videos.
Please download all videos from the [official repo](https://dl3dv-10k.github.io/DL3DV-10K/). Notably, we use the [960P version](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P) to conduct experiments for conserve storage space. 
Then, place all scenes under the same folder named `scene_images`:
```
data/train_data/DL3DV-10K
    - scene_images
        - 102caab29268ec929921158cbc425884754221c124c7d54392f6d4a1155b3edb
            - images_4
                - frame_00000.png
        - 0ff17815ce91c6418645f9d07c23b2abeae54913f1a2d19ddd1063fed01f0c72
        - ....
```

Subsequentially, we create cross-modality training signal, including synthetic depth, thermal and night-time images:
```shell
sh scripts/prepare_training_data/process_DL3DV_modality_transform.sh
```

Then, training pairs with ground-truth matches are generated by the proposed coarse-to-fine framework:
```shell
sh scripts/prepare_training_data/process_DL3DV_construct_ground_truth.sh
```

### Single Image Datasets

#### GoogleLandmark
First, download [GoogleLandmark](https://github.com/cvdfoundation/google-landmark) from the offical repo.
Then, unzip the data into a same folder, named `train`.
The training image list can be downloaded from [here](https://drive.google.com/file/d/1ejxTkYI18I3DZ2etvg2a_IEfDi9tmOZm/view?usp=sharing).
The data structure should be like:
```
data/train_data/google-landmark
    - train
        - 0
            - a
            - b
            - c
            - ...
        - 1
        - 2
        - ...
    - train_gldv2.txt
```
The cross-modality training pairs are generated by:
```shell
sh scripts/prepare_training_data/process_google_landmark_modality_transform.sh
```

#### SA-1B
Download [SA-1B](https://ai.meta.com/datasets/segment-anything/) dataset from the official repo, unzip each image bag and place all images into the same directory named `images`

The cross-modality training pairs are generated by:
```shell
sh scripts/prepare_training_data/process_SA_1B_modality_transform.sh
```
## Model Training
Then, models are trained by our multi-resources, multi-modality mixture training framework:
```shell
# Train ELoFTR model by our framework:
sh scripts/train/train_eloftr.sh

# Train ROMA model by our framework:
sh scripts/train/train_roma.sh
```
The default training scripts are run on two computation nodes, each node has 8 A100-80GB GPUs.
We set a `batchsize=4` for each GPU, where the total `batchsize=64`.
Please modify the computation setting according your device configuration.

# Test:
We evaluate the models pretrained by our framework using a single network weight on all cross-modality matching and registration tasks.

## Data Preparing
Download the `test_data` directory from [here](https://drive.google.com/drive/folders/1jpxIOcgnQfl9IEPPifdXQ7S7xuj9K4j7?usp=sharing) and plase it under `repo_directory/data`. Then, unzip all datasets by:
```shell
cd repo_directiry/data/test_data

for file in *.zip; do
    unzip "$file" -d "${file%.*}" && rm "$file"
done
```

The data structure should looks like:
```
repo_directiry/data/test_data
    - Liver_CT-MR
    - havard_medical_matching
    - remote_sense_thermal
    - MTV_cross_modal_data
    - thermal_visible_ground
    - visible_sar_dataset
    - visible_vectorized_map
```

## Evaluation
```shell
# For Tomography datasets:
sh scripts/evaluate/eval_liver_ct_mr.py
sh scripts/evaluate/eval_harvard_brain.py

# For visible-thermal datasets:
sh scripts/evaluate/eval_thermal_remote_sense.py
sh scripts/evaluate/eval_thermal_mtv.py
sh scripts/evaluate/eval_thermal_ground.py

# For visible-sar dataset:
sh scripts/evaluate/eval_visible_sar.py

# For visible-vectorized map dataset:
sh scripts/evaluate/eval_visible_vectorized_map.py
```