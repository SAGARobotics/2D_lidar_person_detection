# Person Detection in 2D Range Data
This repository is a fork of the original repo [here](https://github.com/VisualComputingInstitute/2D_lidar_person_detection) and contains implementation of DROW3 [(arXiv)](https://arxiv.org/abs/1804.02463) and DR-SPAAM [(arXiv)](https://arxiv.org/abs/2004.14079), real-time person detectors using 2D LiDARs mounted at ankle or knee height.
Pre-trained models (using PyTorch 1.6) can be found in this [Google drive](https://drive.google.com/drive/folders/1Wl2nC8lJ6s9NI1xtWwmxeAUnuxDiiM4W?usp=sharing).

![](imgs/teaser_1.gif)

## Quick start

Use the `Detector` class to run inference
```python
import numpy as np
from dr_spaam.detector import Detector

ckpt = 'path_to_checkpoint'
detector = Detector(
    ckpt,
    model="DROW3",          # Or DR-SPAAM
    gpu=True,               # Use GPU
    stride=1,               # Optionally downsample scan for faster inference
    panoramic_scan=True     # Set to True if the scan covers 360 degree
)

# tell the detector field of view of the LiDAR
laser_fov_deg = 360
detector.set_laser_fov(laser_fov_deg)

# detection
num_pts = 1091
while True:
    # create a random scan
    scan = np.random.rand(num_pts)  # (N,)

    # detect person
    dets_xy, dets_cls, instance_mask = detector(scan)  # (M, 2), (M,), (N,)

    # confidence threshold
    cls_thresh = 0.5
    cls_mask = dets_cls > cls_thresh
    dets_xy = dets_xy[cls_mask]
    dets_cls = dets_cls[cls_mask]
```

## ROS node

![](imgs/dr_spaam_ros_teaser.gif)

Refer to the [people-detection](https://github.com/LCAS/people-detection) repository for details on installation.

Modify the topics and the path to the pre-trained checkpoint at 
`dr_spaam_ros/config/` and launch the node
```
roslaunch dr_spaam_ros dr_spaam_ros.launch
```

For testing, you can play a rosbag sequence from JRDB dataset.
For example,
```
rosbag play JRDB/test_dataset/rosbags/tressider-2019-04-26_0.bag
```
and use RViz to visualize the inference result.
A simple RViz config is located at `dr_spaam_ros/example.rviz`.

## Training and evaluation

For our use case, we would first need to convert the bag file into DROW format and annotate the date to be able to train the detector on the bag data.

To do this preprocessing, you need access to the [people-detection](https://github.com/LCAS/people-detection) repository.
```
python scripts/bag_to_csv.py <your_bag_file>.bag
python scripts/csv_to_drow_format.py <scan_file>.csv
```
To annotate people in the drow_format scan file, run
```
python anno1602.py <drow_scan_file>.csv -p
```
To annotate wheelchairs, run
```
python anno1602.py <drow_scan_file>.csv
```

To train a network from scratch (or evaluate a pretrained checkpoint), run
```
python dr_spaam/utils/train.py --cfg net_cfg.yaml [--ckpt ckpt_file.pth --evaluation]
```
where `net_cfg.yaml` specifies configuration for the training (see examples under `cfgs`).

To finetune a pretrained checkpoint, run
```
python dr_spaam/utils/train_ft.py --cfg net_cfg.yaml --ckpt ckpt_file.pth
```

## Inference time
On DROW dataset (450 points, 225 degrees field of view)
|        | AP<sub>0.3</sub> | AP<sub>0.5</sub> | FPS (RTX 2080 laptop) | FPS (Jetson AGX Xavier) |
|--------|------------------|------------------|-----------------------|------------------|
|DROW3   | 0.638 | 0.659 | 115.7 | 24.9 |
|DR-SPAAM| 0.707 | 0.723 | 99.6 | 22.5 |

On JackRabbot dataset (1091 points, 360 degrees field of view)
|        | AP<sub>0.3</sub> | AP<sub>0.5</sub> | FPS (RTX 2080 laptop) | FPS (Jetson AGX Xavier) |
|--------|------------------|------------------|-----------------------|------------------|
|DROW3   | 0.762 | 0.829 | 35.6 | 10.0 |
|DR-SPAAM| 0.785 | 0.849 | 29.4 | 8.8  |

Note: Evaluation on DROW and JackRabbot are done using different models (the APs are not comparable cross dataset).
Inference time was measured with PyTorch 1.7 and CUDA 10.2 on RTX 2080 laptop,
and PyTorch 1.6 and L4T 4.4 on Jetson AGX Xavier.

## Citation
If you use this repo in your project, please cite:
```BibTeX
@article{Jia2020Person2DRange,
  title        = {{Self-Supervised Person Detection in 2D Range Data using a
                   Calibrated Camera}},
  author       = {Dan Jia and Mats Steinweg and Alexander Hermans and Bastian Leibe},
  journal      = {https://arxiv.org/abs/2012.08890},
  year         = {2020}
}

@inproceedings{Jia2020DRSPAAM,
  title        = {{DR-SPAAM: A Spatial-Attention and Auto-regressive
                   Model for Person Detection in 2D Range Data}},
  author       = {Dan Jia and Alexander Hermans and Bastian Leibe},
  booktitle    = {International Conference on Intelligent Robots and Systems (IROS)},
  year         = {2020}
}
```
