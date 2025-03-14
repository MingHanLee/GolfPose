# GolfPose: From Regular Posture to Golf Swing Posture

<p align="center"> <img src="./images/framework_v13.svg" width="80%"> </p> 

The official implementation of **GolfPose: From Regular Posture to Golf Swing Posture** (ICPR 2024) 

[📜Paper](https://minghanlee.github.io/papers/ICPR_2024_GolfPose.pdf)

## Environment

Create a conda environment named ``golfpose`` from `environment.yml`.
```
conda env create -f environment.yml
conda activate golfpose
```

## Preparation
### Dataset

Download the dataset from this [link](http://gofile.me/4RvCV/LoqF43SeV).<br>
Please email [mhlee.cs09@nycu.edu.tw](mailto:mhlee.cs09@nycu.edu.tw) to authorize the dataset download.

Unzip the dataset and organize it as follows:
```
GolfPose
|-- golfswing
|   |-- coco
|   |-- data_2d_golf_gt.npz
|   |-- data_3d_golf_gt.npz
|   `-- images
```

### Models
Download the checkpoints and and organize it as follows:
```
GolfPose
|-- golfpose_checkpoints
|   |-- golfpose_17+0_35.6.bin
|   |-- golfpose_17+1_30.7_29.5_50.9.bin
|   |-- golfpose_17+2_33.6_30.5_59.5.bin
|   |-- golfpose_17+3_35.6_30.8_62.9.bin
|   |-- golfpose_17+4_37.9_32.3_61.4.bin
|   |-- golfpose_17+5_39.2_32.3_62.8.bin
|   |-- golfpose_club_ViTPose_huge.pth
|   |-- golfpose_club_dekr.pth
|   |-- golfpose_club_hrnetw48.pth
|   |-- golfpose_detector_1cls_faster_rcnn.pth
|   |-- golfpose_detector_1cls_yolox_s.pth
|   |-- golfpose_detector_2cls_faster_rcnn.pth
|   |-- golfpose_detector_2cls_yolox_s.pth
|   |-- golfpose_golfer_ViTPose_huge.pth
|   |-- golfpose_golfer_dekr.pth
|   |-- golfpose_golfer_hrnetw48.pth
|   |-- golfpose_person_ViTPose_huge.pth
|   |-- golfpose_person_dekr.pth
|   `-- golfpose_person_hrnetw48.pth
```
#### Golfpose detectors

| Model | Class | AP | Ckpt | Config |
| - | - | - | - | - |
| Faster R-CNN | 2(golfer, club) | 0.884 | [ckpt](http://gofile.me/4RvCV/0rpv3tZBr) | [config](configs/mmdet/golfpose_detector_2cls.py) |
| Faster R-CNN | 1(golfer-with-club) | 0.918 | [ckpt](http://gofile.me/4RvCV/qQMFqE8Pp) | [config](configs/mmdet/golfpose_detector_1cls.py) |
| YOLOX-s | 2(golfer, club) | 0.916 | [ckpt](http://gofile.me/4RvCV/ALgsmvtPw) | [config](configs/mmdet/golfpose_detector_2cls_yolox_s.py) |
| YOLOX-s | 1(golfer-with-club) | 0.984 | [ckpt](http://gofile.me/4RvCV/HwXebcmpj) | [config](configs/mmdet/golfpose_detector_1cls_yolox_s.py) |

#### GolfPose-2D models

| Model | Source model | AP | Ckpt | Config |
| - | - | - | - | - |
| GolfPose-2D(G) | HRNet-w48 | 0.884 | [ckpt](http://gofile.me/4RvCV/ZeuNina2L) | [config](configs/mmpose/golfpose_person_hrnetw48.py) |
| GolfPose-2D(G) | ViTPose-H | 0.887 | [ckpt](http://gofile.me/4RvCV/HcaHe3i4O) | [config](configs/mmpose/golfpose_person_ViTPose_huge.py) |
| GolfPose-2D(G) | DEKR | 0.869 | [ckpt](http://gofile.me/4RvCV/gXKtqMnd8) | [config](configs/mmpose/golfpose_person_dekr.py) |
| GolfPose-2D(C) | HRNet-w48 | 0.857 | [ckpt](http://gofile.me/4RvCV/brjKDEonU) | [config](configs/mmpose/golfpose_club_hrnetw48.py) |
| GolfPose-2D(C) | ViTPose-H | 0.870 | [ckpt](http://gofile.me/4RvCV/JrI9K96AI) | [config](configs/mmpose/golfpose_club_ViTPose_huge.py) |
| GolfPose-2D(C) | DEKR | 0.858 | [ckpt](http://gofile.me/4RvCV/6HZbJLvIU) | [config](configs/mmpose/golfpose_club_dekr.py) |
| GolfPose-2D(GC) | HRNet-w48 | 0.915 | [ckpt](http://gofile.me/4RvCV/YmRPLzEMc) | [config](configs/mmpose/golfpose_golfer_hrnetw48.py) |
| GolfPose-2D(GC) | ViTPose-H | 0.925 | [ckpt](http://gofile.me/4RvCV/S7Th5DCFc) | [config](configs/mmpose/golfpose_golfer_ViTPose_huge.py) |
| GolfPose-2D(GC) | DEKR | 0.942 | [ckpt](http://gofile.me/4RvCV/gXKtqMnd8) | [config](configs/mmpose/golfpose_golfer_dekr.py) |

#### GolfPose-3D models

| Model | # of Keypoints | MPJPE(mm) | Ckpt |
| - | - | - | - |
| GolfPose-3D(GC) (N=17) | 17+0 | 35.6 | [ckpt](http://gofile.me/4RvCV/0fDtmLac5) |
| GolfPose-3D(GC) (N=18) | 17+1 | 30.7 | [ckpt](http://gofile.me/4RvCV/Jp3LgXPt2) |
| GolfPose-3D(GC) (N=19) | 17+2 | 33.6 | [ckpt](http://gofile.me/4RvCV/7WCM0f7hP) |
| GolfPose-3D(GC) (N=20) | 17+3 | 35.6 | [ckpt](http://gofile.me/4RvCV/dze4upuhT) |
| GolfPose-3D(GC) (N=21) | 17+4 | 37.9 | [ckpt](http://gofile.me/4RvCV/2euaAHSFd) |
| GolfPose-3D(GC) (N=22) | 17+5 | 39.2 | [ckpt](http://gofile.me/4RvCV/Xp00uwEBO) |


## Evaluation
#### Evaluate Golfpose detector:

```shell
python mmdet_test.py configs/mmdet/***.py golfpose_checkpoints/***.pth
```
Example:
```shell
python mmdet_test.py configs/mmdet/golfpose_detector_2cls.py golfpose_checkpoints/golfpose_detector_2cls_faster_rcnn.pth
```

#### Evaluate GolfPose-2D models:

```shell
python mmpose_test.py configs/mmpose/***.py golfpose_checkpoints/***.pth
```
Example:
```shell
python mmpose_test.py configs/mmpose/golfpose_golfer_hrnetw48.py golfpose_checkpoints/golfpose_golfer_hrnetw48.pth
```

#### Evaluate GolfPose-3D models:

```shell
python golfpose_3d.py -k gt -d golf -str G1,G2,G3,G4 -ste G5,G6 -c golfpose_checkpoints --evaluate ***.bin -f 243 -s 243 -gpu 0 -club ***
```
Example:
```shell
python golfpose_3d.py -k gt -d golf -str G1,G2,G3,G4 -ste G5,G6 -c golfpose_checkpoints --evaluate golfpose_17+5_39.2_32.3_62.8.bin -f 243 -s 243 -gpu 0 -club 5
```

## Acknowledgement

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [MixSTE](https://github.com/JinluZhang1126/MixSTE)

## Citation
```
@inproceedings{lee2025golfpose,
  title={GolfPose: From Regular Posture to Golf Swing Posture},
  author={Lee, Ming-Han and Zhang, Yu-Chen and Wu, Kun-Ru and Tseng, Yu-Chee},
  booktitle={International Conference on Pattern Recognition},
  pages={387--402},
  year={2025},
  organization={Springer}
}
```