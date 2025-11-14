# SFDepth
This project focuses on self-supervised depth estimation.
> **SFP-Depth:Self-supervised monocular depth estimation based on semantic fusion and planar constraints**
> Wenhao Li, Chunyu Peng, Zhensong Li, Shoubiao Tan*, Ting Wang, Xiao Wei
> https://doi.org/10.5281/zenodo.17606484

![Local Image](https://github.com/hwlf/SFDepth/blob/main/img/1.png)


## Setup
```shell
pip install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install tensorboardX==2.6.2.2 opencv-python matplotlib
pip install mmcv==2.0.0rc4 mmsegmentation==1.2.2
pip install timm einops IPython
```

## Inference single image
```shell
python test_simple.py --image_path ./  --model_name ./
```

## KITTI training data
You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
The raw KITTI dataset is about 175G.

## Training 
```shell
python train.py --model_name mono_model --learning_rate 5e-5
```

## KITTI evaluation
To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
```
The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ 
```
