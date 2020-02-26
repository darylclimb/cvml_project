# SFM Self Supervised Depth Estimation
TF2 implementation of self supervised depth estimation. 

Code is largely adapted from [Monodepth2](https://github.com/nianticlabs/monodepth2)

## Dataset
You can download the entire raw KITTI dataset by running:

```wget -i splits/kitti_archives_to_download.txt -P kitti_data/```

## Dependency Installation
For pretrained resnet
```
pip install image-classifiers
```

## Training
```python train.py --data_dir path_to_dataset_root```

## Eval & Demo
```python eval.py --data_dir path_to_dataset_root```