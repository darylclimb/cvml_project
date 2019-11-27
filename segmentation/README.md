# Segmentation on Cityscapes
This is a simple and clean implementation of Deeplabv3+ on cityscapes dataset using tensorflow 2.0. It is meant to give a quick
run through of the steps required to implement segmentation in TF2.0

Training methodology follows most implementation of random scale cropping and horizontal flipping

## Setup
From https://www.cityscapes-dataset.com, download 
1. leftImg8bit_trainvaltest 
2. leftImg8bit_demoVideo
3. gtFine_trainvaltest

Note that you will need to register.

## Training
```python train.py --data_dir cityscape_data```

cityscape_data points to the root of the downloaded dataset.
Training parameters are defined in the file ```train.py```

## Saving inference prediction
```python eval.py --data_dir cityscape_data```

Evaluating with the current training setting could achieve a result similar to the [demo](https://youtu.be/V-A_haUIreM)
