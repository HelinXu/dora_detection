# How to Train

1. start an AutoDL training job following:
    - PyTorch 1.10.0
    - Python 3.8(ubuntu20.04)
    - Cuda 11.3

2. install dependencies:
    - opencv-python
    - icecream==2.1.3
    - detectron2==0.6 (python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html)

3. download dataset:
    - (emperically the fastest way) first upload the dataset to Ali Yunpan and then go to AutoDL - Ali Yunpan - download dataset to /root/autodl-tmp/*
    - unzip the dataset to /root/autodl-tmp/dora-sim/[train or test]

4. goto detection/ and run the training script:
    - python train.py --config configs/sim_13classes.yaml --num-gpus 4
    - emperically, IMS_PER_BATCH: 16 * NUM_GPUS is the best choice. set that in the config file.

if it shows something like this, it is running:

```
[08/17 21:29:44 d2.utils.events]:  eta: 1 day, 11:14:26  iter: 679  total_loss: 1.709  loss_box_reg: 0.4565  loss_cls: 0.3607  loss_rpn_cls: 0.2201  loss_rpn_loc: 0.6911  lr: 0.00033966  max_mem: 18496M
[08/17 21:30:13 d2.utils.events]:  eta: 1 day, 11:07:35  iter: 699  total_loss: 1.731  loss_box_reg: 0.4583  loss_cls: 0.3521  loss_rpn_cls: 0.2172  loss_rpn_loc: 0.7267  lr: 0.00034965  max_mem: 18496M
```