# dora_detection

# Setup

- torch 1.10
- cuda 11.3

```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

put ui dataset at /root/autodl-tmp/ui_dataset

train with:

python train.py --config configs/faster_rcnn_R_50_FPN_1x.yaml --num-gpus 4

refer: https://github.com/HelinXu/my_detectron2/blob/main/maskrcnn/configs/InstanceSegmentation/mask_rcnn_R_50_FPN_1x_2.yaml