# dora_detection

# Detectron2 for Dora Design

Author: Helin Xu xuhelin1911@gmail.com

## Requirements

- Python >= 3.8
- PyTorch 1.10
- TorchVision 1.11.0
- CUDA 11.3 or 11.1
- Detectron2 0.6 (Install by `python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html`)
- icecream (Install by `pip install icecream==2.1.3`)


## Personal Notes


put ui dataset at /root/autodl-tmp/ui_dataset

train with:

python train.py --config configs/faster_rcnn_R_50_FPN_1x.yaml --num-gpus 4

refer: https://github.com/HelinXu/my_detectron2/blob/main/maskrcnn/configs/InstanceSegmentation/mask_rcnn_R_50_FPN_1x_2.yaml
