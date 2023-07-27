import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, Metadata, datasets
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
import cv2
import random
import os
from layout import grid_canvas, grid_layout, Square

from icecream import ic, install
install()
ic.configureOutput(includeContext=True, contextAbsPath=True)

datasetpath = '/root/autodl-tmp/real'
datasetpath = '/root/autodl-tmp/DoraAIGC'
# datasetpath = '/root/autodl-tmp/dora_sim/test'
# Load the pre-trained model and config
model_path = './output/model_0039999.pth'
config_path = './configs/sim_11classes.yaml'
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
cfg.MODEL.DEVICE = 'cuda'
ic(cfg)



# Create the predictor
predictor = DefaultPredictor(cfg)

image_paths = os.listdir(datasetpath)

metadata = Metadata().set(thing_classes=["Cont.", "Ttl.", "Img.", "Icon", "Para.", "Bg.", "IrImg.", "BgImg.", "CtPil.", "CtCir.", "ImgCir."])  # Add the object classes

if not os.path.exists('./output/imgs'):
    os.mkdir('./output/imgs')

for image_path in image_paths:
    # dataset_path = "/path/to/dora_data"
    # metadata = MetadataCatalog.get(dataset_name)
    # Load a random image from the dataset

    image = cv2.imread(os.path.join(datasetpath, image_path))

    grid_canvas_ = grid_canvas(image, ratio=1)
    ic(grid_canvas_.canvas_size)
    imagename = image_path.split('/')[-1]
    vis = grid_canvas_.draw_grid_prediction(predictor, metadata, name=imagename)



    
    # cv2.imwrite(f'./output/imgs/pred_{imagename}.jpg', vis)


# # finally, concatenate all the images by dataset name
# import numpy as np
# import cv2
# for dataset_name in dataset_names:
#     images = []
#     for i in range(3):
#         img = cv2.imread(f'./output/{dataset_name}_{i}.jpg')
#         # resize the img to be 1000xN
#         img = cv2.resize(img, (1000, int(img.shape[0] * 1000 / img.shape[1])))
#         images.append(img)
#     vis = cv2.vconcat(images)
#     cv2.imwrite(f'./output/{dataset_name}.jpg', vis)


# # remove the individual images
# import os
# for dataset_name in dataset_names:
#     for i in range(3):
#         os.remove(f'./output/{dataset_name}_{i}.jpg')
