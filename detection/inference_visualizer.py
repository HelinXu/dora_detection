import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, Metadata, datasets
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
import cv2
import random
import os

from icecream import ic, install
install()
ic.configureOutput(includeContext=True, contextAbsPath=True)

datasetpath = '/root/autodl-tmp/DoraAIGC'
# Load the pre-trained model and config
model_path = './output/model_final.pth'
config_path = './configs/sim_5classes.yaml'
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
cfg.MODEL.DEVICE = 'cpu'
ic(cfg)

# Create the predictor
predictor = DefaultPredictor(cfg)

image_paths = os.listdir(datasetpath)

metadata = Metadata().set(thing_classes=["A", "B", "C", "D", "E"])  # Add the object classes

if not os.path.exists('./output/imgs'):
    os.mkdir('./output/imgs')

for image_path in image_paths:
    # dataset_path = "/path/to/dora_data"
    # metadata = MetadataCatalog.get(dataset_name)
    # Load a random image from the dataset

    image = cv2.imread(os.path.join(datasetpath, image_path))
    imagename = image_path.split('/')[-1]

    # Run inference
    outputs = predictor(image)

    # Visualize the predictions
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save the visualization
    pred = out.get_image()[:, :, ::-1]

    # first add some padding to the images
    pred = cv2.copyMakeBorder(pred, 60, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    image = cv2.copyMakeBorder(image, 60, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # then concatenate them
    vis = cv2.hconcat([pred, image])
    # then add labels
    vis = cv2.putText(vis, 'Prediction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # vis = cv2.putText(vis, 'Ground Truth', (pred.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # vis = cv2.putText(vis, image_path.split('/')[-2] + image_path.split('/')[-1], (pred.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(f'./output/imgs/pred_{imagename}.jpg', vis)


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
