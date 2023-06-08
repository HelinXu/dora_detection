import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, datasets
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
import cv2
import random

data_root = '/root/autodl-tmp'

dataset_names = ['train_dora_sim', 'test_dora_sim', 'train_dora_real', 'test_dora_real']
# Register the dataset
datasets.register_coco_instances("train_ui", {},
                                f"{data_root}/ui_dataset/train/_annotations.coco.json",
                                f"{data_root}/ui_dataset/train")
datasets.register_coco_instances("test_ui", {},
                                f"{data_root}/ui_dataset/test/_annotations.coco.json",
                                f"{data_root}/ui_dataset/test")
datasets.register_coco_instances("train_dora_sim", {},
                                f"{data_root}/dora_sim/train/train.json",
                                f"{data_root}/dora_sim/train")
datasets.register_coco_instances("test_dora_sim", {},
                                f"{data_root}/dora_sim/test/test.json",
                                f"{data_root}/dora_sim/test")
datasets.register_coco_instances("train_dora_real", {},
                                 f"{data_root}/dora_real/train.json",
                                 f"{data_root}/dora_real")
datasets.register_coco_instances("test_dora_real", {},
                                    f"{data_root}/dora_real/val.json",
                                    f"{data_root}/dora_real")

# MetadataCatalog.get(dataset_name).set(thing_classes=["A", "B", "C", "D"])  # Add the object classes
metadata = MetadataCatalog.get(dataset_names[3])

# Load the pre-trained model and config
model_path = './output/model_0054999.pth'
config_path = './configs/sim.yaml'
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create the predictor
predictor = DefaultPredictor(cfg)

# Define the dataset and metadata
for dataset_name in dataset_names:
    # dataset_path = "/path/to/dora_data"
    # metadata = MetadataCatalog.get(dataset_name)
    # Load a random image from the dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)

    for i in range(5):
        random_image = random.choice(dataset_dicts)
        image_path = random_image["file_name"]
        image = cv2.imread(image_path)

        # Run inference
        outputs = predictor(image)

        # Visualize the predictions
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Save the visualization
        pred = out.get_image()[:, :, ::-1]

        # Also, visualize the ground truth
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.5)
        out = v.draw_dataset_dict(random_image)
        gt = out.get_image()[:, :, ::-1]

        # first add some padding to the images
        pred = cv2.copyMakeBorder(pred, 60, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        gt = cv2.copyMakeBorder(gt, 60, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # then concatenate them
        vis = cv2.hconcat([pred, gt])
        # then add labels
        vis = cv2.putText(vis, 'Prediction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        vis = cv2.putText(vis, 'Ground Truth', (pred.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imwrite(f'./output/{dataset_name}_{i}.jpg', vis)


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
