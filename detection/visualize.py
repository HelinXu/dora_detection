import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, datasets
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
import cv2
import random


# Define the dataset and metadata
dataset_name = "dora_train"
# dataset_path = "/path/to/dora_data"
# metadata = MetadataCatalog.get(dataset_name)

# Register the dataset
datasets.register_coco_instances("dora_font", {},
                                f"/root/autodl-tmp/dora_font_v1/test/test.json",
                                f"/root/autodl-tmp/dora_font_v1/test")
datasets.register_coco_instances("dora_train", {},
                                f"/root/autodl-tmp/dora_dataset/train.json",
                                f"/root/autodl-tmp/dora_dataset/train")
datasets.register_coco_instances("dora_val", {},
                                f"/root/autodl-tmp/dora_dataset/val.json",
                                f"/root/autodl-tmp/dora_dataset/train")
# MetadataCatalog.get(dataset_name).set(thing_classes=["A", "B", "C", "D"])  # Add the object classes
metadata = MetadataCatalog.get(dataset_name)

# Load the pre-trained model and config
model_path = './output/model_0089999.pth'
config_path = './configs/font.yaml'
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create the predictor
predictor = DefaultPredictor(cfg)

# Load a random image from the dataset
dataset_dicts = DatasetCatalog.get(dataset_name)

for i in range(30):
    random_image = random.choice(dataset_dicts)
    image_path = random_image["file_name"]
    image = cv2.imread(image_path)

    # Run inference
    outputs = predictor(image)

    # Visualize the predictions
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save the visualization
    cv2.imwrite(f"imgs/output_{i}.jpg", out.get_image()[:, :, ::-1])

    # Also, visualize the ground truth
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    out = v.draw_dataset_dict(random_image)
    cv2.imwrite(f"imgs/output_gt_{i}.jpg", out.get_image()[:, :, ::-1])
