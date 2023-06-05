import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# Load the pre-trained model and config
model_path = './model0.pth'
config_path = 'config.json'
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create the predictor
predictor = DefaultPredictor(cfg)

# Load the input image
image_path = 'example.jpg'
image = cv2.imread(image_path)

# Run inference
outputs = predictor(image)

# Get the predicted instances
instances = outputs['instances']

# Get the detected objects and their scores
detected_objects = []
for i in range(len(instances)):
    detected_object = {}
    detected_object['class'] = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[instances[i].pred_classes]
    detected_object['score'] = instances[i].scores.item()
    detected_objects.append(detected_object)

# Print the detected objects
for obj in detected_objects:
    print("Class:", obj['class'], "Score:", obj['score'])