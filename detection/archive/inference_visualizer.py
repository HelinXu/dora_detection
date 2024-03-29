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

datasetpath = '/root/autodl-tmp/real'
datasetpath = '/root/autodl-tmp/DoraAIGC'
# datasetpath = '/root/autodl-tmp/dora_sim/test'
# Load the pre-trained model and config
model_path = './output/model_0039999.pth'
config_path = './configs/sim_11classes.yaml'
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Adjust the threshold as needed
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

    # save the predictions
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    labels = [metadata.thing_classes[i] for i in classes]
    scores = instances.scores.numpy()
    with open(f'./output/imgs/pred_{imagename}.txt', 'w') as f:
        for lab, box, cls, score in zip(labels, boxes, classes, scores):
            f.write(f'{lab} {cls} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')
    
    # copy the original image
    os.system(f'cp "{os.path.join(datasetpath, image_path)}" "./output/imgs/{imagename}"')

