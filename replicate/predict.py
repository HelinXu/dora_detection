# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import cv2
import torch
import random
import os
from copy import copy, deepcopy
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, Metadata, datasets
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
from process import grid_canvas


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        model_path = './model/model_final.pth'
        config_path = './configs/sim_13classes.yaml'
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
        cfg.MODEL.DEVICE = 'cpu'
        predictor = DefaultPredictor(cfg)
        self.model = predictor
        self.metadata = Metadata().set(thing_classes=["Cont.", "Ttl.", "Img.", "Icon", "Para.", "Bg.", "IrImg.", "BgImg.", "CtPil.", "CtCir.", "ImgCir.", "Sec.", "Sec.B"])  # Add the object classes


    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)

        img = cv2.imread(image)

        grid_canvas_ = grid_canvas(img, ratio=1)
        imagename = 'output.jpg'
        vis = grid_canvas_.draw_grid_prediction(self.model, self.metadata, name=imagename)
        cv2.imwrite(f'./img/{imagename}', vis)
        return vis


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    predictor.predict(image="./img/test.webp", scale=1.5)