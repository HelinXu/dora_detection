# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List
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
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed.
        cfg.MODEL.DEVICE = 'cuda' # Note: change to 'cpu' if you don't have a GPU
        predictor = DefaultPredictor(cfg)
        self.model = predictor
        self.metadata = Metadata().set(thing_classes=["Cont.", "Ttl.", "Img.", "Icon", "Para.", "Bg.", "IrImg.", "BgImg.", "CtPil.", "CtCir.", "ImgCir.", "Sec.", "Sec.B"])  # Add the object classes


    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        image_path = str(image)
        img = cv2.imread(image_path)

        print('image read successful.')

        grid_canvas_ = grid_canvas(img, ratio=1)
        # Note: ratio=1 is the default value, but Helin also find that 0.75 works very well.

        print('canvas built')

        imagename = 'dora'
        grid_canvas_.draw_grid_prediction(self.model, self.metadata, name=imagename)
        
        print('prediction done')

        return [Path(f'/tmp/{imagename}_ai.jpg'), Path(f'/tmp/{imagename}_ai.txt'),
                Path(f'/tmp/{imagename}_small_obj_boost.jpg'), Path(f'/tmp/{imagename}_small_obj_boost.txt'),
                Path(f'/tmp/{imagename}_refine.jpg'), Path(f'/tmp/{imagename}_refine.txt')]


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    predictor.predict(image="./img/test.webp", scale=1.5)