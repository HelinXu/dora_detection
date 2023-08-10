# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, Metadata, datasets
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
import cv2
import random
import os

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        model_path = './model/model_final.pth'
        config_path = './configs/sim_11classes.yaml'
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
        cfg.MODEL.DEVICE = 'cuda'
        predictor = DefaultPredictor(cfg)
        self.model = predictor

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
