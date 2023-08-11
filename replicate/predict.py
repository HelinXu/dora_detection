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
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
        cfg.MODEL.DEVICE = 'cuda'
        predictor = DefaultPredictor(cfg)
        self.model = predictor
        self.metadata = Metadata().set(thing_classes=["Cont.", "Ttl.", "Img.", "Icon", "Para.", "Bg.", "IrImg.", "BgImg.", "CtPil.", "CtCir.", "ImgCir.", "Sec.", "Sec.B"])  # Add the object classes


    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        image_path = str(image)
        img = cv2.imread(image_path)

        print('image read successful.')

        grid_canvas_ = grid_canvas(img, ratio=1)

        print('canvas built')

        imagename = 'dora'
        grid_canvas_.draw_grid_prediction(self.model, self.metadata, name=imagename)
        

        # cv2.imwrite(f'/tmp/{imagename}', vis)
        # print(f'image saved to /tmp/{imagename}')
        # # concat the three images
        # img1 = cv2.imread(f'/tmp/{imagename}_ai.png')
        # img2 = cv2.imread(f'/tmp/{imagename}_small_obj_boost.png')
        # img3 = cv2.imread(f'/tmp/{imagename}_refine.png')
        # img4 = cv2.imread(image_path)
        # img1 = cv2.resize(img1, (img4.shape[1], img4.shape[0]))
        # img2 = cv2.resize(img2, (img4.shape[1], img4.shape[0]))
        # img3 = cv2.resize(img3, (img4.shape[1], img4.shape[0]))
        # img = np.concatenate((img4, img1, img2, img3), axis=1)
        # cv2.imwrite(f'/tmp/{imagename}.png', img1)
        # cv2.imwrite(f'/tmp/{imagename}_small_obj_boost.png', img2)
        # cv2.imwrite(f'/tmpt/{imagename}_refine.png', img3)

        print('prediction done')
        os.system('ls /tmp')
        return [Path(f'/tmp/{imagename}_ai.jpg'), Path(f'/tmp/{imagename}_ai.txt'),
                Path(f'/tmp/{imagename}_small_obj_boost.jpg'), Path(f'/tmp/{imagename}_small_obj_boost.txt'),
                Path(f'/tmp/{imagename}_refine.jpg'), Path(f'/tmp/{imagename}_refine.txt')]


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    predictor.predict(image="./img/test.webp", scale=1.5)