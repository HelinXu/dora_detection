# calculate the layout of the page

# original image: h, w
# square: a, a
# square size a = ratio * min(h, w)
# ration : (0.2, 0.5)

# random import
import random
from detectron2.utils.visualizer import Visualizer
import numpy as np
import cv2

class Square(object):
    def __init__(self, x1, y1, a):
        self.x1 = x1
        self.y1 = y1
        self.x2 = self.x1 + a
        self.y2 = self.y1 + a


class grid_layout(object):
    def __init__(self, h, w, ratio=0.25, random=False):
        self.h = h
        self.w = w
        self.random = random
        self.a = int(min(h, w) * ratio)
        self.h_grids = int((h - 1) / self.a) + 1
        self.w_grids = int((w - 1) / self.a) + 1
        self.grid_o = (
            h / 2 - (self.h_grids - 1) * self.a / 2,
            w / 2 - (self.w_grids - 1) * self.a / 2,
        )
    
    def get_square(self, i, j):
        if self.random:
            x1 = int(self.grid_o[0] + i * self.a - random.randint(0, self.a))
            y1 = int(self.grid_o[1] + j * self.a - random.randint(0, self.a))
        else:
            x1 = int(self.grid_o[0] + i * self.a - self.a / 2)
            y1 = int(self.grid_o[1] + j * self.a - self.a / 2)
        x2 = x1 + self.a
        y2 = y1 + self.a
        # clip if out of bound
        if x1 < 0:
            x1 = 0
            x2 = self.a
        if y1 < 0:
            y1 = 0
            y2 = self.a
        if x2 > self.h:
            x2 = self.h
            x1 = self.h - self.a
        if y2 > self.w:
            y2 = self.w
            y1 = self.w - self.a
        return Square(x1, y1, self.a)


class grid_canvas(object):
    def __init__(self, origin_image, ratio=0.5):
        self.image = origin_image
        self.h, self.w, c = self.image.shape
        self.layout = grid_layout(self.h, self.w, ratio)
        self.squares = []
        self.square_imgs = []
        for i in range(self.layout.h_grids):
            for j in range(self.layout.w_grids):
                self.squares.append(self.layout.get_square(i, j))
                # ic the shape
                # ic(self.squares[-1].x1 - self.squares[-1].x2, self.squares[-1].y1 - self.squares[-1].y2)
                self.square_imgs.append(
                    self.image[self.squares[-1].x1 : self.squares[-1].x2, self.squares[-1].y1 : self.squares[-1].y2].copy()
                )
                # ic(self.square_imgs[-1].shape)
        self.canvas_size = (self.layout.h_grids * self.layout.a, self.layout.w_grids * self.layout.a * 2, c)
        
    def draw_grid_prediction(self, predictor, metadata):
        # Run inference
        outputs = predictor(self.image)
        # Visualize the predictions
        v = Visualizer(self.image[:, :, ::-1], metadata=metadata, scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # Save the visualization
        pred = out.get_image()[:, :, ::-1]
        grid_preds = []
        for square_img in self.square_imgs:
            # ic(square_img.shape)
            outputs = predictor(square_img)
            v = Visualizer(square_img[:, :, ::-1], metadata=metadata, scale=1)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            grid_preds.append(out.get_image()[:, :, ::-1])
        # use opencv to draw the grid of predictions
        output_image = np.zeros(self.canvas_size, dtype=np.uint8)
        for i in range(self.layout.h_grids):
            for j in range(self.layout.w_grids):
                output_image[   i * self.layout.a : (i + 1) * self.layout.a, \
                                j * self.layout.a : (j + 1) * self.layout.a] = grid_preds[i * self.layout.w_grids + j]
        
        # draw the grid
        for i in range(self.layout.h_grids):
            cv2.line(output_image, (0, i * self.layout.a), (self.layout.w_grids * self.layout.a, i * self.layout.a), (255, 255, 255), 3)
        for j in range(self.layout.w_grids):
            cv2.line(output_image, (j * self.layout.a, 0), (j * self.layout.a, self.layout.h_grids * self.layout.a), (255, 255, 255), 3)


        # fill the rest of the canvas with the original image, centered
        # output_image[0 : self.h, self.layout.w_grids * self.layout.a : self.layout.w_grids * self.layout.a + self.w] = pred
        output_image[int((self.canvas_size[0] - self.h) / 2) : int((self.canvas_size[0] + self.h) / 2), \
                     self.layout.w_grids * self.layout.a + int((self.layout.w_grids * self.layout.a - self.w) / 2) : self.layout.w_grids * self.layout.a + int((self.layout.w_grids * self.layout.a + self.w) / 2)] = pred
        return output_image