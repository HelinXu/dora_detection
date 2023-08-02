# calculate the layout of the page

# original image: h, w
# square: a, a
# square size a = ratio * min(h, w)
# ration : (0.2, 0.5)

# random import
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
import numpy as np
import cv2

import time


class Square(object):
    def __init__(self, x1, y1, a):
        self.x1 = x1
        self.y1 = y1
        b = int(a * 0.6)
        self.x2 = self.x1 + b
        self.y2 = self.y1 + a

class Rectangle(object):
    def __init__(self, x1, y1, a, b):
        self.x1 = x1
        self.y1 = y1
        self.x2 = self.x1 + a
        self.y2 = self.y1 + b


class grid_layout(object):
    def __init__(self, h, w, ratio=0.25, random=False):
        self.h = h
        self.w = w
        self.random = random
        self.a = int(min(h, w) * ratio)
        self.b = int(self.a * 0.6)
        self.h_grids = int((h - 1) / self.b) + 1
        self.w_grids = int((w - 1) / self.a) + 1
        self.grid_o = (
            h / 2 - (self.h_grids - 1) * self.b / 2,
            w / 2 - (self.w_grids - 1) * self.a / 2,
        )
    
    def get_square(self, i, j):
        '''
        return the Square at (i, j), 0-indexed, and they do not overlap.
        '''
        if self.random:
            x1 = int(self.grid_o[0] + i * self.a - random.randint(0, self.a))
            y1 = int(self.grid_o[1] + j * self.a - random.randint(0, self.a))
        else:
            x1 = int(self.grid_o[0] + i * self.b - self.b / 2)
            y1 = int(self.grid_o[1] + j * self.a - self.a / 2)
        x2 = x1 + self.b
        y2 = y1 + self.a
        # clip if out of bound
        if x1 < 0:
            x1 = 0
            x2 = self.b
        if y1 < 0:
            y1 = 0
            y2 = self.a
        if x2 > self.h:
            x2 = self.h
            x1 = self.h - self.b
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
                # self.square_imgs.append(
                #     self.image[self.squares[-1].x1 : self.squares[-1].x2, self.squares[-1].y1 : self.squares[-1].y2].copy()
                # )
                # ic(self.square_imgs[-1].shape)
        self.canvas_size = (self.layout.h_grids * self.layout.b, self.layout.w_grids * self.layout.a * 2, c)
        
    def draw_grid_prediction(self, predictor, metadata, save=True, name=''):
        # Run inference
        outputs = predictor(self.image)

        # full size
        full_size_output_insts = outputs["instances"]
        # remove those instances with small area
        areas = full_size_output_insts.pred_boxes.area().cpu().numpy()
        keep = np.where(areas > 0.01 * self.h * self.w)[0]
        full_size_output_insts = full_size_output_insts[keep]

        # Visualize the predictions
        v = Visualizer(self.image[:, :, ::-1], metadata=metadata, scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # Save the visualization
        pred = out.get_image()[:, :, ::-1]
        grid_preds = []
        all_output_insts = []
        for square in self.squares:
            square_img = self.image[square.x1 : square.x2, square.y1 : square.y2].copy()
            # ic(square_img.shape)
            outputs = predictor(square_img)

            # draw the original image and write the predictions to txt
            '''
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    labels = [metadata.thing_classes[i] for i in classes]
    scores = instances.scores.numpy()
    with open(f'./output/imgs/pred_{imagename}.txt', 'w') as f:
        for lab, box, cls, score in zip(labels, boxes, classes, scores):
            f.write(f'{lab} {cls} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')
            '''
            ic(outputs["instances"])
            output_insts = outputs["instances"]
            # output_insts.image_height = full_size_output_insts.image_height
            ic(len(output_insts))
            output_insts.pred_boxes.tensor[:, 0] += square.y1
            output_insts.pred_boxes.tensor[:, 1] += square.x1
            output_insts.pred_boxes.tensor[:, 2] += square.y1
            output_insts.pred_boxes.tensor[:, 3] += square.x1
            ic(output_insts)
            areas = outputs["instances"].pred_boxes.area().cpu().numpy()
            keep = np.where(areas < 0.01 * self.h * self.w)[0]
            ic(keep)
            # bbox = bbox[keep]
            # classes = classes[keep]
            # labels = [metadata.thing_classes[i] for i in classes]
            # scores = scores[keep]
            output_insts = output_insts[keep]
            ic(output_insts)
            ic(len(output_insts))

            all_output_insts.append(output_insts)
            
            # bbox = output_insts.pred_boxes.tensor.cpu().numpy()
            # classes = output_insts.pred_classes.cpu().numpy()
            # labels = [metadata.thing_classes[i] for i in classes]
            # scores = output_insts.scores.cpu().numpy()

            # v = Visualizer(square_img[:, :, ::-1], metadata=metadata, scale=1)
            # out = v.draw_instance_predictions(output_insts.to('cpu'))
            # grid_preds.append(out.get_image()[:, :, ::-1])
            # name = str(time.time())
            # with open(f'./output/imgs/{name}.txt', 'w') as f:
            #     for lab, box, cls, score in zip(labels, bbox, classes, scores):
            #         f.write(f'{lab} {cls} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')
            # cv2.imwrite(f'./output/imgs/{name}.png', square_img)
            # cv2.imwrite(f'./output/imgs/{name}_pred.png', grid_preds[-1])


        # # use opencv to draw the grid of predictions
        output_image = np.zeros(self.canvas_size, dtype=np.uint8)
        # for i in range(self.layout.h_grids):
        #     for j in range(self.layout.w_grids):
        #         output_image[   i * self.layout.b : (i + 1) * self.layout.b, \
        #                         j * self.layout.a : (j + 1) * self.layout.a] = grid_preds[i * self.layout.w_grids + j]
        
        # # draw the grid
        # for i in range(self.layout.h_grids):
        #     cv2.line(output_image, (0, i * self.layout.a), (self.layout.w_grids * self.layout.a, i * self.layout.a), (255, 255, 255), 3)
        # for j in range(self.layout.w_grids):
        #     cv2.line(output_image, (j * self.layout.a, 0), (j * self.layout.a, self.layout.h_grids * self.layout.a), (255, 255, 255), 3)

        v = Visualizer(self.image[:, :, ::-1], metadata=metadata, scale=1)
        # merge all the instances
        all_output_insts = [inst.to("cpu") for inst in all_output_insts]
        full_size_output_insts._image_size = all_output_insts[0].image_size
        all_output_insts.append(full_size_output_insts.to("cpu"))
        ic(all_output_insts[-1].image_size)
        all_output_insts = Instances.cat(all_output_insts)

        # remove those instances with large area

        # # None-Maximum Suppression
        # keep = all_output_insts.non_max_suppression(threshold=0.5)

        out = v.draw_instance_predictions(all_output_insts)
        pred_corrected = out.get_image()[:, :, ::-1]


        # fill the rest of the canvas with the original image, centered
        # output_image[0 : self.h, self.layout.w_grids * self.layout.a : self.layout.w_grids * self.layout.a + self.w] = pred
        output_image[int((self.canvas_size[0] - self.h) / 2) : int((self.canvas_size[0] + self.h) / 2), \
                     self.layout.w_grids * self.layout.a + int((self.layout.w_grids * self.layout.a - self.w) / 2) : self.layout.w_grids * self.layout.a + int((self.layout.w_grids * self.layout.a + self.w) / 2)] = pred
        output_image[int((self.canvas_size[0] - self.h) / 2) : int((self.canvas_size[0] + self.h) / 2), \
                     int((self.layout.w_grids * self.layout.a - self.w) / 2) : int((self.layout.w_grids * self.layout.a + self.w) / 2)] = pred_corrected

        bboxs = all_output_insts.pred_boxes.tensor.cpu().numpy()
        classes = all_output_insts.pred_classes.cpu().numpy()
        labels = [metadata.thing_classes[i] for i in classes]
        scores = all_output_insts.scores.cpu().numpy()
        with open(f'./output/imgs/{name}.txt', 'w') as f:
            for lab, box, cls, score in zip(labels, bboxs, classes, scores):
                f.write(f'{lab} {cls} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')
        # save the image
        if save:
            cv2.imwrite(f'./output/imgs/pred_{name}', output_image)
        # also save the original image
        cv2.imwrite(f'./output/imgs/{name}', self.image)

        return output_image