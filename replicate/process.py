# calculate the layout of the page

# original image: h, w
# square: a, a
# square size a = ratio * min(h, w)
# ration : (0.2, 0.5)

# random import
from copy import copy, deepcopy
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

        def save_img(img, img_name):
            cv2.imwrite(f'tmp/{img_name}.jpg', img)

        def save_pred_text(all_output_insts, name):
            bboxs = all_output_insts.pred_boxes.tensor.cpu().numpy()
            classes = all_output_insts.pred_classes.cpu().numpy()
            labels = [metadata.thing_classes[i] for i in classes]
            scores = all_output_insts.scores.cpu().numpy()
            with open(f'tmp/{name}.txt', 'w') as f:
                for lab, box, cls, score in zip(labels, bboxs, classes, scores):
                    f.write(f'{lab} {cls} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')


        # Run inference
        outputs = predictor(self.image)

        v1 = Visualizer(self.image[:, :, ::-1], metadata=metadata, scale=1)
        out_beta = v1.draw_instance_predictions(outputs["instances"].to("cpu"))
        beta_img = out_beta.get_image()[:, :, ::-1]

        #######################
        # save pred from AI
        #######################
        save_img(beta_img, f'{name}_ai')
        save_pred_text(outputs["instances"], f'{name}_ai')
        print(f'save {name}_ai to /tmp/{name}_ai.jpg and /tmp/{name}_ai.txt')




        # outputs["instances"] = Instances.cat(insts)

        # full size
        full_size_output_insts = outputs["instances"]
        # remove those instances with small area
        areas = full_size_output_insts.pred_boxes.area().cpu().numpy()
        keep = np.where(areas > 0.01 * self.h * self.w)[0]
        full_size_output_insts = full_size_output_insts[keep]

        # # Visualize the predictions
        # v = Visualizer(self.image[:, :, ::-1], metadata=metadata, scale=1)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # # Save the visualization
        # pred = out.get_image()[:, :, ::-1]
        # grid_preds = []
        all_output_insts = []
        for square in self.squares:
            square_img = self.image[square.x1 : square.x2, square.y1 : square.y2].copy()
            # ic(square_img.shape)
            outputs = predictor(square_img)
            # refine_algo1(outputs, metadata, square_img)

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
            # ic(outputs["instances"])
            output_insts = outputs["instances"]
            # output_insts.image_height = full_size_output_insts.image_height
            # ic(len(output_insts))
            output_insts.pred_boxes.tensor[:, 0] += square.y1
            output_insts.pred_boxes.tensor[:, 1] += square.x1
            output_insts.pred_boxes.tensor[:, 2] += square.y1
            output_insts.pred_boxes.tensor[:, 3] += square.x1
            # ic(output_insts)
            areas = outputs["instances"].pred_boxes.area().cpu().numpy()
            keep = np.where(areas < 0.01 * self.h * self.w)[0]
            # ic(keep)
            # bbox = bbox[keep]
            # classes = classes[keep]
            # labels = [metadata.thing_classes[i] for i in classes]
            # scores = scores[keep]
            output_insts = output_insts[keep]
            # ic(output_insts)
            # ic(len(output_insts))

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

        # merge all the instances
        all_output_insts = [inst.to("cpu") for inst in all_output_insts]
        full_size_output_insts._image_size = all_output_insts[0].image_size
        all_output_insts.append(full_size_output_insts.to("cpu"))
        # ic(all_output_insts[-1].image_size)
        all_output_insts = Instances.cat(all_output_insts)

        v = Visualizer(self.image[:, :, ::-1], metadata=metadata, scale=1)
        out = v.draw_instance_predictions(all_output_insts)
        pred_boost_small_obj = out.get_image()[:, :, ::-1]
        #######################
        # save pred after small_obj_boost
        #######################
        save_img(pred_boost_small_obj, f'{name}_small_obj_boost')
        save_pred_text(all_output_insts, f'{name}_small_obj_boost')
        print(f'save {name}_ai to /tmp/{name}_small_obj_boost.jpg and /tmp/{name}_small_obj_boost.txt')


        refined_insts = refine_algo1(all_output_insts, metadata, self.image)


        v2 = Visualizer(self.image[:, :, ::-1], metadata=metadata, scale=1)
        out = v2.draw_instance_predictions(refined_insts)
        pred_refined = out.get_image()[:, :, ::-1]
        #######################
        # save pred after refine
        #######################
        save_img(pred_refined, f'{name}_refine')
        save_pred_text(refined_insts, f'{name}_refine')
        print(f'save {name}_ai to /tmp/{name}_refine.jpg and /tmp/{name}_refine.txt')

        # # use opencv to draw the grid of predictions
        # output_image = np.zeros(self.canvas_size, dtype=np.uint8)
        # for i in range(self.layout.h_grids):
        #     for j in range(self.layout.w_grids):
        #         output_image[   i * self.layout.b : (i + 1) * self.layout.b, \
        #                         j * self.layout.a : (j + 1) * self.layout.a] = grid_preds[i * self.layout.w_grids + j]
        
        # # draw the grid
        # for i in range(self.layout.h_grids):
        #     cv2.line(output_image, (0, i * self.layout.a), (self.layout.w_grids * self.layout.a, i * self.layout.a), (255, 255, 255), 3)
        # for j in range(self.layout.w_grids):
        #     cv2.line(output_image, (j * self.layout.a, 0), (j * self.layout.a, self.layout.h_grids * self.layout.a), (255, 255, 255), 3)



        # remove those instances with large area

        # # None-Maximum Suppression
        # keep = all_output_insts.non_max_suppression(threshold=0.5)




        # # fill the rest of the canvas with the original image, centered
        # # output_image[0 : self.h, self.layout.w_grids * self.layout.a : self.layout.w_grids * self.layout.a + self.w] = pred
        # output_image[int((self.canvas_size[0] - self.h) / 2) : int((self.canvas_size[0] + self.h) / 2), \
        #              self.layout.w_grids * self.layout.a + int((self.layout.w_grids * self.layout.a - self.w) / 2) : self.layout.w_grids * self.layout.a + int((self.layout.w_grids * self.layout.a + self.w) / 2)] = beta_img
        # output_image[int((self.canvas_size[0] - self.h) / 2) : int((self.canvas_size[0] + self.h) / 2), \
        #              int((self.layout.w_grids * self.layout.a - self.w) / 2) : int((self.layout.w_grids * self.layout.a + self.w) / 2)] = pred_corrected

        # bboxs = all_output_insts.pred_boxes.tensor.cpu().numpy()
        # classes = all_output_insts.pred_classes.cpu().numpy()
        # labels = [metadata.thing_classes[i] for i in classes]
        # scores = all_output_insts.scores.cpu().numpy()
        # with open(f'/tmp/{name}.txt', 'w') as f:
        #     for lab, box, cls, score in zip(labels, bboxs, classes, scores):
        #         f.write(f'{lab} {cls} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n')
        # save the image
        # if save:
        #     cv2.imwrite(f'./output/imgs/pred_{name}', output_image)
        # also save the original image
        # cv2.imwrite(f'./output/imgs/{name}', self.image)
        return 

zoom_alpha = 1.2



def refine_algo1(outputs, metadata, image):

    def conv_hist(l, r, zoom_alpha, hist):
        # build the responce map, use float
        responce_map = np.zeros_like(hist, dtype=float)
        # build the first responce as a normal distribution with delta = zoom_alpha - 1, centered at x1
        delta = (zoom_alpha - 1) * (r - l) / 2
        for i in range(len(hist)):
            responce_map[i] = np.exp(- (i - l) ** 2 / delta ** 2)
        # build the second responce as a normal distribution with delta = zoom_alpha - 1, centered at x2
        for i in range(len(hist)):
            responce_map[i] += np.exp(- (i - r) ** 2 / delta ** 2)
        # multiply with the hist
        responce_map *= hist
        return responce_map

    if isinstance(outputs, Instances):
        outputs = {"instances": outputs}
    # full size
    insts = outputs["instances"]
    ori_insts = deepcopy(insts)
    v1 = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1)
    original_out = v1.draw_instance_predictions(ori_insts.to("cpu"))
    # use opencv to take the image's edges
    edges = cv2.Canny(image, 1, 50)

    canny_insts = []

    for i in range(len(insts)):
        inst = insts[i]
        # get class
        cls = inst.pred_classes.cpu().numpy()[0]
        # [0 "Cont.", 1 "Ttl.", 2 "Img.", 3 "Icon", 4 "Para.", 5 "Bg.", 6 "IrImg.", 7 "BgImg.", 8 "CtPil.", 9 "CtCir.", 10 "ImgCir."]
        if cls in {1, 4}:
            canny_insts.append(inst)
            continue
        # get the bbox
        bbox = inst.pred_boxes.tensor.cpu().numpy()
        x1, y1, x2, y2 = bbox[0]
        # center coord.
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # scale up the bbox to 1.3x
        x1_ = center_x - (center_x - x1) * zoom_alpha
        x2_ = center_x + (x2 - center_x) * zoom_alpha
        y1_ = center_y - (center_y - y1) * zoom_alpha
        y2_ = center_y + (y2 - center_y) * zoom_alpha
        # clip the bbox
        x1_ = max(0, x1_)
        x2_ = min(image.shape[1], x2_)
        y1_ = max(0, y1_)
        y2_ = min(image.shape[0], y2_)

        # get the bbox's edges
        bbox_edges = edges[int(y1_) : int(y2_), int(x1_) : int(x2_)]

        # find the square's edges
        # # horizontal

        # pixel level histogram analysis to find the peak of horizontal and vertical lines
        # horizontal
        h_hist = np.sum(bbox_edges, axis=0)
        h_hist = conv_hist(x1 - x1_, x2 - x2_ + len(h_hist), zoom_alpha, h_hist)
        H_hist_l = h_hist[:int((x2 - x1) / 2)]
        H_hist_r = h_hist[int((x2 - x1) / 2):]
        # vertical
        v_hist = np.sum(bbox_edges, axis=1)
        v_hist = conv_hist(y1 - y1_, y2 - y2_ + len(v_hist), zoom_alpha, v_hist)
        V_hist_u = v_hist[:int((y2 - y1) / 2)]
        V_hist_d = v_hist[int((y2 - y1) / 2):]

        # # use matplotlib to draw the histogram
        # import matplotlib.pyplot as plt
        # plt.close()
        # plt.plot(h_hist)
        # plt.plot(conv_hist(x1 - x1_, x2 - x2_ + len(h_hist), zoom_alpha, h_hist))
        # plt.savefig(f'./tmp/{i}.hist.jpg')
        # plt.close()

        # # draw the histogram on the horizontal and vertical edges respectively
        # # first turn into BGR
        # bbox_edges = cv2.cvtColor(bbox_edges, cv2.COLOR_GRAY2BGR)
        # # draw the histogram
        # for i in range(len(h_hist)):
        #     bbox_edges = cv2.line(bbox_edges, (i, bbox_edges.shape[0]), (i, bbox_edges.shape[0] - h_hist[i]), (0, 0, 255), 1)
        # for i in range(len(v_hist)):
        #     bbox_edges = cv2.line(bbox_edges, (0, i), (v_hist[i], i), (0, 0, 255), 1)

        # # save image
        # cv2.imwrite(f'{i}.hist.jpg', bbox_edges)

        # find the peak
        H_peak_l = np.argmax(H_hist_l)
        H_peak_r = np.argmax(H_hist_r) + int((x2 - x1) / 2)
        V_peak_u = np.argmax(V_hist_u)
        V_peak_d = np.argmax(V_hist_d) + int((y2 - y1) / 2)
        # draw red lines through the bbox's edges
        bbox_edges = cv2.cvtColor(bbox_edges, cv2.COLOR_GRAY2BGR)

        # draw the bbox's edges
        # bbox_edges = cv2.line(bbox_edges, (0, 0), (bbox_edges.shape[1], 0), (0, 0, 255), 1)
        # bbox_edges = cv2.line(bbox_edges, (0, bbox_edges.shape[0]), (bbox_edges.shape[1], bbox_edges.shape[0]), (0, 0, 255), 1)
        # bbox_edges = cv2.line(bbox_edges, (0, 0), (0, bbox_edges.shape[0]), (0, 0, 255), 1)
        # bbox_edges = cv2.line(bbox_edges, (bbox_edges.shape[1], 0), (bbox_edges.shape[1], bbox_edges.shape[0]), (0, 0, 255), 1)


        bbox_edges = cv2.line(bbox_edges, (H_peak_l, bbox_edges.shape[0]), (H_peak_l, 0), (0, 0, 255), 1)
        bbox_edges = cv2.line(bbox_edges, (H_peak_r, bbox_edges.shape[0]), (H_peak_r, 0), (0, 0, 255), 1)
        bbox_edges = cv2.line(bbox_edges, (0, V_peak_u), (bbox_edges.shape[1], V_peak_u), (0, 0, 255), 1)
        bbox_edges = cv2.line(bbox_edges, (0, V_peak_d), (bbox_edges.shape[1], V_peak_d), (0, 0, 255), 1)

        # # # draw on original image
        # image = cv2.line(image, (int(x1_ + H_peak_l), int(y1_ + V_peak_u)), (int(x1_ + H_peak_l), int(y1_ + V_peak_d)), (0, 0, 255), 3)
        # image = cv2.line(image, (int(x1_ + H_peak_r), int(y1_ + V_peak_u)), (int(x1_ + H_peak_r), int(y1_ + V_peak_d)), (0, 0, 255), 3)
        # image = cv2.line(image, (int(x1_ + H_peak_l), int(y1_ + V_peak_u)), (int(x1_ + H_peak_r), int(y1_ + V_peak_u)), (0, 0, 255), 3)
        # image = cv2.line(image, (int(x1_ + H_peak_l), int(y1_ + V_peak_d)), (int(x1_ + H_peak_r), int(y1_ + V_peak_d)), (0, 0, 255), 3)

        canny_inst = deepcopy(inst)
        canny_inst.pred_boxes.tensor[0][0] = x1_ + H_peak_l
        canny_inst.pred_boxes.tensor[0][1] = y1_ + V_peak_u
        canny_inst.pred_boxes.tensor[0][2] = x1_ + H_peak_r
        canny_inst.pred_boxes.tensor[0][3] = y1_ + V_peak_d
        canny_insts.append(canny_inst)


        # save the bbox's edges
        # cv2.imwrite(f'./tmp/{i}.edges.jpg', bbox_edges)
    

    v2 = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1)
    canny_out = v2.draw_instance_predictions(Instances.cat(canny_insts).to("cpu"))

    canny_img = canny_out.get_image()[:, :, ::-1]
    original_img = original_out.get_image()[:, :, ::-1]


    # concat with original image
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    outputimg = cv2.hconcat([image, edges, original_img, canny_img])
    # save the edges
    outputs["instances"] = Instances.cat(canny_insts)
    return Instances.cat(canny_insts)


# def detect_rectangle(edge_img, i):

    # detect lines
