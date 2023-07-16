# calculate the layout of the page

# original image: h, w
# square: a, a
# square size a = ratio * min(h, w)
# ration : (0.2, 0.5)

# random import
import random

class Square(object):
    def __init__(self, x1, y1, a):
        self.x1 = x1
        self.y1 = y1
        self.x2 = self.x1 + a
        self.y2 = self.y1 + a


class grid_layout(object):
    def __init__(self, h, w, ratio=0.25):
        self.h = h
        self.w = w
        self.a = int(min(h, w) * ratio)
        self.h_grids = int((h - 1) / self.a) + 1
        self.w_grids = int((w - 1) / self.a) + 1
        self.grid_o = (
            h / 2 - (self.h_grids - 1) * self.a / 2,
            w / 2 - (self.w_grids - 1) * self.a / 2,
        )
    
    def get_square(self, i, j):
        x1 = int(self.grid_o[0] + i * self.a - random.randint(0, self.a))
        y1 = int(self.grid_o[1] + j * self.a - random.randint(0, self.a))
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
        return Square(x1, y1, x2, y2)
