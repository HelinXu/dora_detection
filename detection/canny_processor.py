# read all the images and process them with canny

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# read all the images
path = '/root/autodl-tmp/dora_sim/test'
save_path = '/root/autodl-tmp/dora_sim_canny/test'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for image_path in os.listdir(path):
    if not image_path.endswith('webp'):
        continue
    print(image_path)
    image = cv2.imread(os.path.join(path, image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    # image = cv2.resize(image, (256, 256))
    image_edge = cv2.Canny(image, 1, 50)
    # use it as the forth channel
    image[:, :, 3] = image_edge
    # save as png
    image_path = image_path.split('.')[0] + '.png'
    # use 4 channels to save the image
    cv2.imwrite(os.path.join(save_path, image_path), image)
    print(image_path + ' saved')


# run a system command
os.system('cp /root/autodl-tmp/dora_sim/train/train.json /root/autodl-tmp/dora_sim_canny/train/')
os.system("sed -i 's/webp/png/g' /root/autodl-tmp/dora_sim_canny/train/train.json")
os.system('cp /root/autodl-tmp/dora_sim/test/test.json /root/autodl-tmp/dora_sim_canny/test/')
os.system("sed -i 's/webp/png/g' /root/autodl-tmp/dora_sim_canny/test/test.json")