import torch
import cv2
import time
import numpy as np
from PIL import Image
from torchvision import transforms
from model import RegressionResNet


# Load the trained model
model = RegressionResNet(out_channels=5)
model.load_state_dict(torch.load('regression_resnet_model.pth'))
model.eval()
model.to('cuda')

# Prepare the image for inference
for i in range(1,20):
    image_path = f'./img/{i}.jpg'  # Replace with the path to your image
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0).to('cuda')  # Add batch dimension

    tic = time.time()
    # Perform inference
    with torch.no_grad():
        output = model(image, torch.tensor(w).unsqueeze(0).to('cuda'), torch.tensor(h).unsqueeze(0).to('cuda'))

    toc = time.time()

    print('time: ', toc - tic)

    # Retrieve the predicted labels
    output = output.cpu()
    predicted_fontsize = output[0, 0].item() * 300
    predicted_fontweight = output[0, 1].item() * 10
    predicted_color_r = output[0, 2].item() * 300
    predicted_color_g = output[0, 3].item() * 300
    predicted_color_b = output[0, 4].item() * 300


    # paint the color along with the image

    # pred = cv2.copyMakeBorder(cv2.imread(image_path) , 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(predicted_color_b, predicted_color_g, predicted_color_r))
    image = cv2.imread(image_path.replace('.png', '-original.png'))
    predicted_fontsize = min(predicted_fontsize, image.shape[0])
    # the rectangle height is predicted_fontsize (in pixels), centered at the image
    # center
    y1 = int(image.shape[0] / 2 - predicted_fontsize / 2)
    y2 = int(image.shape[0] / 2 + predicted_fontsize / 2)
    x1 = int(10)
    x2 = int(image.shape[1] - 10)
    pred = cv2.rectangle(image, (x1, y1), (x2, y2), (predicted_color_b, predicted_color_g, predicted_color_r), int(predicted_fontweight))
    # overlay the rectangle onto our original image


    gt_image = cv2.imread(image_path)
    # concat horizontally
    pred = np.concatenate((pred, gt_image), axis=1)
    
    # save
    cv2.imwrite(f'./img/pred{i}.png', pred)

# Prepare the image for inference
for i in range(1,100):
    image_path = f'/root/autodl-tmp/dora_font/test/{i}.png'  # Replace with the path to your image
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    tic = time.time()
    # Perform inference
    with torch.no_grad():
        output = model(image, torch.tensor(w).unsqueeze(0), torch.tensor(h).unsqueeze(0))

    toc = time.time()

    print('time: ', toc - tic)

    # Retrieve the predicted labels
    predicted_fontsize = output[0, 0].item() * 300
    predicted_fontweight = output[0, 1].item() * 10
    predicted_color_r = output[0, 2].item() * 300
    predicted_color_g = output[0, 3].item() * 300
    predicted_color_b = output[0, 4].item() * 300


    # paint the color along with the image

    # pred = cv2.copyMakeBorder(cv2.imread(image_path) , 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(predicted_color_b, predicted_color_g, predicted_color_r))
    image = cv2.imread(image_path.replace('.png', '-original.png'))
    predicted_fontsize = min(predicted_fontsize, image.shape[0])
    # the rectangle height is predicted_fontsize (in pixels), centered at the image
    # center
    y1 = int(image.shape[0] / 2 - predicted_fontsize / 2)
    y2 = int(image.shape[0] / 2 + predicted_fontsize / 2)
    x1 = int(10)
    x2 = int(image.shape[1] - 10)
    pred = cv2.rectangle(image, (x1, y1), (x2, y2), (predicted_color_b, predicted_color_g, predicted_color_r), int(predicted_fontweight))
    # overlay the rectangle onto our original image


    gt_image = cv2.imread(image_path)
    # concat horizontally
    pred = np.concatenate((pred, gt_image), axis=1)
    
    # save
    cv2.imwrite(f'out/pred{i}.png', pred)
    