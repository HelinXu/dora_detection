import torch
import cv2
import time
import numpy as np
from PIL import Image
from torchvision import transforms
from model import RegressionResNet
from dataset import ImageRegressionDataset
from math import sqrt

# Load the trained model
model = RegressionResNet(out_channels=5)
model.load_state_dict(torch.load('regression_resnet_model.pth'))
model.eval()
model.to('cuda')


def run_one(image, save_path, w, h, gt_dic=None):
    # image shape 224*224
    inimage = image.unsqueeze(0).to('cuda')  # Add batch dimension

    tic = time.time()
    # Perform inference
    with torch.no_grad():
        output = model(inimage)

    toc = time.time()

    print('time: ', toc - tic)

    np_image = image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    # bgr - rgb
    np_image = np_image[:, :, ::-1].copy()
    
    # Resize the image to the original dimensions (w, h)
    image = cv2.resize(np_image, (w, h))


    # Retrieve the predicted labels
    boarder_size = int(0.2 * sqrt(w*h))
    output = output.cpu()
    predicted_fontsize = output[0, 0].item() * w
    predicted_fontweight = output[0, 1].item() * sqrt(w*h) * 150
    predicted_color_r = output[0, 2].item() * 300
    predicted_color_g = output[0, 3].item() * 300
    predicted_color_b = output[0, 4].item() * 300
    pred_image = cv2.copyMakeBorder(image, boarder_size, boarder_size, boarder_size, boarder_size, cv2.BORDER_CONSTANT, value=(int(predicted_color_b), int(predicted_color_g), int(predicted_color_r)))

    if gt_dic is not None:
        gt_fontsize = gt_dic['fontsize'] * w
        gt_fontweight = gt_dic['fontweight'] * sqrt(w*h) * 150
        gt_color_r = gt_dic['color_r'] * 300
        gt_color_g = gt_dic['color_g'] * 300
        gt_color_b = gt_dic['color_b'] * 300

        print('fontsize: ', gt_fontsize, predicted_fontsize)
        print('fontweight: ', gt_fontweight, predicted_fontweight)
        print('color_r: ', gt_color_r, predicted_color_r)
        print('color_g: ', gt_color_g, predicted_color_g)
        print('color_b: ', gt_color_b, predicted_color_b)


        # write the ground truth on a new image
        gt_image = cv2.copyMakeBorder(image, boarder_size, boarder_size, boarder_size, boarder_size, cv2.BORDER_CONSTANT, value=(int(gt_color_b), int(gt_color_g), int(gt_color_r)))
        # concat
        pred_image = np.concatenate((pred_image, gt_image), axis=1)

        # add border below
        pred_image = cv2.copyMakeBorder(pred_image, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # write the ground truth on the border
        cv2.putText(pred_image, f'size error: {abs(gt_fontsize - predicted_fontsize)/(1+gt_fontsize)*100:.1f}%  weight err: {abs(gt_fontweight-predicted_fontweight)/(gt_fontweight+1)*100:.1f}%', (10, pred_image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # save the image
    cv2.imwrite(save_path, pred_image)


# Prepare the image for inference
for i in range(20):
    image = Image.open(f'./img/{i}.jpg').convert('RGB')
    w, h = image.size
    # resize the image to be 224x224
    image = image.resize((224, 224))
    # Apply transformations if provided
    image = transforms.ToTensor()(image)
    run_one(image, f'./out/real{i}.jpg', w, h)



# dataset
testingset = ImageRegressionDataset(root_dir='/root/autodl-tmp/dora_font/', annotation_file='/root/autodl-tmp/dora_font/test/test.txt',
                                        transform=transforms.ToTensor())

# Prepare the image for inference random choice
for i in range(100):
    image, w, h, fontsize, fontweight, color_r, color_g, color_b = testingset[np.random.randint(0, len(testingset))]

    run_one(image, f'./out/sim-test{i}.jpg', w, h, gt_dic={'fontsize': fontsize, 'fontweight': fontweight, 'color_r': color_r, 'color_g': color_g, 'color_b': color_b})