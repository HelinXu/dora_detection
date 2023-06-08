import torch
import cv2
from torchvision.models import resnet50
import torch.nn as nn
import time

from PIL import Image
from torchvision import transforms


class RegressionResNet(nn.Module):
    def __init__(self, out_channels=6):
        super(RegressionResNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.fc = nn.Linear(1000, out_channels)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

# Load the trained model
model = RegressionResNet(out_channels=6)
model.load_state_dict(torch.load('regression_resnet_model.pth'))
model.eval()

# Prepare the image for inference
image_path = '76.png'  # Replace with the path to your image
image = Image.open(image_path).convert('RGB')
transform = transforms.ToTensor()
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

tic = time.time()
# Perform inference
with torch.no_grad():
    output = model(image)

toc = time.time()

print('time: ', toc - tic)

# Retrieve the predicted labels
predicted_fontsize = output[0, 0].item() * 300
predicted_fontweight = output[0, 1].item() * 1000
predicted_color_r = output[0, 2].item() * 300
predicted_color_g = output[0, 3].item() * 300
predicted_color_b = output[0, 4].item() * 300
predicted_color_a = output[0, 5].item() * 300

# paint the color along with the image

pred = cv2.copyMakeBorder(cv2.imread(image_path) , 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(predicted_color_b, predicted_color_g, predicted_color_r))
# save
cv2.imwrite('pred.png', pred)