import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision import transforms

from dataset import ImageRegressionDataset
from model import RegressionResNet


# Define the L2 loss function
criterion = nn.MSELoss()

# Define the dataset and dataloader (replace with your own implementation)
dataset = ImageRegressionDataset(root_dir='/root/autodl-tmp/dora_font/', annotation_file='/root/autodl-tmp/dora_font/train/train.txt',
                                     transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# Create an instance of the ResNet model
model = RegressionResNet(out_channels=5)

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)

# Training loop
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "You need to enable GPU in order to train this model."
model.to(device).to(torch.float32)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, w, h, fontsize, fontweight, color_r, color_g, color_b) in enumerate(dataloader):
        images = images.to(device)
        fontsize = fontsize.to(device).to(torch.float32)
        fontweight = fontweight.to(device).to(torch.float32)
        color_r = color_r.to(device).to(torch.float32)
        color_g = color_g.to(device).to(torch.float32)
        color_b = color_b.to(device).to(torch.float32)
        w = w.to(device).to(torch.float32)
        h = h.to(device).to(torch.float32)

        # # assert the inputs
        # assert torch.max(fontsize) <= 1.0
        # assert torch.max(fontweight) <= 1.0
        # assert torch.max(color_r) <= 1.0
        # assert torch.max(color_g) <= 1.0
        # assert torch.max(color_b) <= 1.0
        # assert torch.max(color_a) <= 1.0, (fontsize, fontweight, color_r, color_g, color_b, color_a)

        # # assert the inputs
        # assert torch.min(fontsize) >= 0.0
        # assert torch.min(fontweight) >= 0.0
        # assert torch.min(color_r) >= 0.0
        # assert torch.min(color_g) >= 0.0
        # assert torch.min(color_b) >= 0.0
        # assert torch.min(color_a) >= 0.0


        # print(fontsize, fontweight, color_r, color_g, color_b, color_a)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # print(outputs)
        
        # Calculate the loss
        loss = criterion(outputs[:, 0], fontsize) + criterion(outputs[:, 1], fontweight) + criterion(outputs[:, 2], color_r) + criterion(outputs[:, 3], color_g) + criterion(outputs[:, 4], color_b)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'regression_resnet_model.pth')
