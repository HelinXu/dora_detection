import os
from PIL import Image
from torch.utils.data import Dataset

class ImageRegressionDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.data = []

        # Read annotation file
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        # Parse annotation file
        for line in lines:
            line = line.strip().split(' ')
            image_name = line[0]
            labels = [float(label) for label in line[1:]]
            self.data.append((image_name, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name, labels = self.data[index]

        # Load image
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        # resize the image to be 224x224
        image = image.resize((224, 224))

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, labels[0]/300, labels[1]/1000, labels[2]/300, labels[3]/300, labels[4]/300, labels[5]/300


if __name__ == '__main__':
    import torch
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Define the dataset
    dataset = ImageRegressionDataset(root_dir='/root/autodl-tmp/dora_font/', annotation_file='/root/autodl-tmp/dora_font/train/train.txt',
                                     transform=transforms.ToTensor())

    # iter through the dataset to make sure it works
    for i in range(len(dataset)):
        image, fontsize, fontweight, color_r, color_g, color_b, color_a = dataset[i]
        assert color_a <= 1.0, (fontsize, fontweight, color_r, color_g, color_b, color_a)
        print(image.shape, fontsize, fontweight, color_r, color_g, color_b, color_a)
