import os
from PIL import Image
from torch.utils.data import Dataset
from math import sqrt

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
        #   [file] [fontSize] [fontWeight] [r] [g] [b] [a]
        image_name, labels = self.data[index]

        # Load image
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        width, height = image.size

        # resize the image to be 224x224
        image = image.resize((224, 224))

        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, width, height, labels[0]/width, labels[1]/sqrt(width*height)/150, labels[2]/300, labels[3]/300, labels[4]/300


if __name__ == '__main__':
    import torch
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Define the dataset
    dataset = ImageRegressionDataset(root_dir='/root/autodl-tmp/dora_font/', annotation_file='/root/autodl-tmp/dora_font/train/train.txt',
                                     transform=transforms.ToTensor())

    # iter through the dataset to make sure it works, plot the histogram of the labels
    fontsize = []
    fontweight = []
    color_r = []
    color_g = []
    color_b = []
    for i in range(len(dataset)):
        image, w, h, f, fw, r, g, b = dataset[i]
        fontsize.append(f)
        fontweight.append(fw)
        color_r.append(r)
        color_g.append(g)
        color_b.append(b)
        # print(image.shape, w, h, f, fw, r, g, b)
        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()
    plt.hist(fontsize)
    # save the histogram
    plt.savefig('fontsize.png')
    plt.clf()
    plt.hist(fontweight)
    plt.savefig('fontweight.png')
    plt.clf()
    plt.hist(color_r)
    plt.savefig('color_r.png')
    plt.clf()
    plt.hist(color_g)
    plt.savefig('color_g.png')
    plt.clf()
    plt.hist(color_b)
    plt.savefig('color_b.png')
    plt.clf()

