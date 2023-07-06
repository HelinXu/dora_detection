from detectron2.data import DatasetCatalog, MetadataCatalog, datasets
from detectron2.utils.visualizer import Visualizer
import cv2
import random

num_to_visualize = 1000

# Define the dataset and metadata
dataset_name = "dora_sim_train"

# Register the dataset
datasets.register_coco_instances("dora_sim_train", {}, # dataset name
                                f"/root/autodl-tmp/dora_sim/train/train.json", # path to the json file
                                f"/root/autodl-tmp/dora_sim/train")  # folder to the images.
datasets.register_coco_instances("dora_sim_test", {},
                                f"/root/autodl-tmp/dora_sim/test/test.json",
                                f"/root/autodl-tmp/dora_sim/test")
metadata = MetadataCatalog.get(dataset_name)

dataset_dicts = DatasetCatalog.get(dataset_name)

for i in range(num_to_visualize):
    random_image = random.choice(dataset_dicts)
    image_path = random_image["file_name"]
    image = cv2.imread(image_path)

    # Also, visualize the ground truth
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    out = v.draw_dataset_dict(random_image)
    pred = out.get_image()[:, :, ::-1]
    pred = cv2.copyMakeBorder(pred, 60, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    out = cv2.putText(pred, image_path.split('/')[-2] + image_path.split('/')[-1], (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(f"imgs/gt_{image_path.split('/')[-2] + image_path.split('/')[-1]}.jpg", out)
