import json
import numpy as np

from icecream import ic, install
install()
ic.configureOutput(includeContext=True, contextAbsPath=True)

def load_coco_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# ['info', 'licenses', 'images', 'annotations', 'categories']

def split_train_val(data, val_ratio=0.1):
    img_ids = [img['id'] for img in data['images']]
    img_ids = np.array(img_ids)
    np.random.shuffle(img_ids)
    val_num = int(len(img_ids) * val_ratio)
    val_ids = img_ids[:val_num]
    train_ids = img_ids[val_num:]
    train_data = {'info': data['info'], 'licenses': data['licenses'], 'images': [], 'annotations': [], 'categories': data['categories']}
    val_data = {'info': data['info'], 'licenses': data['licenses'], 'images': [], 'annotations': [], 'categories': data['categories']}
    for img in data['images']:
        if img['id'] in train_ids:
            train_data['images'].append(img)
        else:
            val_data['images'].append(img)
    for ann in data['annotations']:
        if ann['image_id'] in train_ids:
            train_data['annotations'].append(ann)
        else:
            val_data['annotations'].append(ann)
    return train_data, val_data

data = load_coco_json('/root/autodl-tmp/dora_dataset/train/_annotations.coco.json')
train_data, val_data = split_train_val(data)
# save train_data and val_data
with open('/root/autodl-tmp/dora_dataset/train.json', 'w') as f:
    json.dump(train_data, f)
with open('/root/autodl-tmp/dora_dataset/val.json', 'w') as f:
    json.dump(val_data, f)

