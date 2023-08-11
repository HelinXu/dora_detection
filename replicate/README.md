# Inference the Dora Detection Model

Helin Xu, xuhelin1911@gmail.com (c) Dora AI

Updated: 2023/08/11

## Requirements

If you run this model on AutoDL, the tested environment is as follows:

- Mirror by AutoDL as follows:
    - PyTorch  1.10.0
    - Python  3.8(ubuntu20.04)
    - Cuda  11.3

- Install after creating an AutoDL instance:
    - opencv-python
    - icecream==2.1.3
    - detectron2==0.6 (`python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html`)
    - replicate (optional, check the official website of replicate)
    - cog (optional, check the official website of replicate)

## Usage

First, put the model file `model_final.pth` in the `model` folder.

The img folder contains a demo iamge. Use this command to inference:

```bash
python predict.py
```

The demo should produce something like this:

```text
$ python predict.py
image read successful.
canvas built
/Users/helin/opt/anaconda3/envs/py39arm/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:3484.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]


Here is the prediction:

Para. 4 0.9999911785125732 1208.5035400390625 63.49684524536133 1299.177734375 139.95587158203125
Para. 4 0.999029278755188 837.7548828125 33.308753967285156 1072.1285400390625 651.4390258789062
Ttl. 1 0.9989843964576721 1194.7222900390625 6.875363826751709 1251.875 21.554277420043945
Img. 2 0.9989525079727173 0.8805546164512634 0.0 607.759521484375 740.1917114257812
Ttl. 1 0.9630513787269592 837.3397827148438 6.2567572593688965 908.671875 20.68829917907715
Img. 2 0.9614969491958618 851.504150390625 234.4654083251953 1068.57373046875 343.4640808105469
Bg. 5 0.8049775958061218 0.0 0.0 1440.0 784.2698364257812

The end.

save dora_ai to /tmp/dora_ai.jpg and /tmp/dora_ai.txt


Here is the prediction:

Para. 4 0.9999796152114868 1209.9248046875 61.93326187133789 1300.355712890625 140.12655639648438
Ttl. 1 0.999806821346283 1194.267822265625 6.0712199211120605 1253.3035888671875 22.349748611450195
Ttl. 1 0.9981582760810852 837.391357421875 7.507558345794678 911.9822998046875 20.556766510009766
Ttl. 1 0.9724526405334473 839.9649658203125 320.4610595703125 1061.0208740234375 334.8432922363281
Cont. 0 0.5944395661354065 844.6984252929688 320.46966552734375 1065.88427734375 337.00762939453125
Para. 4 0.999029278755188 837.7548828125 33.308753967285156 1072.1285400390625 651.4390258789062
Img. 2 0.9989525079727173 0.8805546164512634 0.0 607.759521484375 740.1917114257812
Img. 2 0.9614969491958618 851.504150390625 234.4654083251953 1068.57373046875 343.4640808105469
Bg. 5 0.8049775958061218 0.0 0.0 1440.0 784.2698364257812

The end.

save dora_ai to /tmp/dora_small_obj_boost.jpg and /tmp/dora_small_obj_boost.txt


Here is the prediction:

Para. 4 0.9999796152114868 1209.9248046875 61.93326187133789 1300.355712890625 140.12655639648438
Ttl. 1 0.999806821346283 1194.267822265625 6.0712199211120605 1253.3035888671875 22.349748611450195
Ttl. 1 0.9981582760810852 837.391357421875 7.507558345794678 911.9822998046875 20.556766510009766
Ttl. 1 0.9724526405334473 839.9649658203125 320.4610595703125 1061.0208740234375 334.8432922363281
Cont. 0 0.5944395661354065 839.579833984375 319.81585693359375 1070.579833984375 337.81585693359375
Para. 4 0.999029278755188 837.7548828125 33.308753967285156 1072.1285400390625 651.4390258789062
Img. 2 0.9989525079727173 7.0 8.0 599.0 748.0
Img. 2 0.9614969491958618 839.7971801757812 233.56553649902344 1070.7972412109375 337.5655517578125
Bg. 5 0.8049775958061218 7.0 8.0 1294.0 748.0

The end.

save dora_ai to /tmp/dora_refine.jpg and /tmp/dora_refine.txt
prediction done
```

And the prediction result is saved in `/tmp`. You can change the path in `process.py:grid_canvas.draw_grid_prediction`.

Follow predict.py to inference on multiple images (just a few lines and you can do it):

```python
# First modify the output path in `process.py:grid_canvas.draw_grid_prediction` as needed.
if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    predictor.predict(image=...)
    predictor.predict(image=...)
    ...
```