"""
Get started with our docs, https://docs.beam.cloud
"""
import beam

app = beam.App(
    name="dora-detection-1",
    cpu=1,
    memory="16Gi",
    # gpu=1, # TODO
    python_version="python3.10",
    python_packages=[
    ],
    commands=[
        "pip install torch torchvision icecream==2.1.3 opencv-python pycocotools",
        "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'",
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"
    ]
)

# app.py
app.Mount.PersistentVolume(name="pvc", path="./pvc")

# Triggers determine how your app is deployed
app.Trigger.RestAPI(
    inputs={"image_base64": beam.Types.String()},
    outputs={"response": beam.Types.String()},
    handler="run.py:inference"
)
