"""
Get started with our docs, https://docs.beam.cloud
"""
import beam

app = beam.App(
    name="dora-detection-1",
    cpu=1,
    memory="16Gi",
    gpu=1, # TODO
    python_version="3.10",
    python_packages=[
        "torch==1.10.0",
        "torchvision==0.11.1",
        "detectron2==0.6",
        "icecream==2.1.3",
        "opencv-python==4.7.0.72",
        "pycocotools==2.0.6",
    ]
)

# Triggers determine how your app is deployed
app.Trigger.RestAPI(
    inputs={"text": beam.Types.String()},
    outputs={"response": beam.Types.String()},
    handler="run.py:hello_world"
)
