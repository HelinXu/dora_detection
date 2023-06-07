import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata
import cv2
import base64
import numpy as np

# Load the pre-trained model and config
model_path = './pvc/model0.pth'
config_path = './configs/faster_rcnn_R_50_FPN_1x.yaml'
cfg = get_cfg()
cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

metadata = Metadata(thing_classes=["Container", "Text", "Image", "Icon"])

# Create the predictor
predictor = DefaultPredictor(cfg)

def inference(**inputs):
    print('Start of inference')

    jpg_as_text = bytes(inputs['image_base64'], 'utf-8')

    # decode image base64
    jpg_original = base64.b64decode(jpg_as_text)

    image = cv2.imdecode(np.frombuffer(jpg_original, dtype=np.uint8), -1)
    # # save current image
    # cv2.imwrite('pvc/current_image.png', image)
    print("Image Decoded")

    # Run inference
    outputs = predictor(image)

    # Get the predicted instances
    instances = outputs['instances']
    print("inst number detected:", len(instances))

    # Get the detected objects and their scores
    detected_objects = []
    for i in range(len(instances)):
        detected_object = {}
        detected_object['class'] = instances[i].pred_classes
        detected_object['score'] = instances[i].scores.item()
        detected_object['bbox'] = instances[i].pred_boxes.tensor.tolist()
        detected_objects.append(detected_object)

    # # Print the detected objects
    # for obj in detected_objects:
    #     print("Class:", obj['class'], "Score:", obj['score'], "BBox:", obj['bbox'])

    # Visualize the detections
    v = Visualizer(image[:, :, ::-1], metadata, scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite('pvc/output.png', v.get_image()[:, :, ::-1])
    print("output image generated")

    # encode image base64
    retval, buffer = cv2.imencode('.png', v.get_image()[:, :, ::-1])
    return_image_text = str(base64.b64encode(buffer).decode('utf-8'))

    results = {
        "response": return_image_text
    }
    print('End of inference')
    return results


if __name__ == "__main__":
    import cv2
    import base64
    import numpy as np
    # Load the input image
    image_name = './pvc/example.jpeg'

    # encode image base64
    retval, buffer = cv2.imencode('.jpg', cv2.imread(image_name))
    text = str(base64.b64encode(buffer).decode('utf-8'))

    result = inference(image_base64=text)

    # decode image base64
    jpg_original = base64.b64decode(bytes(result['response'], 'utf-8'))
    image = cv2.imdecode(np.frombuffer(jpg_original, dtype=np.uint8), -1)
    cv2.imwrite('./pvc/received_output.png', image)

    

