def inference(**inputs):
    import torch
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, datasets, get_detection_dataset_dicts
    import cv2
    import base64
    import numpy as np

    category = {"id":1,"name":"Text","supercategory":"UI"},{"id":2,"name":"Image","supercategory":"UI"},{"id":3,"name":"Icon","supercategory":"UI"}

    # Load the pre-trained model and config
    model_path = './pvc/model0.pth'
    config_path = './configs/faster_rcnn_R_50_FPN_1x.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    datasets.register_coco_instances("dora_ui", {}, f"data.json", f"./")

    # Create the predictor
    predictor = DefaultPredictor(cfg)

    # # Load the input image
    # image_path = 'example.jpeg'
    # image = cv2.imread(image_path)

    # # encode image base64
    # retval, buffer = cv2.imencode('.jpg', image)
    # jpg_as_text = base64.b64encode(buffer)

    jpg_as_text = inputs['image_base64']

    # decode image base64
    jpg_original = base64.b64decode(jpg_as_text)

    image = cv2.imdecode(np.frombuffer(jpg_original, dtype=np.uint8), -1)

    # Run inference
    outputs = predictor(image)

    # Get the predicted instances
    instances = outputs['instances']

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
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite('output.png', v.get_image()[:, :, ::-1])

    return {"objects": detected_objects, "picture": v.get_image()[:, :, ::-1]}


if __name__ == "__main__":
    import cv2
    import base64
    # Load the input image
    image_path = 'example.jpeg'
    image = cv2.imread(image_path)

    # encode image base64
    retval, buffer = cv2.imencode('.jpg', image)
    text = base64.b64encode(buffer)
    result = inference(image_base64=text)
    print(result)
    

