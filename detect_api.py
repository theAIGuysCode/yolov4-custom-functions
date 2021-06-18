
# API imports
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import cv2
import time

#Detect imports
import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from core.yolov4 import filter_boxes
import core.utils as utils
from core.functions import *

app = Flask(__name__)

def find_and_ocr(image):
    # Setup Parameters
    weights_path = './checkpoints/custom-416.tflite'
    input_size = 416
    iou = 0.45
    score = 0.5
    fast_ocr = False
    deskew = False

    config = ConfigProto()
    config.gpu_options.allow_growth = True

    interpreter = tf.lite.Interpreter(model_path=weights_path)


    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    # Get license plate image
    crop_license_plate = custom_crop_and_return_objects(original_image, pred_bbox, allowed_classes)
    
    license_plate = utils.custom_recognize_plate(crop_license_plate, fast_ocr, deskew)

    return license_plate    

@app.route("/anpr", methods=["POST"])
def process_image():
    start_time = time.time()
    file = request.files.get('image')
    print("file = ", file)
    # Read the image via file.stream
    img = Image.open(file.stream).convert('RGB')

    image = np.array(img)

    license_plate = find_and_ocr(image)

    elapsed_time = (time.time() - start_time)

    if license_plate is not None:
        return jsonify({
            'license_plate':license_plate,
            'time': elapsed_time
        })
    else:
        return jsonify({'vechileId':'unable to decode'})
    
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=3333)