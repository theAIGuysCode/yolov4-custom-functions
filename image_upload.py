import streamlit as st
import os
import io

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, logging
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
        return opencv_image
    else:
        return None

def load_model():
    model = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    return model

def predict(saved_model_loaded, image):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416

    # load image
    original_image = image
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    # load_weights
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.5
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

    # custom allowed classes (uncomment line below to allow detections for only people)
    # allowed_classes = ['person']

    image = utils.draw_bbox(original_image, pred_bbox, False, allowed_classes=allowed_classes,
                            read_plate=False)

    image = Image.fromarray(image.astype(np.uint8))

    st.image(image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite('./detections/' + 'detection' + str(1) + '.png', image)


def main():
    st.title('Pretrained model demo')
    original_image = load_image()
    model = load_model()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, original_image)


if __name__ == '__main__':
    main()
