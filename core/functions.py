import os
import cv2
import random
import numpy as np
import tensorflow as tf
from core.config import cfg
from core.utils import read_class_names

# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            counts[class_name] = counts.get(class_name, 0) + 1

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes = None):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        counts[class_name] = counts.get(class_name, 0) + 1
        # get box coords
        xmin, ymin, xmax, ymax = boxes[i]
        # crop detection from image (take an additional 5 pixels around all edges)
        cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # construct image name and join it to path for saving crop properly
        img_name = class_name + '_' + str(counts[class_name]) + '.png'
        img_path = os.path.join(path, img_name )
        # save image
        cv2.imwrite(img_path, cropped_img)