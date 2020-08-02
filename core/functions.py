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