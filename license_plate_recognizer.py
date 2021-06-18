
import pytesseract
import cv2
import os
from os import path
import numpy as np
import re
import statistics
import time
import logging
import sys, getopt

# Deskew
import math
from typing import Tuple, Union
from deskew import determine_skew

def optical_image_recognition(image, char_whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", psm = 13, oem = 1):
    config = f"-c tessedit_char_whitelist={char_whitelist} --psm {psm} --oem {oem}"
    if logging.DEBUG >= logging.root.level:
       print(f"Optical Image Recognition config: {config}")
    return pytesseract.image_to_string(image, config=config)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                    for im in im_list]
    return cv2.hconcat(im_list_resize)

def find_check_contours(sorted_contours, height_org, width_org, image):
    correct_contours = []
    for idx, cnt in enumerate(sorted_contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if logging.DEBUG >= logging.root.level:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,255),2)
            cv2.imshow("Highlight Contour", image)
            cv2.waitKey(0)


        # if height of box is not a quarter of total height then skip
        if height_org / float(h) > 6:
            if logging.DEBUG >= logging.root.level: 
                print("Height Failed")
            continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.4 skip
        if ratio < 1.25: 
            if logging.DEBUG >= logging.root.level: 
                print("Ratio Failed")
            continue

        area = h * w
        # if width is not more than 25 pixels skip
        if width_org / float(w) > 30: 
            if logging.DEBUG >= logging.root.level: 
                print("Width Failed")
            continue
        # if area is less than 100 pixels skip
        if area < 100: 
            if logging.DEBUG >= logging.root.level: 
                print("Area Failed")
            continue
        
        value_index = [cnt, idx]
        correct_contours.append(value_index)

    return correct_contours

def calculate_average_height_of_letter(contour_list):
    average_height_list = contour_list

    # remove smallest and largest number for a better average
    average_height_list.remove(max(average_height_list))
    average_height_list.remove(min(average_height_list))

    # calculate letter area average
    return statistics.mean(average_height_list)

def most_frequent(list):
    counter = 0
    num = list[0]
      
    for i in list:
        curr_frequency = list.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
  
    return num

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def main(args):
    # Prepare image parameter 
    image = None
    enable_fast = False

    # Get full command-line arguments
    full_cmd_arguments = sys.argv

    # Keep all but the first
    argument_list = full_cmd_arguments[1:]

    short_options = "hdfi:"
    long_options = ["help", "debug", "fast", "input="]

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print (str(err))
        sys.exit(2)
    
    if(len(argument_list) == 0):
        print(
            "Usage: \n"
            "  python license_plate_recognizer.py <command> [option]"
            "\n \n"
            "Options: \n"
            "  -h, --help \t \t Show help. \n"
            "  -d, --debug\t \t Enable debugmode.\n"
            "  -f, --fast\t \t Less accuracy, but faster"
        )
        sys.exit(2)

    # Evaluate given options
    for current_argument, current_value in arguments:
        if current_argument in ("-d", "--debug"):
            print ("Enabling debug mode")
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        elif current_argument in ("-h", "--help"):
            print(
                "Usage: \n"
                "  python license_plate_recognizer.py <command> [option]"
                "\n \n"
                "Options: \n"
                "  -h, --help \t \t Show help. \n"
                "  -d, --debug\t \t Enable debugmode.\n"
                "  -f, --fast\t \t Less accuracy, but faster"
            )
            sys.exit(2)
        elif current_argument in ("-i", "--input"):
            if(path.exists(current_value)):
                image = current_value
            else:
                print("File doesn't exist!")
                sys.exit(2)
        elif current_argument in ("-f", "--fast"):
            enable_fast = True

    # Setup Parameters
    image = cv2.imread(image)
    plate_num = ""
    offset_size_letter = 0.1 # Maximum offset of the average letter

    #region Filters
    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if logging.DEBUG >= logging.root.level:
       cv2.imshow("Grayscale Image", gray)
       cv2.waitKey(0)

    # angle = determine_skew(gray)
    # rotated = rotate(gray, angle, (0, 0, 0))
    # if logging.DEBUG >= logging.root.level:
    #   cv2.imshow("Rotated Image", rotated)
    #   print(f"Image Angle:{ angle}")
    #   cv2.waitKey(0)
    
    # Resize image to three times as large as original for better readability
    resize = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    if logging.DEBUG >= logging.root.level:
       cv2.imshow("Resize Image", resize)
       cv2.waitKey(0)

    # perform gaussian blur to smoothen image
    # blur = cv2.GaussianBlur(resize, (5,5), 0)
    # if logging.DEBUG >= logging.root.level:
    #    cv2.imshow("Blur Image", blur)
    #    cv2.waitKey(0)

    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(resize, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    if logging.DEBUG >= logging.root.level:
       cv2.imshow("Dilation Image", dilation)
       cv2.waitKey(0)

    #endregion

    #region Contours
    # find contours of regions of interest within license plate
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #endregion

    # create copy of gray image
    copy_orginal = resize.copy()

    # Help to calculate if a contours is valid, this gives the height and width of the orginal image
    height, width = copy_orginal.shape

    # Returns the correct contours and the average height of a letter
    correct_contours = find_check_contours(contours, height, width, resize.copy())

    # Get Plate Level
    
    list_plate_level = []
    for cnt, index in correct_contours:
        list_plate_level.append(hierarchy[0,index,3])
    plate_level = most_frequent(list_plate_level)

    print(f"PlateLv {plate_level}")

    correct_contours_no_child = []
    correct_contours_height_list = []
    for cnt, index in correct_contours:
        if plate_level == hierarchy[0,index,3]:
            x,y,w,h = cv2.boundingRect(cnt)
            correct_contours_no_child.append(cnt)
            correct_contours_height_list.append(h)

    # Calculate the average height of a letter to increase the accuracy
    average_height = calculate_average_height_of_letter(correct_contours_height_list)

    # sort contours left-to-right
    sorted_correct_contours = sorted(correct_contours_no_child, key=lambda ctr: cv2.boundingRect(ctr)[0])

    images = []
    for cnt in sorted_correct_contours:
        x,y,w,h = cv2.boundingRect(cnt)

        if h/average_height > (1 + offset_size_letter) or h/average_height < (1 - offset_size_letter):
            continue
            
        # draw the rectangle
        cv2.rectangle(copy_orginal, (x,y), (x+w, y+h), (0,255,0),2)
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)
        images.append(roi)

    if enable_fast:
        # Concatenate all the valid images horizontally to save time  
        con_h_valid_images = hconcat_resize_min(images)
        if logging.DEBUG >= logging.root.level:
            cv2.imshow("Concatenated Valid Images", con_h_valid_images)
            cv2.waitKey(0)
        # Try to read what the license plate is
        try:
            text = optical_image_recognition(con_h_valid_images)
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except: 
            text = None
    else:
        for image in images:
            # Try to read what the license plate is
            if logging.DEBUG >= logging.root.level:
                cv2.imshow("Valid Images", image)
                cv2.waitKey(0)
            try:
                text = optical_image_recognition(image)
                # clean tesseract text by removing any unwanted blank spaces
                clean_text = re.sub('[\W_]+', '', text)
                plate_num += clean_text
            except: 
                text = None

    if plate_num != None:
        print("License Plate #: ", plate_num)
    
    if logging.DEBUG >= logging.root.level:
        cv2.imshow("All Found Characters", copy_orginal)
        cv2.waitKey(0)

    #If there is a window open destroy it or it will wait forever
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Setup Timing
    start_time = time.time()

    # Run main script
    main(sys.argv[1:])

    # Print Execution time
    print("--- Total Execution Time: %s seconds ---" % (time.time() - start_time))




