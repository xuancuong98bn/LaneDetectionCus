#!/usr/bin/env python
import logging
# The above line is essential for ros python file

import sys, time
import numpy as np

import detectLane

# import opencv
import cv2 as cv

detect = detectLane.Detected_Lane()

def main(args):
    
    img = cv.imread("image_500.jpg")
    if img is None:
        sys.exit("Could not read the image.")
    scale_percent = 1920/img.shape[1] * 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    cv.imshow("Display window", resized)

    detect.detect(resized)

    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite("out1.png", img)
    cv.destroyAllWindows()

        

if __name__ == '__main__':
    try:
        main(sys.argv)
    except Exception as e:
        logging.exception(e)
        pass