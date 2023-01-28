#!/usr/bin/env python
import logging
# The above line is essential for ros python file

import sys, time
import numpy as np

import detectLane

# import opencv
import cv2

detect = detectLane.Detected_Lane()

def save(cap):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'h264')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (width,  height))
    # out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (width,  height))

def main(args):
    cap = cv2.VideoCapture('Video14.mp4')
    capture = -1
    if cap.isOpened():
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Exiting ...")
                    break
                capture = capture + 1
                if capture % 2 == 0:
                    detect.process(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.2)
            except:
                pass

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

        

if __name__ == '__main__':
    try:
        main(sys.argv)
    except Exception as e:
        logging.exception(e)
        pass