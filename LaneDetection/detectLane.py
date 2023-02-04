#!/usr/bin/env python

# The above line is essential for ros python file
# Use equation axx+bx+c
import sys, time
import numpy as np
import matplotlib.pyplot as plt

# import ros libraries
import Line
import Utility as Utils
import CameraCalibration as CamCali

# import opencv
import cv2

class Detected_Lane:
    # constructor
    def __init__(self):
        self.leftLine = Line.Line()
        self.rightLine = Line.Line()

    def calc_line_fits(self, img):
        ### Settings
        # Number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 20
        # Set minimum number of pixels found to recenter window
        minpix = 10

        height, width = img.shape
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.uint(histogram.shape[0]/2)

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.uint(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        min_x_left = width
        max_x_left = 0
        min_x_right = width
        max_x_right = 0

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            if leftx_current < min_x_left:
                min_x_left = leftx_current
            win_xleft_high = leftx_current + margin
            if leftx_current > max_x_left:
                max_x_left = leftx_current
            win_xright_low = rightx_current - margin
            if rightx_current < min_x_right:
                min_x_right = rightx_current
            win_xright_high = rightx_current + margin
            if rightx_current > max_x_right:
                max_x_right = rightx_current
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 1)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 1)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.uint(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.uint(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # a = np.column_stack((leftx, lefty))
        # b = np.column_stack((rightx, righty))
        # Fit a second order polynomial to each
        try:
            left_fit = np.polyfit(leftx, lefty, 2)
        except:
            left_fit = None

        try:
            right_fit = np.polyfit(rightx, righty, 2)
        except:
            right_fit = None

        try:
            left_fit_e = np.polyfit(leftx, lefty, 1)
        except:
            left_fit_e = None

        try:
            right_fit_e = np.polyfit(rightx, righty, 1)
        except:
            right_fit_e = None

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

        return left_fit, right_fit, left_fit_e, right_fit_e, out_img, leftx, rightx

    def __get_left_line__(self):
        best_fit_px, best_fit_m = self.leftLine.__get_line__()
        return best_fit_px
    
    def __get_right_line__(self):
        best_fit_px, best_fit_m = self.rightLine.__get_line__()
        return best_fit_px
    
    def slice_road(self, image, SKY_LINE = 555, HALF_ROAD = 110, CAR_LINE = 0): #14 555 110 // 21 700 110 // 8 600 110
        height, width = image.shape[:2]
        IMAGE_H = height - CAR_LINE
        IMAGE_W = width

        SRC_W1 = IMAGE_W / 2 - HALF_ROAD / 2
        SRC_W2 = IMAGE_W / 2 + HALF_ROAD / 2

        IMAGE_W1_TF = IMAGE_W / 2 - HALF_ROAD
        IMAGE_W2_TF = IMAGE_W / 2 + HALF_ROAD

        src = np.float32([[SRC_W1, SKY_LINE], [SRC_W2, SKY_LINE], [0, IMAGE_H], [width, IMAGE_H]])
        dst = np.float32([[IMAGE_W1_TF, 0], [IMAGE_W2_TF, 0], [IMAGE_W1_TF, IMAGE_H], [IMAGE_W2_TF, IMAGE_H]])
        image = image[0:(height - CAR_LINE), 0:IMAGE_W]  # Apply np slicing for ROI crop

        warper_img = Utils.warper(image, src, dst)
        warper_img = warper_img[0:(height - CAR_LINE), int(IMAGE_W / 2 - HALF_ROAD*2):int(IMAGE_W / 2 + HALF_ROAD*2)]

        cv2.imshow('warper_img', warper_img)
        # unwarper_img = Utils.unwarp(warper_img, src, dst)
        # cv2.imshow('unwarper_img', unwarper_img)


        return warper_img

    # callback function for processing image
    def preprocess(self, image):
        # calibration = CamCali.CameraCalibration('camera_cal', 9, 6)
        # cam = calibration.undistort(image)
        warper_img = self.slice_road(image)

        line_normal = Utils.binary_HSV(warper_img)
        # cv2.imshow('line_normal', line_normal)

        line_shadow = Utils.shadow_HSV(warper_img)
        #cv2.imshow('line_shadow', line_shadow)

        result_img = cv2.bitwise_or(line_normal, line_shadow)
        #cv2.imshow('result_img', result_img)

        canny_img = Utils.run_canny(warper_img)
        # cv2.imshow('canny_img', canny_img)

        test_img = cv2.bitwise_or(canny_img, result_img)
        cv2.imshow('test_img', test_img)
        return test_img

    def detect(self, origin_img, result_img):
        left_fit, right_fit, left_fit_e, right_fit_e, out_img, leftx, rightx = self.calc_line_fits(result_img)

        self.leftLine.__add_new_fit__(left_fit, leftx)
        self.rightLine.__add_new_fit__(right_fit, rightx)

        quadratic_img = np.zeros_like(out_img)
        quadratic_img = Utils.draw_quadratic(quadratic_img, self.leftLine.__get_line__(), self.leftLine.__get_limitx__())
        quadratic_img = Utils.draw_quadratic(quadratic_img, self.rightLine.__get_line__(), self.rightLine.__get_limitx__(), color=(255, 0, 0))

        equation_img = np.zeros_like(out_img)
        equation_img = Utils.draw_equation(equation_img, left_fit_e, (0, origin_img.shape[1]))
        equation_img = Utils.draw_equation(equation_img, right_fit_e, (0, origin_img.shape[1]), color=(255, 0, 0))

        # canvas = Utils.draw_equation(canvas, mid, (0, 1024), (0, 255, 0))
        return quadratic_img, equation_img, out_img #(max(leftx), min(rightx))

    def resize_prepro(self, image):
        if image is not None:
            scale_percent = 1920 / image.shape[1]  # percent of original size
            width = int(image.shape[1] * scale_percent)
            height = int(image.shape[0] * scale_percent)
            dim = (width, height)
            # resize image
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            return resized

    def process(self, frame):
        if frame is not None:
            resized = self.resize_prepro(frame)
            Utils.imshow("Display window", resized, (1/2, 1/2))

            pre_img = self.preprocess(resized)
            quadratic_img, equation_img, out_img = self.detect(resized, pre_img)

            Utils.imshow("Processed window 1", quadratic_img, (3/4, 3/4))
            # Utils.imshow("Processed window 2", equation_img, (1,1/3))
            Utils.imshow("Processed window 3", out_img, (3/4, 3/4))

    def save_preprocess(self, image, folder):
        # calibration = CamCali.CameraCalibration('camera_cal', 9, 6)
        # cam = calibration.undistort(image)
        warper_img = self.slice_road(image)
        cv2.imwrite('./' + folder + '/' + 'warper_img' + '.jpg', warper_img)
        line_normal = Utils.binary_HSV(warper_img)
        # cv2.imshow('line_normal', line_normal)
        cv2.imwrite('./' + folder + '/' + 'line_normal' + '.jpg', line_normal)
        line_shadow = Utils.shadow_HSV(warper_img)
        #cv2.imshow('line_shadow', line_shadow)
        cv2.imwrite('./' + folder + '/' + 'line_shadow' + '.jpg', line_shadow)
        result_img = line_normal + line_shadow
        #cv2.imshow('result_img', result_img)
        cv2.imwrite('./' + folder + '/' + 'result_img' + '.jpg', result_img)
        canny_img = Utils.run_canny(warper_img)
        # cv2.imshow('canny_img', canny_img)
        cv2.imwrite('./' + folder + '/' + 'canny_img' + '.jpg', canny_img)
        test_img = cv2.bitwise_or(canny_img, result_img)
        cv2.imwrite('./' + folder + '/' + 'mix_img' + '.jpg', test_img)
        return test_img

    def save_process(self, frame, folder):
        if frame is not None:
            resized = self.resize_prepro(frame)
            cv2.imwrite('./' + folder + '/' + 'resized' + '.jpg', resized)

            pre_img = self.save_preprocess(resized, folder)
            quadratic_img, equation_img, out_img = self.detect(resized, pre_img)
            cv2.imwrite('./' + folder + '/' + 'quadratic_img' + '.jpg', quadratic_img)
            cv2.imwrite('./' + folder + '/' + 'out_img' + '.jpg', out_img)