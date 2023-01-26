import cv2
import numpy as np

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0] + 90)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)  # keep same size as input image
    return warped


def unwarp(img, src, dst):
    # Compute and apply inverse perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size)
    return unwarped

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def binary_HSV(img):
    minThreshold = (0, 0, 150)
    maxThreshold = (179, 255, 255)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    out = cv2.inRange(hsv_img, minThreshold, maxThreshold)
    kernel = np.ones((3, 3), np.uint8)
    out1 = cv2.erode(out, kernel, iterations=1)
    out2 = cv2.dilate(out1, kernel, iterations=2)
    #cv2.imshow('out', out)
    #cv2.imshow('out2', out2)
    return out2

def shadow_HSV(img):
    minShadowTh = (90, 43, 36)
    maxShadowTh = (120, 81, 171)

    minLaneInShadow = (90, 43, 97)
    maxLaneInShadow = (120, 80, 171)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);

    shadowMask = cv2.inRange(imgHSV, minShadowTh, maxShadowTh)

    shadow = cv2.bitwise_and(img,img, mask= shadowMask)

    shadowHSV = cv2.cvtColor(shadow, cv2.COLOR_BGR2HSV);

    out = cv2.inRange(shadowHSV, minLaneInShadow, maxLaneInShadow)

    kernel = np.ones((3, 3), np.uint8)
    out1 = cv2.erode(out, kernel, iterations=1)
    out2 = cv2.dilate(out1, kernel, iterations=2)

    return out2

def run_canny(img, kernel_size=5, low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply a Gaussian Blur
    gausImage = gaussian_blur(gray, kernel_size)

    # Run the canny edge detection
    cannyImage = cv2.Canny(gausImage, low_thresh, high_thresh)

    return cannyImage

def kirsch_filter(img, kernel_size=5):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gausImage = gaussian_blur(gray, kernel_size)
    kernel = []
    kernel.append(np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]))
    kernel.append(np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))
    kernel.append(np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))
    kernel.append(np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]))
    kernel.append(np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
    kernel.append(np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
    kernel.append(np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))
    kernel.append(np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]))
    img_filter = []
    for i in range(8):
        img_filter.append(cv2.filter2D(gausImage, -1, kernel[i]))
    img_out = img_filter[0]
    for i in range(1, 7):
        img_out = img_out + img_filter[i]
    return img_out

