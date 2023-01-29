import cv2
import numpy as np

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
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

def houghlines_trans(img, rho=1, theta=np.pi/180, threshold = 50, minLineLength=5, maxLineGap=250):
    # Run the canny edge detection
    cannyImage = run_canny(img)

    lines = cv2.HoughLinesP(cannyImage, rho, theta, threshold, minLineLength, maxLineGap)
    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line[0]
            cv2.line(cannyImage, (x1, y1), (x2, y2), (255, 0, 0), 30)
            # cv2.waitKey(0)
            # if (y1 > 200 or y2 > 200):  # Filter out the lines in the top of the image
            #     cv2.line(cannyImage, (x1, y1), (x2, y2), (255, 0, 0), 3)

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

def draw_quadratic(img, abc, limit, color=(0, 0, 255)):
    canvas = img.copy()
    height, width = canvas.shape[:2]
    a, b, c = abc
    for x in np.arange(limit[0], limit[1], 0.5):
        y = int(a * x**2 + b * x + c)
        if 0 <= y < height:
            canvas = cv2.circle(canvas,(int(x), y), 5, color, -1)
    return canvas

def draw_equation(img, ab, limit, color=(0, 0, 255)):
    canvas = img.copy()
    height, width = canvas.shape[:2]
    a, b = ab
    for x in  np.arange(limit[0], limit[1], 0.5):
        y = int(a * x + b)
        if 0 <= y < height:
            canvas = cv2.circle(canvas,(int(x), y), 5, color, -1)
    return canvas

def imshow(name, img, size):
    dim = (int(img.shape[0] * size[0]), int(img.shape[1] * size[1]))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(name, resized)
