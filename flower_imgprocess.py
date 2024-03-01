from __future__ import division
import cv2
# to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

def show(flower_image):
    plt.figure(figsize=(10, 10))

    # Show image, with nearest neighbour interpolation
    plt.imshow(flower_image, interpolation='nearest')



def overlay_mask(mask, flower_image):
    #since the image is RGB and mask is Grayscale
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # calculates the weightes sum of two arrays. in our case image arrays
    # optional depth value set to 0 no need
    img = cv2.addWeighted(rgb_mask, 0.5, istock_image, 0.5, 0)
    return img    



def flower_contour(flower_image):
    flower_image = flower_image.copy()
    _, contours, hierarchy = cv2.findContours(flower_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(flower_image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask



def circle_contour(flower_image, contour): #depends on the final output needing, this equation is unset

    return




def find_stock(flower_image):
    flower_image = cv2.cvtColor(flower_image,cv2.COLOR_BGR2RGB)
    max_dimension = max(flower_image.shape)

`` unclear max window size
    scale = 700 / max_dimension

#rescale ratios
    flower_image = cv2.resize(flower_image, None, fx=scale, fy=scale)
    flower_image1 = flower_image.copy()
    cv2.imwrite('image.jpg', flower_image)

#blur image using Gaussian, transfering RGB to HSV to eliminate shining light
`values unsure`
    flower_imageblur = cv2.GaussianBlur(flower_image, (7, 7), 0)
    cv2.imwrite('image_blur.jpg', flower_imageblur)

    # just want to focus on color, segmentation
    image_blur_hsv = cv2.cvtColor(flower_imageblur, cv2.COLOR_RGB2HSV)
    cv2.imwrite('image_blur_hsv.jpg', image_blur_hsv)

    # HSV hue range value, For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
`values unsure`

    min_white = np.array([0, 100, 80])
    max_white = np.array([10, 256, 256])

    mask1 = cv2.inRange(image_blur_hsv, min_white, max_white)
    cv2.imwrite('mask1.jpg', mask1)

    min_white2 = np.array([170, 100, 80])
    max_white2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_white2, max_white2)
    cv2.imwrite('mask2.jpg', mask2)

    #combining two  blured masks
    mask = mask1 + mask2
    cv2.imwrite('mask.jpg', mask)

    #clean up, morph image. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('mask_closed.jpg', mask_closed)

    # erosion followed by dilation. useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('mask_clean.jpg', mask_clean)

    #overlay cleaned mask on image
    overlay = overlay_mask(mask_clean, flower_image)
    cv2.imwrite('overlay.jpg', overlay)

    #circle_contour?
    circle = circle_contour(overlay, flower_contour)

    return bgr