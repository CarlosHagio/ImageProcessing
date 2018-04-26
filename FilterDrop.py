import numpy as np
import cv2
import re
from functools import reduce
import math
import time

#FIND BEST SUBWELL MASK AND APPLY OVER THE IMAGE ON IMGFILE
#RETURN MASKED IMG
#initial_ri: initial internal radius
#initial_ro: initial outer radius
#size_step: radius increment in each step
#number_steps: maximum number of steps to find mask
#white_thresh: number of whites pixels threshold to evalue masked image

## simple coordinates conversion
def polar2cartesian(r,theta):
    z = r*np.exp(1j*theta)
    x = np.real(z)
    y = np.imag(z)
    return x,y

## crop circular ROI from img
def crop_circle(img, center, radius):
    return img_drop

## classify img as clean or not clean
def classify_region(img):
    return 0

## return subwell circular contour coordinates
def find_circle_pixels(img_subwell):
    _, thresh = cv2.threshold(img_subwell, 10, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh, 10,20)
    _,filtered, hierarchy =  cv2.findContours(thresh,1,2)
    filtered = sorted(filtered, key = lambda img: cv2.contourArea(img), reverse= True)

    return filtered

## main function, find drop boundary coordinates
def find_drop(img_subwell):
    s = time.time()
    
    blur = cv2.medianBlur(img_subwell, 7)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    
    pixels_circle = find_circle_pixels (img_subwell)


    (x,y),radius = cv2.minEnclosingCircle(pixels_circle[0])
    center = (int(x), int(y))
    radius = int(radius)

    img_drop = img_subwell.copy()


    loop_number_steps_radius = 20
    loop_number_steps_theta = 80
    loop_number_steps_theta_decay = loop_number_steps_theta/loop_number_steps_radius
    drawing_circle_radius = int(radius/15)
    loop_radius = radius - drawing_circle_radius
    loop_theta = 0
    loop_step_radius = loop_radius/loop_number_steps_radius
    loop_step_theta = 2*np.pi/loop_number_steps_theta

    while loop_radius > 0:
        while loop_theta < 2*np.pi:
            print(polar2cartesian(loop_radius, loop_theta))
            x_new,y_new = polar2cartesian(loop_radius, loop_theta)
            center_new = (int(x_new+x), int(y_new+y))
            cv2.circle(img_drop, center_new, drawing_circle_radius, (255,255,255), 2)
            loop_theta += loop_step_theta

            cv2.waitKey()
        loop_radius -= loop_step_radius
        loop_number_steps_theta -= loop_number_steps_theta_decay
        if(loop_number_steps_theta == 0):
            break
        loop_step_theta = 2*np.pi/loop_number_steps_theta
        loop_theta = 0
        
    
    f = time.time()
    print("drop time: %f\n" %(f-s))

    return img_drop            
if __name__ =="__main__":
##  TEST FOR ONE IMAGE
    path = "/mnt/c/Users/carlos.hagio/Desktop/OpenCV/OpenCV/RankerPython/Image Processing/Subwell/Subwell21.jpg"
    img = cv2.imread(path,0)
    find_drop(img)
else:
    print("Importing FilterDrop.py")
