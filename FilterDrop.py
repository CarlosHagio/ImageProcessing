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

def polar2cartesian(r,theta):
    z = r*np.exp(1j*theta)
    x = np.real(z)
    y = np.imag(z)
    return x,y

def find_circle_pixels(img_subwell):
    _, thresh = cv2.threshold(img_subwell, 10, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh, 10,20)
    _,filtered, hierarchy =  cv2.findContours(thresh,1,2)
    filtered = sorted(filtered, key = lambda img: cv2.contourArea(img), reverse= True)

    return filtered

def find_drop(img_subwell):
    s = time.time()
    
##    cv2.imshow("", thresh)
##    cv2.waitKey()
##USAR TEXTURA - OFFICE/TASK
  ##em cima do CAnny ou da img_subwell direto?
##preciso achar o círculo externo? Como?
    blur = cv2.medianBlur(img_subwell, 7)
##    blur = cv2.GaussianBlur(img_subwell, (7,7), 0)
##    blur = cv2.bilateralFilter(img_subwell,9,5,5)
##    canny = cv2.Canny(blur, 10,20)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
##    cv2.imshow('',thresh)
##    cv2.waitKey()
    
##    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
##    filtered = cv2.findNonZero(canny)
##    cv2.imshow('', canny)
##    cv2.waitKey()
##

##    img_drop=cv2.drawContours(canny, filtered, 1, (255,255,255), 10)
    
    pixels_circle = find_circle_pixels (img_subwell)

##    print("\nnumber of dots: %d"%(len(filtered)))
    (x,y),radius = cv2.minEnclosingCircle(pixels_circle[0])
    center = (int(x), int(y))
    radius = int(radius)

    img_drop = img_subwell.copy()
##    img_drop = cv2.ellipse(img_subwell, center, (radius-5,radius-5), 0, 0, 45, (255,255,255), 10)
##    img_drop = cv2.circle(img_subwell, center, radius, (255,255,255), 10)
##    img_drop = cv2.drawContours(img_subwell, pixels_circle, 0, (255,255,255), -1)
##    img_drop = canny

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
##            cv2.imshow('', img_drop)
            cv2.waitKey()
        loop_radius -= loop_step_radius
        loop_number_steps_theta -= loop_number_steps_theta_decay
        if(loop_number_steps_theta == 0):
            break
        loop_step_theta = 2*np.pi/loop_number_steps_theta
        loop_theta = 0
        
##    cv2.imshow('', img_drop)
##    cv2.waitKey()
    
    f = time.time()
    print("drop time: %f\n" %(f-s))
##    img_drop = thresh
    return img_drop            
if __name__ =="__main__":
##  TEST FOR ONE IMAGE
    path = "/mnt/c/Users/carlos.hagio/Desktop/OpenCV/OpenCV/RankerPython/Image Processing/Subwell/Subwell21.jpg"
    img = cv2.imread(path,0)
    find_drop(img)
##    find_subwell(path, 300, 400, 30, 20, 50000,35000) 
else:
    print("Importing FilterDrop.py")
