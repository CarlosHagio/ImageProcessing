import numpy as np
import cv2
from LBP import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import matplotlib.pyplot as plt
import matplotlib.image as mping

import re
from functools import reduce
import math
import time
from random import randint


#FIND BEST SUBWELL MASK AND APPLY OVER THE IMAGE ON IMGFILE
#RETURN MASKED IMG
#initial_ri: initial internal radius
#initial_ro: initial outer radius
#size_step: radius increment in each step
#number_steps: maximum number of steps to find mask
#white_thresh: number of whites pixels threshold to evalue masked image

def random_color():
    B = randint(0,255)
    G = randint(0,255)
    R = randint(0,255)
    
    return B,G,R

## simple coordinates conversion
def polar2cartesian(r,theta):
    z = r*np.exp(1j*theta)
    x = np.real(z)
    y = np.imag(z)
    return x,y    

## normalize to 0/255 gray scale
def normalize_gray(circle_roi_image):
    gray_points = cv2.findNonZero(circle_roi_image)
    gray_mask = cv2.inRange(circle_roi_image, 1,255)

    circle_roi_image = cv2.normalize(circle_roi_image, circle_roi_image, 0, 255, cv2.NORM_MINMAX, mask = gray_mask)
    
    return circle_roi_image

## crop circular ROI from img
def crop_circle(img, center, radius):
    xm= np.uint16(np.around(center[0]/2))
    ym= np.uint16(np.around(center[1]/2))

    mask = np.zeros((2*radius+1,2*radius+1), np.uint8)
    mask = cv2.circle(mask, (radius, radius), radius, (255,255,255), -1)

    roi = img[(center[1]-radius):(center[1]+radius+1), (center[0]-radius):(center[0]+radius+1)]

##  ignore roi if it is out of subwell limits
    if(roi.shape != mask.shape):
        return None

    cropped_circle = cv2.bitwise_and(roi, roi, mask = mask)
##    cropped_circle = normalize_gray(cropped_circle)
    
    return cropped_circle

## classify img using willpower
def classify_roi(img_roi):
    roi2 = normalize_gray(img_roi)
    blur = cv2.GaussianBlur(roi2, (13,13),0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
##    _, thresh = cv2.threshold(img_roi, 127, 255, cv2.THRESH_BINARY)
    thresh = 255-cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,2)
    canny = cv2.Canny(blur,10,20)
##    canny = cv2.Canny(thresh, 5,10)
##    _,filtered, hierarchy =  cv2.findContours(thresh,1,2)
##    filtered = sorted(filtered, key = lambda img: cv2.arcLength(img, False), reverse= True)

##    print(filtered)
    minLineLengthVal = 20
    maxLineGapVal = 1
    lines = cv2.HoughLinesP(canny,1,np.pi/180,15,minLineLength=minLineLengthVal, maxLineGap=maxLineGapVal)
##    roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
##    print(lines[0])
    if lines is None:
        return 'Clean', None
    else:
        return 'NotClean', lines

## check if line found on roi is close enough to the center of it
def check_distance(x1,y1,x2,y2, radius):
    xc = radius+1
    yc = radius+1
    num = abs(xc*(y2-y1)-yc*(x2-x1)+x2*y1-x1*y2)
    den = math.sqrt((x2-x1)**2+(y2-y1)**2)

    distance = int(num/den)

    if abs(distance-radius) < 5:
        return 0
    else:
        return 1

## compute distance between two roi's
def compute_roi_distance(circle_test, circle_compare):
##    threshold_radius = circle_test[1]
    x1,y1 = circle_test[0]
    x2,y2 = circle_compare[0]

    distance = math.sqrt((x2-x1)**2+(y2-y1)**2)
    
    return int(distance)

## compute distance between roi's table
def compute_roi_distance_table(circles_list_src):
    if circles_list_src is not None:
        n = len(circles_list_src)
        table = np.zeros((n, n))
        for x in range(1,n):
            for y in range(0,x):
                table[y][x] = compute_roi_distance(circles_list_src[x], circles_list_src[y])
                table[x][y] = table[y][x]
##        print(table)
        return table

    else:
        return None    
def compare_clusters(cluster_x,cluster_y):
    nx = len(cluster_x)
    ny = len(cluster_y)

    for x in cluster_x:
        for y in cluster_y:
            if x == y:
##                print("\n\n\n\nmerge")
##                print(cluster_x)
##                print(x)
##                print(cluster_y)
##                print(y)
                return 'merge'            

    return 'dont merge'

def organize_cluster_list(cluster_list):
    n = len(cluster_list)
##    print(cluster_list)
    organized_clusters = cluster_list.copy()
    x = 0
    while(x < n-1):
##    for x in range(0,n-1):
        y = x+1
        while(y < n):
##        for y in range(x+1,n):
##            check = compare_clusters(cluster_list[x],cluster_list[y])
            check = compare_clusters(organized_clusters[x], organized_clusters[y])
            if check == 'merge':
##                print(cluster_list)
##                print(organized_clusters)
##                print(cluster_list[x])
                cluster_x = organized_clusters[x]
                cluster_y = organized_clusters[y]
                new = set(organized_clusters[x]+ organized_clusters[y]) 
##                print(cluster_x, cluster_y, new)
##                print(organized_clusters)
                organized_clusters[x] = list(new)
##                cluster_x = organized_clusters[x]
##                cluster_y = organized_clusters[y]
##                organized_clusters.remove(cluster_x)            
                organized_clusters.remove(cluster_y)
##                organized_clusters.append(list(new))
##                organized_clusters[x] = [new]
                n = len(organized_clusters)
##            elif check == 'dont merge':
##                print('dont merge')
##                print(organized_clusters)
                if x > 0:
                    x = x-1
                if y > 0:
                    y = y-1
            y = y+1
        x = x+1
##    print("desorganizado")
##    for cluster in cluster_list:
##        print(cluster)
##    print("organizado")
##    for cluster in organized_clusters:
##        print(cluster)
    return organized_clusters

## clusterize roi's by distance between
def clusterize_roi(circles):
##    print(circles)
    table = compute_roi_distance_table(circles)
##    list_clusters = []
##    n = len(circles_list_src)
    distance = 2*circles[0][1]

    cluster_list= []
    n = table.shape[0]
    
    for line in range(0,n):
        cluster = []
        if(all((element >= distance or element == 0) for element in table[line])):
##            print("table: ",table[line])
            cluster = [(circles[line][0],circles[line][1])]
            cluster_list = cluster_list+[cluster]

    for line in range(0,n):
        cluster = []
        if not(all(element > distance for element in table[line])):
            for x in range(line+1, n):
                if table[line][x] != 0 and table[line][x] <= distance:
##                    print("\nline: %d   x: %d" %(line, x))
                    cluster = cluster + [(circles[line][0],circles[line][1])]
                    cluster = cluster + [(circles[x][0], circles[x][1])]
                    table[line][x] = 0
                    table[x][line] = 0
                    for y in range(line+1, n):
                        if table[y][x] != 0 and table[y][x] <= distance:
                            cluster = cluster + [(circles[y][0], circles[y][1])]
                            table[y][x] = 0
                            table[x][y] = 0
##                            print("\ny: %d   x: %d" %(y, x))
        if (len(cluster)!= 0):
            cluster = list(set(cluster))
            cluster_list = cluster_list+[cluster]
            cluster = []
    
    cluster_list = organize_cluster_list(cluster_list)
    
##    for line in range(0, n-1):
##        cluster = []
##        if table[line].all() == True:
##            print("alone")
##            print(table[line])
##        for x in range(line+1, n):
##            if table[line][x] < distance and table[line][x] != 0:
####                cluster = cluster+[table[line][x]]
##                cluster = cluster+[circles_list_src[line][0], circles_list_src[x][0]]
##                table[line][x] = 0
##                table[x][line] = 0
##                for y in range(line, n):
##                    if table[y][x] < distance and table[y][x] != 0:
####                        cluster = cluster+[table[y][x]]
##                        cluster = cluster+[circles_list_src[y][0], circles_list_src[x][0]]
##                        table[y][x] = 0
##                        table[x][y] = 0
##
##        list_clusters = list_clusters+[cluster]
        
    return cluster_list

## correct classify_roi eliminating convexities, suposing that drop edges don't have any
def correct_classification(clean_circles, not_clean_circles):
    return

## return subwell circular contour coordinates
def find_circle_pixels(img_subwell):
    _, thresh = cv2.threshold(img_subwell, 10, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh, 5,10)
    _,filtered, hierarchy =  cv2.findContours(thresh,1,2)
    filtered = sorted(filtered, key = lambda img: cv2.contourArea(img), reverse= True)

    return filtered

def find_dif(circles, clusters):
    found = 'not found'
    missing = []
    for circle in circles:
        for cluster in clusters:
            for point in cluster:
                if (point[0][0] == circle[0][0] and point[0][1] == circle[0][1]):
##                    print('found: ',circle, point)
                    found = 'found'
                    break
            if found == 'found':
                break
        if found == 'not found':
            missing = missing+[circle]
##            print('not found: ', circle)
        found = 'not found'
    return missing

## main function, find drop boundary coordinates
##def find_drop(img_subwell, modelSVC, desc):
def find_drop(img_subwell):
    clean_circles = []
    not_clean_circles = []
##    contour_circles = []

    blur = cv2.medianBlur(img_subwell, 7)

##    aux_function(blur)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    pixels_circle = find_circle_pixels (img_subwell)
    
    (x,y),radius = cv2.minEnclosingCircle(pixels_circle[0])
    center = (int(x), int(y))
    radius = int(radius)

    img_drop = cv2.cvtColor(img_subwell.copy(), cv2.COLOR_GRAY2BGR)


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
            x_new,y_new = polar2cartesian(loop_radius, loop_theta)
            center_new = (int(x_new+x), int(y_new+y))
            cropped = crop_circle(img_subwell, center_new, drawing_circle_radius)
            if cropped is not None:
##                roi_class = classify_roi(cropped, modelSVC, desc)
                roi_class, lines = classify_roi(cropped)
                if roi_class == 'Clean':
##                    cv2.circle(img_drop, center_new, int(drawing_circle_radius/3), (0,0,255), 2)
##                    cv2.putText(img_drop, roi_class, (center_new), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255),1)
                    clean_circles.append(center_new)
                elif roi_class == 'NotClean':
##                    cv2.circle(img_drop, center_new, int(drawing_circle_radius), (255,0,0), 2)
##                    for line in lines:
##                        for x1,y1,x2,y2 in line:
##                            if(check_distance(x1,y1,x2,y2, int(drawing_circle_radius)) == 0):
##                                cv2.line(img_drop,(center_new[0]+(x1-int(drawing_circle_radius)-1),center_new[1]+(y1-int(drawing_circle_radius)-1)),((x2-int(drawing_circle_radius)-1)+center_new[0],(y2-int(drawing_circle_radius)-1)+center_new[1]),(0,0,255),1)
##                            else:
##                                cv2.line(img_drop,(center_new[0]+(x1-int(drawing_circle_radius)-1),center_new[1]+(y1-int(drawing_circle_radius)-1)),((x2-int(drawing_circle_radius)-1)+center_new[0],(y2-int(drawing_circle_radius)-1)+center_new[1]),(0,255,0),3)
##                    
##                    cv2.putText(img_drop, roi_class, (center_new), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0,255,0),1)
                    not_clean_circles.append((center_new, drawing_circle_radius))
##                elif roi_class == 'Contour':
##                    cv2.circle(img_drop, center_new, int(drawing_circle_radius/3), (255,0,0), 2)
##                    cv2.putText(img_drop, roi_class, (center_new), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255,0,0),1)
##                    contour_circles.append(center_new)
                    
            loop_theta += loop_step_theta
        loop_radius -= loop_step_radius
        loop_number_steps_theta -= loop_number_steps_theta_decay
        if(loop_number_steps_theta == 0):
            break
        loop_step_theta = 2*np.pi/loop_number_steps_theta
        loop_theta = 0

    cv2.putText(img_drop, str(len(not_clean_circles)), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255),3)
    cluster_test = clusterize_roi(not_clean_circles)

    organized_counter = 0
    for cluster in cluster_test:    
##        print(cluster)
        B,G,R = random_color()
        radius = randint(5,15)
##        print(cluster)
        for point in cluster:
##            print(point[0])
            cv2.circle(img_drop, point[0], point[1], (B,G,R), 2)
            organized_counter = organized_counter+1
    cv2.putText(img_drop, str(organized_counter), (50,750), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255),3)

    if(organized_counter != len(not_clean_circles)):
        missing = find_dif(not_clean_circles, cluster_test)

        for point in missing:
            cv2.circle(img_drop, point[0], 5, (0,0,0), -1)
            print(point)
##        cv2.imshow("",img_drop)
##        cv2.waitKey()
####        print("\n\n\ncircle:")
####        for circle in not_clean_circles:
####            print(circle)
####        print("\n\n\ncluster:")
####        for cluster in cluster_test:
####            print(cluster)

##        print(not_clean_circles)
##        print(cluster_test)
##    print(cluster_test)
##    if(len(not_clean_circles)!=len(cluster_test)):
##        print("erro: %d %d"%(len(not_clean_circles),len(cluster_test)))
####    for cluster in cluster_test:
####        print(cluster_test)
####        for circle in cluster:
####        print(circle[0])
####        cv2.circle(img_drop, cluster[0], cluster[1], (0,0,255), 2)

       
    return img_drop            
if __name__ =="__main__":
##  TEST FOR ONE IMAGE
    path = "/mnt/c/Users/carlos.hagio/Desktop/OpenCV/OpenCV/RankerPython/Image Processing/Subwell/Subwell2.jpg"
    img = cv2.imread(path,0)
    cv2.imshow("",find_drop(img))
    cv2.waitKey()
else:
    print("Importing FilterDrop.py")
