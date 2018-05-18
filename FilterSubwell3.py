import numpy as np
import cv2
import re
from functools import reduce
import math

#MINOR SUBWELL MASK SHIFT

def shift_mask(mask, white, white_counter, xm, ym,ri,ro,rstep):
    avrg = reduce(lambda x,y:x+y, white)/white_counter
    avrg = np.uint16(np.around(avrg))

    white_x = avrg[0][0]
    white_y = avrg[0][1]

    delta_x = white_x.astype(np.int16) - xm.astype(np.int16) 
    delta_y = white_y.astype(np.int16) - ym.astype(np.int16)

    delta_tan = delta_y/(delta_x+0.00000001)
    d = ro - math.hypot(delta_x, delta_y)
    ri = ri + np.uint16(np.around(d/10))
    ro = ro + np.uint16(np.around(d/10))
    
    shift_x = np.uint16(np.around(d/(10*(math.sqrt(1+delta_tan**2)))))
    shift_y = np.uint16(np.around(d/(10*(math.sqrt(1+(1/(0.00000001+delta_tan**2)))))))
    
    if delta_x > 0:
        shift_x = -1*shift_x
    if delta_y > 0:
        shift_y = -1*shift_y

    xm = xm + shift_x.astype(np.int16)
    ym = ym + shift_y.astype(np.int16)

    return xm,ym,ri,ro

#FIND BEST SUBWELL MASK AND APPLY OVER THE IMAGE ON IMGFILE
#RETURN MASKED IMG
#initial_ri: initial internal radius
#initial_ro: initial outer radius
#size_step: radius increment in each step
#number_steps: maximum number of steps to find mask
#white_thresh: number of whites pixels threshold to evalue masked image
def find_subwell(imgfile, initial_ri, initial_ro, size_step, number_steps, white_thresh, shift_thresh):
    #treating parameters
    numbers = re.compile(r'(\d+)')
    img = cv2.imread(imgfile, 0)
    ri = initial_ri
    ro = initial_ro
    delta_radius = ro - ri
    rm = np.uint16(np.around((ri+ro)/2))
    rstep = size_step
    i = int(numbers.findall(imgfile)[1])

    counter_aux = 0

    #image dimensions and center of the circles
    shape = img.shape
    ym = shape[0]
    xm = shape[1]
    xm = np.uint16(np.around(xm/2))
    ym = np.uint16(np.around(ym/2))

    ##blur and thresh to accentuate desired contour (experimental parameters values)
    blur = cv2.medianBlur(img, 57)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,111,5)
    cv2.subtract(255, thresh, thresh)

##    cv2.imshow("",img)
##    cv2.waitKey()
##    cv2.imshow("",blur)
##    cv2.waitKey()
##    cv2.imshow("",thresh)
##    cv2.waitKey()


    while(True):        
        mask = np.zeros(shape, np.uint8)
        mask = cv2.circle(mask, (xm, ym), rm, (255,255,255), delta_radius) #circular crown mask
        mask_thresh = cv2.bitwise_and(thresh, thresh, mask = mask) #apply over contour image
        
        white = cv2.findNonZero(mask_thresh) #evaluate by the number of whites pixels
        if white is not None:
            break
        else:
            rm = rm+1
        if rm > shape[1]:
            print("FilterSubwell3 error: low image quality")
            return None

##    cv2.imshow("",mask_thresh)
##    cv2.waitKey()

    white_counter = len(white)
    counter_aux = white_counter
    delta = 0
    
    while(True):
        for x in range(0, number_steps): #loop to find best subwell mask
            if (white_counter > white_thresh):
                mask_return = cv2.circle(np.zeros(shape, np.uint8), (xm, ym), rm, (255,255,255), -1)
                masked = cv2.bitwise_and(img, img, mask = mask_return)
                break
            elif (white_counter > shift_thresh) and (white_counter<white_thresh): #shift crown mask and make it a little bigger
                xm,ym,ri,ro = shift_mask(mask, white, white_counter, xm,ym,ri,ro,rstep)
                rm = np.uint16(np.around((ri+ro)/2))
            else:   #try bigger circular crown mask
                ri = ri + rstep
                ro = ro + rstep
                rm = np.uint16(np.around((ri+ro)/2))

            mask = np.zeros(shape, np.uint8)
            mask = cv2.circle(mask, (xm, ym), rm, (255,255,255), delta_radius) #circular crown mask
            mask_thresh = cv2.bitwise_and(thresh, thresh, mask = mask) #apply over contour image

            white = cv2.findNonZero(mask_thresh) #evaluate by the number of whites pixels
            aux_counter = white_counter
            if white is not None:
                white_counter = len(white)
            else:
                white_counter = 0
            
            delta = white_counter - aux_counter

        if white_counter <= white_thresh:
            white_thresh = white_thresh - 50
            numbers = re.compile(r'(\d+)')
            ri = initial_ri
            ro = initial_ro
            delta_radius = ro - ri
            rm = np.uint16(np.around((ri+ro)/2))
            ym = shape[0]
            xm = shape[1]
            xm = np.uint16(np.around(xm/2))
            ym = np.uint16(np.around(ym/2))
        elif white_counter > white_thresh:
            break
    return masked
##    cv2.imshow("",masked)
##    cv2.waitKey()       
if __name__ =="__main__":
##  TEST FOR ONE IMAGE
    path = "/mnt/c/Users/carlos.hagio/Desktop/OpenCV/OpenCV/RankerPython/Media/vis2/image vis79.jpg"
    find_subwell(path, 300, 400, 30, 20, 50000,35000) 
else:
    print("Importing FilterSubwell.py")
