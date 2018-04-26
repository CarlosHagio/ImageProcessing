import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import time
import subprocess
import argparse
import re
import glob
import numbers

##import FilterSubwell2 as fs
import FilterSubwell3 as fs
import FilterDrop as fd

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

##GET ARGUMENTS FROM EACH LINE OF --input ARGUMENT FILE
##PATH IS THE PATH OF THE IMAGES
##OUTPUT IS THE OUTPUT FILE - SIMPLE REPORT FOR NOW
##RI IS THE INITIAL INTERNAL RADIUS OF THE CIRCULAR CROWN
##RO IS THE INITIAL EXTERNAL RADIUS OF THE CIRCULAR CROWN
##STEPSIZE IS THE RADIUS INCREMENT FOR SEARCHING BEST CIRCULAR CROWN
##NUMBERSTEPS IS THE MAXIMUM NUMBER OF STEPS TO FIND BEST CIRCULAR CROWN
##THRESHOLD IS THE NUMBER OF WHITE PIXELS TO EVALUEATE MASKED IMAGE
def get_args():
    print("get_args")

    dirName = os.path.dirname(__file__)
    filename = os.path.join(dirName, 'Image Processing/Subwell')
    if not os.path.exists(filename):
        os.makedirs(filename)
    filename = os.path.join(dirName, 'Image Processing/Drop')
    if not os.path.exists(filename):
        os.makedirs(filename)

    parser = argparse.ArgumentParser(description = 'Image Processing')
    parser.add_argument('--input', action = 'store', dest = 'input', required = True)

    args = parser.parse_args()

    name_inputFile = args.input
    file_input = open(name_inputFile, "r")
    lines_input = file_input.readlines()
    file_input.close()

    path = lines_input[0] = (lines_input[0].replace('path = ', '')).rstrip('\n')
    name_outputFile = (lines_input[1].replace('output = ', '')).rstrip('\n')
    file_output = open(name_outputFile + ".txt", "w")

    initial_ri = int((lines_input[2].replace('ri = ', '')).rstrip('\n'))
    initial_ro = int((lines_input[3].replace('ro = ', '')).rstrip('\n'))
    size_step = int((lines_input[4].replace('stepsize = ', '')).rstrip('\n'))
    number_steps = int((lines_input[5].replace('numbersteps = ', '')).rstrip('\n'))
    threshold1 = int((lines_input[6].replace('threshold1 = ', '')).rstrip('\n'))
    threshold2 = int((lines_input[7].replace('threshold2 = ', '')).rstrip('\n'))
    
    delta = number_steps*size_step
    file_output.write("ri = %d-%d\r\nro = %d-%d\r\n\nstep size = %d\r\nnumber of steps = %d\r\/\
white counter threshold = %d\r\nshift threshold = %d\r\n\n"%(initial_ri, initial_ri + delta, initial_ro,
                                     initial_ro + delta, size_step, number_steps, threshold1, threshold2))

    return(path,initial_ri,initial_ro, size_step, number_steps, threshold1, threshold2)

if __name__ =="__main__":
    numbers = re.compile(r'(\d+)')
    subwellList = []
    dropList = []
##    MAKE SUBWELL FILTERED LIST OF IMAGES
    path,initial_ri,initial_ro, size_step, number_steps, threshold1, threshold2 = get_args()
    for imgfile in sorted(glob.glob(path), key=numericalSort):
        i = int(numbers.findall(imgfile)[1])
        subwell = fs.find_subwell(imgfile, initial_ri,initial_ro, size_step, number_steps, threshold1, threshold2)
        cv2.imwrite("Image Processing/Subwell/Subwell%d.jpg"%i, subwell)
        drop = fd.find_drop(subwell)
        cv2.imwrite("Image Processing/Drop/Drop%d.jpg"%i, drop)
        
##        subwellList.append(subwell)
##    print(len(subwellList))
else:
    print("Importing Teste.py")
