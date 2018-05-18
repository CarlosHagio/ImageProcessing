import os
import cv2
import FilterSubwell3 as fs
import FilterDrop as fd

import argparse
from glob import glob

def find_beginning(save_dir, root_dir):

    beginning_path = root_dir

    plate_list = glob(save_dir+'/*')

    plate_index = len(plate_list)-1
    
    if plate_index >= 0:

        plate_string = max(plate_list, key = os.path.getctime)
        
        batch_list = glob(plate_string+'/*')
        print(batch_list)
        batch_index = len(batch_list)-1
        if batch_index >= 0:

            batch_string = max(batch_list, key = os.path.getctime)
            batch_index = batch_list.index(batch_string)
            well_list = glob(batch_string+'/*')
            print(well_list)
            well_index = len(well_list)-1
            if well_index >= 0:
                well_string = max(well_list, key = os.path.getctime)
                well_index = well_list.index(well_string)
            else:
                well_index = 0
        else:
            batch_index = 0

    else:
        plate_string = os.listdir(root_dir)[0]
        plate_list = os.listdir(root_dir)
        plate_index = 0
        
        batch_string = os.listdir(root_dir+'/'+plate_string)[0]
        batch_list = os.listdir(root_dir+'/'+plate_string)
        batch_index = 0
        
        well_string = os.listdir(root_dir+'/'+plate_string+'/'+batch_string)[0]
        well_list = os.listdir(root_dir+'/'+plate_string+'/'+batch_string)
        well_index = 0

    return plate_index, batch_index, well_index

def check_folder():
    return
def get_args():
    print("get_args")


    dirName = os.path.dirname(__file__)
    filename = os.path.join(dirName, 'Image Processing2')
    if not os.path.exists(filename):
        os.makedirs(filename)
##    dirName = os.path.dirname(__file__)
##    filename = os.path.join(dirName, 'Image Processing2/Subwell')
##    if not os.path.exists(filename):
##        os.makedirs(filename)
##    filename = os.path.join(dirName, 'Image Processing2/Drop')
##    if not os.path.exists(filename):
##        os.makedirs(filename)

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

    path_training = lines_input[8] = (lines_input[8].replace('path_training = ', '')).rstrip('\n')
    
    delta = number_steps*size_step
    file_output.write("ri = %d-%d\r\nro = %d-%d\r\n\nstep size = %d\r\nnumber of steps = %d\r\/\
white counter threshold = %d\r\nshift threshold = %d\r\n\n"%(initial_ri, initial_ri + delta, initial_ro,
                                     initial_ro + delta, size_step, number_steps, threshold1, threshold2))

    return(path,initial_ri,initial_ro, size_step, number_steps, threshold1, threshold2,path_training)             

if __name__ == '__main__':
    _,initial_ri,initial_ro, size_step, number_steps, threshold1, threshold2,path_training = get_args()
    
##    plate_index, batch_index, well_index, plate_list, batch_list, well_list = find_beginning('Image Processing2', 'ImagensSSH')
    plate_index, batch_index, well_index = find_beginning('Image Processing2', 'ImagensSSH')

##    print(plate_index, batch_index, well_index, plate_list, batch_list, well_list)
    print(plate_index, batch_index, well_index)

    plate_list = os.listdir('ImagensSSH')
    plate_list_size = len(plate_list)
    for plate_count in range(plate_index, plate_list_size):
        batch_list = os.listdir('ImagensSSH/'+plate_list[plate_count])
        batch_list_size = len(batch_list)
        for batch_count in range(batch_index, batch_list_size):
            well_list = os.listdir('ImagensSSH/'+plate_list[plate_count]+'/'+batch_list[batch_count])
            print(well_list)
            well_list_size = len(well_list)
            for well_count in range(well_index, well_list_size):
                profiles = os.listdir('ImagensSSH/'+plate_list[plate_count]+'/'+batch_list[batch_count]+'/'+well_list[well_count])
                for profile in profiles:
                    if profile == 'profileID_1':                
                        files = os.listdir('ImagensSSH/'+plate_list[plate_count]+'/'+batch_list[batch_count]+'/'+well_list[well_count]+'/'+profile)
## 
                        dirName = os.path.dirname(__file__)
                        folderName = os.path.join(dirName, 'Image Processing2/'+plate_list[plate_count]+'/'+batch_list[batch_count]+'/'+well_list[well_count]+'/'+profile)
                        if not os.path.exists(folderName):
##                            print("folder: ",folderName)
                            os.makedirs(folderName)

                        for image in files:
                            if image.endswith('_ef.jpg'):
                                image_path = 'ImagensSSH/'+plate_list[plate_count]+'/'+batch_list[batch_count]+'/'+well_list[well_count]+'/'+profile+'/'+image
##                                image_save_path_root = image_path.replace('/', '$')
                                image_save_path_root = image_path.replace('ImagensSSH', 'Image Processing2')
                                print(image_path)
                                
                                subwell = fs.find_subwell(image_path, initial_ri,initial_ro, size_step, number_steps, threshold1, threshold2)
                                if subwell is not None:
                                    image_save_path = image_path.replace('.jpg','subwell.jpg')
                                    cv2.imwrite(image_save_path, subwell)
                                    image_save_path_root = image_save_path_root.replace('.jpg','subwell.jpg')
                                    cv2.imwrite(image_save_path_root, subwell)
##                                    print('root: ',image_save_path_root)

                                    drop = fd.find_drop(subwell)
##                                    image_save_path = image_path+'drop.jpg'
                                    image_save_path = image_path.replace('.jpg','drop.jpg')
                                    cv2.imwrite(image_save_path, drop)
                                    image_save_path_root = image_save_path_root.replace('subwell', 'drop')
                                    cv2.imwrite(image_save_path_root, drop)
##                                    print('root: ',image_save_path_root)
            well_index = 0
        batch_index = 0

else:
    print('Importing Dir.py')
