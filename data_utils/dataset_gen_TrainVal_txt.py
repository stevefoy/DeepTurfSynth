'''
Conversion tool -  which can convert grassclover into the a size to quicker dataset loading, 
After this is completed you need to "base_size": null, the base trainer will then no resize the image

Created on 30 Jan 2023

@author: stephen foy
'''

import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
from utils_dir import *
from PIL import Image
import torch
from torchvision import transforms
import numpy 
import csv
import os
import random

def find_jpg_filenames(path_to_dir, suffix=".jpg"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def find_png_filenames(path_to_dir, suffix=".png"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# Log file
LogFileName = "logfile.csv"
# initializing the titles and rows list
fields = ['FileName', 'w','h','grass=0', 'white clover=1', 'red clover=2', 'weeds=3', 'soil=4' , 'other colver=5'] 
rows = []


# ToDo Profile timing
def count_unique(img):
    uniqueColors = set()
    
    w, h, c = img.shape
    for x in range(w):
        for y in range(h):
            pixel = img[x, y]  
            lst = str(int(pixel[0]))+" "+str(int(pixel[1]))+" "+str(int(pixel[2]))
            uniqueColors.add(lst) 
    print("unique BGR=",uniqueColors )


def np_count_unique(img):
    np_img = np.array(img)
    # Arrange all pixels into a tall column of 3 RGB values and find unique rows (colours)
    colours, counts = np.unique(np_img.reshape(-1,3), axis=0, return_counts=1)
    print("np unique BGR=",colours)


def main():
    
    parser = argparse.ArgumentParser(
        description='tool for processing a folder of images')
    parser.add_argument('-i', '--inputDir', action=readable_dir,
                        help='input directory to process')


    args  = parser.parse_args()
    root_image_path = args.inputDir 
    label_path = os.path.join(root_image_path, 'ImageLabels')
    image_path = os.path.join(root_image_path, 'Images')

    label_paths =  find_png_filenames(label_path)
    image_paths =  find_png_filenames(image_path)

    print(len(image_paths))
    print(len(label_paths))


    val_sublist = []
    for i in range(0, 1000):
        random_element = random.choice(image_paths)
        val_sublist.append(random_element)
        image_paths.remove(random_element)
    
    val_file = os.path.join(root_image_path,'ImagesValidation.txt')

    with open( val_file, 'w') as fp:
        for item in val_sublist: 
            # write each item on a new line
            fp.write("%s\n" % item)
    fp.close()

    train_file = os.path.join(root_image_path,'ImagesTrain.txt')
    with open(train_file, 'w') as fp:
        for item in image_paths:
            # write each item on a new line
            fp.write("%s\n" % item)
    fp.close()
    
    print('--------------Done----------------------------------')
            


    # for fileName in tqdm(images):
    #    pass

    



if __name__ == '__main__':
    main()

    os.system("shutdown /s /t 1")