'''
Conversion tool -  which can convert blender renders to image mask
Created on 30 March 2022

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

def find_png_filenames(path_to_dir, suffix=".png"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

# TODO: load from json file: Note opencv format BGR 
# Note: This is where you can change the mapping, BGR to MASK, 
mappingBGR_16b= {(50712,1,  4708):50,  # sky
                (51564,31017,36202):10,  # ground
                ( 1 , 37985, 52902): 20, # object
                (1, 1, 45589) : 30, # flat
                (45589 ,50086, 50086) :40, # Construction
                (45589, 37027, 45589 ) : 50,  # Nature
                (49346, 37027, 51912): 60,  # Weeds
                (34249, 48531, 34249): 70,  # Clover
                (50959, 41263, 42934): 80,  # Boundary
                (1, 37985, 45544): 90,  # Rubish
                (1, 37985, 34124): 100,   #person
                (50414, 50712, 20200):110, #Ego
                (1, 48992, 50712) :120, #Animal
                ( 36455,15805,19734):130, # Fence
                (32781, 45544, 52902):140,   # Bunker
                (20570,26176, 37126) :150# Damage
                }    
mappingBGR_8b= {
                (201,121,141):10,  # ground
                (0, 0, 178) : 30, # flat
                } 


def rgb8bit_mask_to_mono8bit(mask):
        mask = torch.from_numpy(np.array(mask))

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.empty(h, w, dtype=torch.uint8)

        for k in mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))   
                 
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(mapping[k], dtype=torch.uint8)
       
        return mask_out

def rgb8bit_Render_to_mask(mask):
    w, h, c = mask.shape

    image = np.zeros((w, h, 1), dtype="uint8")
    # ref helper: https://answers.opencv.org/question/97416/replace-a-range-of-colors-with-a-specific-color-in-python/ 

    for k in mappingBGR_8b:
        image[np.where((mask==k).all(axis = 2))] = mappingBGR_8b[k]

    return image

def rgb16bit_Render_to_mask(mask):
    w, h, c = mask.shape

    image = np.zeros((w, h, 1), dtype="uint8")
    # ref helper: https://answers.opencv.org/question/97416/replace-a-range-of-colors-with-a-specific-color-in-python/ 

    for k in mappingBGR_16b:
        image[np.where((mask==k).all(axis = 2))] = mappingBGR_16b[k]

    return image

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
    parser.add_argument('-f', '--fileinput',  action=readable_file,
                        help='input file')
    parser.add_argument('-o', '--outputDir', action=writable_dir,
                        help='output directory to process')
    
    args  = parser.parse_args()
    root_image_path = args.inputDir
    results_path=  args.outputDir
    images =  find_png_filenames(root_image_path)

    # inter for debug
    for fileName in tqdm(images):

        # format file names correct
        rawFile = os.path.join(args.inputDir, fileName)
        DLFile = os.path.join(args.outputDir, fileName)
        print(rawFile)
        
        if os.path.isfile(rawFile):

            # BGR
            img16 = cv2.imread(rawFile, cv2.IMREAD_UNCHANGED)
            # Convert from 16bit depth to 8 bit depth direct
            #img8 = (img16/256).astype('uint8')
            #count_unique(img8)

            np_count_unique(img16)
            # mask_rgb = rgb16bit_Render_to_mask(img16)
            mask_rgb = rgb8bit_Render_to_mask(img16)            
            image = transforms.ToPILImage()(mask_rgb)
            image.save(DLFile)


if __name__ == '__main__':
    main()