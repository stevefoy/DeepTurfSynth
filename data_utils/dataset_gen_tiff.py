'''
Tool to generate a single .tiff file for annotation in gimp, 
Layer for each of the main object types in an image
Created on 14 Feb 2023

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
from PIL import Image, ImageSequence,  TiffTags, TiffImagePlugin

import torch
from torchvision import transforms
import numpy 
import csv
import os
from subprocess import call
from sys import argv
from os.path import abspath


def find_jpg_filenames(path_to_dir, suffix=".jpg"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def find_JPEG_filenames(path_to_dir, suffix=".JPEG"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def find_png_filenames(path_to_dir, suffix=".png"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def find_tiff_filenames(path_to_dir, suffix=".tiff"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

# TODO: load from json file: Note opencv format 
ignore_label = 255
# class_label_id;class_name
# 0;soil
# 1;clover
# 2;grass
# 3;weeds
# 4;white_clover
# 5;red_clover
# 6;dandelion 
# 7;shepherds_purse
# 8;thistle
# 9;white_clover_flower
# 10;white_clover_leaf
# 11;red_clover_flower
# 12;red_clover_leaf
# 13;unknown_clover_leaf
# 14;unknown_clover_flower 

#Note Grass could be mapped all black and soil will over write it
# mapping must match orginal dataset
mapping_grass = {(0, 0, 0, 255): 2}  # 0 = Grass


mapping_soil = {(0, 0, 0, 255): 0}  # 4 = Soil
           # On tiff layer (0, 0, 0, 0) : 255  is Not Soil Mask
            

num_classes = 6

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
    parser.add_argument('-f', '--fileinput',  action=readable_file,
                        help='input file')
    parser.add_argument('-o', '--outputDir', action=writable_dir,
                        help='output directory to process')
    
    args  = parser.parse_args()
    root_image_path = args.inputDir 
    image_path = os.path.join(root_image_path, '')
    out_path=  args.outputDir
    label_output_path = os.path.join(out_path, 'ImageLabels')
    image_output_path = os.path.join(out_path, 'Images_tiffs')

    if not os.path.exists( image_output_path):
        os.makedirs( image_output_path)

    images =   find_jpg_filenames(root_image_path)


    tag_ids = {info.name: info.value for info in TiffTags.TAGS_V2.values()}
    ImageJMetaData = tag_ids['PageName']

    for fileName in tqdm(images):
        # format file names correct
        image_full_path = os.path.join(image_path, fileName) 
        im = Image.open(image_full_path )     
        w, h = im.size
 
        image_pages = []
        info = TiffImagePlugin.ImageFileDirectory()
        info[ImageJMetaData] = fileName
        im.encoderinfo = {'tiffinfo': info}
        image_pages.append(im)
        
        ID_TO_TRAINID = { -1: ignore_label,
                        0:0,  # sky
                        10:5,  # ground Dirt
                        20: 4, # object Golfball
                        30 : 2, # flat grass
                        40 :3, # Construction Stone
                        50 : 1,  # Nature and Clover
                        60: 1,  # Weeds
                        70: 1,  # Clover
                        80: 3,  # Boundary
                        90: 4,  # Rubish
                        100: 4,   #person
                        110:0, #Ego
                        120:4, #Animal
                        130:3, # Fence
                        140:3,   # Bunker
                        150:0# water
                        } 

        # Create pages for the tiff file
        page_names_list = ["grass" ,"soil" ,"weeds" , "boundary", "objects" ]
        for page_txt in page_names_list:
            
            if page_txt == 'soil':
                fullPathSoftmaxL4 = os.path.join(root_image_path, fileName.replace(".JPEG","")+'_Soil.png')
                
            
            image_page = np.full((h,w,4) , [0,0,0,255], dtype=np.uint8)
            image_data = Image.fromarray(image_page)
            info = TiffImagePlugin.ImageFileDirectory()
            info[ImageJMetaData] = page_txt
            image_data.encoderinfo = {'tiffinfo': info}
            image_pages.append(image_data)
        
        fileName = fileName.replace(".JPEG","")+str('.tiff')
        image_output_path_out = os.path.join(image_output_path, fileName)
        with open(image_output_path_out, "w+b") as fp:
            with TiffImagePlugin.AppendingTiffWriter(fp) as tf:
                for page in image_pages:
                    page.encoderconfig = ()
                    TiffImagePlugin._save(page, tf, image_output_path_out)
                    tf.newFrame()


    

if __name__ == '__main__':
    main()
