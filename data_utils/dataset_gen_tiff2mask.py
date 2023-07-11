'''
Conversion tool -  which can convert gimp xcf annotation into a single mask
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
from PIL import Image, ImageSequence,  TiffTags

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


mapping_soil = {(0, 0, 0, 200): 0}  # 4 = Soil >200
           # On tiff layer (0, 0, 0, 0) : 255  is Not Soil Mask
            
mapping_objects = {(0, 0, 0, 255): 16}  # 6 = objects

num_classes = 6

# Log file
LogFileName = "logfile.csv"
# initializing the titles and rows list
fields = ['FileName', 'w','h','grass=0', 'white clover=1', 'red clover=2', 'weeds=3', 'soil=4' , 'other colver=5', 'objects=6'] 
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
    image_output_path = os.path.join(out_path, 'Images')
    
    if not os.path.exists( label_output_path):
        os.makedirs( label_output_path)

    if not os.path.exists( image_output_path):
        os.makedirs( image_output_path)

    images =   find_tiff_filenames(image_path)
    print(len(images))
    # inter for debug
    '''
    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0
    '''

    #image_output_pathLog = os.path.join(out_path, LogFileName)
    #print("logfile location :",  image_output_pathLog)
    #csvlogFile = open(image_output_pathLog , 'w') 
    #write_outfile = csv.writer(csvlogFile )
    # write_outfile.writerow(fields)

    tag_ids = {info.name: info.value for info in TiffTags.TAGS_V2.values()}



    for fileName in tqdm(images):

        # format file names correct
        image_full_path = os.path.join(image_path, fileName)
        
        #image_full_path_out = os.path.join(image_output_path, fileName)
        print("Image Load:", fileName  )
        im = Image.open(image_full_path )    
        tiffinfo = im.tag_v2    
        
        ImageJMetaData = tag_ids['PageName']
        label  = None
        label_remapped = None
        label_full_path_out = None
        jpeg_txt =  '.jpg'
        for i, page in enumerate(ImageSequence.Iterator(im)):
            
            layerName = page.tag[ImageJMetaData]
            layerName= layerName[0]
            label = np.array(page)
            print("page", layerName,label.shape)
            if jpeg_txt in layerName :
                fileName = fileName.replace(".tiff","")+str('.png')
                fileName_soil = fileName.replace(".tiff","")+str('_soilProb.png')
                print("Setup filename.jpg:", fileName)
                label_full_path_out = os.path.join(label_output_path, fileName)
                print("Full label_full_path_out:", image_output_path)
                image_full_path_out = os.path.join(image_output_path, fileName.replace(".png", ".png"))
                page.save(image_full_path_out )
                #create the black map 
                w, h, c = label.shape
                print("Layer shape ", label.shape)
                # Really visual that image is blank white image
                label_remapped = np.full((w,h,1) , 255, dtype=np.uint8)
                # print(label_remapped.shape)
                
            else:
                #  fileName = fileName.replace(".tiff","")+str('.png')
                # label_full_path_out = os.path.join(label_output_path, fileName)
                if layerName=="grass":
                    print("Grass")    
                    # Remapping here
                    print("grass ",label.shape," ", label_remapped.shape)
                    for k, v in mapping_grass.items():
                        #label_remapped[label == k] = v
                        label_remapped[np.where((label==k).all(axis = 2))] = v

                elif layerName=="soil":
                    print("Soil") 
                    # uniqueBefore = np.unique(label, return_counts=True)
                    # Remapping here
                    if False:
                        soil_prob_channel = label[:,:,3]
                        label_soil_full_path_out = os.path.join(label_output_path, fileName_soil)
                        heatmap_img = cv2.applyColorMap(soil_prob_channel, cv2.COLORMAP_JET)
                        soil_image_label = transforms.ToPILImage()(heatmap_img)
                        soil_image_label.save(label_soil_full_path_out)
                        
                    for k, v in mapping_soil.items():
                        label_remapped[np.where((label>=k).all(axis = 2))] = v
                
                elif layerName=="objects":
                    print("objects") 
                    # uniqueBefore = np.unique(label, return_counts=True)
                    # Remapping here
                    for k, v in mapping_objects.items():
                        label_remapped[np.where((label>=k).all(axis = 2))] = v
                
        # Write to the file here
        image_label = transforms.ToPILImage()(label_remapped)            
        image_label.save(label_full_path_out)


        '''
        if os.path.isfile(label_full_path):

            # print(label_full_path, " ", image_full_path)
            fileName = str(fileName).replace(".jpg",".png")
            label_full_path_out = os.path.join(label_output_path, fileName)
            image_full_path_out = os.path.join(image_output_path, fileName)

            # Images data 
            image = np.asarray(
                    Image.open(image_full_path ).convert('RGB'))
            label = np.asarray(Image.open(label_full_path ), dtype=np.uint8)
            label_remapped = np.full(label.shape, 255, dtype=np.uint8)
            
            # uniqueBefore = np.unique(label, return_counts=True)
            # Remapping here
            for k, v in ID_TO_TRAINID.items():
                label_remapped[label == k] = v

            # uniqueAfter = np.unique(label_remapped, return_counts=True)
            # print(uniqueBefore,":", uniqueAfter  )
            base_size = 800

            if base_size:

                longside = base_size
                h, w, _ = image.shape
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
    
                # dsize
                dsize = (w, h)

                # resize image
                image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
                label_remapped = cv2.resize(label_remapped, dsize, interpolation=cv2.INTER_NEAREST)
            
                maskCountClass = []
                maskCountClass.append(fileName)
                maskCountClass.append(str(w))
                maskCountClass.append(str(h))
                
                for trainId in range(num_classes):
                    # count how many pixels in label_img which are of object class trainId:
                    trainId_mask = np.equal(label_remapped, trainId)
                    trainId_count = np.sum(trainId_mask)


                    # add to the total count, type numpy.int64:
                    trainId_to_count[trainId] += trainId_count
                    #convet to normal form
                    maskCountClass.append(str(trainId_count.item()))
                    
                    
                    

                write_outfile.writerow(maskCountClass)

                # Write to the file here
                image_label = transforms.ToPILImage()(label_remapped)
                image_rgb = transforms.ToPILImage()(image)
                
                image_label.save(label_full_path_out)
                image_rgb.save(image_full_path_out)
                '''
    
    

if __name__ == '__main__':
    main()
