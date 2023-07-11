'''
Tool -  Analysis of data for training augmentation 

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

def find_jpg_filenames(path_to_dir, suffix=".jpg"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def find_png_filenames(path_to_dir, suffix=".png"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

# TODO: load from json file: Note opencv format 
ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label, 
                0:4, 
                1: 5, 
                2: 0, 
                3: 3,
                4: 1,
                5: 2, 
                6: 3, 
                7: 3, 
                8: 3, 
                9: 1, 
                10: 1,
                11: 2, 
                12: 2, 
                13: 5, 
                14: 5}

num_classes = 6

# Log file
LogFileName = "logfile.csv"
# initializing the titles and rows list
fields = ['FileName', 'w','h','grass=0', 'white clover=1', 'red clover=2', 'weeds=3', 'soil=4' , 'other colver=5'] 
rows = []


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
    label_path = os.path.join(root_image_path, 'ImageLabels')
    image_path = os.path.join(root_image_path, 'Images')

    out_path=  args.outputDir
    label_output_path = os.path.join(out_path, 'ImageLabels')
    image_output_path = os.path.join(out_path, 'Images')
    
    if not os.path.exists( label_output_path):
        os.makedirs( label_output_path)

    if not os.path.exists( image_output_path):
        os.makedirs( image_output_path)

    images =  find_jpg_filenames(image_path)
    print(len(images))
    # inter for debug
    trainId_to_count = {}
    for trainId in range(num_classes):
        trainId_to_count[trainId] = 0
    

    image_output_pathLog = os.path.join(out_path, LogFileName)
    print("logfile location :",  image_output_pathLog)
    csvlogFile = open(image_output_pathLog , 'w') 
    write_outfile = csv.writer(csvlogFile )
    write_outfile.writerow(fields)

    for fileName in tqdm(images):

        # format file names correct
        label_full_path = os.path.join(label_path, str(fileName).replace(".jpg",".png"))
        image_full_path = os.path.join(image_path, fileName)
        


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
        
    
    
    # compute the class weights according to the ENet paper:
    class_weights = []
    maxCount= 0 
    total_count = sum(trainId_to_count.values())
    for trainId, count in trainId_to_count.items():
        trainId_prob = float(count)/float(total_count)
        trainId_weight = 1/np.log(1.02 + trainId_prob)
        class_weights.append(trainId_weight)
        maxCount +=1
    
    print (class_weights)
    write_outfile.writerow(class_weights)
    csvlogFile.close()


if __name__ == '__main__':
    main()

    # os.system("shutdown /s /t 1")