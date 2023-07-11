import argparse
import scipy
import os
import numpy as np
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import PIL



def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def sliding_predict_avg(model, image, num_classes, flip=True, crop_size=400):
    image_size = image.shape

    tile_size = (int(crop_size), int(crop_size))
    overlap = 0 

    stride = ceil(tile_size[0] * (1 - overlap))
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    
    tile_counter = 0
    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)

            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions

def get_files(folder_path, fileList_text='ImagesValidation.txt'):

    label_path = os.path.join(folder_path, 'ImagesLabels')
    image_path = os.path.join(folder_path, 'Images')

    image_paths, label_paths = [], []
    ImagesNames = []

    fileReaderPath = os.path.join(folder_path, fileList_text)         
    ImagesNames = [line.strip() for line in open(fileReaderPath, 'r')]


    # From the file of name , make the label and the image path
    label_paths = [os.path.join(label_path, x) for x in ImagesNames]
    # image_paths = [os.path.join(image_path, x.replace(".png",".jpg")) for x in ImagesNames]
    image_paths = [os.path.join(image_path, x) for x in ImagesNames]

    check_flag = 0
    check_flag_labels = 0
    for i in image_paths:
        if os.path.exists(i):
            check_flag += 1
        else:
            print("missing", i)
    for i in label_paths:
        if os.path.exists(i):
            check_flag_labels += 1
        else:
            print("missing", i)

    print("check_flag", check_flag, "check_flag_labels", check_flag_labels)
    if check_flag_labels != check_flag:
        raise Exception("Please force an end here now, issue with the images missing, Label:",check_flag_labels,"image", check_flag )

    return list(zip(image_paths, label_paths))


def sliding_windows_save(filename, img_path, label_path, output_path, crop_size=400):

    mask = Image.open(label_path).convert("L")
    image = Image.open(img_path).convert('RGB')  
    
    # img2np_arr = np.array(image) 
    # np_arr2img = Image.fromarray(img2np_arr)
    image = np.array(image) 
    mask = np.array(mask) 

    label_path_save = os.path.join(output_path, 'ImagesLabels')
    image_path_save = os.path.join(output_path, 'Images')
    
    image_size = image.shape
    tile_size = (int(crop_size), int(crop_size))
    overlap = 0 

    stride = ceil(tile_size[0] * (1 - overlap))
    num_rows = int(ceil((image_size[0] - tile_size[0]) / stride) ) # don't add + 1, we drop padded region
    num_cols = int(ceil((image_size[1] - tile_size[1]) / stride) ) # don't add + 1, we drop padded region

    tile_counter = 0
    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[1])
            y_max = min(y_min + tile_size[0], image_size[0])

            img = image[y_min:y_max, x_min:x_max, :]
            img_mask = mask[y_min:y_max, x_min:x_max]

            # Note we drop the padded region
            # padded_img = pad_image(img, tile_size)
            

            save_filename = filename+str("_tile_{}.png")
            save_filename = save_filename.format(tile_counter)

            save_label_path = os.path.join(label_path_save, save_filename )
            save_png_path = os.path.join(image_path_save, save_filename )

            print("save_png_path ", save_png_path)
            print("save_label_path ", save_label_path )
            
            img = PIL.Image.fromarray(img)
            img.save(save_png_path)
            
            mask_img = PIL.Image.fromarray(img_mask)
            mask_img.save(save_label_path)

            tile_counter += 1

def sliding_predict(model, image, num_classes, flip=True, crop_size=400):
    image_size = image.shape

    tile_size = (int(crop_size), int(crop_size))
    overlap = 0 

    stride = ceil(tile_size[0] * (1 - overlap))
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    layer_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    
    tile_counter = 0
    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)

            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            
            layer_predictions[:, y_min:y_max, x_min:x_max] = predictions.data.cpu().numpy().squeeze(0)
            # draw outer boarder to debug put layer 5 to a high value , bad method but ok  
            #layer_predictions[5, y_min:y_max, x_max-6:x_max] = layer_predictions[:, y_min:y_max, x_max-6:x_max].max() +0.2
            #layer_predictions[5, y_max-6:y_max, x_min:x_max] = layer_predictions[:, y_max-6:y_max, x_min:x_max].max() +0.2

    
    return layer_predictions

def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions





def main():
    args = parse_arguments()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    if args.datadir :
        
        imagelist_filename='ImagesValidation.txt'
        # print("get_files(root_path, imagelist_filename)", get_files(args.datadir , imagelist_filename))
        images_paths = get_files(args.datadir, imagelist_filename)
        root_path_save =  os.path.join(args.datadir, 'Processed_SlidingWindow')
        # save paths

        label_path_save = os.path.join(root_path_save, 'ImagesLabels')
        image_path_save = os.path.join(root_path_save, 'Images')


    tbar = tqdm(images_paths, ncols=100)
    for img_label_index in tbar:
        
        if args.datadir:
            print("index:",img_label_index)
            img_path, label_path = img_label_index
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            image_filename = os.path.basename(img_path).split('.')[0] 

            print("image_filename", img_path)
            sliding_windows_save(image_filename , img_path, label_path, root_path_save, crop_size=400)


          

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')

    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-d', '--datadir', default=None, type=str,
                        help='input root path for image list')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
