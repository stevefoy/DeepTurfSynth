import os 
import random
import numpy as np
import cv2
import numpy as np
from PIL import Image
import numpy as np
import argparse
import cv2


# Tutorial PyImage 
# https://pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
# import the necessary packages
# Second version 
# https://github.com/chia56028/Color-Transfer-between-Images/blob/master/color_transfer.py# 


def color_transfer(source, target):

    # convert the images from the RGB to L*ab* color space, being
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations using paper proposed factor
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b


    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    
    # return the color transferred image
    return transfer

def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)





def show_image(title, image, width = 300):
    # resize the image to have a constant width, just to
    # make displaying the images take up less screen real
    # estate
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # show the resized image
    cv2.imshow(title, resized)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_file_into_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            return lines
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

def write_list_to_file(file_path, input_list):
    try:
        with open(file_path, 'w') as file:
            for item in input_list:
                file.write(str(item) + '\n')
        print("Successfully.")
    except IOError:
        print(f"Error writing")


    
# Could load this via arg
path_learn =  '/media/freddy/vault/datasets/greenway/all/test/Divot_SimonV2/Images_Processed/image_patches'
file_path = 'FullImageList.txt'  # Replace 'example.txt' with the path to your file
path_learn_images = os.path.join(path_learn, "Images" )

# Image to get colour from
full_images_learn_path = os.path.join(path_learn, file_path)
full_images_list_learn = read_file_into_list(full_images_learn_path)



# Image to apply colour to
path_apply ='/media/freddy/vault/datasets/greenway/m4k_16mmLens/Processed_SlidingWindow'
file_path = 'FullImageList.txt'  # Replace 'example.txt' with the path to your file
path_apply_images = os.path.join(path_apply, "Images" )
path_apply_labels = os.path.join(path_apply, "ImagesLabels" )

full_images_apply_path = os.path.join(path_apply, file_path)
full_images_list_apply = read_file_into_list(full_images_apply_path )


# 20 percent of the data
ratio_count = int(len(full_images_list_apply)*0.2)
selected_items = random.sample(full_images_list_apply, ratio_count)


path_save = '/media/freddy/vault/datasets/greenway/m4k_16mmLens/Processed_SlidingWindow/Images_transfer'

for i, img_file in enumerate(selected_items):
    print("index ", i, " ", img_file)
    file_name = full_images_list_learn[i]
    
    
    image_learn = os.path.join(path_learn_images, file_name)
    image_apply = os.path.join(path_apply_images, img_file)
    source = cv2.imread(image_learn )
    target = cv2.imread(image_apply)
    image_label_apply = os.path.join(path_apply_labels , img_file)
    label = np.asarray(Image.open(image_label_apply), dtype=np.uint8)
    grass_pixel_num = np.count_nonzero(label == 30)
    soil_pixel_num = np.count_nonzero(label ==10)
    total = grass_pixel_num + soil_pixel_num 
    # if grass  high percentage
    perC = grass_pixel_num /total
    print(perC)
    if(perC>0.95):
          
        transfer = color_transfer(source, target)
        image_save= os.path.join(path_save, img_file)
        cv2.imwrite(image_save, transfer)
        if i < 40:
            show_image("Source", source)
            show_image("Target", target)
            show_image("Transfer", transfer)
            
            cv2.waitKey(0)
    
    
    
    
    

    
    
    
    
    
        


    
    
    
    
    
    # load the images
    
    
    

    # transfer the color distribution from the source image
    # to the target image
    

    
    
    

    # show the images and wait for a key press
    
    
    
    