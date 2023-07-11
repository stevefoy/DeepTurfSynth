#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:47:28 2023

@author: freddy
"""

import os
import cv2

def remove_salt_and_pepper_noise(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Remove salt and pepper noise using median blur
            denoised_image = cv2.medianBlur(image, 5)

            # Write the denoised image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, denoised_image)

    print("Denoising completed. Denoised images saved to", output_folder)

# Example usage
input_folder = '/media/freddy/vault/datasets/greenway/all/test/Divot_SimonV2/new_set/16BitMasks'  # Replace with the path to your input folder
output_folder = '/media/freddy/vault/datasets/greenway/all/test/Divot_SimonV2/new_set/16BitMaskSmooth'  # Replace with the desired output folder path

remove_salt_and_pepper_noise(input_folder, output_folder)
