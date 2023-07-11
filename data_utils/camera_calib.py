'''
Camera calibration based on opencv
https://docs.opencv.org/4.x/dd/d92/tutorial_corner_subpixels.html
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
import json 
import matplotlib.pyplot as plt


def find_filenames(path_to_dir, suffix=".png"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def create_target( nCols, nRows, squareSizemm):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    boardPoints = np.zeros((nRows*nCols,3), np.float32)

    boardPoints[:,:2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1,2)*squareSizemm
    #plt.plot(boardPoints)
    #plt.show() 

    return boardPoints

def main():
     
    parser = argparse.ArgumentParser(
        description='tool for calibration a folder of images')
    parser.add_argument('-i', '--inputDir', action=readable_dir,
                        help='input directory to process')
    parser.add_argument('-j', '--json',  action=readable_file,
                        help='json calibration config')
    parser.add_argument('-o', '--outputDir', action=writable_dir,
                        help='output directory to process')
    

    #json_data =  '{ "numRows":7, "numCols":9,"scalePercent":100, "squareDia":20,"calSizeW":1280,"calSizeH":966, "imageType":".png", "calibrationTestImg":"0000.png"}'
    json_data_test1 =  '{  "numCols":9,"numRows":7,"scalePercent":100,"squareDia":20,"calSizeW":1280,"calSizeH":966, "imageType":".jpg", "calibrationTestImg":"t7_9_20mm_780mm_Dz.jpg"}'
    json_data =  '{  "numCols":9,"numRows":7,"scalePercent":100,"squareDia":20,"calSizeW":1280,"calSizeH":966, "imageType":".jpg", "calibrationTestImg":"13.jpg"}'

    cal = json.loads(json_data)

    args  = parser.parse_args()
    input_path = args.inputDir
    if(args.outputDir== None):
        output_path =  args.outputDir
    else:
        output_path =  args.outputDir

    images =  find_filenames(input_path, cal["imageType"])
    print(images)
    boardPoints = create_target(cal["numCols"], cal["numRows"], cal["squareDia"] )

    # world 3d points
    boardPointsTargets = [] 
    # image plane 2d points
    imgCornerDetections = []

    countGoodImage = 0
    width = 0
    height = 0

    # inter for debug
    for fileName in tqdm(images):

        # format file names correct
        rawFile = os.path.join(input_path, fileName)

        outFile = os.path.join(args.outputDir, fileName)
        print(rawFile)
        
        if os.path.isfile(rawFile):
            img = cv2.imread(rawFile)
            scale_percent = cal["scalePercent"] # percent of original size
            width = int(img.shape[1])
            height = int(img.shape[0])

            if scale_percent != 100:
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (height, width)
                print("Set scale", dim)
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
                #img = cv2.resize(img, (cal["calSizeW"], cal["calSizeH"])) 

            gray    = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


            retval, corners = cv2.findChessboardCorners(gray, ( cal["numCols"], cal["numRows"]), cv2.CALIB_CB_FAST_CHECK)

            if retval == True:
                # Calculate the refined corner locations
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
                cornersSubPix = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

                # Append to list world 3d and 2d corners detected
                boardPointsTargets.append(boardPoints)
                imgCornerDetections.append(cornersSubPix)
                
                countGoodImage = countGoodImage + 1
                # Debug corners
                #for i in range(corners.shape[0]):
                #    print(" -- Refined Corner [", i, "]  (", corners[i,0,0], ",", corners[i,0,1], ")")

                img = cv2.drawChessboardCorners(img, ( cal["numCols"], cal["numRows"]), cornersSubPix, retval)
                scale_percent = 25
                width2 = int(img.shape[1] * scale_percent / 100)
                height2 = int(img.shape[0] * scale_percent / 100)
                dim = (width2, height2)
                img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow('imgage',img2)
 
                k = cv2.waitKey(1) & 0xFF
                cv2.destroyAllWindows()
            else:
                print(fileName, " Failed to calibrate")

    if countGoodImage > 10:
        print("Trying to calibrate the cameras")
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(boardPointsTargets, imgCornerDetections, gray.shape[::-1],None, None)
        

        # transform the matrix and distortion coefficients to writable lists
        calData = {'camera_matrix': np.asarray(matrix).tolist(),
                'dist_coeff': np.asarray(distortion).tolist()}

        # Calibration data
        print(" Camera matrix:")
        print(matrix)
        
        print("\n Distortion coefficient:")
        print(distortion)
        
        print("\n Rotation Vectors:")
        print(r_vecs)
        
        print("\n Translation Vectors:")
        print(t_vecs)
        
        # Raspberry pi HQ sensor
        sensor_width= 7.564 
        sensor_height= 5.476
        dim = (width, height)
        # Center coordinates -- in percent, with (0, 0) being image center
        c_x = (matrix[0, 2] *1.0 / width - 0.5) * 100.0
        c_y = (matrix[1, 2] *1.0 / width - 0.5) * 100.0

        # f_x/f_y - if object size is same a distance to object, how much of a
        # frame will it take? in percent
        f_x = matrix[0, 0] * 100.0 / width
        f_y = matrix[1, 1] * 100.0 / width

        fov_x, fov_y, focal_len, principal, aspect = \
            cv2.calibrationMatrixValues(matrix, dim,
                                        sensor_width, sensor_height)
        

        print("Values", fov_x, " ",fov_y," ", focal_len," ", principal," ", aspect )
        
        with open(os.path.join(args.outputDir, "calibration_matrix.yaml"), "w") as f:
            json.dump(calData, f)

        # Test image format file names correct
        rawFile = os.path.join(input_path, cal["calibrationTestImg"])
        print("Test image", rawFile)
        outFile = os.path.join(args.outputDir, cal["calibrationTestImg"])
        
    
        if os.path.isfile(rawFile):
            img = cv2.imread(rawFile)
            
            #img = cv2.resize(img, (cal["calSizeW"], cal["calSizeH"])) 
            

            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(matrix, distortion, (width, height), 1, (width, height))

            # undistort
            mapx,mapy = cv2.initUndistortRectifyMap(matrix, distortion, None, newcameramtx,(width, height),5)
            dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

            # crop the image
            #x,y,w,h = roi
            #dst = dst[y:y+h, x:x+w]
            #print("ROI: ", x, y, w, h)

            cv2.imwrite(outFile, dst);
            img = cv2.resize(img, (cal["calSizeW"], cal["calSizeH"])) 
            cv2.imshow('imgage dst', img)
            k = cv2.waitKey(0) & 0xFF
            dst = cv2.resize(dst, (cal["calSizeW"], cal["calSizeH"])) 
            cv2.imshow('imgage dst', dst)
            k = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

    # Distroy incase CV2 window is hanging



if __name__ == '__main__':
    main()



