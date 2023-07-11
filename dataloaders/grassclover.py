from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import numpy.ma as ma
import os
from PIL import Image
from tqdm import tqdm

ignore_label = 255



ID_TO_TRAINID_OLD = {-1: ignore_label, 0: 0, 1: 1, 2: 2, 3: 3,
                 4: 1, 5: 1, 6: 3, 7: 3, 8: 3, 9: 1, 10: 1,
                 11: 1, 12: 1, 13: 1, 14: 1}


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


## Info for epoch 20 ## 
# val_loss       : 0.57903
# Pixel_Accuracy : 0.763
# Mean_IoU       : 0.5600000023841858
# Class_IoU      : {0: 0.558, 1: 0.727, 2: 0.491, 3: 0.463}
# Class Soil     : {soil: 0.56, clover: 0.73,grass: 0.491 , weeds: 0.463}

## Info for epoch 40 ## 
# Pixel_Accuracy : 0.655
# Mean_IoU       : 0.4560000002384186
# Class_IoU      : {grass: 0.48378417, white clover: 0.27844968, red clover: 0.5372614, weeds: 0.50919664, soil: 0.5881317, clover other: 0.3373777}

# colour augmentation 
# val_loss       : 0.58007
# Pixel_Accuracy : 0.747
# Mean_IoU       : 0.5799999833106995
# Class_IoU      : {0: 0.5421305, 1: 0.5886883, 2: 0.70850825, 3: 0.52477425, 4: 0.5689178, 5: 0.54952604}


# Submissions for the semantic segmentation challenge must be in the form of indexed png-images, 
# where: grass=0, white clover=1, red clover=2, weeds=3 soil=4 
# where minie: grass=0, white clover=1, red clover=2, weeds=3 soil=4, clover other = 5

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
GrassClover_ID_TO_TRAINID = {-1: ignore_label, 
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
                14: 5,
                16: 4} # temp fix for testing real data annotation , ID 16 object class mapped to soil 

#"weight":  [3.9170224385759638, 3.8018232098154527, 3.589300823087543, 11.357607000838737, 50.4983497918439, 11.02286545044973],
# soil:5
data_live = {'class_names':  [ 'grass', 'white clover', 'red clover', 'weeds', 'soil', 'clover other'],
        'class_ids': [0, 1, 2, 3, 4, 5]
        }
class GrassCloverDataset(BaseDataSet):
    def __init__(self,mode='fine', **kwargs):
        
        if False:
            self.num_classes = 4
            self.id_to_trainId = ID_TO_TRAINID_OLD
        else:
            self.num_classes = 6
            self.id_to_trainId = GrassClover_ID_TO_TRAINID 

        self.mode = mode      
        self.split=kwargs["split"]
        self.palette = palette.GrassClover_palette

        super(GrassCloverDataset, self).__init__(**kwargs)


    def _set_files(self):
        label_path = os.path.join(self.root, 'ImageLabels')
        image_path = os.path.join(self.root, 'Images')

        image_paths, label_paths = [], []

        ImagesNames = []
        if self.split == 'train':
            print("Train mode")
            fileReaderPath = os.path.join(self.root, 'ImagesTrain.txt')         
            ImagesNames = [line.strip() for line in open(fileReaderPath, 'r')]

        elif self.split == "val":
            fileReaderPath = os.path.join(self.root, 'ImagesValidation.txt')
            ImagesNames = [line.strip() for line in open(fileReaderPath, 'r')]

        else:
            image_paths.extend(sorted(glob(os.path.join(image_path, '*.png'))))
            label_paths.extend(sorted(glob(os.path.join(label_path, '*.png'))))

        # From the file of name , make the label and the image path
        label_paths = [os.path.join(label_path, x) for x in ImagesNames]
        # image_paths = [os.path.join(image_path, x.replace(".png",".jpg")) for x in ImagesNames]
        image_paths = [os.path.join(image_path, x) for x in ImagesNames]

        check_flag = 0
        check_flag_labels = 0
        for i in image_paths:
            if os.path.exists(i):
                check_flag += 1
        for i in label_paths:
            if os.path.exists(i):
                check_flag_labels += 1

        print("check_flag", check_flag)
        print("check_flag", check_flag_labels)
        if check_flag_labels != check_flag:
            raise Exception("Please force an end here now")

        self.files = list(zip(image_paths, label_paths))


        # Once off thing to calculate the weights
        #self.compute_class_weights()
        #self.calcualte_mean() # weight=config['weight']


    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(
                Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        label_remapped = np.full(label.shape, 255, dtype=np.uint8)
        
        #print("Unique IDS before: ", np.unique(label, return_counts=True))
        # Remapping here
        if self.map_label:
            for k, v in self.id_to_trainId.items():
                label_remapped[label == k] = v
            #print("Unique IDS after: ", np.unique(label_remapped, return_counts=True))
            return image, label_remapped, image_id
        
        else:
            return image, label, image_id

        
    
        



    # Ref version mod from here: https://github.com/fregu856/deeplabv3/blob/master/utils/preprocess_data.py
    def compute_class_weights(self):
        ################################################################################
        # compute the class weigths:
        ################################################################################
        print ("computing class weights")

        trainId_to_count = {}
        for trainId in range(self.num_classes):
            trainId_to_count[trainId] = 0

        maxCount= 0 
        # get the total number of pixels in all train label_imgs that are of each object class:
        #for step, label_img_path in enumerate(train_label_img_paths):
        for step, (image_data, label_data) in tqdm(enumerate(self.files)):
            # print(image_data,":", label_data)
            image_data, label_img, image_id = self._load_data(step)

            for trainId in range(self.num_classes):
                # count how many pixels in label_img which are of object class trainId:
                trainId_mask = np.equal(label_img, trainId)
                trainId_count = np.sum(trainId_mask)

                # add to the total count:
                trainId_to_count[trainId] += trainId_count
            
            maxCount +=1
            if maxCount==100:
                break

        # compute the class weights according to the ENet paper:
        class_weights = []
        maxCount= 0 
        total_count = sum(trainId_to_count.values())
        for trainId, count in trainId_to_count.items():
            trainId_prob = float(count)/float(total_count)
            trainId_weight = 1/np.log(1.02 + trainId_prob)
            class_weights.append(trainId_weight)
            maxCount +=1
            if maxCount==100:
                break

        print (class_weights)
        #[49.859577140450305, 48.208871284054766, 2.037260085442715, 19.528667068658073, 41.367888421769656, 3.190874152837387]
        #with open("./class_weights.pkl", "wb") as file:
        #    pickle.dump(class_weights, file, protocol=2)
        raise Exception("Please force an end here now")

    #Ref code : https://androidkt.com/calculate-mean-and-std-for-the-pytorch-image-dataset/
    def calcualte_mean(self):

        mean = np.array([0.,0.,0.])
        stdTemp = np.array([0.,0.,0.])
        std = np.array([0.,0.,0.])

        maxCount = 0
        for index, (image_data, label_data) in tqdm(enumerate(self.files)):
            # print(image_data,":", label_data)
            image_data, label, image_id = self._load_data(index)
            im = image_data.astype(float) / 255.

            for j in range(3):
                mean[j] += np.mean(im[:,:,j])
            maxCount=index

            if maxCount==500:
                break

        
        mean = (mean/maxCount)
        print("Mean ", mean)

        maxCount = 0
        for index, (image_data, label_data) in tqdm(enumerate(self.files)):
            # print(image_data,":", label_data)
            image_data, label, image_id = self._load_data(index)
            im = image_data.astype(float) / 255.
            maxCount=index
            for j in range(3):
                stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])


            if maxCount==1000:
                break
        
        std = np.sqrt(stdTemp/maxCount)
        print("STD", std)
        raise Exception("Please end here now")




class GrassClover(BaseDataLoader):
    def __init__(self, data_dir, batch_size=1, split=False,
                 crop_size=None, map_label=True, base_size=None, scale=True,
                 num_workers=1, mode='fine', val=False, shuffle=False,
                 flip=False, rotate=False, blur=False, augment=False,
                 val_split=False, return_id=False):

        # Calculated values
        self.MEAN = [0.36036222, 0.42922836, 0.21562772] 
        self.STD = [0.13999425, 0.17160976, 0.10774312]

        # if encoder is froozen to ImageNet the RGB Image feature are seen as ImageNet 
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'map_label': map_label,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = GrassCloverDataset(mode=mode, **kwargs)
        super(GrassClover, self).__init__(
                self.dataset, batch_size, shuffle, num_workers, val_split)
