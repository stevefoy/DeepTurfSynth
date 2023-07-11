from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import numpy.ma as ma
import os
from PIL import Image
from tqdm import tqdm

ignore_label = 255

Trufgrass_ID_TO_TRAINID_Binary = { -1: ignore_label,
                0:1,  # Soil to Soil
                16: 1, # Objects to Soil
                2:0 # follage grass to 1
                } 

TrufgrassID_MappedTo_GrassClover = { -1: ignore_label,
                0:4,  # Soil to Soil
                16: 4, # Objects to Soil
                2:0 # follage to grass
                } 

Trufgrass_ID_TO_TRAINID = {-1: ignore_label, 
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
# Quick check on real data
#"weight":  [1.475434602990349, 50.4983497918439, 50.4983497918439, 50.4983497918439, 14.673449004400611, 50.4983497918439],
# soil:5
# Mean  [0.52782338 0.6290964  0.30889719]
# STD [0.17453718 0.16221044 0.20381077]

data_live = {'class_names':  [ 'grass', 'white clover', 'red clover', 'weeds', 'soil', 'clover other', 'object'],
        'class_ids': [0, 1, 2, 3, 4, 5, 6]
        }
data_live_binary = {'class_names':  [ 'grass', 'soil'],
        'class_ids': [0, 1]
        }

class TurfGrassDataset(BaseDataSet):
    def __init__(self,mode='fine', **kwargs):
        
        if False:
            self.num_classes = 4
            self.id_to_trainId = ID_TO_TRAINID_OLD
        else:
            self.num_classes = 6
            self.id_to_trainId = TrufgrassID_MappedTo_GrassClover
            self.palette = palette.turfGrass_palette
        # Trufgrass_ID_TO_TRAINID_Binary 
        # self.num_classes = 2
        # self.id_to_trainId = Trufgrass_ID_TO_TRAINID_Binary
        # self.palette = palette.turfGrass_palette_2Class

        self.mode = mode      
        self.split=kwargs["split"]
        

        super(TurfGrassDataset, self).__init__(**kwargs)


    def _set_files(self):
        label_path = os.path.join(self.root, 'ImagesLabels')
        image_path = os.path.join(self.root, 'Images')

        image_paths, label_paths = [], []

        ImagesNames = []
        if self.split == 'train':
            print("Train mode")
            fileReaderPath = os.path.join(self.root, 'ImagesTrain.txt')         
            ImagesNames = [line.strip() for line in open(fileReaderPath, 'r')]

        elif self.split == "val":
            # fileReaderPath = os.path.join(self.root, 'ImagesValidation.txt')
            print("Validation on full manual  annotated dataset")
            fileReaderPath = os.path.join(self.root, 'ImagesValidation.txt')
            ImagesNames = [line.strip() for line in open(fileReaderPath, 'r')]

        else:
            image_paths.extend(sorted(glob(os.path.join(image_path, '*.png'))))
            label_paths.extend(sorted(glob(os.path.join(label_path, '*.png'))))

        # From the file of name , make the label and the image path
        print("label_path", label_path)
        print("image_path", image_path)
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
        # self.compute_class_weights()
        # self.calcualte_mean() # weight=config['weight']


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
            if maxCount==1000:
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

            #if maxCount==4000:
            #    break

        
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


            #if maxCount==4000:
            #    break
        
        std = np.sqrt(stdTemp/maxCount)
        print("STD", std)
        raise Exception("Please end here now")




class TurfGrass(BaseDataLoader):
    def __init__(self, data_dir, batch_size=1, split=False,
                 crop_size=None, map_label=True, base_size=None, scale=True,
                 num_workers=1, mode='fine', val=False, shuffle=False,
                 flip=False, rotate=False, blur=False, augment=False,
                 val_split=False, return_id=False):



        # If you train with a MEAN the deploy, 
        # Anothe idea if you freezee backbone on ImageNet, then should deploy on  ImageNet Mean SD 
        # Use this GrassClover if trained with this 
        self.MEAN = [0.36036222, 0.42922836, 0.21562772] 
        self.STD = [0.13999425, 0.17160976, 0.10774312]
        
        # Real data Annotation - calculated on the 4000 Sliding windows 400x400 dataset 
        # self.MEAN = [0.49601265, 0.62375998, 0.36324892] 
        # self.STD = [0.17580946, 0.1584281,  0.18407977]

        #Full Greeway dataset  if trained with this
        self.MEAN = [0.24020349, 0.29680853, 0.19672048]
        self.STD = [0.18339201, 0.17367418, 0.17249223]



        # Real data Annotation +AWB - calculated on the 4000 Sliding windows 400x400 dataset 
        # self.MEAN = [0.50652866, 0.61484343, 0.35649721] 
        # self.STD = [0.19723097, 0.15700872, 0.19747502]]

        # Low value test 
        #self.MEAN = [0.1, 0.1, 0.1] 
        #self.STD = [0.1, 0.1,  0.1]

        # Deployt with None
        #self.MEAN = [None, None, None]
       # self.STD = [None, None, None]


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

        self.dataset = TurfGrassDataset(mode=mode, **kwargs)
        super(TurfGrass, self).__init__(
                self.dataset, batch_size, shuffle, num_workers, val_split)
