from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
from torchvision import transforms
from PIL import Image, ImageStat
from tqdm import tqdm

ignore_label = 255

class Stats(ImageStat.Stat):
  def __add__(self, other):
    # add self.h and other.h element-wise
    return Stats(list(np.add(self.h, other.h)))

#                 class_label_id;class_name
# 0

data_live = {'class_names':  [ 'grass', 'white clover', 'red clover', 'weeds', 'soil', 'clover other'],
        'class_ids': [0, 1, 2, 3, 4, 5]
        }

GreenWay_ID_TO_TRAINID_V1 = { -1: ignore_label,
                0:0,    # sky
                10:5,   # ground Dirt
                20: 4,  # object Golfball
                30 : 2, # flat grass
                40 :3,  # Construction Stone
                50 : 1, # Nature and Clover
                60: 1,  # Weeds
                70: 1,  # Clover
                80: 3,  # Boundary
                90: 4,  # Rubish
                100: 4, # person
                110:0,  # Ego
                120:4,  # Animal
                130:3,  # Fence
                140:3,  # Bunker
                150:0   # water
                } 


GreenWay_ID_TO_TRAINID = { -1: ignore_label,
                0:6,    # sky
                10:4,   # ground Dirt
                20: 6,  # object Golfball
                30 :0,  # flat grass
                40 :6,  # Construction Stone
                50 :6,  # Nature 
                60: 3,  # Weeds
                70: 5,  # Clover
                80: 6,  # Boundary
                90: 6,  # Rubish
                100:6,  # person
                110:6,  # Ego
                120:6,  # Animal
                130:6,  # Fence
                140:6,  # Bunker   ----> Boundary class
                150:6   # water
                } 

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
                14: 5}

# These are the results of weights from the 5000 synthetic images 
# "weight":  [49.859577140450305, 48.208871284054766, 2.037260085442715, 19.528667068658073, 41.367888421769656, 3.190874152837387],
# "weight":  [3.000, 48.208871284054766, 2.037260085442715, 19.528667068658073, 41.367888421769656, 3.190874152837387],
#                 class_label_id;class_name
ID_TO_TRAINID_Binary = { -1: ignore_label,
                0:0,  # sky
                10:1,  # ground Dirt
                20: 1, # object Golfball
                30 : 0, # flat grass
                40 :1, # Construction Stone
                50 : 0,  # Nature and clover
                60: 0,  # Weeds
                70: 0,  # Clover
                80: 1,  # Boundary
                90: 1,  # Rubish
                100: 1,   #person
                110:0, #Ego
                120:1, #Animal
                130:1, # Fence
                140:1,   # Bunker
                150:0# water
                } 
                
#"weight":  [0.2, 0.8 ],

class SyntheticGrassDataset(BaseDataSet):
    def __init__(self, mode='fine', **kwargs):
        print("Kwards", kwargs.items)
        self.num_classes = 2
        self.mode = mode
        self.palette = palette.turfGrass_palette_2Class
        self.id_to_trainId = ID_TO_TRAINID_Binary
        self.counter = 0
        super(SyntheticGrassDataset, self).__init__(**kwargs)

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

        print("check files flag", check_flag)
        print("check_flag", check_flag_labels)
        if check_flag_labels != check_flag:
            raise Exception("Please force an end here now")

        self.files = list(zip(image_paths, label_paths))

        #self.compute_class_weights()
        # self.calcualte_mean()


    def _set_files_glob(self):
        label_path = os.path.join(self.root, 'ImagesLabels')
        image_path = os.path.join(self.root, 'Images')

        image_paths, label_paths = [], []

        image_paths.extend(sorted(glob(os.path.join(image_path, '*.png'))))
        label_paths.extend(sorted(glob(os.path.join(label_path, '*.png'))))

        if len(image_paths) != len(label_paths) and len(image_paths)!=0 :
            print("Count images:",len(image_paths)  )
            print("Image path:", image_path)
            print("Count labels:",len(label_paths)  )
            print("Label path:", label_path)
            raise Exception("Issue with input data")


        self.files = list(zip(image_paths, label_paths))
         
        # Once off thing to calculate the weights
        
        # self.calcualte_mean()weight=config['weight']
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
            if maxCount==20000:
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
            if maxCount==20000:
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

           # if maxCount==1000:
          #      break

        
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


          #  if maxCount==1000:
         #       break
        
        std = np.sqrt(stdTemp/maxCount)
        print("STD", std)
        raise Exception("Please end here now")





    def calcualte_meanV2(self):

        toPIL=transforms.ToPILImage()
        
        statistics = None
        statistics_labels = None

        mean = 0
        count =0 
        for index, (image_data, label_data) in tqdm(enumerate(self.files)):
            # print(image_data,":", label_data)
            image_data, label, image_id = self._load_data(index)
            # for ij in range(image_data.shape[0]):
            if statistics is None:
                statistics = Stats(toPIL(np.uint8(image_data)))
            else:
                statistics += Stats(toPIL(np.uint8(image_data)))



            count +=1 
            if count >100:
                break
        
        print(f'mean:{statistics.mean}, std:{statistics.stddev}')
        
        
        
        

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(
                Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        label_remapped = np.full(label.shape, 255, dtype=np.uint8)
        # Map image maps to labels
      
        #print("Unique IDS before: ", np.unique(label, return_counts=True))
        # Remapping here
        if self.map_label:
            for k, v in self.id_to_trainId.items():
                label_remapped[label == k] = v


        #print("Unique IDS before: ", np.unique(label, return_counts=True)," ", np.unique(label_remapped, return_counts=True) )
        # Debug Raw data being loaded into trainner 
        if False:
            _debug_data(image, label)


        return image, label_remapped, image_id
    
    def _debug_data(self, image, label):
            
            import PIL
            print("RAW data as per input into CNN DEBUG CODE")
            fileLabelSave = "/home/freddy/projects/pytorch_D65/outputs/label_"+str(self.counter)+ ".png"
            fileImageSave = "/home/freddy/projects/pytorch_D65/outputs/Image_"+str(self.counter)+ ".png"

            img = PIL.Image.fromarray(image.astype(np.uint8), "RGB")
            img.save(fileImageSave)

            img = PIL.Image.fromarray(label.astype(np.uint8), mode='L')
            img.save(fileLabelSave)
            self.counter +=1
            if self.counter >20:
                raise Exception("Debug of raw input data") 


class SyntheticGrass(BaseDataLoader):
    def __init__(self, data_dir, batch_size=1, split=False,
                 crop_size=None, map_label=True, base_size=None, scale=True,
                 num_workers=1, mode='fine', val=False, shuffle=False,
                 flip=False, rotate=False, blur=False, augment=False,
                 val_split=False, return_id=False):

        #ToDo check on ROI basic, possible difference, but this should be good 
        #self.MEAN = [0.25277309, 0.33264832, 0.18483537]
        #self.STD = [0.16721844, 0.18176823, 0.12718696]

        # Full Greeway dataset  
        self.MEAN = [0.24020349, 0.29680853, 0.19672048]
        self.STD = [0.18339201, 0.17367418, 0.17249223]

        #Full Greeway dataset + AWB, 
        # self.MEAN = [0.17885829, 0.22282441, 0.13670546]
        # self.STD = [0.11858419, 0.10887117, 0.1075377]


        # Deploy with None
        # self.MEAN = [None, None, None] 
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

        self.dataset = SyntheticGrassDataset(mode=mode, **kwargs)
        super(SyntheticGrass, self).__init__(
                self.dataset, batch_size, shuffle, num_workers, val_split)
