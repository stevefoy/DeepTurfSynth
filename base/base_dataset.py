import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms



class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, map_label=True, base_size=None, augment=True, val=False,
                crop_size=None, scale=False, flip=False, rotate=False, blur=False, return_id=False, tranform_colorJitter=None):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.map_label = map_label
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        
        # self.transforms_compose = transforms.Compose([transforms.Normalize(mean, std),transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0) transforms.ToTensor()])
       
        if self.mean[0]==None:
            #self.transforms_compose = None
            # Used when calculating the mean and std of the dataset
            self.transforms_compose = transforms.Compose([ transforms.ToTensor()])
            self.transforms_compose_val = transforms.Compose([ transforms.ToTensor()])
            # 1 Trained with all the datasets 
            # self.transforms_compose = transforms.Compose([transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0), transforms.ToTensor()])
            # 2. Tried and not great invert
            #self.transforms_compose = transforms.Compose([transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0), transforms.RandomInvert(p = 0.5)])
            # 3
            
            #self.transforms_compose = transforms.Compose( [transforms.ColorJitter(brightness=(0.01, 0.5), contrast=(0.1, 0.5), saturation=(0.01, 0.5), hue=(-0.1, 0.1)), transforms.ToTensor()])

           # self.transforms_compose = transforms.Compose([transforms.AutoAugmentPolicy(1)])
        
            print("______NO MEAN or STD__transform")

        else:

             
            #self.transforms_compose = transforms.Compose([transforms.Normalize(mean, std)])
            self.transforms_compose = transforms.Compose([ transforms.ToTensor(), 
                                                            transforms.Normalize(mean, std)])

            
            
            self.transforms_compose_val = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean, std)])
            
            
            print("MEAN", str(mean), "STD", str(std))
         


        self.return_id = return_id
        self.counter =0
        cv2.setNumThreads(0)
        random.seed(10)

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        h, w, _ = image.shape
        
        if self.base_size < w :
            print("self.base_size2", self.base_size)
            raise Exception("Sorry, no numbers below zero") 
            longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)

            # dsize
            dsize = (w, h)

            # resize image
            image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
        # Padding to return the correct crop size
    
        h, w, _ = image.shape
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't
        # have any obj in the image, re-do the processing
        if self.base_size < w or self.scale == True:
            
            #print("self.base_size", self.base_size, "w",w,"h",h)
            if self.scale:
                longside = random.randint(
                        int(self.base_size), int(self.base_size*2.5))
            else:
                longside = self.base_size

            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            # dsize
            dsize = (w, h)
            # resize image
            image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
           # print("Config is doing  resize", dsize, "self.base_size", self.base_size, "w",w,"h",h)
    
        h, w, _ = image.shape
        # Rotate the image with an angle
        if self.rotate:
            angle = random.randint(-25, 25)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix,
                                   (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix,
                                   (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(5 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma,
                                     sigmaY=sigma,
                                     borderType=cv2.BORDER_REFLECT_101)

        return image, label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)


        # Debug code to check dataloader on raw image RGB and Masks
        if False:
            
            import PIL

            print("RAW data as per input into CNN DEBUG CODE")
            fileLabelSave = "./outputs/label_"+str(self.counter)+ ".png"
            fileImageSave = "./outputs/Image_"+str(self.counter)+ ".png"

            img = PIL.Image.fromarray(image.astype(np.uint8), "RGB")
            img.save(fileImageSave)

            img = PIL.Image.fromarray(label.astype(np.uint8), mode='L')
            # img.save(fileLabelSave)
            self.counter +=1
            if self.counter == 100:
                raise Exception("Please end here now") 



        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        
        if  self.val:
            # print("V MEAN", str(self.mean), "STD", str(self.std))
            return self.transforms_compose_val(image), label
        elif self.return_id:
            print("HEREH 2")
            return self.transforms_compose(self.to_tensor(image)), label
        elif self.transforms_compose == None:
            print("HEREH 3")
            return self.to_tensor(image), label
        else:
            # print("T MEAN", str(self.mean), "STD", str(self.std))
            return self.transforms_compose(image), label


    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
