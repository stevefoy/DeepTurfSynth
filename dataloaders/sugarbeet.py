from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt


plt.rcParams["savefig.bbox"] = 'tight'
ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label} #0:1, 255: 0}
#             //"data_dir": "/home/freddy/projects/sugarbeet_dataset/sugarbeet_dataset_512",


class SugarBeetDataset(BaseDataSet):
    def __init__(self, mode='fine', **kwargs):
        self.num_classes = 3
        self.mode = mode
        self.palette = palette.SugarBeet_palette
        self.id_to_trainId = ID_TO_TRAINID
        self.mapping = {(0, 0, 0): 0,  # 0 = background
                        (255, 0, 0): 1,  # 1 = plant
                        (0, 255, 0): 2,  # 2 = weeds
                        (0, 0, 255): 3}  # 3 = class 3
        self.regenerateMask = False

        super(SugarBeetDataset, self).__init__(**kwargs)

    def show(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = transforms.functional.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    def _set_files(self):
        assert (self.mode == 'fine' and self.split in ['train', 'valid'])


        print("root", self.root)
        #base_path_full = os.path.join(self.root, self.split)    
        #ToDo: if folder list does not exist , should be created
        # Folder column
        #for root, dir, files in os.walk(self.root):
        #files = [f for f in files if not f.startswith('~') and f!='Thumbs.db']
        #paths = [os.path.join(root, f) for f in files]
        #for root, dirs, files in os.walk(self.root, topdown=True):
        #    for name in dirs:
        #        print(os.path.join(root, name))
        #print(folders)

        base_folderList = open(os.path.join(self.root, "folderList.txt"))
        print(base_folderList)

        self.filenames_rgb = []
        self.filenames_nir = []
        self.filenames_mask = []

        for file_base_path in base_folderList:
            file_base_path = os.path.join(self.root, file_base_path.strip())
            print(file_base_path)

            imgs_path_rgb = os.path.join(file_base_path, "images/rgb")
            imgs_path_nir = os.path.join(file_base_path, "images/nir")
            imgs_path_mask = os.path.join(
                    file_base_path, "annotations/dlp/colorCleaned")
            imgs_path_mask_gray = os.path.join(
                    file_base_path, "annotations/dlp/maskGen")

            if not os.path.exists(imgs_path_mask_gray):
                os.makedirs(imgs_path_mask_gray)
            else:
                if self.regenerateMask:
                    pass
                    # ToDo clear folder for regen

            rgb_files =[f for f in os.listdir(imgs_path_rgb) if os.path.isfile(os.path.join(imgs_path_rgb, f))]
            print(rgb_files)

            for filename in rgb_files:
                file_path_mask_gray = os.path.join(imgs_path_mask_gray, filename)
                file_path_rgb = os.path.join(imgs_path_rgb, filename )
                file_path_mask = os.path.join(imgs_path_mask, filename )

                #Create a Gray Mask file if one doesn;t exist
                if os.path.isfile(file_path_mask_gray) != True:
                    if os.path.isfile(file_path_mask) == True:
                        mask_rgb = Image.open(file_path_mask)
                        mask_rgb = self._mask_to_class_rgb(mask_rgb)
                        image = transforms.ToPILImage()(mask_rgb)
                        image.save(file_path_mask_gray)
                        # import matplotlib.pyplot as plt
                        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize = (6,6))
                        # img1 = ax1.imshow(mask_rgb, cmap='gray')
                        # ax1.axis('on')
                        # #img2 = ax2.imshow(mask_rgb)
                        # #ax2.axis('on')        
                        # plt.show()  
                    else:
                        pass
                        #print("File error: Mask RGB  Not Found -", file_path_mask)
                    
                else:
                    pass
                    #print("File exists for Masks: ", str(file_path_mask_gray) )
                
                #
                if os.path.isfile(file_path_rgb) and os.path.isfile(file_path_mask_gray ):
                    
                    self.filenames_rgb.append(file_path_rgb)
                    self.filenames_nir.append(os.path.join(imgs_path_nir, filename))
                    self.filenames_mask.append(file_path_mask_gray )
                    
                else:
                    
                    print("Path Error", imgs_path_rgb) 
                
        print("self.filenames_rgb", len(self.filenames_rgb))
        print("self.filenames_masks", len(self.filenames_mask))
            
            
                
        self.files = list(zip(self.filenames_rgb, self.filenames_mask))
    
    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask
    
    
    def _mask_to_class_rgb(self, mask):

        mask = torch.from_numpy(np.array(mask))
        #print(mask.shape)
        #mask = torch.squeeze(mask)  # remove 1

        # check the present values in the mask, 0 and 255 in my case
        if len(torch.unique(mask))>2:
            print('unique values rgb    ', len(torch.unique(mask))) 
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.empty(h, w, dtype=torch.uint8)

        for k in self.mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))         
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.uint8)

        # check the present values after mapping, in my case 0, 1, 2, 3
        #print('unique values mapped ', torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])
       
        return mask_out
    
    
    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int8)
        for k, v in self.id_to_trainId.items():
            label[label == k] = v
        
        if False:    
            pil_img = Image.fromarray(label)    
            import matplotlib.pyplot as plt
            plt.imshow(pil_img)
            plt.show()
        
        return image, label, image_id



class SugarBeet(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, mode='fine', val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = SugarBeetDataset(mode=mode, **kwargs)
        super(SugarBeet, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)


