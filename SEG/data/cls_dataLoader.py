from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import torch
from collections import Counter

from torchvision import transforms as T
import torch.nn.functional as F
import random

class Cls_Dataset(Dataset):
    def __init__(self,
                 img_path,
                 txt_path,
                 size = 512,
                 augmentation=None,
                 ):
        self.img_path = img_path
        self.txt_path = txt_path
        self.size = size
        self.augmentation = augmentation

        with open(txt_path, 'r') as f:
            txt_list = f.readlines()
        # random.shuffle(txt_list)
        # trainval_txt = txt_list[:int(0.8*len(txt_list))]
        # test_txt = txt_list[int(0.8*len(txt_list)):]
        # trainval = "/media/work/data/zbt/dataset/TNSCUI2020_train/trainval.txt"
        # test = "/media/work/data/zbt/dataset/TNSCUI2020_train/test.txt"
        # with open(trainval, 'w') as f:
        #     f.writelines(trainval_txt)
        # with open(test, 'w') as f:
        #     f.writelines(test_txt)
        # quit()
        
        txt_list = [txt.strip() for txt in txt_list]
        self.ids = [txt.split(',')[0] for txt in txt_list]
        self.cate = [txt.split(',')[1] for txt in txt_list]
        

        # img_names = os.listdir(self.img_path)
        # for id in self.ids:
        #     if id not in img_names:
        #         print(id)
        # quit()
        # print(set(self.ids)^set(img_name))


        # print(len(list(set(self.ids))))
        # print(len(list(set(img_name))))
        # quit()

        





    def __len__(self):
        return len(self.ids)
    

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.ids[index])


        img_cate = self.cate[index]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        
        
        if self.augmentation:
            img = self.augmentation(image=img)['image']
        

        img = img / 255.0
        img = img.transpose(2,0,1)
        img = img.astype('float32')
        img_cate = np.asarray(img_cate).astype(int)
        img_cate = torch.from_numpy(img_cate).type(torch.int64)
        
        
        
        # img_cate = F.one_hot(img_cate, 2)

    
        

        return img, img_cate