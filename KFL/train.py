import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torch.nn as nn

import random
import argparse
import datetime
from tqdm import tqdm




# from model.swin_transformer3d import SwinTransformer3D
from model.key_frame_locator import KeyFrameLocator
from model.key_frame_loss import KeyFrameLoss
# from data.coco_utils import get_coco_api_from_dataset
# from data.dataloader import DataLoader, default_collate_fn
from engine.trainer import TrainEpoch, VaildEpoch
# from engine.ema import ModelEMA
# from data.dataset_register import doctor_trainDataset, doctor_testDataset,\
#                                     tjk_ljy_trainDataset, tjk_ljy_testDataset,\
#                                     tjk_wx_trainDataset, tjk_wx_testDataset,\
#                                     tjk_zcz_trainDataset, tjk_zcz_testDataset,\
#                                     tjk_trainDataset, tjk_testDataset,\
#                                     xy_2022_trainDataset, xy_2022_testDataset,\
#                                     xy_2021_trainDataset, xy_2021_testDataset,\
#                                     xy_2020_trainDataset, xy_2020_testDataset,\
#                                     qz_trainDataset, qz_testDataset
from dataset_register import ljy_TrainVideoDataset, ljy_TestVideoDataset,\
                                wx_TrainVideoDataset, wx_TestVideoDataset, test_VideoDataset


import sys
sys.path.append('..')
from backbone import ConvNeXt, ViTBaseline, HYBNet

import json
import pprint
import copy
import re 

def set_global_random_seed(seed):
    os.environ['PYTHONASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('Detection', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--output_dir', default='./logs/', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    # parser.add_argument('--checkpoint', default='./logs/202312312027___bs1_512_seed3407/checkpoint_1159.pth', type=str)
    # parser.add_argument('--checkpoint', default='./USFM_20231222.pth', type=str)

    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--seed', default=3407, type=int,
                        help='random seed')
    return parser



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    set_global_random_seed(args.seed)

    DEVICE = 'cuda'
    in_chans = 3
    is_rgb =True
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    t_size = args.input_size
    lr = 1e-6

    

    model = KeyFrameLocator().to(DEVICE)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = KeyFrameLoss().to(DEVICE)


    if checkpoint:
        checkpoint = torch.load(checkpoint)
        miss_key = model.load_state_dict(checkpoint, strict=True)
    # ema = ModelEMA(model=model, decay=0.9999, warmups=2000)

    






    # scaler = torch.cuda.amp.GradScaler()
    # clip_max_norm = 0.1


    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=lr,
                                  betas=[0.9, 0.999],
                                  weight_decay=0.0001)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50], gamma=0.1)
    



    

    model_dir = args.output_dir + "/" + datetime.datetime.now().strftime('%Y%m%d%H%M_')  + "_" + \
         "_bs" + str(batch_size) + "_"  + str(t_size) + "_seed" + str(args.seed) + "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)



    # train_dataset = test_VideoDataset
    train_dataset = ljy_TrainVideoDataset
    test_dataset = ljy_TestVideoDataset
   


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    
    print ("Train:", len(train_dataset))
    print ("Valid:", len(test_dataset))
    
    # frames, labels = tjk_ljy_VideoDataset[30]
    # print(frames.shape)
    # img = frames[4].clone().detach().double().to(torch.device('cpu'))
    # img = np.ascontiguousarray(img.numpy().transpose((1,2,0)))*255
    # cv2.imwrite(f'frame_test.png', img)
    # quit()


    train_epoch = TrainEpoch(model=model,
                             criterion=criterion,
                             optimizer=optimizer,
                             device=DEVICE)
    valid_epoch = VaildEpoch(model=model,
                             criterion=criterion,
                             device=DEVICE)





    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    best_epoch = 0
    best_dis = 10

    # epoch
    for epoch in range(0, 10000):
        print('\nEpoch: {}'.format(epoch))

        

        loss_mean = train_epoch.run(train_loader, epoch)
        lr_scheduler.step()
        mean_distance = valid_epoch.run(valid_loader, epoch)


        # cur_lr = lr_scheduler.get_last_lr()[0]
        # record_dict = {'epoch': epoch, 'lr': cur_lr, 'mean_distance': mean_distance}
        # with open(os.path.join(model_dir, 'log.log'), 'a') as f:
        #     f.write(json.dumps(record_dict)+'\n')
        


        last_checkpoint_path = os.path.join(model_dir, f'checkpoint_{epoch-1}.pth')
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
        checkpoint_path = os.path.join(model_dir, f'checkpoint_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        
        
        if mean_distance < best_dis:
            best_dis = mean_distance
            best_epoch = epoch
            checkpoint_path = os.path.join(model_dir, f'checkpoint_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
        print(f'best_dis:{best_dis}  epoch:{best_epoch}')
        
        # log_stats = {'epoch': epoch, 
        #              'train_loss': loss_mean.item(),
        #             **{f'test_{k}': v for k, v in test_stats.items()}}
        
        # with open(os.path.join(model_dir, "log.txt"), 'a') as f:
        #     f.write(json.dumps(log_stats) + "\n")

    

    
        
        
        

        


            
 
            


        