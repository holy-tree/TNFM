import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn


import plots
import shutil
import argparse
import random
import datetime



from engine.ema import ModelEMA
from data import data_loader
from data import augumentations as augu

from model.segmentor import SegmentationModel
from utils import losses, metrics
from engine.train import TrainEpoch, ValidEpoch
from data.dataset_register import tjk_trainDataset, tjk_testDataset,\
                                    xy_2022_trainDataset, xy_2022_testDataset,\
                                    xy_2021_trainDataset, xy_2021_testDataset,\
                                    xy_2020_trainDataset, xy_2020_testDataset,\
                                    qz_trainDataset, qz_testDataset,\
                                    tn3k_trainDataset, tn3k_testDataset,\
                                    breast_nodule_trainDataset, breast_nodule_testDataset,\
                                    BR_BUSI_trainDataset, BR_BUSI_testDataset,\
                                    BR_BUSI_all_trainDataset, BR_BUSI_all_testDataset,\
                                    BR_BUSI_10_trainDataset, BR_BUSI_10_testDataset,\
                                    BR_BUSI_20_trainDataset, BR_BUSI_20_testDataset,\
                                    BR_busi_all_trainDataset, BR_busi_all_testDataset,\
                                    BR_busi_10_trainDataset, BR_busi_10_testDataset,\
                                    BR_busi_20_trainDataset, BR_busi_20_testDataset
                                    







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
    parser = argparse.ArgumentParser('Segmentation', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size')
    
    # parser.add_argument('--encoder_weights', default='./convmae_base.pth', type=str,
    #                     help='encoder weights')
    # parser.add_argument('--encoder_weights', default='/media/work/data/zbt/ConvMAE/output_dir1/checkpoint.pth', type=str,
    #                     help='encoder weights')
    parser.add_argument('--encoder_weights', default='imagenet', type=str,
                        help='encoder weights')
    
    
    parser.add_argument('--datapath', default='/media/work/data/zbt/dataset/xiangya/Tijian/deblur', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./logs/', type=str,
                        help='output directory')
    # parser.add_argument('--checkpoint', default='./logs/202312071508_unetpp_resnet50_deblurTrue_bs2_512_seed3407/50_dice_0.79058_ema.pth', type=str,
    #                     help='encoder weights')
    # parser.add_argument('--checkpoint', default='./38_dice_0.76114_ema.pth', type=str,
    #                     help='encoder weights')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='encoder weights')

    
    
    
    # unet, unetpp
    parser.add_argument('--model', default='unetpp', type=str,
                        help='segmentation model (unet, unetpp)')
    # parser.add_argument('--encoder', default='resnet101', type=str,
    #                     help='encoder (convmae or dconvmae)')
    parser.add_argument('--encoder', default='resnet50', type=str,
                        help='encoder (convmae or dconvmae)')
    parser.add_argument('--is_deblurring', default=True, type=bool,
                        help='is deblurring, should be consistent with encoder_weights')
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size')
    parser.add_argument('--seed', default=3407, type=int,
                        help='random seed')
    

    return parser



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    set_global_random_seed(args.seed)
    


    ENCODER = args.encoder
    ENCODER_WEIGHTS = args.encoder_weights
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'
    n_class = 1
    batch_size = args.batch_size
    in_chans = 3
    is_rgb =True



    model = SegmentationModel(encoder_name=ENCODER, 
                              encoder_weights=ENCODER_WEIGHTS, 
                              in_channels=in_chans, 
                              classes=n_class, 
                              activation=ACTIVATION).to(DEVICE)
    

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)

        # for k, v in checkpoint.module.items():
        #     print(f"keys:{k}")
        # quit()
        # print(checkpoint.module)
        miss_key = model.load_state_dict(checkpoint, strict=False)
        # print(miss_key)
    # quit()


    # if torch.cuda.is_available():
    #     print ("CUDA is available, using GPU.")
    #     num_gpu = list(range(torch.cuda.device_count()))
    #     model = nn.DataParallel(model, device_ids=num_gpu)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)



    t_size = args.input_size
    if ENCODER_WEIGHTS == None:
        ENCODER_WEIGHTS = 'None'
    model_dir = args.output_dir + "/" + datetime.datetime.now().strftime('%Y%m%d%H%M_') + args.model + "_" + ENCODER + \
        "_deblur" + str(args.is_deblurring) + "_bs" + str(batch_size) + "_"  + str(t_size) + "_seed" + str(args.seed) + "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # DATA_DIR = "/media/work/data/zbt/dataset/xiangya/Tijian/deblur"
    # x_valid_dir = os.path.join(DATA_DIR, 'images_ori/val/')
    # y_valid_dir = os.path.join(DATA_DIR, 'masks/val/')

    

    # train_dataset = ConcatDataset([tjk_trainDataset, xy_2022_trainDataset, xy_2021_testDataset])
    # test_dataset = tjk_testDataset
    train_dataset = BR_busi_all_trainDataset
    test_dataset = BR_busi_all_testDataset
    # test_dataset = ConcatDataset([train_dataset, test_dataset])
    # test_dataset = tn3k_testDataset

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    print ("Train:", len(train_dataset))
    print ("Valid:", len(test_dataset))





    loss = losses.DiceLoss()
    # loss = losses.BCEWithLogitsLoss()
    metrics = [metrics.Fscore(), metrics.IoU(),]

    # optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=1e-4),])
    optimizer = torch.optim.AdamW(params=[{'params': model.encoder.parameters(), 'lr': 0.00001},
                                          {'params': model.decoder.parameters(), 'weight_decay': 0.},],
                                  lr=0.0001,
                                  betas=[0.9, 0.999],
                                  weight_decay=0.0001)


    ema = ModelEMA(model=model, decay=0.9999, warmups=2000)
    # ema = None

    train_epoch = TrainEpoch(
        ema,
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        ema,
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_dice = 0
    best_epoch = 0
    EARLY_STOPS = 100
    train_dict = {'loss': [], 'dice': [], 'iou': [] }
    val_dict = {'loss': [], 'dice': [], 'iou': [] }

    for epoch in range(0, 100000):

        print('\nEpoch: {}'.format(epoch))
        print ("Best epoch:", best_epoch, "\tDICE:", max_dice)



        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        train_dict['loss'].append(train_logs['dice_loss'])  # 'dice_loss + jaccard_loss'
        train_dict['dice'].append(train_logs['fscore'])
        train_dict['iou'].append(train_logs['iou_score'])
        val_dict['loss'].append(valid_logs['dice_loss'])
        val_dict['dice'].append(valid_logs['fscore'])
        val_dict['iou'].append(valid_logs['iou_score'])

        plots.save_loss_dice(train_dict, val_dict, model_dir)



        last_checkpoint_path = os.path.join(model_dir, f'checkpoint_{epoch-1}.pth')
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
        checkpoint_path = os.path.join(model_dir, f'checkpoint_{epoch}.pth')
        torch.save(ema.module.state_dict(), checkpoint_path)



        # do something (save model, change lr, etc.)
        if max_dice < valid_logs['fscore']:
            if max_dice != 0:
                old_filepath = model_dir + str(best_epoch) + "_dice_" + str(max_dice) + "_ema.pth"
                os.remove(old_filepath)

            max_dice = np.round(valid_logs['fscore'], 5)
            # torch.save(model, model_dir + str(epoch) + "_dice_" + str(max_dice) + ".pth")
            torch.save(ema.module.state_dict(), model_dir + str(epoch) + "_dice_" + str(max_dice) + "_ema.pth")
            print('Model saved!')
            best_epoch = epoch


        if epoch - best_epoch > EARLY_STOPS:
            print (str(EARLY_STOPS), "epoches didn't improve, early stop.")
            print ("Best dice:", max_dice)
            break


