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


from model import *
from data.coco_utils import get_coco_api_from_dataset
from data.dataloader import DataLoader, default_collate_fn
from engine.trainer import TrainEpoch, VaildEpoch
from engine.ema import ModelEMA
from data.dataset_register import doctor_trainDataset, doctor_testDataset,\
                                    tjk_ljy_trainDataset, tjk_ljy_testDataset,\
                                    tjk_wx_trainDataset, tjk_wx_testDataset,\
                                    tjk_zcz_trainDataset, tjk_zcz_testDataset,\
                                    tjk_trainDataset, tjk_testDataset,\
                                    xy_2022_trainDataset, xy_2022_testDataset,\
                                    xy_2021_trainDataset, xy_2021_testDataset,\
                                    xy_2020_trainDataset, xy_2020_testDataset,\
                                    qz_trainDataset, qz_testDataset



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
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    

    
    parser.add_argument('--datapath', default='/media/work/data/zbt/dataset/xiangya/Tijian/deblur', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./logs/', type=str,
                        help='output directory')


    # parser.add_argument('--checkpoint', default='', type=str,
    #                     help='encoder weights')
    parser.add_argument('--checkpoint', default='./logs/202312151553___bs1_512_seed3407/checkpoint_best.pth', type=str,
                        help='encoder weights')


    # parser.add_argument('--is_deblurring', default=True, type=bool, help='is deblurring, should be consistent with encoder_weights')
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size')
    parser.add_argument('--seed', default=3407, type=int,
                        help='random seed')

    
    

    return parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    set_global_random_seed(args.seed)


    DATA_DIR = args.datapath
    # ENCODER = args.encoder
    # ENCODER_WEIGHTS = args.encoder_weights
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda'
    n_class = 1
    batch_size = args.batch_size
    in_chans = 3
    is_rgb =True


    # if args.backbone == 'resnet':
    #     backbone = PResNet(depth=50,
    #                     variant='d',
    #                     freeze_at=0,
    #                     return_idx=[1,2,3],
    #                     num_stages=4,
    #                     freeze_norm=True,
    #                     pretrained=True)
    #     encoder_in_channels = [512, 1024, 2048]

    # elif args.backbone == 'convmae':
    #     backbone = ConvMAE()
    #     encoder_in_channels = [384, 768, 768]

    # elif args.backbone == 'sam':
    #     resnet = PResNet(depth=50,
    #                     variant='d',
    #                     freeze_at=0,
    #                     return_idx=[1,2,3],
    #                     num_stages=4,
    #                     freeze_norm=True,
    #                     pretrained=True)
    #     sam = ImageEncoderViT(pretrained='./sam_vit_b_01ec64.pth'
    #                                )
    #     backbone = vit_res(vit=sam,
    #                        resnet=resnet,
    #                        is_sam=True)
    #     encoder_in_channels = [512, 1280, 2048]
    #     ismae = True


    # elif args.backbone == 'mae':
        # backbone = MAE_S(pretrained='./USFMpretrained1.ckpt')
        # backbone = BEiTBackbone4Seg(num_classes=1, pretrained='./USFMpretrained1.ckpt')
    convnext = ConvNeXt(depths=[3,3,9,3],
                        dims=[96, 192, 384, 768],
                        out_indices=[0, 1, 2, 3])
    url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth"
    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
    convnext.load_state_dict(checkpoint["model"], strict=False)


    vit = ViTBaseline(out_indices=[3, 5, 7, 11],
                        # pretrained = "../SEG/deit_base_patch16_224-b5f2ef4d.pth",
                        pretrained = "../MAE/7/checkpoint-999.pth"
                        
                        )
    # vit._freeze_parameters(vit)

    backbone = HYBNet(vit=vit,
                        resnet=convnext,
                        is_sam=False,
                        catn=True)
    # backbone = convnext
    # encoder_in_channels = [1280, 1792, 2816]
    encoder_in_channels = [192, 384, 768]
        # backbone = ViTAdapter(pretrained=None,
        #                     #   pretrain_size=512,
        #                       patch_size=16,
        #                       embed_dim=768,
        #                       depth=12,
        #                       num_heads=12,
        #                       mlp_ratio=4,
        #                       drop_path_rate=0.3,
        #                       conv_inplane=64,
        #                       n_points=4,
        #                       deform_num_heads=12,
        #                       cffn_ratio=0.25,
        #                       deform_ratio=0.5,
        #                       interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        #                       window_attn=[True, True, False, True, True, False,
        #                                   True, True, False, True, True, False],
        #                       window_size=[14, 14, None, 14, 14, None,
        #                                   14, 14, None, 14, 14, None],
        #                       )
    
        # encoder_in_channels = [768, 768, 768]
        
    encoder = HybridEncoder(in_channels=encoder_in_channels,
                            feat_strides=[8, 16, 32],
                            hidden_dim=256,
                            use_encoder_idx=[2],
                            num_encoder_layers=1,
                            nhead=8,
                            dim_feedforward=1024,
                            dropout=0.,
                            enc_act='gelu',
                            pe_temperature=10000,
                            expansion=1.0,
                            depth_mult=1.0,
                            act='silu',
                            # eval_size=[512, 512]
                            )
    decoder = RTDETRTransformer(num_classes=1,
                                feat_channels=[256, 256, 256],
                                feat_strides=[8, 16, 32],
                                hidden_dim=256,
                                num_levels=3,
                                num_queries=300,
                                num_decoder_layers=6,
                                num_denoising=100,
                                eval_idx=-1,
                                # eval_spatial_size=[512, 512]
                                )
    
    model = RTDETR(backbone, encoder, decoder).to(DEVICE)
    matcher = HungarianMatcher(weight_dict={'cost_class':2, 'cost_bbox': 5, 'cost_giou':2},
                               use_focal_loss=True,
                               alpha=0.25,
                               gamma=2.0)
    criterion = SetCriterion(matcher=matcher,
                             weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
                             losses=['vfl', 'boxes'],
                             alpha=0.75,
                             gamma=2.0,
                             num_classes=1).to(DEVICE)
    postprocessor = RTDETRPostProcessor(num_classes=1, num_top_queries=300, use_focal_loss=True)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        miss_key = model.load_state_dict(checkpoint, strict=True)
    ema = ModelEMA(model=model, decay=0.9999, warmups=2000)


    scaler = torch.cuda.amp.GradScaler()
    clip_max_norm = 0.1


    optimizer = torch.optim.AdamW(params=[{'params': backbone.parameters(), 'lr': 0.00001},
                                          {'params': encoder.parameters(), 'weight_decay': 0.},
                                          {'params': decoder.parameters(), 'weight_decay': 0.},],
                                  lr=0.0001,
                                  betas=[0.9, 0.999],
                                  weight_decay=0.0001)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50], gamma=0.1)
    checkpoint_step = 5




    t_size = args.input_size

    model_dir = args.output_dir + "/" + datetime.datetime.now().strftime('%Y%m%d%H%M_')  + "_" + \
         "_bs" + str(batch_size) + "_"  + str(t_size) + "_seed" + str(args.seed) + "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)



    # train_dataset = ConcatDataset([tjk_trainDataset, xy_2022_trainDataset, xy_2021_trainDataset])
    train_dataset = xy_2022_trainDataset
    test_dataset = doctor_testDataset
    # test_dataset = ConcatDataset([xy_2022_testDataset, xy_2021_testDataset])
    # test_dataset = doctor_testDataset   
    


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=default_collate_fn)
    valid_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=default_collate_fn)

    
    print ("Train:", len(train_dataset))
    print ("Valid:", len(test_dataset))
    

    train_epoch = TrainEpoch(ema=ema,
                             model=model,
                             criterion=criterion,
                             optimizer=optimizer,
                             device=DEVICE)
    valid_epoch = VaildEpoch(ema=ema,
                             criterion=criterion,
                             postprocessors=postprocessor,
                             device=DEVICE)





    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    best_epoch = 0
    best_ap = 0

    # epoch
    for epoch in range(0, 10000):
        print('\nEpoch: {}'.format(epoch))

        # loss_mean = train_epoch.run(train_loader)
        lr_scheduler.step()
        test_stats, coco_evaluator = valid_epoch.run(valid_loader)



        last_checkpoint_path = os.path.join(model_dir, f'checkpoint_{epoch-1}.pth')
        if os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)
        checkpoint_path = os.path.join(model_dir, f'checkpoint_{epoch}.pth')
        torch.save(ema.module.state_dict(), checkpoint_path)
        
        
        if test_stats['coco_eval_bbox'][1] > best_ap:
            best_ap = test_stats['coco_eval_bbox'][1]
            best_epoch = epoch
            checkpoint_path = os.path.join(model_dir, f'checkpoint_best.pth')
            torch.save(ema.module.state_dict(), checkpoint_path)
        print(f'best_ap50:{best_ap}  epoch:{best_epoch}')
        
        log_stats = {'epoch': epoch, 
                     'train_loss': loss_mean.item(),
                    **{f'test_{k}': v for k, v in test_stats.items()}}
        
        with open(os.path.join(model_dir, "log.txt"), 'a') as f:
            f.write(json.dumps(log_stats) + "\n")

    

    
        
        
        

        


            
 
            


        