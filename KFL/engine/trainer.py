import sys
import torch
from tqdm import tqdm
import cv2
# import sklearn
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
# from data.coco_eval import CocoEvaluator
# from data.coco_utils import get_coco_api_from_dataset

import pprint


class TrainEpoch():
    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        # self.criterion.to(self.device)

    def on_epoch_start(self):
        self.model.train()
        # self.criterion.train()

    def run(self, train_loader, epoch):
        self.on_epoch_start()

        tqdm_length = train_loader.__len__()
        # tqdm_length = 100
        loss_list = []

        with tqdm(total=tqdm_length, ncols=120) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch+1, 1000))
            for frames, labels in train_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
   
                outputs = self.model(frames)   
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                # if self.ema is not None:
                #     self.ema.update(self.model)
               
                loss_list.append(loss)
                loss_mean = sum(loss_list) / len(loss_list)
                _tqdm.set_postfix(loss='{:.6f}'.format(loss_mean))
                _tqdm.update(1)
                

        return loss_mean





class VaildEpoch():
    def __init__(self, model, criterion, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.device = device
        # if self.ema:
        #     self.model = self.ema.module
        
        self._to_device()
        
    def _to_device(self):
        self.model.to(self.device)
        self.criterion.to(self.device)
    
    def on_epoch_start(self):
        self.model.eval()
        # self.criterion.eval()

    def run(self, test_loader, epoch):
        self.on_epoch_start()
        
        tqdm_length = test_loader.__len__()

        all_label_gt = []
        all_label_pred = []
        all_pred = []
        key_distance_list = []

        with tqdm(total=tqdm_length, ncols=120) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch+1, 1000))

            for frames, labels in test_loader:
                
                frames = frames.to(self.device)
                labels = labels.squeeze().cpu().numpy().astype(np.int64)
                # cate = F.one_hot(cate, 2)
                
                with torch.no_grad():
                    outputs = self.model(frames)
                
                pred_label = torch.sigmoid(outputs).squeeze().cpu().detach().numpy()
               
                # pred_label = np.where(pred_label > 0.5, 1, 0)
                max_index = np.argmax(pred_label)
                result = np.zeros_like(pred_label)
                result[max_index] = 1
                

                key_pred = np.where(result == 1)[0]
                key_gt = np.where(labels == 1)[0]
                key_distance = abs(key_gt - key_pred)
                key_distance_list.append(key_distance)
                

                

                all_label_pred.append(result)
                all_label_gt.append(labels)
                
                # all_pred.append(outputs[0, 1].cpu().numpy())
                # _tqdm.set_postfix(loss='{:.6f}'.format(loss_mean))
                _tqdm.update(1)
        all_label_pred =  np.concatenate(all_label_pred)
        all_label_gt = np.concatenate(all_label_gt)

        # print(all_label_gt)
        # print(all_label_pred)
        # print(len(all_label_gt))
        # print(len(all_label_pred))
        acc = metrics.accuracy_score(all_label_gt, all_label_pred)
        f1_micro = metrics.f1_score(all_label_gt, all_label_pred, average='micro')
        precision = metrics.precision_score(all_label_gt, all_label_pred, average='binary')
        recall = metrics.recall_score(all_label_gt, all_label_pred, average='binary')
        mean_distance = sum(key_distance_list) / len(key_distance_list)
      
        print(f'Acc:{acc}————F1_micro:{f1_micro}————Precision:{precision}————Recall:{recall}')
        print(f'mean_distance: {mean_distance}')

        return mean_distance





