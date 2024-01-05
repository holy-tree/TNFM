import sys
import torch
from tqdm import tqdm
import cv2
import numpy as np

from data.coco_eval import CocoEvaluator
from data.coco_utils import get_coco_api_from_dataset

import pprint

class Epoch:
    def __init__(self, ema, model, criterion, device='cpu'):
        self.ema = ema
        self.model = model
        self.criterion = criterion
        self.device = device

        self._to_device()
    
    def _to_device(self):
        self.model.to(self.device)
        self.criterion.to(self.device)
    
    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, train_loader):
        self.on_epoch_start()

        tqdm_length = train_loader.__len__()
        # tqdm_length = 100
        loss_list = []

        with tqdm(total=tqdm_length, ncols=120) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(10000 + 1, 1000))


            # for i in range(tqdm_length):
            #     data_iter = iter(train_loader)
            #     imgs, targets = next(data_iter)

            for imgs, targets in train_loader:
                imgs = imgs.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]


                loss_dict = self.batch_update(imgs, targets)

                if self.ema is not None:
                    self.ema.update(self.model)


                with torch.no_grad():
                    loss_value = sum(loss_dict.values())
                loss_list.append(loss_value)
                loss_mean = sum(loss_list) / len(loss_list)
                _tqdm.set_postfix(loss='{:.6f}'.format(loss_mean))
                _tqdm.update(1)
        return loss_mean
         

class TrainEpoch(Epoch):
    def __init__(self, ema, model, criterion, optimizer, device='cpu'):
        super().__init__(
            ema=ema,
            model=model,
            criterion=criterion,
            device=device,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()
        self.criterion.train()



    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        
        outputs = self.model(x, y)
        loss_dict = self.criterion(outputs, y)
        loss = sum(loss_dict.values())
        loss.backward()

        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)


        self.optimizer.step()

        return loss_dict


class VaildEpoch():
    def __init__(self, ema, criterion, postprocessors, device='cpu'):
        self.ema = ema
        self.criterion = criterion
        self.device = device
        self.postprocessors = postprocessors
        if self.ema:
            self.model = self.ema.module
        
        self._to_device()
        
    def _to_device(self):
        self.model.to(self.device)
        self.criterion.to(self.device)
    
    def on_epoch_start(self):
        self.model.eval()
        self.criterion.eval()

    def run(self, test_loader):
        self.on_epoch_start()

        iou_types = self.postprocessors.iou_types
        base_ds = get_coco_api_from_dataset(test_loader.dataset)
        coco_evaluator = CocoEvaluator(base_ds, iou_types)
        panoptic_evaluator = None
        

        for samples, targets in tqdm(test_loader):

            # print(samples.shape)
            # pprint.pprint(targets)
            # quit()
            
            # for i in [0,1,2,3]:
            #     img = samples[i].clone().detach().double().to(torch.device('cpu'))
            #     img = np.ascontiguousarray(img.numpy().transpose((1,2,0)))*255


            #     # print(targets[i]['boxes'].data)
            #     for box in targets[i]['boxes'].data:
            #         xminVal = int(box[0])   # object_info列表中的元素都是字符串类型
            #         yminVal = int(box[1])
            #         xmaxVal = int(box[2])
            #         ymaxVal = int(box[3])
            #         img = cv2.rectangle(img, (xminVal, yminVal), (xmaxVal, ymaxVal), (0,255,0),2)
            #     cv2.imwrite(f"img{i}.png", img)

        





            # print(targets)
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            outputs = self.model(samples)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0) 
            results = self.postprocessors(outputs, orig_target_sizes)
            

            # for i, result in enumerate(results):
            #     boxes = result['boxes']
            #     scores = result['scores']
                
            #     thre = 0.75
            #     nodule_num = 0
            #     for index in range(10):
            #         if scores[index] < thre:
            #             nodule_num = index
            #             break
            #     boxes = boxes[:nodule_num]
            #     scores = scores[:nodule_num]
            #     print(nodule_num)
                
            #     img = samples[i].clone().detach().double().to(torch.device('cpu'))
            #     img = np.ascontiguousarray(img.numpy().transpose((1,2,0)))*255
            #     img = cv2.resize(img, (int(orig_target_sizes[i][0]), int(orig_target_sizes[i][1])))

            #     for box in boxes:
            #         xminVal = int(box[0])   # object_info列表中的元素都是字符串类型
            #         yminVal = int(box[1])
            #         xmaxVal = int(box[2])
            #         ymaxVal = int(box[3])
            #         img = cv2.rectangle(img, (xminVal, yminVal), (xmaxVal, ymaxVal), (0,255,0),2)
            #     cv2.imwrite(f"pred{i}.png", img)

            # quit()





            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)


        # 不知道干啥的
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
        if panoptic_evaluator is not None:
            panoptic_evaluator.synchronize_between_processes()
        
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
        
        stats = {}

        if coco_evaluator is not None:
            if 'bbox' in iou_types:
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in iou_types:
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        return stats, coco_evaluator