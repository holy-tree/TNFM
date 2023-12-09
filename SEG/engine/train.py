import sys
import torch
from tqdm import tqdm
import cv2
import numpy as np



class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next result to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan





class Epoch:
    def __init__(self, ema, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.ema = ema
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose), ncols=120) as iterator:
            for x, y, _ in iterator:
                x, y = x.to(self.device), y.to(self.device)

                

                # img = x[0].clone().detach().double().to(torch.device('cpu'))
                # img = np.ascontiguousarray(img.numpy().transpose((1,2,0)))*255
                # cv2.imwrite(f"img_ori.jpg", img)
               
                # y[y == 1] = 255


                # img = y[0].clone().detach().double().to(torch.device('cpu'))
                # img = np.ascontiguousarray(img.numpy().transpose((1,2,0)))*255
                # cv2.imwrite(f"img_mask.jpg", img)
                # quit()
                # print(y.shape)
                # y = resize(y, (128,128),mode='bilinear')
                # y[y == 1] = 255


                # img = y[0].clone().detach().double().to(torch.device('cpu'))
                # img = np.ascontiguousarray(img.numpy().transpose((1,2,0)))*255
                # cv2.imwrite(f"img_mask.jpg", img)
                # quit()
                # print(y.shape)




                loss, y_pred = self.batch_update(x, y)

                
                if self.ema:
                    self.ema.update(self.model)
                

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, ema, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            ema=ema,
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        
        prediction = self.model.forward(x)
        
        
        # print(prediction.dtype)
        # print(y.dtype)
        # quit()

        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()

        
        
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, ema, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            ema=ema,
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            # module = self.model
            if self.ema:
                module = self.ema.module
            prediction = module.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
