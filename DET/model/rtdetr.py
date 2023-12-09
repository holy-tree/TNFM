import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 


__all__ = ['RTDETR', ]


class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super(RTDETR, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

        
    def forward(self, x, targets=None):
        # if self.multi_scale and self.training:
        #     sz = np.random.choice(self.multi_scale)
        #     x = F.interpolate(x, size=[sz, sz])
        
        # print(x.shape)

        x = self.backbone(x)
        
        # print(len(x))
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # print(x[3].shape)
        # quit()
        # if self.ismae:
        #     x = [x[1], x[2], x[3]]
        x = [x[1], x[2], x[3]]
        x = self.encoder(x)     
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 