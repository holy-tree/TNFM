import os
import torch

from .swin_transformer3d import SwinTransformer3D
from .cls_head import ClsHead

class KeyFrameLocator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.video_transformer = SwinTransformer3D(pretrained=None,
                                                    pretrained2d=True,
                                                    patch_size=(1,4,4),
                                                    embed_dim=96,
                                                    depths=[2,2,6,2],
                                                    num_heads=[3,6,12,24],
                                                    window_size=(8,7,7),
                                                    mlp_ratio=4.,
                                                    qkv_bias=True,
                                                    qk_scale=None,
                                                    drop_rate=0.,
                                                    attn_drop_rate=0.,
                                                    drop_path_rate=0.1,
                                                    patch_norm=True)
        self.cls_head = ClsHead(num_classes=1,
                                in_channels=768,
                                dropout_ratio=0.5,
                                init_std=0.01)
                                

    def forward(self, x):

        features = self.video_transformer(x)
        
        cls_score = self.cls_head(features)
        
        return cls_score

