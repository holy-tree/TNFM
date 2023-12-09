import torch
import torch.nn as nn


from typing import Optional, Union, List

# from .decoder.unetpp import UnetPlusPlusDecoder
from .decoder.segformer_head import SegFormerHead
# from .decoder.uper_head import UPerHead1

from .base.modules import Flatten, Activation

import sys
sys.path.append('./..')

from backbone import *





def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)




class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)




class SegmentationModel(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()


        # self.encoder = ResNetEncoder(
        #     encoder_name,
        #     in_channels=in_channels,
        #     depth=encoder_depth,
        #     weights=encoder_weights,
        # )

        # resnet = PResNet(depth=50,
        #                 variant='d',
        #                 freeze_at=0,
        #                 return_idx=[0,1,2,3],
        #                 num_stages=4,
        #                 freeze_norm=True,
        #                 pretrained=True)
        # vit = ViTBaseline(out_indices=[3, 5, 7, 11])

        # backbone = hybnet(vit=vit,
        #                    resnet=resnet,
        #                    is_sam=False,
        #                    catn=True)
        
        convnext = ConvNeXt(depths=[3,3,9,3],
                            dims=[96, 192, 384, 768],
                            out_indices=[0, 1, 2, 3])
        url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        convnext.load_state_dict(checkpoint["model"], strict=False)

        vit = ViTBaseline(out_indices=[3, 5, 7, 11],
                          pretrained = "./deit_base_patch16_224-b5f2ef4d.pth")
        backbone = HYBNet(vit=vit,
                           resnet=convnext,
                           is_sam=False,
                           catn=True)
        # backbone = convnext


        # backbone = hybnet






        # backbone = resnet

        self.encoder = backbone
        # # self.encoder.out_channels = (64, 256, 512, 1024, 2048)
        # self.encoder.out_channels = (64, 64, 128, 256, 512)
        # print(self.encoder.out_channels)
        # print(decoder_channels)
        # print(123)
        # quit()



        # self.decoder = UnetPlusPlusDecoder(
        #     encoder_channels=self.encoder.out_channels,
        #     decoder_channels=decoder_channels,
        #     n_blocks=encoder_depth,
        #     use_batchnorm=decoder_use_batchnorm,
        #     center=True if encoder_name.startswith("vgg") else False,
        #     attention_type=decoder_attention_type,
        #     is_mae=is_mae
        # )
        # in_channels = [64, 128, 256, 512]
        # in_channels = [256, 512, 1024, 2048]
        in_channels = [96,192,384,768]
        # in_channels = [768,768,768,768]

        self.decoder = SegFormerHead(feature_strides=[4, 8, 16, 32],
                                     in_channels=in_channels,
                                     num_classes=classes,
                                     )
        # self.decoder = UPerHead(in_channels=[64, 128, 256, 512],
        #                         num_classes=classes,
        #                         channels=512,
        #                         )


        # self.segmentation_head = SegmentationHead(
        #     in_channels=decoder_channels[-1],
        #     out_channels=classes,
        #     activation=activation,
        #     kernel_size=3,
        #     upsampling=1,
        # )
        self.segmentation_head = SegmentationHead(
            in_channels=768,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=4,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()




    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        
        # for ii in range(len(features)):
        #     print (ii, features[ii].shape)
        # quit()

        # decoder_output = self.decoder(*features)
        decoder_output = self.decoder(features)
        # print(decoder_output.shape)
        # quit()
        # for ii in range(len(decoder_output)):
        #     print (ii, decoder_output[ii].shape)
        # quit()

        masks = self.segmentation_head(decoder_output)
        # masks = decoder_output
        # print(masks)
        # quit()
        # print(masks.shape)
        # quit()
        

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)