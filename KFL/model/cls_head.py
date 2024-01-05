import torch
import torch.nn as nn


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)



class ClsHead(torch.nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.5,
                 init_std=0.01):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std


        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        

        # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
        # self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [8, in_channels, 7, 7]
        x = x.squeeze().permute(1,0,2,3)
        
        # [8, in_channels, 1, 1]
        x = self.avg_pool(x)

        # [8, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [8, in_channels, 1, 1]
        x = x.view(x.shape[0], -1)
        # [8, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
