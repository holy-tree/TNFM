import torch 
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F


from torchvision import datapoints

from typing import Any, Dict, List, Optional


__all__ = ['train_transforms', 'val_transforms']




class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt


class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)
    


train_transforms = T.Compose([
    T.RandomPhotometricDistort(p=0.8), 
    T.RandomZoomOut(fill=0), 
    RandomIoUCrop(p=0.8), 
    T.SanitizeBoundingBox(min_size=1),
    T.RandomHorizontalFlip(),
    T.Resize(size=[512, 512]),
    T.ToImageTensor(),
    T.ConvertDtype(),
    T.SanitizeBoundingBox(min_size=1),
    ConvertBox(out_fmt='cxcywh', normalize=True)
])


val_transforms = T.Compose([
    T.Resize(size=[512, 512]),
    T.ToImageTensor(),
    T.ConvertDtype(),
])




