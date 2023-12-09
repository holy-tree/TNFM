import os

from torch.utils.data import ConcatDataset

from .coco_dataset import CocoDetection
from .transforms import train_transforms, val_transforms

doctor_path = '/media/work/data/zbt/dataset/xiangya/doctor_coco'

path = '/media/work/data/zbt/dataset/thyroid/detection/coco'

tjk_ljy_path = os.path.join(path, 'tjk_ljy')
tjk_wx_path = os.path.join(path, 'tjk_wx')
tjk_zcz_path = os.path.join(path, 'tjk_zcz')

xy_2022_path = os.path.join(path, 'xy_2022')
xy_2021_path = os.path.join(path, 'xy_2021')
xy_2020_path = os.path.join(path, 'xy_2020')
qz_path = os.path.join(path, 'qz')


def register_dataset(dataset_path):
    x_train = os.path.join(dataset_path, 'JPEGImages/train')
    x_test = os.path.join(dataset_path, 'JPEGImages/test')
    y_train = os.path.join(dataset_path, 'Annotations/train.json')
    y_test = os.path.join(dataset_path, 'Annotations/test.json')

    train_dataset = CocoDetection(img_folder=x_train,
                                  ann_file=y_train,
                                  transforms=train_transforms,
                                  return_masks=False)
    valid_dataset = CocoDetection(img_folder=x_test,
                                  ann_file=y_test,
                                  transforms=val_transforms,
                                  return_masks=False)
    
    return train_dataset, valid_dataset


doctor_trainDataset, doctor_testDataset = register_dataset(doctor_path)

tjk_ljy_trainDataset, tjk_ljy_testDataset = register_dataset(tjk_ljy_path)
tjk_wx_trainDataset, tjk_wx_testDataset = register_dataset(tjk_wx_path)
tjk_zcz_trainDataset, tjk_zcz_testDataset = register_dataset(tjk_zcz_path)
tjk_trainDataset = ConcatDataset([tjk_ljy_trainDataset, tjk_wx_trainDataset, tjk_zcz_trainDataset])
tjk_testDataset = ConcatDataset([tjk_ljy_testDataset, tjk_wx_testDataset, tjk_zcz_testDataset])

xy_2022_trainDataset, xy_2022_testDataset = register_dataset(xy_2022_path)
xy_2021_trainDataset, xy_2021_testDataset = register_dataset(xy_2021_path)
xy_2020_trainDataset, xy_2020_testDataset = register_dataset(xy_2020_path)
qz_trainDataset, qz_testDataset = register_dataset(qz_path)