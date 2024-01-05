import os


from torch.utils.data import ConcatDataset


from .data_loader import Dataset, VocDataset
from .augumentations import get_training_augmentation


path = "/media/work/data/zbt/dataset/thyroid/segmentation/seg/"

tjk_ljy_path = os.path.join(path, 'tjk_ljy')
tjk_wx_path = os.path.join(path, 'tjk_wx')
tjk_zcz_path = os.path.join(path, 'tjk_zcz')

xy_2022_path = os.path.join(path, 'xy_2022')
xy_2021_path = os.path.join(path, 'xy_2021')
xy_2020_path = os.path.join(path, 'xy_2020')
qz_path = os.path.join(path, 'qz')


doctor_path = "/media/work/data/zbt/dataset/xiangya/Tijian/deblur/"
tn3k_path = "/media/work/data/zbt/dataset/tn3k/tn3k_segmentation"



breast_nodule_path = "/media/work/data/zbt/dataset/breast_nodule/br"
BR_BUSI_path = "/media/work/data/zbt/dataset/BR_BUSI/busi/Dataset_BUSI_with_GT/seg"

BR_BUSI_all_path = "/media/work/data/zbt/dataset/BR_BUSI/dataset/BUSI_all/"
BR_BUSI_10_path = "/media/work/data/zbt/dataset/BR_BUSI/dataset/BUSI_10/"
BR_BUSI_20_path = "/media/work/data/zbt/dataset/BR_BUSI/dataset/BUSI_20/"

BR_busi_all_path = "/media/work/data/zbt/dataset/BR_BUSI/BUSI/busi_all"
BR_busi_10_path = "/media/work/data/zbt/dataset/BR_BUSI/BUSI/busi_10"
BR_busi_20_path = "/media/work/data/zbt/dataset/BR_BUSI/BUSI/busi_20"




def register_dataset(dataset_path):
    x_train = os.path.join(dataset_path, 'images/train/')
    x_test = os.path.join(dataset_path, 'images/test/')
    y_train = os.path.join(dataset_path, 'masks/train/')
    y_test = os.path.join(dataset_path, 'masks/test/')

    train_dataset = Dataset(x_train, y_train, augmentation=get_training_augmentation(), t_size=512)
    valid_dataset = Dataset(x_test, y_test, t_size=512)
    return train_dataset, valid_dataset


tjk_ljy_trainDataset, tjk_ljy_testDataset = register_dataset(tjk_ljy_path)
tjk_wx_trainDataset, tjk_wx_testDataset = register_dataset(tjk_wx_path)
tjk_zcz_trainDataset, tjk_zcz_testDataset = register_dataset(tjk_zcz_path)
tjk_trainDataset = ConcatDataset([tjk_ljy_trainDataset, tjk_wx_trainDataset, tjk_zcz_trainDataset])
tjk_testDataset = ConcatDataset([tjk_ljy_testDataset, tjk_wx_testDataset, tjk_zcz_testDataset])


xy_2022_trainDataset, xy_2022_testDataset = register_dataset(xy_2022_path)
xy_2021_trainDataset, xy_2021_testDataset = register_dataset(xy_2021_path)
xy_2020_trainDataset, xy_2020_testDataset = register_dataset(xy_2020_path)
qz_trainDataset, qz_testDataset = register_dataset(qz_path)
# tn3k_trainDataset, tn3k_testDataset = register_dataset(tn3k_path)
doctor_trainDataset, doctor_testDataset = register_dataset(doctor_path)



breast_nodule_trainDataset, breast_nodule_testDataset = register_dataset(breast_nodule_path)
BR_BUSI_trainDataset, BR_BUSI_testDataset = register_dataset(BR_BUSI_path)

BN_all_trainDataset, BN_all_testDataset = register_dataset(BR_BUSI_all_path)
BN_10_trainDataset, BN_10_testDataset = register_dataset(BR_BUSI_10_path)
BN_20_trainDataset, BN_20_testDataset = register_dataset(BR_BUSI_20_path)

BR_busi_all_trainDataset, BR_busi_all_testDataset = register_dataset(BR_busi_all_path)
BR_busi_10_trainDataset, BR_busi_10_testDataset = register_dataset(BR_busi_10_path)
BR_busi_20_trainDataset, BR_busi_20_testDataset = register_dataset(BR_busi_20_path)




tn3k_trainDataset = VocDataset(images_dir='/media/work/data/zbt/dataset/tn3k/tn3k/JPEGImages',
                                masks_dir='/media/work/data/zbt/dataset/tn3k/tn3k/SegmentationClass',
                                txt_path='/media/work/data/zbt/dataset/tn3k/tn3k/ImageSets/train.txt')
tn3k_testDataset = VocDataset(images_dir='/media/work/data/zbt/dataset/tn3k/tn3k/JPEGImages',
                                masks_dir='/media/work/data/zbt/dataset/tn3k/tn3k/SegmentationClass',
                                txt_path='/media/work/data/zbt/dataset/tn3k/tn3k/ImageSets/new_test.txt')