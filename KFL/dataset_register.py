import os
from VideoDataset import VideoDataset


ljy_videos_dir = '/media/work/data/zbt/dataset/key_frame_location/videos_clip/ljy'
wx_videos_dir = '/media/work/data/zbt/dataset/key_frame_location/videos_clip/wx'
test_videos_dir = '/media/work/data/zbt/dataset/key_frame_location/videos_clip/test'

ljy_train_txt = '/media/work/data/zbt/dataset/key_frame_location/videos_clip/ljy_train.txt'
ljy_test_txt = '/media/work/data/zbt/dataset/key_frame_location/videos_clip/ljy_test.txt'
wx_train_txt = '/media/work/data/zbt/dataset/key_frame_location/videos_clip/wx_train.txt'
wx_test_txt = '/media/work/data/zbt/dataset/key_frame_location/videos_clip/wx_test.txt'
test_txt = '/media/work/data/zbt/dataset/key_frame_location/videos_clip/test.txt'


ljy_TrainVideoDataset = VideoDataset(videos_dir=ljy_videos_dir,
                                     txt_dir=ljy_train_txt,
                                     batch_frames=15,
                                     img_size=224)
ljy_TestVideoDataset = VideoDataset(videos_dir=ljy_videos_dir,
                                     txt_dir=ljy_test_txt,
                                     batch_frames=15,
                                     img_size=224)

wx_TrainVideoDataset = VideoDataset(videos_dir=wx_videos_dir,
                                     txt_dir=wx_train_txt,
                                     batch_frames=15,
                                     img_size=224)
wx_TestVideoDataset = VideoDataset(videos_dir=wx_videos_dir,
                                     txt_dir=wx_test_txt,
                                     batch_frames=15,
                                     img_size=224)
                   
test_VideoDataset = VideoDataset(videos_dir=ljy_videos_dir,
                                 txt_dir=test_txt,
                                 batch_frames=15,
                                 img_size=224)




# tjk_ljy_VideoDataset = VideoDataset(videos_dir='/media/work/data/zbt/dataset/key_frame_location/videos_clip/test',
#                                     txt_dir='/media/work/data/zbt/dataset/key_frame_location/videos_clip/test.txt',
#                                     batch_frames=15,
#                                     img_size=224)
# tjk_ljy_VideoDataset = VideoDataset(videos_dir='/media/work/data/zbt/dataset/key_frame_location/videos_clip/test',
#                                     txt_dir='/media/work/data/zbt/dataset/key_frame_location/videos_clip/test.txt',
#                                     batch_frames=15,
#                                     img_size=224)

# tjk_wx_VideoDataset = VideoDataset(videos_dir='/media/work/data/zbt/dataset/key_frame_location/videos_clip/wx',
#                                     txt_dir='/media/work/data/zbt/dataset/key_frame_location/videos_clip/wx_videos.txt',
#                                     batch_frames=15,
#                                     img_size=224)

