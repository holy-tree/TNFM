import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


from tqdm import tqdm
import pprint


class VideoDataset(Dataset):
    def __init__(self, videos_dir, txt_dir, batch_frames, img_size=224):
        self.videos_dir = videos_dir
        self.txt_dir = txt_dir
        self.batch_frames = batch_frames
        self.img_size = img_size

        
        with open(txt_dir, 'r') as f:
            txt_lines = f.readlines()
        txt_lines = [txt_line.strip() for txt_line in txt_lines]
        
        # self.videos_name = [txt_line.split(',')[0] for txt_line in txt_lines]
        # # start_frame_list = [int(txt_line.split(',')[1]) for txt_line in txt_lines]
        # # end_frames_list = [int(txt_line.split(',')[3]) for txt_line in txt_lines]
        # self.key_frames_list = [int(txt_line.split(',')[2])-int(txt_line.split(',')[1]) for txt_line in txt_lines]
        # self.videos_frames_num = [int(txt_line.split(',')[3])-int(txt_line.split(',')[1])+1 for txt_line in txt_lines]
        
        self.videos_info = []

        for txt_line in txt_lines:
            video_name = txt_line.split(',')[0]
            video_frames_num = int(txt_line.split(',')[3])-int(txt_line.split(',')[1])+1
            key_frame = int(txt_line.split(',')[2])-int(txt_line.split(',')[1])
            if video_frames_num <= key_frame or key_frame <= 0:
                continue
            self.videos_info.append({'video_name': video_name, 
                                     'video_frames_num': video_frames_num,
                                     'key_frame': key_frame})


        # self.total_frames = []
        # self.total_labels = []

        # for video_name, video_frames_num, key_frame in tqdm(zip(self.videos_name, self.videos_frames_num, self.key_frames_list)):
        #     if int(video_frames_num) <= int(key_frame) or int(key_frame) <= 0:
        #         continue
        #     # print(f'{video_name},{video_frames_num}, {key_frame}')
            
        #     print(video_frames_num)
            

        #     video_path = os.path.join(videos_dir, video_name)
        #     video = cv2.VideoCapture(video_path)

        #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        #     clip_start_h = int(0.12*height)
        #     clip_end_h = int(0.75*height)
        #     clip_start_w = int(0.18*width)
        #     clip_end_w = int(0.85*width)

        #     # video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #     # print(video_frames_num == video_frames)
        #     video_labels = [0] * video_frames_num
        #     video_labels[key_frame] = 1
        #     self.total_labels += video_labels

        #     while True:
        #         ret, frame = video.read()
        #         if not ret:
        #             break
        #         frame = frame[clip_start_h:clip_end_h, clip_start_w:clip_end_w, :]
        #         frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        #         self.total_frames.append(frame)
        #     video.release()

        # print(f'total videos frames:{len(self.total_frames)}')
        # print(f'total frames labels:{len(self.total_labels)}')
        
        # self.key_frame_indexs = [index for index, value in enumerate(self.total_labels) if value == 1]
        # quit()
    

    def __len__(self):
        # return len(self.total_frames) / self.batch_frames
        return len(self.videos_info)

    def __getitem__(self, index):
        # key_frame_index = self.key_frame_indexs[index]

        # half_length = self.batch_frames / 2
        # start_index = int(max(0, key_frame_index-half_length))
        # end_index = int(min(len(self.total_labels), start_index + self.batch_frames))
        
        # frames = self.total_frames[start_index:end_index]
        # labels = self.total_labels[start_index:end_index]

        # frames = [frame / 255.0 for frame in frames]
        
        # frames = torch.Tensor(frames).permute(3,0,1,2)
        # labels = torch.Tensor(labels)
        
        video_info = self.videos_info[index]
        video_name = video_info['video_name']
        video_frames_num = video_info['video_frames_num']
        key_frame = video_info['key_frame']
        # print(f'video_name:{video_name}, video_frames_num:{video_frames_num}, key_frame:{key_frame}')


        frames = []
        labels = []
        
        if video_frames_num <= self.batch_frames:
            batch_frames_num = video_frames_num
            start_frame_index = 0
            end_frame_index = video_frames_num-1
        else:
            batch_frames_num = self.batch_frames
            temp_num = int(self.batch_frames / 2)
            if key_frame < temp_num:
                start_frame_index = 0
                end_frame_index = self.batch_frames-1
            elif (video_frames_num-key_frame-1) < temp_num:
                start_frame_index = video_frames_num - self.batch_frames
                end_frame_index = video_frames_num - 1
            else:
                start_frame_index = key_frame - temp_num
                end_frame_index = key_frame + temp_num
                
        # print(f'start_frame_index: {start_frame_index}, end_frame_index: {end_frame_index}')



        video_path = os.path.join(self.videos_dir, video_name)
        video = cv2.VideoCapture(video_path)
        
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        clip_start_h = int(0.12*height)
        clip_end_h = int(0.75*height)
        clip_start_w = int(0.18*width)
        clip_end_w = int(0.85*width)

        # video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(video_frames)
        # print(video_frames_num == video_frames)
        labels = [0] * batch_frames_num
        labels[key_frame-start_frame_index] = 1
        
        frame_index = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if start_frame_index <= frame_index <= end_frame_index:
                frame = frame[clip_start_h:clip_end_h, clip_start_w:clip_end_w, :]
                frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                frames.append(frame)
            frame_index += 1
        video.release()

        # print(len(frames))
        # print(len(labels))
        

        frames = [frame / 255.0 for frame in frames]
        
        frames = torch.Tensor(frames).permute(3,0,1,2)
        labels = torch.Tensor(labels)

        
        
        return frames, labels
        
