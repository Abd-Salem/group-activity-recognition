import os, torch
from torch.utils.data import Dataset
from volleyball_annot_loader_utils import  load_volleyball_dataset
from helper_utils.configs import CONFIG
from torchvision import transforms
from PIL import Image

# get configs
config = CONFIG()

class PersonLevelDataset(Dataset):
    def __init__(self, paths, labels ,processor=transforms.ToTensor(), temporal=True, clips_info=None):
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.temporal = temporal
        self.clips_info = clips_info

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = torch.tensor(config.LABELS[self.labels[idx]], dtype=torch.long)

        if self.temporal:
            frames_paths = self.paths[idx]
            frames_info = self.clips_info[idx]      # clip's frames info
            clip = []
            for path_idx in range(len(frames_paths)):
                frame = Image.open(frames_paths[path_idx]).convert('RGB')
                frame_boxes_info = frames_info[path_idx]    # frame's boxes info
                crops = []
                for box_info in frame_boxes_info:
                    processed_crop = self.processor(frame.crop(box_info.bounding_box))
                    crops.append(processed_crop)
                crops = torch.stack(crops)      # (players, C, H, W)
                clip.append(crops)
            clip = torch.stack(clip)    # (frames, players, C, H, W)
            return clip, label

        else:
            # get target frame
            img_path = self.paths[idx][config.TARGET_FRAME_IDX]
            img_boxes = self.clips_info[idx][config.TARGET_FRAME_IDX]
            img = Image.open(img_path).convert('RGB')

            crops = []
            for box_info in img_boxes:
                processed_crop = self.processor(img.crop(box_info.bounding_box))
                crops.append(processed_crop)
            crops = torch.stack(crops)      # (players x C x H x W)
            return crops, label





class ImageLevelDataset(Dataset):
    def __init__(self, paths, labels, processor=transforms.ToTensor(), temporal=True):
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.temporal = temporal

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = torch.tensor(config.LABELS[self.labels[idx]], dtype=torch.long)

        if self.temporal:
            frames_paths = self.paths[idx]
            clip = []
            for path_idx in range(len(frames_paths)):
                frame = Image.open(frames_paths[path_idx]).convert('RGB')
                processed_frame = self.processor(frame)
                clip.append(processed_frame)
            clip = torch.stack(clip)            # (frames, C, H, W)

            return clip, label

        else:
            img = Image.open(self.paths[idx][config.TARGET_FRAME_IDX]).convert('RGB')     # target frame
            img = self.processor(img)       # (C, H, W)

            return img, label





def load_clips_and_labels(split:list, image_level=True):
    videos_annots = load_volleyball_dataset()
    clips, labels, clips_info = [], [], []
    for vid_id in split:
        video_path = os.path.join(config.VIDEO_ROOT_DIR, vid_id)

        if not os.path.isdir(video_path):
            continue

        clips_ids = os.listdir(video_path)
        clips_ids.sort()

        for clip_id in clips_ids:

            if not os.path.isdir(os.path.join(video_path, clip_id)):
                continue

            label = videos_annots[vid_id][clip_id]['label']
            labels.append(label)

            frames_ids = list(videos_annots[vid_id][clip_id]['frames_boxes_dct'].keys())
            frames_ids.sort()          # frames in each clip are in order

            if image_level:
                clip = [os.path.join(video_path, clip_id, f'{frame_id}.jpg') for frame_id in frames_ids]
                clips.append(clip)
            else:
                clip, clip_info = [], []
                for frame_id in frames_ids:
                    path = os.path.join(video_path, clip_id, f'{frame_id}.jpg')
                    clip.append(path)

                    # get frame info object and add it to clip info list
                    frame_info = videos_annots[vid_id][clip_id]['frames_boxes_dct'][frame_id]
                    frame_boxes_info = frame_info.boxes_info    # list of each frame's boxes info
                    clip_info.append(frame_boxes_info)

                clips.append(clip)
                clips_info.append(clip_info)

    if image_level:
        return clips, labels
    else:
        return clips, labels, clips_info