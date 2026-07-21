import os, torch
import numpy as np
from torch.utils.data import Dataset
from volleyball_annot_loader_utils import  load_volleyball_dataset
from helper_utils.configs import CONFIG
from helper_utils.more_helpers import extract_features, check, load_extractor
from PIL import Image

# get configs
config = CONFIG()

class FeatureDataset(Dataset):
    def __init__(self, features, labels, image_level=True):
        super().__init__()
        self.features = features
        self.labels = labels
        self.image_leve = True


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        # 9 x 2k
        # 9 x 12 x 2k
        if self.image_leve:
            feature = torch.cat(self.features[idx])        # 9 x 2k
        else:
            feature =  torch.stack(self.features[idx])     # 9 x 12 x 2k
        label = torch.tensor(config.LABELS[self.labels[idx]], dtype=torch.long)

        return feature, label



class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, processor=None):
        self.paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.processor:
            img = self.processor(img)

        labels = torch.tensor(config.LABELS[self.labels[idx]], dtype=torch.long)

        return img, labels




def prepare_dataset_input_label(split, image_level=True, sequence=True, features=True):
    '''
    loading images paths and labels
    :param split: train or val or test
    :param image_level: True -> full image, False -> players' crops
    :param sequence: True -> stack frames as one input,   False -> get each frame as one input with clip label
    :param features: True -> load saved extracted features,  False -> load frames' path
    :return X: paths
    :return y: labels
    '''

    videos_annots = load_volleyball_dataset()

    if image_level:
        if sequence:
            if features:
                pass
            else:
                pass
        else:
            if features:
                pass
            else:
                X, y = [], []
                for vid_id in split:
                    for clip_id in videos_annots[vid_id]:

                        frames_ids = list(videos_annots[vid_id][clip_id]['frames_boxes_dct'].keys())
                        for frame_id in frames_ids:
                            frame_path = os.path.join(videos_annots[vid_id][clip_id]['clip_dir_path'],
                                                      f'{frame_id}.jpg')
                            X.append(frame_path)
                            y.append(videos_annots[vid_id][clip_id]['label'])

                return X, y





def load_extracted_features(split, image_level=True):
    '''
    loading extracted features for split and corresponding labels
    :param split: train or eval or test
    :return X: paths
    :return y: labels
    '''

    videos_annots = load_volleyball_dataset()
    X, y = [], []
    for vid_id in split:
        video_path = os.path.join(config.VIDEO_ROOT_DIR, vid_id)

        if not os.path.isdir(video_path):
            continue

        clips_ids = os.listdir(video_path)
        clips_ids.sort()
        for clip_id in clips_ids:

            if not os.path.isdir(os.path.join(video_path, clip_id)):
                continue

            if image_level:
                features_dir = os.path.join(config.IMAGE_LEVEL_DIR, 'resnet')
                frames_ids = videos_annots[vid_id][clip_id]['frames_boxes_dct']

                sequence = []
                for frame_id in frames_ids:
                    feature_path = os.path.join(features_dir, vid_id, f'{frame_id}.npy')
                    feature = np.load(feature_path)
                    sequence.append(torch.from_numpy(feature))

            else:
                features_dir = os.path.join(config.PLAYER_LEVEL_DIR, 'resnet')
                frames_ids = videos_annots[vid_id][clip_id]['frames_boxes_dct']

                sequence = []
                for frame_id in frames_ids:
                    feature_path = os.path.join(features_dir, vid_id, f'{frame_id}.npy')
                    feature = np.load(feature_path)
                    sequence.append(torch.from_numpy(feature).unsqueeze(0))

            X.append(sequence)
            y.append(videos_annots[vid_id][clip_id]['label'])

    return X, y


if __name__ == '__main__':
    check()         # versions and machines

    full_image = False      # full frame or crops

    model = load_extractor()        # extractor

    if full_image:
        output_root = f'{config.IMAGE_LEVEL_DIR}/resnet'
    else:
        output_root = f'{config.PLAYER_LEVEL_DIR}/resnet'

    # extract_features(config.VIDEO_ROOT_DIR, config.TRACKING_ANNOTS_ROOT_DIR, output_root
    #                  , model, full_image=full_image)   # extract features and save them

    x, y = load_extracted_features(['7', '10'], full_image)
    dataset = FeatureDataset(features=x, labels=y, image_level=full_image)
    feature, label = dataset.__getitem__(0)
    print(f'feature shape: {feature.shape}')
    print(f'label shape: {label.shape}')