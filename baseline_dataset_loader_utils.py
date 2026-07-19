import os, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from volleyball_annot_loader_utils import load_annotations, save_annotations, config
from PIL import Image


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float().squeeze()
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




def load_image_paths_labels(split):
    '''
    loading images paths and labels
    :param split: train or val or test
    :return X: paths
    :return y: labels
    '''

    videos_annots = load_annotations()
    X, y = [], []
    for vid_id in split:
        for clip_id in videos_annots[vid_id]:
            target_frame_path = os.path.join(videos_annots[vid_id][clip_id]['clip_dir_path'], f'{clip_id}.jpg')
            X.append(target_frame_path)
            y.append(videos_annots[vid_id][clip_id]['label'])

    return X, y


def load_feat_annot(split):
    '''
    loading extracted features for split and corresponding labels
    :param split: train or eval or test
    :return X: paths
    :return y: labels
    '''

    annots_file = f'{config.ANNOT_ROOT}/annots.pickle'

    # check annotations file existence
    if not os.path.exists(annots_file):
        save_annotations()

    vid_annots = load_annotations()
    X, y = [], []
    for vid_id in split:
        vid_path = os.path.join(config.VIDEO_ROOT, vid_id)

        if not os.path.isdir(vid_path):
            continue

        clips_dir = os.listdir(vid_path)
        clips_dir.sort()
        for t_frame in clips_dir:
            # if not t_frame in ['38025','51725','18360','20500','20525']:
            #     continue

            if not os.path.isdir(os.path.join(vid_path, t_frame)):
                continue

            feature_path = os.path.join(config.IMAGE_LEVEL_DIR, 'resnet', vid_id, f'{t_frame}.npy')
            feature = np.load(feature_path)
            X.append(feature)
            y.append(vid_annots[vid_id][t_frame]['label'])

    return X, y


if __name__ == '__main__':
    X, y = load_feat_annot(split=config.TRAIN_IDS)
    b1_dataset = FeatureDataset(features=X, labels=y)
    dataloader = DataLoader(b1_dataset, batch_size=2, shuffle=True)
    for idx, (x, y) in enumerate(dataloader):
        print(f'batch_idx:{idx},    x={x},      y={y}')