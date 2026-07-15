import os, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from volleyball_annot_loader_utils import load_annotations, train_ids,save_annotations, config


class Bl1Dataset(Dataset):
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


def load_feat_annot(split):
    '''
    loading extracted features for split and corresponding labels
    :param split: train or eval or test
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
    X, y = load_feat_annot(split=train_ids)
    b1_dataset = Bl1Dataset(features=X, labels=y)
    dataloader = DataLoader(b1_dataset, batch_size=2, shuffle=True)
    for idx, (x, y) in enumerate(dataloader):
        print(f'batch_idx:{idx},    x={x},      y={y}')