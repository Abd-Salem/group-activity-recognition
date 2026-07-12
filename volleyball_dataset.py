import os, torch
import numpy as np
from torch.utils.data import Dataset
from volleyball_annot_loader_utils import dataset_root, load_annotations, train_ids

cat = {
    'l-pass': 0,
    'r-pass': 1,
    'l-spike': 2,
    'r_spike': 3,
    'l_set': 4,
    'r_set': 5,
    'l_winpoint': 6,
    'r_winpoint': 7
}

class Volleyball_Image(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        feature = self.features[idx]
        label = cat[self.labels[idx]]

        return torch.from_numpy(feature).float(), torch.tensor(label, dtype=torch.long)


def load_feat_annot():
    features_root = f'{dataset_root}/features/image-level/resnet'
    vid_annots = load_annotations()

    X, y = [], []
    for vid_id in train_ids:
        vid_path = os.path.join(features_root, vid_id)
        if not os.path.isdir(vid_path):
            continue

        clips_dir = os.listdir(vid_path)
        clips_dir.sort()
        for t_frame in clips_dir:
            feature_path = os.path.join(vid_path, f'{t_frame}.npy')
            feature = np.load(feature_path)
            X.append(feature)
            y.append(vid_annots[vid_id][t_frame]['label'])

    return X, y


if __name__ == '__main__':
    pass