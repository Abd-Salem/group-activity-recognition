import torch
from torch.utils.data import Dataset
from helper_utils.configs import CONFIG
from torchvision import transforms
from PIL import Image
from datetime import datetime, timezone
from abc import ABC, abstractmethod



# === Abstract base ===
class VolleyballDatasetBase(Dataset, ABC):
    def __init__(self):
        super().__init__()


    def _load_frame(self, path):
        try:
            frame = Image.open(path).convert('RGB')
            return frame

        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            # get error logs
            timestamp = datetime.now(timezone.utc).isoformat()
            print(f'Error at: {timestamp}  |  Failed to load: {path}  |  '
                  f'Exceptions: {e}')

            raise RuntimeError(f"Frame failed to load after validation: {path}") from e


    @abstractmethod
    def __getitem__(self, idx):
        ...

    @abstractmethod
    def __len__(self):
        ...


# === Dataset implementations ===
class PersonLevelDataset(VolleyballDatasetBase):
    def __init__(self, paths, labels ,processor=transforms.ToTensor(), temporal=True, clips_info=None, config=None):
        super().__init__()

        self.config = CONFIG() if config is None else config
        self.paths = paths
        self.labels = [torch.tensor(self.config.LABELS[l], dtype=torch.long) for l in labels]
        self.processor = processor
        self.temporal = temporal
        self.clips_info = clips_info


    def __getitem__(self, idx):

        if self.temporal:
            frames_paths = self.paths[idx]
            frames_info = self.clips_info[idx]      # clip's frames info
            clip = []
            for path_idx in range(len(frames_paths)):
                frame = self._load_frame(frames_paths[path_idx])
                frame_boxes_info = frames_info[path_idx]    # frame's boxes info
                crops = []
                for box_info in frame_boxes_info:
                    processed_crop = self.processor(frame.crop(box_info.bounding_box))
                    crops.append(processed_crop)
                crops = torch.stack(crops)      # (players, C, H, W)
                clip.append(crops)
            clip = torch.stack(clip)    # (frames, players, C, H, W)
            return clip, self.labels[idx]

        else:
            # get target frame
            img_path = self.paths[idx][self.config.TARGET_FRAME_IDX]
            img = self._load_frame(img_path)

            img_boxes = self.clips_info[idx][self.config.TARGET_FRAME_IDX]
            crops = []
            for box_info in img_boxes:
                processed_crop = self.processor(img.crop(box_info.bounding_box))
                crops.append(processed_crop)
            crops = torch.stack(crops)      # (players x C x H x W)
            return crops, self.labels[idx]

    def __len__(self):
        return len(self.labels)


class ImageLevelDataset(VolleyballDatasetBase):
    def __init__(self, paths, labels, processor=transforms.ToTensor(), temporal=True,  config=None):
        super().__init__()

        self.config = CONFIG() if config is None else config
        self.paths = paths
        self.labels = [torch.tensor(self.config.LABELS[l], dtype=torch.long) for l in labels]
        self.processor = processor
        self.temporal = temporal

    def __getitem__(self, idx):

        if self.temporal:
            frames_paths = self.paths[idx]
            clip = []
            for path_idx in range(len(frames_paths)):
                frame = self._load_frame(frames_paths[path_idx])
                processed_frame = self.processor(frame)
                clip.append(processed_frame)
            clip = torch.stack(clip)            # (frames, C, H, W)

            return clip, self.labels[idx]

        else:
            img = self._load_frame(self.paths[idx][self.config.TARGET_FRAME_IDX])
            img = self.processor(img)       # (C, H, W)

            return img, self.labels[idx]

    def __len__(self):
        return len(self.labels)