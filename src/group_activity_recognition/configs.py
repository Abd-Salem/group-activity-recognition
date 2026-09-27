import yaml, os, torch
from pathlib import Path

class CONFIG:
    '''
    preparing configurations with respect to yaml file
    '''
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    

    def __init__(self, path='configs.yml'):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        self.ENV = data['environment']
        self.ROOT = data['roots'][self.ENV]

        if self.ENV == 'local':
            self.DATASET_ROOT_DIR = Path(self.ROOT) / data['local_dataset_dir_name']
            self.VIDEOS_DIR = Path(self.DATASET_ROOT_DIR) / f'samples/videos'
            self.TRACKING_ANNOTS_DIR = Path(self.DATASET_ROOT_DIR) / f'annotations/volleyball_tracking_annotation'
            self.BALL_ANNOT_DIR = Path(self.DATASET_ROOT_DIR) / f'volleyball_ball_annotation'
            self.FEATURES_DIR = Path(self.DATASET_ROOT_DIR) / f'extracted_features'
            self.ANNOT_SAVE_DIR = Path(self.ROOT) / f'saved_annotations'
            self.BACKBONE_DIR = Path(self.ROOT) / f'backbones'
            self.TEST_DIR = Path(self.ROOT) / f'test'

        elif self.ENV == 'kaggle':
            self.DATASET_ROOT_DIR = Path(self.ROOT) / data['kaggle_dataset_dir_name']
            self.VIDEOS_DIR = Path(self.DATASET_ROOT_DIR) / f'videos'
            self.TRACKING_ANNOTS_DIR = Path(self.DATASET_ROOT_DIR) / f'volleyball_tracking_annotation'
            self.BALL_ANNOT_DIR = Path(f'/kaggle/input/ball_annots')
            self.FEATURES_DIR = Path(f'/kaggle/working/extracted_features')
            self.ANNOT_SAVE_DIR = Path(f'/kaggel/working/saved_annotations')
            self.BACKBONE_DIR = Path(f'/kaggel/working/backbones')
            self.TEST_DIR = Path(f'/kaggel/working/test')

        self.IMAGE_LEVEL_DIR = f'{self.FEATURES_DIR}/image-level'
        self.PLAYER_LEVEL_DIR = f'{self.FEATURES_DIR}/player-level'

        self.LABELS = data['labels']
        self.ACTIONS = data['actions']
        self.TRAIN_IDS = data['train_ids']
        self.VAL_IDS = data['val_ids']
        self.TEST_IDS = data['test_ids']
        self.TARGET_FRAME_IDX = data['target_frame_idx']
        self.BATCH_SIZE = data['batch_size']
        self.OPTIMS = {'adam': torch.optim.Adam,
                       'adamw': torch.optim.AdamW,
                       'sgd' : torch.optim.SGD}

        self.CUSTOM_KEY = lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)     # for dir sorting

        self._create_dirs()

    def _create_dirs(self):
        dirs = [self.FEATURES_DIR,self.IMAGE_LEVEL_DIR, self.PLAYER_LEVEL_DIR ,self.ANNOT_SAVE_DIR, self.BACKBONE_DIR, self.TEST_DIR]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

    @staticmethod
    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")