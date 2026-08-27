import yaml, os

class CONFIG:
    '''
    preparing configurations with respect to yaml file
    '''
    def __init__(self, path='configs.yml'):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        self.DATASET_ROOT_DIR = data['dataset_root']
        self.VIDEO_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/samples/videos'
        self.ANNOT_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/annotations'
        self.BALL_ROOT_DIR = f'{self.ANNOT_ROOT_DIR}/volleyball_ball_annotation'
        self.TRACKING_ANNOTS_ROOT_DIR = f'{self.ANNOT_ROOT_DIR}/volleyball_tracking_annotation'
        self.FEATURES_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/samples/features'

        self.ANNOT_SAVE_DIR = f'{self.ANNOT_ROOT_DIR}/all-annotations'
        self.IMAGE_LEVEL_DIR = f'{self.FEATURES_ROOT_DIR}/image-level'
        self.PLAYER_LEVEL_DIR = f'{self.FEATURES_ROOT_DIR}/player-level'

        self.LABELS = data['labels']
        self.TRAIN_IDS = data['train_ids']
        self.VAL_IDS = data['val_ids']
        self.TEST_IDS = data['test_ids']
        self.TARGET_FRAME_IDX = data['target_frame_idx']

        self.CUSTOM_KEY = lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)     # for dir sorting

        self._create_dirs()

    def _create_dirs(self):
        dirs = [self.FEATURES_ROOT_DIR,self.IMAGE_LEVEL_DIR, self.PLAYER_LEVEL_DIR ,self.ANNOT_SAVE_DIR]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)