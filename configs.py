import yaml, os

class CONFIG:
    '''
    preparing configurations with respect to yaml file
    '''
    def __init__(self, path='configs.yml'):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        self.DATASET_ROOT_DIR = data['dataset_root']
        self.VIDEO_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/videos_sample'
        self.BALL_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/volleyball_ball_annotation'
        self.ANNOT_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/all-annotations'
        self.FEATURES_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/features'
        self.IMAGE_LEVEL_DIR = f'{self.FEATURES_ROOT_DIR}/image-level'
        self.PLAYER_LEVEL_DIR = f'{self.FEATURES_ROOT_DIR}/player-level'
        self.TRACKING_ANNOTS_DIR = f'{self.DATASET_ROOT_DIR}/volleyball_tracking_annotations'
        self.LABELS = data['labels']
        self.TRAIN_IDS = data['train_ids']
        self.VAL_IDS = data['val_ids']
        self.TEST_IDS = data['test_ids']

        self._create_dirs()

    def _create_dirs(self):
        dirs = [self.FEATURES_ROOT_DIR,self.IMAGE_LEVEL_DIR, self.PLAYER_LEVEL_DIR ,self.ANNOT_ROOT_DIR]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)