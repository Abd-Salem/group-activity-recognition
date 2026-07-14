import yaml, os

class CONFIG:
    '''
    preparing configurations with respect to yaml file
    '''
    def __init__(self, path='configs.yml'):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        self.DATASET_ROOT = data['dataset_root']
        self.VIDEO_ROOT = f'{self.DATASET_ROOT}/videos'
        self.FEATURES_ROOT_DIR = f'{self.DATASET_ROOT}/features'
        self.IMAGE_LEVEL_DIR = f'{self.FEATURES_ROOT_DIR}/image-level'
        self.PLAYER_LEVEL_DIR = f'{self.FEATURES_ROOT_DIR}/player-level'
        self.ANNOT_ROOT = f'{self.DATASET_ROOT}/all-annotations'
        self.TRACKING_ANNOTS = f'{self.DATASET_ROOT}/volleyball_tracking_annotations'
        self.LABELS = data['labels']

        self._create_dirs()

    def _create_dirs(self):
        dirs = [self.FEATURES_ROOT_DIR,self.IMAGE_LEVEL_DIR, self.PLAYER_LEVEL_DIR ,self.ANNOT_ROOT]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)